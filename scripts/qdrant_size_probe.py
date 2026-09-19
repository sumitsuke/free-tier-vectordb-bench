"""Investigate whether Qdrant (Cloud free tier, client 1.18) exposes a per-
collection BYTE size, so consumption.md can honestly say whether the RAM-bound
free tier's usage is measurable.

IMPORTANT (audit 2026-06-25): an earlier version ran against an ALREADY-TORN-DOWN
collection, so steps 1-2 just 404'd and the "structurally unmeasurable" claim
rested on an absent collection (absence-of-collection != absence-of-field). This
version CREATES and POPULATES a live collection with the real 8,674 ArguAna
vectors, probes every surface against live data, persists the evidence to
results/qdrant_size_probe.txt, then tears the collection down.

Surfaces probed:
  1. client.get_collection() — ALL CollectionInfo fields + any byte-ish field
  2. REST GET /collections/{name}              (raw CollectionInfo json)
  3. REST GET /telemetry?details_level=10      (per-collection block: segments etc.)
  4. REST GET /metrics                          (Prometheus *_bytes)
Plus a node resident-memory delta around the insert (node-level, NOT collection-
attributable — reported only as an order-of-magnitude footnote).

    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m scripts.qdrant_size_probe
"""

from __future__ import annotations

import os
import re

import numpy as np
import requests
from dotenv import load_dotenv

from harness import config

load_dotenv()

URL = os.environ["QDRANT_URL"].rstrip("/")
KEY = os.environ.get("QDRANT_API_KEY") or None
# Use a DISTINCT name so we never collide with / clobber a real bench collection.
COLL = "arguana_sizeprobe"
H = {"api-key": KEY} if KEY else {}
OUT = config.RESULTS_DIR / "qdrant_size_probe.txt"

_lines: list[str] = []


def out(s: str = "") -> None:
    print(s)
    _lines.append(s)


def get(path: str, **params):
    r = requests.get(f"{URL}{path}", headers=H, params=params, timeout=120)
    return r.status_code, r


def byte_keys(obj, path="") -> list[str]:
    """Paths whose key mentions byte/disk/ram/size/memory with a scalar value."""
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}"
            if re.search(r"byte|disk|ram|size|memory", str(k), re.I) and not isinstance(v, (dict, list)):
                hits.append(f"{p} = {v}")
            hits += byte_keys(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:5]):
            hits += byte_keys(v, f"{path}[{i}]")
    return hits


def resident_bytes() -> int | None:
    code, r = get("/telemetry", details_level=3)
    if code != 200:
        return None
    mem = r.json().get("result", {}).get("memory") or {}
    return mem.get("resident_bytes")


def main() -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    out(f"# qdrant_size_probe (LIVE collection) — {URL}")
    out(f"# collection={COLL}\n")

    client = QdrantClient(url=URL, api_key=KEY, timeout=120)
    vecs = np.load(config.CORPUS_VECS_NPY)
    n = len(vecs)
    out(f"loaded {n} real corpus vectors, dim={vecs.shape[1]}")

    rss_before = resident_bytes()
    try:
        # fresh collection
        try:
            client.delete_collection(COLL)
        except Exception:
            pass
        client.create_collection(
            collection_name=COLL,
            vectors_config=VectorParams(size=vecs.shape[1], distance=Distance.COSINE),
        )
        batch = 256
        for i in range(0, n, batch):
            pts = [
                PointStruct(id=j, vector=vecs[j].tolist(), payload={"doc_id": str(j)})
                for j in range(i, min(i + batch, n))
            ]
            client.upsert(collection_name=COLL, points=pts, wait=True)
        out(f"upserted {n} vectors into live collection\n")
        rss_after = resident_bytes()

        # 1. client CollectionInfo — ALL fields
        out("== 1. client.get_collection() — every field ==")
        info = client.get_collection(COLL)
        fields = list(getattr(info, "model_fields", {}) or {f: None for f in dir(info) if not f.startswith("_")})
        out(f"   CollectionInfo fields ({len(fields)}): {fields}")
        byteish = [f for f in fields if re.search(r"byte|disk|ram|size|memory", f, re.I)]
        out(f"   byte-ish fields: {byteish or 'NONE'}")
        out(
            f"   points_count={getattr(info, 'points_count', None)} "
            f"segments_count={getattr(info, 'segments_count', None)} "
            f"indexed_vectors_count={getattr(info, 'indexed_vectors_count', None)}"
        )

        # 2. raw /collections json
        out("\n== 2. GET /collections/{name} (raw json) ==")
        code, r = get(f"/collections/{COLL}")
        out(f"   status {code}")
        if code == 200:
            out(f"   byte-ish keys: {byte_keys(r.json().get('result', {})) or 'NONE'}")

        # 3. telemetry — the PER-COLLECTION block (live, non-empty now)
        out("\n== 3. GET /telemetry?details_level=10 (per-collection block) ==")
        code, r = get("/telemetry", details_level=10)
        out(f"   status {code}")
        if code == 200:
            res = r.json().get("result", {})
            colls = (res.get("collections") or {}).get("collections")
            out(
                f"   collections.collections present: {colls is not None}, "
                f"count={len(colls) if isinstance(colls, list) else 'n/a'}"
            )
            if colls:
                ck = byte_keys(colls[0], ".collections[0]")
                out(f"   per-collection byte-ish keys: {ck or 'NONE'}")
            node = byte_keys(res.get("memory") or {}, ".memory")
            out(f"   node-level memory keys (NOT collection-attributable): {node or 'NONE'}")

        # 4. prometheus
        out("\n== 4. GET /metrics (Prometheus *_bytes) ==")
        code, r = get("/metrics")
        out(f"   status {code}")
        if code == 200:
            bl = [
                ln
                for ln in r.text.splitlines()
                if re.search(r"byte|disk|ram|memory", ln, re.I) and not ln.startswith("#")
            ]
            out(f"   *_bytes lines ({len(bl)}): {bl[:12]}")

        # node RSS delta (node-level only)
        if rss_before is not None and rss_after is not None:
            out("\n== node resident_bytes delta (node-level, NOT collection-attributable) ==")
            out(f"   before={rss_before:,} after={rss_after:,} delta={rss_after - rss_before:,} bytes")

        out("\n== CONCLUSION ==")
        out(
            "   A per-collection BYTE total was found on a surface: "
            "see byte-ish results above. If all four are NONE for collection scope, "
            "Qdrant exposes no per-collection size even on a LIVE, populated collection."
        )
    finally:
        try:
            client.delete_collection(COLL)
            out("\n[teardown] dropped probe collection")
        except Exception as e:
            out(f"\n[teardown] failed: {e}")
        OUT.write_text("\n".join(_lines), encoding="utf-8")
        print(f"\n[written] {OUT}")


if __name__ == "__main__":
    main()
