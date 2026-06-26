"""Critical accuracy check (2026-06-25): did Qdrant actually use HNSW at N=8,674,
or did it brute-force? Qdrant's default `indexing_threshold` is 10,000 vectors —
segments below it are kept UNINDEXED and searched exhaustively (exact), so a
near-perfect ANN Recall there is exact-by-construction, NOT evidence of HNSW
fidelity. The benchmark's QdrantStore.create() uses the DEFAULT config, so this
matters for how the article frames "Qdrant reaches the ceiling".

Two collections, both with the real 8,674 ArguAna vectors:
  A) DEFAULT config (== QdrantStore): read indexing_threshold, wait for the
     optimizer, report indexed_vectors_count. If it stays 0 -> brute force.
  B) indexing_threshold=1 (force HNSW): wait until indexed_vectors_count==8674,
     then read sizes WITH the HNSW graph (vectors+ram+disk usage) -> the
     complete per-collection footprint the size-probe couldn't capture.

    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m scripts.qdrant_index_probe
"""
from __future__ import annotations

import os
import time

import numpy as np
import requests
from dotenv import load_dotenv

from harness import config

load_dotenv()
URL = os.environ["QDRANT_URL"].rstrip("/")
KEY = os.environ.get("QDRANT_API_KEY") or None
H = {"api-key": KEY} if KEY else {}
OUT = config.RESULTS_DIR / "qdrant_index_probe.txt"
_lines: list[str] = []


def out(s=""):
    print(s)
    _lines.append(str(s))


def seg_sizes(name):
    """Sum per-segment vectors/ram/disk bytes for a collection from telemetry."""
    r = requests.get(f"{URL}/telemetry", headers=H, params={"details_level": 10}, timeout=60)
    colls = (r.json().get("result", {}).get("collections") or {}).get("collections") or []
    for c in colls:
        # match by shard collection id if present; else take the only one
        shards = c.get("shards") or []
        vec = ram = disk = 0
        for sh in shards:
            loc = sh.get("local") or {}
            vec += loc.get("vectors_size_bytes", 0) or 0
            for seg in (loc.get("segments") or []):
                info = seg.get("info") or {}
                ram += info.get("ram_usage_bytes", 0) or 0
                disk += info.get("disk_usage_bytes", 0) or 0
        if shards:
            return {"vectors_size_bytes": vec, "ram_usage_bytes": ram, "disk_usage_bytes": disk}
    return {}


def make(client, name, vecs, threshold):
    from qdrant_client.models import Distance, OptimizersConfigDiff, PointStruct, VectorParams
    try:
        client.delete_collection(name)
    except Exception:
        pass
    kw = {}
    if threshold is not None:
        kw["optimizers_config"] = OptimizersConfigDiff(indexing_threshold=threshold)
    client.create_collection(name, vectors_config=VectorParams(size=vecs.shape[1],
                                                               distance=Distance.COSINE), **kw)
    n = len(vecs)
    for i in range(0, n, 256):
        pts = [PointStruct(id=j, vector=vecs[j].tolist(), payload={"doc_id": str(j)})
               for j in range(i, min(i + 256, n))]
        client.upsert(name, points=pts, wait=True)


def wait_indexed(client, name, want, secs):
    t0 = time.perf_counter()
    last = -1
    while time.perf_counter() - t0 < secs:
        info = client.get_collection(name)
        idx = info.indexed_vectors_count
        st = str(info.status)
        if idx != last:
            out(f"   ...indexed={idx}/{want} status={st} segments={info.segments_count}")
            last = idx
        if idx and idx >= want and "green" in st.lower():
            return info
        time.sleep(2.0)
    return client.get_collection(name)


def main():
    from qdrant_client import QdrantClient
    client = QdrantClient(url=URL, api_key=KEY, timeout=120)
    vecs = np.load(config.CORPUS_VECS_NPY)
    want = len(vecs)
    out(f"# qdrant_index_probe — N={want} real vectors, default indexing_threshold question\n")

    A, B = "arguana_idxprobe_default", "arguana_idxprobe_forced"
    try:
        # --- A) DEFAULT config (mirrors QdrantStore.create) ---
        out("== A) DEFAULT config (== benchmark QdrantStore) ==")
        make(client, A, vecs, threshold=None)
        info = client.get_collection(A)
        oc = info.config.optimizer_config
        thr = getattr(oc, "indexing_threshold", None)
        out(f"   indexing_threshold (default) = {thr}")
        info = wait_indexed(client, A, want, secs=60)
        out(f"   FINAL: points={info.points_count} indexed={info.indexed_vectors_count} "
            f"segments={info.segments_count}")
        brute = (info.indexed_vectors_count or 0) == 0
        out(f"   => {'BRUTE-FORCE (no HNSW built; ANN Recall is exact-by-construction)' if brute else 'HNSW BUILT'}"
            f"  [threshold={thr}, segment≈{want//max(1,info.segments_count)} < {thr}]")
        out(f"   sizes (default): {seg_sizes(A)}")

        # --- B) force HNSW (indexing_threshold=1) ---
        out("\n== B) indexing_threshold=1 (force HNSW) -> complete footprint ==")
        make(client, B, vecs, threshold=1)
        info = wait_indexed(client, B, want, secs=180)
        out(f"   FINAL: points={info.points_count} indexed={info.indexed_vectors_count} "
            f"segments={info.segments_count} status={info.status}")
        sz = seg_sizes(B)
        out(f"   sizes WITH HNSW: {sz}")
        if sz.get("vectors_size_bytes"):
            raw = want * vecs.shape[1] * 4
            tot = sz["vectors_size_bytes"] + sz.get("ram_usage_bytes", 0) + sz.get("disk_usage_bytes", 0)
            out(f"   raw vectors={raw:,}  vectors_size={sz['vectors_size_bytes']:,}  "
                f"ram={sz.get('ram_usage_bytes',0):,}  disk={sz.get('disk_usage_bytes',0):,}")
    finally:
        for nm in (A, B):
            try:
                client.delete_collection(nm)
            except Exception:
                pass
        OUT.write_text("\n".join(_lines), encoding="utf-8")
        out(f"\n[written] {OUT}")


if __name__ == "__main__":
    main()
