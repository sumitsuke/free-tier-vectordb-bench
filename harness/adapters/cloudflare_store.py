"""Cloudflare Vectorize adapter (v2 REST API, cosine).

The 4th DB and the "managed black-box" of the bench: a globally-distributed edge
vector index. We use the Vectorize v2 REST API directly (no Worker deploy needed)
so the harness stays uniform with the other adapters.

Notes / gotchas (why this adapter looks different):
- **Async indexing**: upsert is asynchronous — the call returns a `mutationId`
  immediately, but vectors are NOT queryable until Vectorize processes the
  mutation. We push all NDJSON batches (copy_s) then POLL /info until vectorCount
  catches up (index_s) before any query. Skipping this yields empty/partial
  results — the single biggest footgun.
- **Text ids** kept directly (like pgvector/Turso); no int mapping (cf. Qdrant).
- Cosine metric -> query `score` is already a similarity (higher = better).
- **No query-time ANN knob** (managed): tuning layer is N/A (like Turso DiskANN).
- **No cheap server-side timing**: the query response carries no processing-time
  field, so sample_server_ms returns None -> CF reports e2e only (honest, §5-1).
- **Stored-dims is the free-tier meter** (§7): stats() returns vectorCount and
  stored_dims = vectorCount * dim (CF bills/limits on dimensions, not bytes).
- Region is global/edge (no single AWS region) -> region label is "global (edge)".
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from harness.adapters.base import VectorStore


class CloudflareStore(VectorStore):
    name = "cloudflare"
    # Vectorize builds/propagates the index asynchronously server-side after upsert.
    upsert_strategy = "async server-side (Vectorize)"
    region = "global (edge)"

    def __init__(self):
        self.account = os.environ["CF_ACCOUNT_ID"]
        self.token = os.environ["CF_API_TOKEN"]
        self.index = os.environ.get("CF_VECTORIZE_INDEX", "arguana-minilm")
        self.base = (f"https://api.cloudflare.com/client/v4/accounts/{self.account}"
                     f"/vectorize/v2/indexes")
        self.h = {"Authorization": f"Bearer {self.token}"}
        self.upsert_breakdown: dict | None = None

    # --- HTTP helper with transient retry (429/5xx/gateway) ---
    def _req(self, method: str, path: str = "", *, json_body=None, data=None,
             ctype: str | None = None, tries: int = 5):
        import requests
        url = self.base + path
        headers = dict(self.h)
        if ctype:
            headers["Content-Type"] = ctype
        last = None
        for a in range(tries):
            try:
                r = requests.request(method, url, headers=headers, json=json_body,
                                     data=data, timeout=120)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"transient {r.status_code}: {r.text[:200]}")
                return r
            except Exception as e:  # noqa: BLE001
                last = e
                if a == tries - 1:
                    raise
                time.sleep(1.5 * (a + 1))
        raise last  # unreachable

    def _ok(self, r) -> dict:
        try:
            body = r.json()
        except ValueError:
            r.raise_for_status()
            raise
        if not body.get("success", False):
            raise RuntimeError(f"CF API error {r.status_code}: {body.get('errors')}")
        return body.get("result", {})

    def create(self, dim: int, metric: str = "cosine") -> None:
        if metric != "cosine":
            raise ValueError(f"this bench is cosine-only, got {metric}")
        # We do NOT delete+recreate: Vectorize index deletion is eventually-
        # consistent and a just-deleted name keeps returning 410 "index deleted"
        # for a while (a DELETE-then-upsert race that 410s mid-run, observed).
        # Since upsert overwrites by id and our id set is fixed (ArguAna corpus),
        # reusing an existing index is idempotent — so: reuse if present, else create.
        info = self._req("GET", f"/{self.index}/info")
        if info.status_code == 200:
            return  # exists with our dim/cosine config -> reuse as-is
        r = self._req("POST", "", json_body={
            "name": self.index,
            "config": {"dimensions": dim, "metric": "cosine"},
        })
        body = r.json() if r.content else {}
        errs = str(body.get("errors", ""))
        if not body.get("success") and "already exists" not in errs.lower():
            raise RuntimeError(f"CF create failed {r.status_code}: {body.get('errors')}")
        # readiness: /info must answer 200 before we upsert — fail loudly otherwise
        for _ in range(30):
            if self._req("GET", f"/{self.index}/info").status_code == 200:
                return
            time.sleep(1.0)
        raise RuntimeError("Vectorize index did not become ready within 30s after create")

    def upsert(self, ids, vecs, payloads=None) -> None:
        # 1) push NDJSON batches (copy_s). Vectorize caps a request at 1000 vectors.
        batch = 1000
        n = len(ids)
        last_mutation = None
        t0 = time.perf_counter()
        for i in range(0, n, batch):
            lines = []
            for j in range(i, min(i + batch, n)):
                lines.append(json.dumps({"id": str(ids[j]), "values": vecs[j].tolist()}))
            ndjson = "\n".join(lines)
            r = self._req("POST", f"/{self.index}/upsert", data=ndjson.encode("utf-8"),
                          ctype="application/x-ndjson")
            res = self._ok(r)
            last_mutation = res.get("mutationId") or last_mutation
        copy_s = time.perf_counter() - t0

        # 2) wait for async indexing to catch up (index_s). Gate on
        # processedUpToMutation == last upsert's mutationId when available (the
        # authoritative "this mutation is live" signal); fall back to vectorCount.
        # Raise on timeout so a partially-indexed run can NEVER be shipped as a
        # quietly-wrong recall row.
        want = len(set(str(x) for x in ids))
        t1 = time.perf_counter()
        deadline = t1 + 900
        ok = False
        last = None
        while time.perf_counter() < deadline:
            info = self._info()
            cnt = int(info.get("vectorCount") or 0)
            proc = info.get("processedUpToMutation")
            done = (proc == last_mutation) if (last_mutation and proc) else (cnt >= want)
            if done and cnt >= want:
                ok = True
                break
            sig = (cnt, proc)
            if sig != last:
                print(f"[cf] async indexing: vectorCount={cnt}/{want} processedUpTo={proc}")
                last = sig
            time.sleep(3.0)
        index_s = time.perf_counter() - t1
        if not ok:
            raise RuntimeError(
                f"Vectorize indexing did not catch up: vectorCount={cnt}/{want} "
                f"after {index_s:.0f}s — refusing to query a partial index")
        self.upsert_breakdown = {"copy_s": round(copy_s, 3), "index_s": round(index_s, 3)}

    def search(self, qvec, k: int) -> list[tuple[str, float]]:
        r = self._req("POST", f"/{self.index}/query", json_body={
            "vector": qvec.tolist(),
            "topK": k,
            "returnValues": False,
            "returnMetadata": "none",
        })
        res = self._ok(r)
        return [(m["id"], float(m["score"])) for m in res.get("matches", [])]

    def _info(self) -> dict:
        """Raw /info result: vectorCount, dimensions, processedUpToMutation."""
        return self._ok(self._req("GET", f"/{self.index}/info"))

    def stats(self) -> dict:
        # Vectorize's vectorCount is EVENTUALLY-CONSISTENT and non-monotonic — it
        # oscillates (e.g. 8674 -> 5000 -> 8674) for a while after a mutation even
        # though every vector is queryable (verified: ANN Recall stays ~1.0). A
        # single read can catch a transient low, so take the MAX over a few reads.
        cnt = dim = None
        for _ in range(3):
            res = self._info()
            c = res.get("vectorCount")
            dim = res.get("dimensions") or dim
            if c is not None:
                cnt = c if cnt is None else max(cnt, c)
            time.sleep(0.5)
        out = {"rows": cnt, "dimensions": dim}
        if cnt is not None and dim:
            out["stored_dims"] = int(cnt) * int(dim)  # the CF free-tier meter (§7)
        return out

    def teardown(self) -> None:
        try:
            self._req("DELETE", f"/{self.index}")
        except Exception:
            pass
