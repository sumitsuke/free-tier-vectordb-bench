"""Qdrant Cloud adapter (HNSW, cosine).

Notes:
- Qdrant point ids must be uint/UUID, but ArguAna ids are strings. We use the
  enumerated integer as the point id and stash the original id in the payload;
  search reads it back. (Only Qdrant needs this; pgvector/Turso/CF take text ids.)
- Cosine distance -> returned score is already a similarity (higher = better).
- `hnsw_ef` / `exact` are the tuning knobs for the "reach the ceiling" run (§7).
- Server-side time: the python client's `query_points` returns only `points`
  (no `time` field in qdrant-client 1.18.0; the old `search()` is gone), so we
  read the `time` field from a separate sampled REST call (`sample_server_ms`).
  Verified against the installed client; to be re-confirmed on a live cluster.
"""
from __future__ import annotations

import os

from harness.adapters.base import VectorStore, aws_region_label


class QdrantStore(VectorStore):
    name = "qdrant"
    # Qdrant's default optimizer only builds HNSW once a segment exceeds
    # indexing_threshold (10,000). At our N=8,674 NO HNSW is built -> queries are
    # exact brute-force (verified: results/qdrant_index_probe.txt, indexed=0).
    upsert_strategy = "batch-upsert (brute-force <10k threshold)"

    def __init__(self):
        from qdrant_client import QdrantClient

        self.url = os.environ["QDRANT_URL"]
        self.region = aws_region_label(self.url)
        self.api_key = os.environ.get("QDRANT_API_KEY") or None
        self.collection = os.environ.get("QDRANT_COLLECTION", "arguana_minilm")
        self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=120)
        self.hnsw_ef: int | None = None   # set by orchestrator for tuning runs
        self.exact: bool = False

    def create(self, dim: int, metric: str = "cosine") -> None:
        from qdrant_client.models import Distance, VectorParams

        if metric != "cosine":
            raise ValueError(f"this bench is cosine-only, got {metric}")
        try:
            self.client.delete_collection(self.collection)
        except Exception:
            pass
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def upsert(self, ids, vecs, payloads=None) -> None:
        from qdrant_client.models import PointStruct

        batch = 256
        n = len(ids)
        for i in range(0, n, batch):
            pts = [
                PointStruct(
                    id=j,
                    vector=vecs[j].tolist(),
                    payload={"doc_id": ids[j]},
                )
                for j in range(i, min(i + batch, n))
            ]
            self.client.upsert(collection_name=self.collection, points=pts, wait=True)

    def search(self, qvec, k: int) -> list[tuple[str, float]]:
        from qdrant_client.models import SearchParams

        params = None
        if self.exact:
            params = SearchParams(exact=True)
        elif self.hnsw_ef is not None:
            params = SearchParams(hnsw_ef=self.hnsw_ef)

        res = self.client.query_points(
            collection_name=self.collection,
            query=qvec.tolist(),
            limit=k,
            with_payload=True,
            search_params=params,
        )
        return [(p.payload["doc_id"], float(p.score)) for p in res.points]

    def sample_server_ms(self, qvec, k: int) -> float | None:
        """Server-side time via the REST query endpoint, which returns `time`
        (seconds). Separate sampled path (the grpc/python client omits it)."""
        import requests

        params: dict = {}
        if self.exact:
            params["exact"] = True
        elif self.hnsw_ef is not None:
            params["hnsw_ef"] = self.hnsw_ef
        # with_payload=True to match the e2e search() (which fetches doc_id),
        # so server-side time measures the same work as the e2e path.
        body = {"query": qvec.tolist(), "limit": k, "with_payload": True}
        if params:
            body["params"] = params
        headers = {"api-key": self.api_key} if self.api_key else {}
        r = requests.post(
            f"{self.url}/collections/{self.collection}/points/query",
            json=body, headers=headers, timeout=60,
        )
        r.raise_for_status()
        t = r.json().get("time")
        return float(t) * 1000.0 if t is not None else None

    def stats(self) -> dict:
        info = self.client.get_collection(self.collection)
        return {
            "points": getattr(info, "points_count", None),
            "vectors": getattr(info, "vectors_count", None),
            "status": str(getattr(info, "status", "")),
        }

    def teardown(self) -> None:
        try:
            self.client.delete_collection(self.collection)
        except Exception:
            pass
