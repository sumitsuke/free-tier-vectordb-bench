"""Supabase / pgvector adapter (HNSW, cosine).

Notes:
- Table holds text ids directly. Index is built AFTER the bulk COPY (building
  HNSW once on full data is far faster than incremental inserts).
- `<=>` is cosine distance; we return similarity = 1 - distance (higher = better).
- `ef_search` is the tuning knob (SET hnsw.ef_search) for the ceiling run (§7).
- Server-side time via EXPLAIN (ANALYZE) "Execution Time" — sampled, not in the
  hot loop, to feed the e2e-vs-server split (§5-1).
- stats() reports table + index byte sizes for the consumption table (§8).
"""

from __future__ import annotations

import os
import time

import numpy as np

from harness.adapters.base import VectorStore, aws_region_label


def _vec_literal(vec: np.ndarray) -> str:
    # .9g round-trips float32 losslessly, so pgvector sees the *same* vector that
    # numpy (exact-kNN ceiling) and Qdrant (full float) see -> identical-vector
    # control condition stays exact across DBs.
    return "[" + ",".join(f"{x:.9g}" for x in vec.tolist()) + "]"


class PgvectorStore(VectorStore):
    name = "supabase_pgvector"
    # HNSW is built once, post-hoc, over the full table after the bulk COPY.
    upsert_strategy = "post-hoc HNSW"

    def __init__(self):
        import psycopg

        self.dsn = os.environ["SUPABASE_DB_URL"]
        self.region = aws_region_label(self.dsn)
        self.table = os.environ.get("SUPABASE_TABLE", "docs_arguana")
        self.index = f"{self.table}_hnsw"
        self.conn = psycopg.connect(self.dsn, autocommit=True)
        self.ef_search: int | None = None  # set by orchestrator for tuning runs
        self.dim: int | None = None
        # split insert vs index-build time so "TPS" isn't mis-read (audit H)
        self.upsert_breakdown: dict | None = None

    def create(self, dim: int, metric: str = "cosine") -> None:
        if metric != "cosine":
            raise ValueError(f"this bench is cosine-only, got {metric}")
        self.dim = dim
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"DROP TABLE IF EXISTS {self.table}")
            cur.execute(f"CREATE TABLE {self.table} (id text PRIMARY KEY, embedding vector({dim}))")

    def upsert(self, ids, vecs, payloads=None) -> None:
        with self.conn.cursor() as cur:
            t0 = time.perf_counter()
            with cur.copy(f"COPY {self.table} (id, embedding) FROM STDIN") as copy:
                for i in range(len(ids)):
                    copy.write_row((ids[i], _vec_literal(vecs[i])))
            copy_s = time.perf_counter() - t0
            # Single-threaded HNSW build -> deterministic-ish graph for the
            # "reproducible" claim (parallel build varies the graph & ANN recall).
            cur.execute("SET max_parallel_maintenance_workers = 0")
            t1 = time.perf_counter()
            # build HNSW once on the full data (cosine ops)
            cur.execute(f"CREATE INDEX {self.index} ON {self.table} USING hnsw (embedding vector_cosine_ops)")
            index_s = time.perf_counter() - t1
            cur.execute(f"ANALYZE {self.table}")
        # report insert vs index-build separately ("Tursoは挿入が遅い" 等の誤読回避)
        self.upsert_breakdown = {"copy_s": round(copy_s, 3), "index_s": round(index_s, 3)}

    def _maybe_set_ef(self, cur) -> None:
        if self.ef_search is not None:
            cur.execute(f"SET hnsw.ef_search = {int(self.ef_search)}")

    def search(self, qvec, k: int) -> list[tuple[str, float]]:
        lit = _vec_literal(qvec)
        with self.conn.cursor() as cur:
            self._maybe_set_ef(cur)
            cur.execute(
                f"SELECT id, 1 - (embedding <=> %s::vector) AS sim "
                f"FROM {self.table} ORDER BY embedding <=> %s::vector LIMIT %s",
                (lit, lit, k),
            )
            return [(row[0], float(row[1])) for row in cur.fetchall()]

    def sample_server_ms(self, qvec, k: int) -> float | None:
        """Server-side execution time via EXPLAIN ANALYZE (sampled, separate path).

        Caveat (reported in the article): EXPLAIN ANALYZE includes planning +
        instrumentation overhead, and repeated runs warm shared_buffers, so this
        trends optimistic vs cold e2e. Use p50 primarily; p95 here is a guide."""
        lit = _vec_literal(qvec)
        with self.conn.cursor() as cur:
            self._maybe_set_ef(cur)
            # TIMING OFF removes per-node instrumentation overhead; total
            # "Execution Time" is still reported. Buffers are already warm here
            # (the e2e pass ran every query 3x first), so this is a warm number.
            cur.execute(
                f"EXPLAIN (ANALYZE, TIMING OFF, FORMAT JSON) "
                f"SELECT id FROM {self.table} ORDER BY embedding <=> %s::vector LIMIT %s",
                (lit, k),
            )
            plan = cur.fetchone()[0]
            try:
                return float(plan[0]["Execution Time"])  # milliseconds
            except (KeyError, IndexError, TypeError):
                return None

    def stats(self) -> dict:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self.table}")
            rows = cur.fetchone()[0]
            cur.execute("SELECT pg_total_relation_size(%s)", (self.table,))
            total = cur.fetchone()[0]
            try:
                cur.execute("SELECT pg_relation_size(%s)", (self.index,))
                idx = cur.fetchone()[0]
            except Exception:
                idx = None
        return {"rows": rows, "total_bytes": total, "index_bytes": idx}

    def teardown(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.table}")

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
