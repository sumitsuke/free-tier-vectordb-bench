"""Turso / libSQL adapter (DiskANN vector index, cosine).

The "odd one out" of the bench: a SQLite-family DB (libSQL) with a native vector
index (DiskANN via `libsql_vector_idx`). We talk to the *remote* Turso DB over
HTTP (libsql_client sync) so latency/size reflect the cloud instance, not a local
embedded replica.

Notes:
- Text ids stored directly (no int mapping). Vectors via `vector32('[...]')`.
- ANN search uses the DiskANN index: `vector_top_k(idx, vector32(q), k)` -> rowids,
  joined back to docs; score = 1 - vector_distance_cos (higher = better).
- Index built BEFORE bulk insert, on the EMPTY table: libSQL DiskANN is maintained
  incrementally per-INSERT. (Post-hoc CREATE INDEX over 8,674 rows on Turso Cloud
  either 502s at the HTTP gateway or returns ~instantly without a usable graph ->
  garbage search. So index_s is just DDL; the real index cost folds into copy_s.
  This is the OPPOSITE of pgvector's separate post-hoc HNSW build — see upsert().)
- stats() reports row count + total DB bytes (page_count*page_size) — the libSQL
  DiskANN index-bloat story (PLAN §4-1 / RESEARCH C5) is measured here at M3.
- No cheap server-side timing (no EXPLAIN ANALYZE equiv over the client) ->
  sample_server_ms returns None; Turso reports e2e only (honest, per §5-1).
"""
from __future__ import annotations

import os

import numpy as np

from harness.adapters.base import VectorStore, aws_region_label


def _vec_literal(vec: np.ndarray) -> str:
    # libSQL vector32() parses a JSON-style text array; .9g round-trips float32.
    return "[" + ",".join(f"{x:.9g}" for x in vec.tolist()) + "]"


class TursoStore(VectorStore):
    name = "turso"
    # DiskANN is maintained incrementally; the index is created on the EMPTY table
    # FIRST, then INSERTs build it row-by-row (post-hoc bulk build is broken here).
    upsert_strategy = "incremental-on-empty (DiskANN)"

    def __init__(self):
        from libsql_client import create_client_sync

        self.url = os.environ["TURSO_DATABASE_URL"]
        self.region = aws_region_label(self.url)
        self.auth = os.environ.get("TURSO_AUTH_TOKEN") or None
        self.table = os.environ.get("TURSO_TABLE", "docs_arguana")
        self.index = f"{self.table}_vidx"
        # libsql_client wants ws/wss/http/https; map libsql:// -> https (HTTP hrana)
        url = self.url
        if url.startswith("libsql://"):
            url = "https://" + url[len("libsql://"):]
        self.client = create_client_sync(url=url, auth_token=self.auth)
        self.upsert_breakdown: dict | None = None

    def create(self, dim: int, metric: str = "cosine") -> None:
        if metric != "cosine":
            raise ValueError(f"this bench is cosine-only, got {metric}")
        self.client.execute(f"DROP TABLE IF EXISTS {self.table}")
        self.client.execute(
            f"CREATE TABLE {self.table} (id TEXT PRIMARY KEY, "
            f"embedding F32_BLOB({dim}))"
        )

    def _exec_retry(self, fn, tries: int = 5):
        """Turso Cloud throws transient 502/503 under load; retry with backoff."""
        import time
        last = None
        for a in range(tries):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                # Turso Cloud transients seen in practice: HTTP 502/503 gateway,
                # generic SERVER_ERROR/stream resets, and a sporadic DiskANN
                # "failed to insert shadow row" (recovers on retry). Substring match
                # is fragile (see AUDIT) but libsql_client doesn't expose codes here.
                transient = ("502" in msg or "503" in msg or "SERVER_ERROR" in msg
                             or "stream" in msg.lower()
                             or "failed to insert shadow row" in msg)
                last = e
                if a == tries - 1 or not transient:
                    raise
                time.sleep(1.5 * (a + 1))
        raise last  # unreachable

    def upsert(self, ids, vecs, payloads=None) -> None:
        import time
        from libsql_client import Statement

        # libSQL DiskANN: create the vector index on the EMPTY table FIRST so it is
        # maintained incrementally per-INSERT. Building it post-hoc over 8,674 rows
        # on Turso Cloud either times out at the HTTP gateway (502) or returns in
        # ~0.03s without a usable graph (search then yields garbage) — both observed.
        # So index_s here is just the DDL on an empty table; the real index cost is
        # amortized into copy_s (unlike pgvector's separate post-hoc build).
        # IF NOT EXISTS makes the retry idempotent: if a 502 fires AFTER the server
        # actually created the index, the retry must not die on "already exists".
        t1 = time.perf_counter()
        self._exec_retry(lambda: self.client.execute(
            f"CREATE INDEX IF NOT EXISTS {self.index} ON {self.table} "
            f"(libsql_vector_idx(embedding))"
        ))
        index_s = time.perf_counter() - t1

        batch = 256
        n = len(ids)
        t0 = time.perf_counter()
        for i in range(0, n, batch):
            stmts = [
                Statement(
                    # OR REPLACE keeps a retried batch idempotent: if a batch partly
                    # applied before a transient error, the retry must not hit a
                    # PRIMARY KEY conflict on the already-inserted ids.
                    f"INSERT OR REPLACE INTO {self.table} (id, embedding) "
                    f"VALUES (?, vector32(?))",
                    [ids[j], _vec_literal(vecs[j])],
                )
                for j in range(i, min(i + batch, n))
            ]
            self._exec_retry(lambda s=stmts: self.client.batch(s))
        copy_s = time.perf_counter() - t0
        self.upsert_breakdown = {"copy_s": round(copy_s, 3), "index_s": round(index_s, 3)}

    def search(self, qvec, k: int) -> list[tuple[str, float]]:
        lit = _vec_literal(qvec)
        rs = self.client.execute(
            f"SELECT {self.table}.id AS id, "
            f"vector_distance_cos({self.table}.embedding, vector32(?)) AS dist "
            f"FROM vector_top_k('{self.index}', vector32(?), ?) AS vt "
            f"JOIN {self.table} ON {self.table}.rowid = vt.id "
            f"ORDER BY dist ASC",
            [lit, lit, k],
        )
        out = []
        for row in rs.rows:
            out.append((row["id"], 1.0 - float(row["dist"])))
        return out

    def stats(self) -> dict:
        rows = self.client.execute(f"SELECT count(*) AS c FROM {self.table}").rows[0]["c"]
        # Use dbstat (REAL used bytes per object). WARNING: pragma_page_count is
        # unreliable on Turso Cloud — it returns a CONSTANT ~695 MB regardless of row
        # count (mmap/max-size allocation), so it cannot measure data or index bloat.
        # We instead sum dbstat pgsize and isolate the DiskANN index, which libSQL
        # stores in shadow tables ({index}_shadow, {index}, libsql_vector_meta_shadow).
        # This gives index_bytes for parity with pgvector and grounds the bloat ratio.
        out: dict = {"rows": rows, "total_bytes": None, "index_bytes": None}
        try:
            rs = self.client.execute(
                "SELECT name, SUM(pgsize) AS b FROM dbstat GROUP BY name"
            )
            sizes = {row["name"]: (row["b"] or 0) for row in rs.rows}
            total = sum(sizes.values())
            idx = sum(b for n, b in sizes.items()
                      if n.startswith(self.index) or n.startswith("libsql_vector_meta"))
            out["total_bytes"] = total
            out["index_bytes"] = idx
            out["table_bytes"] = total - idx
        except Exception as e:  # noqa: BLE001
            out["dbstat_error"] = str(e)
        return out

    def teardown(self) -> None:
        try:
            self.client.execute(f"DROP TABLE IF EXISTS {self.table}")
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass
