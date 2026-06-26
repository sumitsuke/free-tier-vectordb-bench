"""Standalone probe for the Turso adapter against the real cloud DB.

Validates the risky SQL (vector32 insert, libsql_vector_idx build, vector_top_k
JOIN on rowid, vector_distance_cos, pragma page_count) on a tiny synthetic set
BEFORE running the full bench. Run:

    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m scripts.turso_probe
"""
from __future__ import annotations

import numpy as np
from dotenv import load_dotenv

from harness.adapters.turso_store import TursoStore

load_dotenv()

DIM = 384
rng = np.random.default_rng(0)
vecs = rng.standard_normal((6, DIM)).astype("float32")
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)  # normalize (cosine)
ids = [f"d{i}" for i in range(6)]

store = TursoStore()
try:
    print("[create]")
    store.create(DIM, "cosine")
    print("[upsert]")
    store.upsert(ids, vecs)
    print("  breakdown:", store.upsert_breakdown)
    print("[stats]", store.stats())
    # query with d0 itself -> expect d0 as nearest (score ~1.0)
    print("[search] q=d0")
    hits = store.search(vecs[0], 3)
    for h in hits:
        print("   ", h)
    assert hits and hits[0][0] == "d0", f"expected d0 first, got {hits[:1]}"
    assert abs(hits[0][1] - 1.0) < 1e-3, f"self score not ~1.0: {hits[0][1]}"
    print("[OK] adapter SQL validated against real Turso DB")
finally:
    store.teardown()
    store.close()
    print("[teardown] done")
