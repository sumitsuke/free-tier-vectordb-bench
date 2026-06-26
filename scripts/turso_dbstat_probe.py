"""Probe whether Turso/libSQL exposes per-object byte sizes (dbstat) so we can
isolate the DiskANN index size (index_bytes) from the table — for the bloat story.

Creates a tiny table + vector index, then lists sqlite_master objects and, if the
dbstat virtual table is available, per-object page bytes. Run:

    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m scripts.turso_dbstat_probe
"""
from __future__ import annotations

import numpy as np
from dotenv import load_dotenv

from harness.adapters.turso_store import TursoStore

load_dotenv()

DIM = 384
rng = np.random.default_rng(0)
vecs = rng.standard_normal((50, DIM)).astype("float32")
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
ids = [f"d{i}" for i in range(50)]

s = TursoStore()
c = s.client
try:
    s.create(DIM, "cosine")
    s.upsert(ids, vecs)
    print("[rows]", c.execute(f"SELECT count(*) AS n FROM {s.table}").rows[0]["n"])

    print("\n[sqlite_master objects]")
    rs = c.execute("SELECT type, name, tbl_name FROM sqlite_master ORDER BY type, name")
    for r in rs.rows:
        print(f"   {r['type']:8} {r['name']:40} (tbl={r['tbl_name']})")

    print("\n[dbstat per-object pgsize]")
    try:
        rs = c.execute("SELECT name, SUM(pgsize) AS bytes, COUNT(*) AS pages "
                       "FROM dbstat GROUP BY name ORDER BY bytes DESC")
        for r in rs.rows:
            print(f"   {r['name']:40} {r['bytes']:>12} bytes  ({r['pages']} pages)")
    except Exception as e:
        print("   dbstat unavailable:", e)

    print("\n[total]", s.stats())
finally:
    s.teardown()
    s.close()
    print("[teardown] done")
