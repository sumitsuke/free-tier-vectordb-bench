"""Measure Turso/libSQL DiskANN index bloat AND whether the space is reclaimable
(RESEARCH C5: "VACUUM / 回収可否はM3で実測"). Run AFTER a full `--db turso --no-teardown`
run leaves the indexed 8,674-row table in place:

    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m scripts.turso_reclaim

Steps: report sizes (dbstat) -> try VACUUM (is it even supported on Turso?) ->
DROP the vector index -> sizes -> VACUUM -> sizes. Finally DROP TABLE (cleanup).
All sizes via dbstat (pragma_page_count is a useless constant on Turso Cloud).
"""

from __future__ import annotations

from dotenv import load_dotenv

from harness.adapters.turso_store import TursoStore

load_dotenv()
RAW_VEC_BYTES = 8674 * 384 * 4  # F32_BLOB payload, the honest baseline


def show(s: TursoStore, label: str) -> None:
    st = s.stats()
    tot, idx, tbl = st.get("total_bytes"), st.get("index_bytes"), st.get("table_bytes")
    print(f"\n[{label}] rows={st.get('rows')}")
    if tot is not None:
        print(f"   total={tot:,}  index={idx:,}  table={tbl:,} bytes")
        if idx:
            print(f"   index/raw-vectors = {idx / RAW_VEC_BYTES:.1f}x  (raw={RAW_VEC_BYTES:,})")
            print(f"   index/table       = {idx / tbl:.1f}x")
    else:
        print(f"   stats: {st}")


def run_sql(s: TursoStore, sql: str) -> None:
    try:
        s.client.execute(sql)
        print(f"   OK: {sql}")
    except Exception as e:  # noqa: BLE001
        print(f"   FAILED: {sql}\n        -> {e}")


def main() -> None:
    s = TursoStore()
    try:
        show(s, "after full run (indexed)")

        print("\n== VACUUM supported on Turso? ==")
        run_sql(s, "VACUUM")
        show(s, "after VACUUM (index still present)")

        print(f"\n== DROP the DiskANN index ({s.index}) ==")
        run_sql(s, f"DROP INDEX IF EXISTS {s.index}")
        show(s, "after DROP INDEX")

        print("\n== VACUUM after dropping index ==")
        run_sql(s, "VACUUM")
        show(s, "after DROP INDEX + VACUUM")
    finally:
        print("\n== cleanup: DROP TABLE ==")
        s.teardown()
        s.close()
        print("[done]")


if __name__ == "__main__":
    main()
