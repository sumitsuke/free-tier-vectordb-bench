"""One-off, idempotent migration: add `region` and `upsert_strategy` as structured
columns to results/summary.csv and backfill the pre-existing rows (which only
carried this info inside the free-text `note`). New columns let the same-region
control (Supabase vs Turso both ap-northeast-1) be checked mechanically.

Existing rows can't be re-run cheaply (Turso upsert alone is ~881s), so they are
backfilled from the known, already-recorded provenance per `db`. Reruns are safe:
rows that already have non-empty values are left untouched. Column order follows
bench.SUMMARY_FIELDS so future appends stay aligned.

    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m scripts.migrate_summary_columns
"""

from __future__ import annotations

import csv

from harness import config
from harness.bench import SUMMARY_FIELDS

# Provenance per db (matches the adapters' region/upsert_strategy attributes;
# regions from the recorded endpoints — Qdrant Oregon, Supabase/Turso Tokyo).
BACKFILL = {
    "qdrant": {"region": "us-west-2 (Oregon)", "upsert_strategy": "batch-upsert (brute-force <10k threshold)"},
    "supabase": {"region": "ap-northeast-1 (Tokyo)", "upsert_strategy": "post-hoc HNSW"},
    "turso": {"region": "ap-northeast-1 (Tokyo)", "upsert_strategy": "incremental-on-empty (DiskANN)"},
    "cf": {"region": "global (edge)", "upsert_strategy": "async server-side (Vectorize)"},
}


def main() -> None:
    path = config.RESULTS_DIR / "summary.csv"
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    changed = 0
    for r in rows:
        db = r.get("db", "")
        bf = BACKFILL.get(db)
        if not bf:
            # The db column carries the CLI name (qdrant/supabase/turso), not
            # adapter.name ('supabase_pgvector'). Warn loudly rather than silently
            # skip, so a future schema change can't quietly drop a row's backfill.
            print(f"[warn] row db={db!r} has no BACKFILL entry — region/upsert_strategy left as-is")
            continue
        for col, val in bf.items():
            if not (r.get(col) or "").strip():  # only fill blanks (idempotent)
                r[col] = val
                changed += 1

    # rewrite with the canonical column order; DictWriter fills any missing key ""
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in SUMMARY_FIELDS})

    print(f"[migrate] {len(rows)} rows rewritten, {changed} cells backfilled -> {path}")


if __name__ == "__main__":
    main()
