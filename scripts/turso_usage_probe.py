"""Fetch Turso's real consumption from the Platform API and persist it, so the
§7 'measured' numbers (rows_read / rows_written / storage_bytes) are reproducible
like the rest of consumption.md instead of being hand-copied constants.

    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m scripts.turso_usage_probe

Writes results/turso_usage.json (raw 'total' usage block + a fetched_at stamp
passed in via TURSO_USAGE_DATE env, default unset). consumption_report.py reads
this file if present and falls back to its built-in constants otherwise (so the
report still regenerates offline). The Platform token (TURSO_PLATFORM_TOKEN) is
management-scoped and only used here, never by the benchmark harness.
"""

from __future__ import annotations

import json
import os

import requests
from dotenv import load_dotenv

from harness import config

load_dotenv()

ORG = os.environ.get("TURSO_ORG", "<your-org>")
DB = os.environ.get("TURSO_DB_NAME", "arguana-bench")
TOKEN = os.environ["TURSO_PLATFORM_TOKEN"]
OUT = config.RESULTS_DIR / "turso_usage.json"


def main() -> None:
    url = f"https://api.turso.tech/v1/organizations/{ORG}/databases/{DB}/usage"
    r = requests.get(url, headers={"Authorization": f"Bearer {TOKEN}"}, timeout=60)
    r.raise_for_status()
    total = r.json().get("total", {})
    # date is not derivable in-script (no clock); pass via env to stamp the artifact.
    record = {
        "fetched_at": os.environ.get("TURSO_USAGE_DATE", ""),
        "org": ORG,
        "database": DB,
        "rows_read": total.get("rows_read"),
        "rows_written": total.get("rows_written"),
        "storage_bytes": total.get("storage_bytes"),
        "bytes_synced": total.get("bytes_synced"),
        "reads_free_per_mo": 500_000_000,
        "writes_free_per_mo": 10_000_000,
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"\n[written] {OUT}")


if __name__ == "__main__":
    main()
