"""Guard against the §7 numbers drifting between the generated consumption.md and
the hand-typed REPORT.md §2.6 table (audit improvement I1: single source of truth).

consumption_report.py emits results/consumption_values.json (the load-bearing
display tokens). This script asserts every token appears inside REPORT.md's
"## 2.6" section. Run AFTER scripts.consumption_report; exits non-zero on drift,
so it can gate a pre-publish check.

    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m scripts.check_consumption_consistency
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VALUES = ROOT / "results" / "consumption_values.json"
REPORT = ROOT / "REPORT.md"


def section_2_6(text: str) -> str:
    """Slice REPORT.md from '## 2.6' to the next top-level '## ' heading, so a
    bare token like '49' is checked against §2.6 specifically, not the whole doc."""
    start = text.find("## 2.6")
    if start < 0:
        return ""
    rest = text[start + 6:]
    nxt = rest.find("\n## ")
    return rest if nxt < 0 else rest[:nxt]


def main() -> int:
    if not VALUES.exists():
        print(f"[error] {VALUES} missing — run scripts.consumption_report first")
        return 2
    tokens = json.loads(VALUES.read_text(encoding="utf-8"))["report_tokens"]
    sec = section_2_6(REPORT.read_text(encoding="utf-8"))
    if not sec:
        print("[error] REPORT.md has no '## 2.6' section")
        return 2

    missing = [t for t in tokens if t not in sec]
    if missing:
        print(f"[DRIFT] REPORT.md §2.6 is missing {len(missing)} canonical token(s) "
              f"from consumption.md:")
        for t in missing:
            print(f"   - {t!r}")
        print("→ update REPORT.md §2.6 to match results/consumption.md (regenerate first).")
        return 1
    print(f"[ok] all {len(tokens)} canonical §7 tokens present in REPORT.md §2.6 "
          f"(consumption.md and REPORT agree)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
