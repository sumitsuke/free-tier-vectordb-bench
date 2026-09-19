"""Guard against the §7 numbers drifting between the generated consumption.md and
the hand-typed article (audit improvement I1: single source of truth).

consumption_report.py emits results/consumption_values.json (the load-bearing
display tokens). This script asserts every token appears in the PUBLIC canonical
ARTICLE.md (whole document: the tokens live in §3 and §7). REPORT.md is an
internal, git-ignored draft; when it is present its "## 2.6" table is checked
too, but its absence is not an error (2026-09-17: a reader running the public
repo hit a missing-file crash here — the gate was pointing at the private file).
Run AFTER scripts.consumption_report; exits non-zero on drift.

    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m scripts.check_consumption_consistency
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VALUES = ROOT / "results" / "consumption_values.json"
REPORT = ROOT / "REPORT.md"  # internal draft (git-ignored); optional
ARTICLE = ROOT / "ARTICLE.md"  # public canonical
# Tokens consumption.md prints but the article deliberately does not (the article
# shows the derived 49 runs/month, not the per-evaluation queried dims behind it).
ARTICLE_EXEMPT = {"3,870,720"}
# The article prints two values in a different form than consumption.md (rounded
# Supabase DB size; Qdrant shown as RAM %, not disk %). Checked via the printed form.
ARTICLE_ALIASES = {"32.8MB": "約33MB", "0.33%": "~3.8%"}


def section_2_6(text: str) -> str:
    """Slice REPORT.md from '## 2.6' to the next top-level '## ' heading, so a
    bare token like '49' is checked against §2.6 specifically, not the whole doc."""
    start = text.find("## 2.6")
    if start < 0:
        return ""
    rest = text[start + 6 :]
    nxt = rest.find("\n## ")
    return rest if nxt < 0 else rest[:nxt]


def main() -> int:
    if not VALUES.exists():
        print(f"[error] {VALUES} missing — run scripts.consumption_report first")
        return 2
    tokens = json.loads(VALUES.read_text(encoding="utf-8"))["report_tokens"]
    if not ARTICLE.exists():
        print(f"[error] {ARTICLE} missing")
        return 2
    art = ARTICLE.read_text(encoding="utf-8")
    missing = [t for t in tokens if ARTICLE_ALIASES.get(t, t) not in art and t not in ARTICLE_EXEMPT]
    if missing:
        print(f"[DRIFT] ARTICLE.md is missing {len(missing)} canonical token(s) from consumption.md:")
        for t in missing:
            print(f"   - {t!r}")
        print("→ update ARTICLE.md (§3 / §7) to match results/consumption.md (regenerate first).")
        return 1
    checked = len(tokens) - len([t for t in tokens if t in ARTICLE_EXEMPT])
    print(
        f"[ok] {checked}/{len(tokens)} canonical tokens present in ARTICLE.md "
        f"({sorted(ARTICLE_EXEMPT)} not printed by the article by design; "
        f"{ARTICLE_ALIASES} checked in the article's printed form)"
    )
    if REPORT.exists():
        sec = section_2_6(REPORT.read_text(encoding="utf-8"))
        miss2 = [t for t in tokens if t not in sec] if sec else tokens
        if miss2:
            print(f"[DRIFT] internal REPORT.md §2.6 is missing {len(miss2)} token(s): {miss2}")
            return 1
        print(f"[ok] internal REPORT.md §2.6 also agrees ({len(tokens)}/{len(tokens)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
