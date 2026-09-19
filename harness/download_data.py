"""Download + extract the ArguAna BEIR dataset (no `beir` dependency).

We avoid the heavy `beir` package and just pull the canonical zip, which
contains corpus.jsonl / queries.jsonl / qrels/test.tsv. This keeps reader
reproducibility high (one small dependency-free script).

Usage:
    python -m harness.download_data
"""

from __future__ import annotations

import sys
import zipfile

from harness import config


def download(url: str, dest) -> None:
    import requests  # imported lazily so extraction works without the dep

    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    print(f"\r[download] {done / 1e6:.1f}/{total / 1e6:.1f} MB ({pct}%)", end="")
        print()


def main() -> int:
    config.DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    if config.CORPUS_JSONL.exists() and config.QRELS_TSV.exists():
        print(f"[skip] dataset already present at {config.DATASET_DIR}")
        return 0

    zip_path = config.DATASETS_DIR / f"{config.DATASET}.zip"
    if not zip_path.exists():
        download(config.ARGUANA_URL, zip_path)
    else:
        print(f"[skip] zip already downloaded: {zip_path}")

    print(f"[extract] -> {config.DATASETS_DIR}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(config.DATASETS_DIR)

    ok = config.CORPUS_JSONL.exists() and config.QUERIES_JSONL.exists() and config.QRELS_TSV.exists()
    if not ok:
        print("[error] expected files missing after extraction:", file=sys.stderr)
        print(f"  corpus:  {config.CORPUS_JSONL} exists={config.CORPUS_JSONL.exists()}", file=sys.stderr)
        print(f"  queries: {config.QUERIES_JSONL} exists={config.QUERIES_JSONL.exists()}", file=sys.stderr)
        print(f"  qrels:   {config.QRELS_TSV} exists={config.QRELS_TSV.exists()}", file=sys.stderr)
        return 1

    print(f"[ok] dataset ready at {config.DATASET_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
