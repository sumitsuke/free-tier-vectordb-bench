"""RECON ONLY (M5 scale-sweep de-risk, NOT article body): download + embed the
FiQA BEIR corpus (57,638 docs) so scale_sweep.py can test whether HNSW ANN
fidelity (the article's "天井到達" climax, currently proven only at 8,674) holds
as the SAME embedding model scales from 8.7k -> ~57k.

Kept separate from the ArguAna harness: writes only datasets/fiqa/*.npy, touches
nothing the 3-DB benchmark depends on. Same model/dim/normalize as the main run
so the vectors are directly comparable.

    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m scripts.recon_embed_fiqa
"""
from __future__ import annotations

import json
import time
import zipfile

import numpy as np

from harness import config
from harness.dataset import doc_text

FIQA_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
FIQA_DIR = config.DATASETS_DIR / "fiqa"
CORPUS_JSONL = FIQA_DIR / "corpus.jsonl"
QUERIES_JSONL = FIQA_DIR / "queries.jsonl"
CORPUS_VECS = FIQA_DIR / "corpus_vecs.npy"
CORPUS_IDS = FIQA_DIR / "corpus_ids.npy"
QUERY_VECS = FIQA_DIR / "query_vecs.npy"
QUERY_IDS = FIQA_DIR / "query_ids.npy"


def _download(url, dest):
    import requests
    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r[download] {done/1e6:.1f}/{total/1e6:.1f} MB", end="")
        print()


def _read_jsonl(path, is_corpus):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if is_corpus:
                out[o["_id"]] = doc_text(o.get("title", ""), o.get("text", ""))
            else:
                out[o["_id"]] = (o.get("text", "") or "").strip()
    return out


def main() -> int:
    config.DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    if not (CORPUS_JSONL.exists() and QUERIES_JSONL.exists()):
        zip_path = config.DATASETS_DIR / "fiqa.zip"
        if not zip_path.exists():
            _download(FIQA_URL, zip_path)
        print(f"[extract] -> {config.DATASETS_DIR}")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(config.DATASETS_DIR)

    corpus = _read_jsonl(CORPUS_JSONL, True)
    queries = _read_jsonl(QUERIES_JSONL, False)
    print(f"[load] FiQA corpus={len(corpus):,} queries={len(queries):,}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.MODEL_NAME, device="cpu")
    model.max_seq_length = config.MAX_SEQ_LEN

    cids = list(corpus.keys())
    qids = list(queries.keys())
    t0 = time.perf_counter()
    cvecs = model.encode([corpus[i] for i in cids], normalize_embeddings=True,
                         batch_size=64, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    print(f"[embed] corpus {cvecs.shape} in {time.perf_counter()-t0:.0f}s")
    qvecs = model.encode([queries[i] for i in qids], normalize_embeddings=True,
                         batch_size=64, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)

    np.save(CORPUS_VECS, cvecs)
    np.save(CORPUS_IDS, np.array(cids, dtype=object))
    np.save(QUERY_VECS, qvecs)
    np.save(QUERY_IDS, np.array(qids, dtype=object))
    print(f"[save] {CORPUS_VECS} {cvecs.shape}; {QUERY_VECS} {qvecs.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
