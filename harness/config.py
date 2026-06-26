"""Shared configuration: paths, dataset, model, and benchmark constants.

These are the *controlled conditions* of the experiment (PLAN §5). Every DB
adapter reads the same dim / metric / topK so the only variable is the DB.
"""
from __future__ import annotations

from pathlib import Path

# --- paths (everything relative to the repo root) ---
ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"
RESULTS_DIR = ROOT / "results"

# --- dataset (PLAN §4-2: ArguAna chosen so it fits Cloudflare's 5M stored-dim cap) ---
DATASET = "arguana"
# Canonical BEIR mirror (TU Darmstadt). Zip extracts to datasets/arguana/.
ARGUANA_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip"
DATASET_DIR = DATASETS_DIR / DATASET
CORPUS_JSONL = DATASET_DIR / "corpus.jsonl"
QUERIES_JSONL = DATASET_DIR / "queries.jsonl"
QRELS_TSV = DATASET_DIR / "qrels" / "test.tsv"

# --- embedding model (PLAN §4-3: 384-dim, CPU, Apache-2.0) ---
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384
# all-MiniLM-L6-v2 truncates beyond 256 tokens; absolute nDCG is therefore low
# but identical across DBs (PLAN §4-3). Documented, not a bug.
MAX_SEQ_LEN = 256

# --- benchmark constants (controlled across all DBs) ---
METRIC = "cosine"
TOP_K = 10
# Fetch a buffer so we can drop the self-match (>=1 per query) AND absorb the
# ArguAna duplicate docs (48 pairs / 96 docs) / per-DB tie ordering, keep top-10.
# Every adapter retrieves FETCH_K, excludes self, truncates to TOP_K — identical
# depth = fair ANN-recall comparison. 30 stays within Cloudflare Vectorize's topK
# cap (<=100 when returnValues:false, which the CF adapter uses).
# (audit: bumped 15->30 for margin; the bench logs any query left with <TOP_K.)
FETCH_K = TOP_K + 20

# --- artifact paths (gitignored *.npy) ---
CORPUS_VECS_NPY = DATASET_DIR / "corpus_vecs.npy"
CORPUS_IDS_NPY = DATASET_DIR / "corpus_ids.npy"
QUERY_VECS_NPY = DATASET_DIR / "query_vecs.npy"
QUERY_IDS_NPY = DATASET_DIR / "query_ids.npy"
