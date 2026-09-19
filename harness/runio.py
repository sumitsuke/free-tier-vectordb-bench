"""Save/load a *run* = per-query ranked results, as CSV.

A run is {query_id: [(doc_id, score), ...]} ordered best-first. Persisting the
raw id/score/rank (DoD: results must be recomputable) lets us re-derive every
metric without re-querying a DB. CSV columns: query_id, rank, doc_id, score.
"""

from __future__ import annotations

import csv
from pathlib import Path

from harness import config

Run = dict[str, list[tuple[str, float]]]


def run_path(name: str) -> Path:
    return config.RESULTS_DIR / f"run_{name}.csv"


def save_run(name: str, run: Run) -> Path:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = run_path(name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "rank", "doc_id", "score"])
        for qid, hits in run.items():
            for rank, (did, score) in enumerate(hits, start=1):
                w.writerow([qid, rank, did, f"{score:.6f}"])
    return path


def save_latency(name: str, e2e_by_query: dict[str, list[float]]) -> Path:
    """Persist raw e2e samples (query_id, rep, e2e_ms) so p50/p95 are independently
    recomputable later (DoD: every reported number must be reproducible from CSV).
    """
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"latency_{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "rep", "e2e_ms"])
        for qid, samples in e2e_by_query.items():
            for rep, ms in enumerate(samples):
                w.writerow([qid, rep, f"{ms:.4f}"])
    return path


def load_run(name: str) -> Run:
    run: Run = {}
    with open(run_path(name), encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            run.setdefault(row["query_id"], []).append((row["doc_id"], float(row["score"])))
    # ensure best-first by rank (file is already ordered, but be safe)
    return run
