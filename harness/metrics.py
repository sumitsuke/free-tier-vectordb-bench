"""Evaluation metrics — the two recalls kept strictly separate (PLAN §7).

1. Task quality  : nDCG@10 / Recall@10 vs BEIR qrels (pytrec_eval).
                   Depends on embedding+data, ~identical across DBs.
2. ANN fidelity  : ANN Recall@10 / overlap vs the numpy exact-kNN *ceiling*.
                   Depends on the DB's index + search params.

All runs must pass through `exclude_self` first (ArguAna self-match, verified
in BUILD_LOG: doc_id==query_id is never the gold answer, so dropping it is safe).
"""
from __future__ import annotations

import pytrec_eval

Run = dict[str, list[tuple[str, float]]]


def exclude_self(run: Run) -> Run:
    """Drop the query's own document (ArguAna self-match). Safe: gold != qid."""
    return {qid: [(d, s) for d, s in hits if d != qid] for qid, hits in run.items()}


def truncate(run: Run, k: int) -> Run:
    return {qid: hits[:k] for qid, hits in run.items()}


def _to_pytrec(run: Run) -> dict[str, dict[str, float]]:
    return {qid: {d: float(s) for d, s in hits} for qid, hits in run.items()}


def task_metrics(run: Run, qrels: dict[str, dict[str, int]], k: int = 10) -> dict:
    """nDCG@k and Recall@k vs qrels. Returns {'ndcg': mean, 'recall': mean, 'n': N}."""
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {f"ndcg_cut.{k}", f"recall.{k}"}
    )
    per_query = evaluator.evaluate(_to_pytrec(run))
    ndcg_key, rec_key = f"ndcg_cut_{k}", f"recall_{k}"
    if not per_query:
        return {"ndcg": 0.0, "recall": 0.0, "n": 0}
    n = len(per_query)
    ndcg = sum(v[ndcg_key] for v in per_query.values()) / n
    recall = sum(v[rec_key] for v in per_query.values()) / n
    return {"ndcg": ndcg, "recall": recall, "n": n}


def ann_recall(run: Run, reference: Run, k: int = 10) -> dict:
    """Mean overlap of top-k doc ids vs the exact-kNN reference (the ceiling).

    ANN Recall@k = |topk(run) ∩ topk(ref)| / |topk(ref)|, averaged over the
    queries actually present in `run` (intersected with the reference).

    NB: iterate over `run`, NOT `reference` — otherwise a partial run (e.g.
    --limit smoke test) is penalized with 0 for every un-run query, which made
    a 100-query run report 100/1406 = 0.071 instead of the true ~1.0.
    """
    overlaps: list[float] = []
    for qid, run_hits in run.items():
        ref_ids = {d for d, _ in reference.get(qid, [])[:k]}
        if not ref_ids:
            continue
        run_ids = {d for d, _ in run_hits[:k]}
        overlaps.append(len(run_ids & ref_ids) / len(ref_ids))
    n = len(overlaps)
    return {"ann_recall": (sum(overlaps) / n if n else 0.0), "n": n}
