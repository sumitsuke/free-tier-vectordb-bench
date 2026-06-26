"""Exact brute-force cosine kNN over the corpus = the ANN ceiling (PLAN §5-2).

Vectors are L2-normalized, so cosine == dot product. We compute the true top-K
locally with numpy; every DB's ANN result is later scored against THIS as the
ground truth (ANN Recall@10). Also reports the task-quality ceiling (the best
nDCG/Recall the embedding can reach), since exact search has no ANN error.

Usage:
    python -m harness.exact_knn
"""
from __future__ import annotations

import time

import numpy as np

from harness import config, dataset, metrics, runio


def exact_topk(query_vecs, corpus_vecs, corpus_ids, query_ids, fetch_k):
    """Return {qid: [(doc_id, cosine), ...]} of the top `fetch_k` per query."""
    sims = query_vecs @ corpus_vecs.T  # (Nq, Nc); normalized -> cosine
    run: dict[str, list[tuple[str, float]]] = {}
    # argpartition for the top fetch_k, then sort those descending
    part = np.argpartition(-sims, kth=fetch_k - 1, axis=1)[:, :fetch_k]
    for i, qid in enumerate(query_ids):
        idx = part[i]
        idx = idx[np.argsort(-sims[i, idx])]
        run[qid] = [(corpus_ids[j], float(sims[i, j])) for j in idx]
    return run


def main() -> int:
    corpus_vecs = np.load(config.CORPUS_VECS_NPY)
    corpus_ids = np.load(config.CORPUS_IDS_NPY, allow_pickle=True).tolist()
    query_vecs = np.load(config.QUERY_VECS_NPY)
    query_ids = np.load(config.QUERY_IDS_NPY, allow_pickle=True).tolist()
    qrels = dataset.load_qrels()
    print(f"[load] corpus {corpus_vecs.shape} queries {query_vecs.shape}")

    t0 = time.perf_counter()
    raw = exact_topk(query_vecs, corpus_vecs, corpus_ids, query_ids, config.FETCH_K)
    dt = time.perf_counter() - t0
    print(f"[exact] brute-force top-{config.FETCH_K} for {len(query_ids):,} queries "
          f"in {dt:.2f}s")

    # diagnostic: how often is the query's own doc the #1 raw hit? (why we exclude)
    self_top1 = sum(1 for qid, hits in raw.items() if hits and hits[0][0] == qid)
    print(f"[self-match] query's own doc is raw #1 for {self_top1}/{len(raw)} queries "
          f"-> excluded before scoring")

    run = metrics.truncate(metrics.exclude_self(raw), config.TOP_K)
    path = runio.save_run("exact", run)
    print(f"[save] {path.name}")

    tm = metrics.task_metrics(run, qrels, k=config.TOP_K)
    print(f"[ceiling] exact search is the ANN-Recall@10 ceiling (= 1.0 by definition).")
    print(f"[ref]     task quality of exact top-10 vs qrels: "
          f"nDCG@10={tm['ndcg']:.4f}  Recall@10={tm['recall']:.4f} (n={tm['n']})")
    print("          ^ this is a REFERENCE, not an upper bound on task quality: a DB's")
    print("            approximate top-10 can score slightly HIGHER on qrels when a tie")
    print("            at the rank-10 boundary swaps a relevant doc in (seen: +<=0.0014).")
    print("          Absolute values are low by design (MiniLM 256-tok truncation);")
    print("          identical basis for every DB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
