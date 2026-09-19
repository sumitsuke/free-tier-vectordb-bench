"""RECON ONLY (M5 de-risk, NOT article body): does HNSW ANN fidelity — the
article's "どのDBも天井(厳密top10)に届く" climax, currently proven only at
N=8,674 — hold as the corpus scales up? We sweep N over a real FiQA embedding
(same model/dim/normalize as the benchmark) and, at each N, compare a local HNSW
index (faiss IndexHNSWFlat, the same algorithm Qdrant/pgvector use) against the
exact top-10 (faiss IndexFlatIP). Cloud-free, local, reproducible.

Output: results/scale_sweep.csv + a printed verdict. This informs the Part1
GO/NO-GO. Tested N maxes at the full FiQA corpus (57,638); a "~100k" claim is
EXTRAPOLATION beyond the tested range and is labeled as such in the verdict.

    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m scripts.scale_sweep
"""

from __future__ import annotations

import csv
import time

import faiss
import numpy as np

from harness import config

FIQA_DIR = config.DATASETS_DIR / "fiqa"
OUT_CSV = config.RESULTS_DIR / "scale_sweep.csv"
TOP_K = 10
# HNSW params: M=16 / efConstruction=200 are common library defaults (≈ what the
# cloud DBs ship). efSearch is the "reach the ceiling" knob — sweep a low default
# and a higher value to see if any scale needs more ef to stay at the ceiling.
HNSW_M = 16
HNSW_EFC = 200
EF_SEARCH = [16, 64]  # 16 = faiss default; 64 ≈ a modest bump
N_GRID = [8674, 25000, 50000]  # + the full corpus, appended at runtime
QUERY_SAMPLE = 2000  # cap queries for stable, fast recall stats


def exact_top_k(corpus: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    idx = faiss.IndexFlatIP(corpus.shape[1])  # inner product == cosine (normalized)
    idx.add(corpus)
    _, I = idx.search(queries, k)
    return I


def hnsw_top_k(corpus, queries, k, ef_search):
    idx = faiss.IndexHNSWFlat(corpus.shape[1], HNSW_M)  # L2; == cosine for unit vecs
    idx.hnsw.efConstruction = HNSW_EFC
    t0 = time.perf_counter()
    idx.add(corpus)
    build_s = time.perf_counter() - t0
    idx.hnsw.efSearch = ef_search
    t1 = time.perf_counter()
    _, I = idx.search(queries, k)
    q_ms = (time.perf_counter() - t1) * 1000.0 / len(queries)
    return I, build_s, q_ms


def recall_at_k(approx: np.ndarray, exact: np.ndarray, k: int) -> float:
    hits = sum(len(set(a[:k]) & set(e[:k])) for a, e in zip(approx, exact))
    return hits / (len(exact) * k)


def main() -> int:
    cvecs = np.load(FIQA_DIR / "corpus_vecs.npy")
    qvecs = np.load(FIQA_DIR / "query_vecs.npy")
    rng = np.random.default_rng(0)
    if len(qvecs) > QUERY_SAMPLE:
        qvecs = qvecs[rng.choice(len(qvecs), QUERY_SAMPLE, replace=False)]
    print(f"[load] FiQA corpus={len(cvecs):,} dim={cvecs.shape[1]} queries(sampled)={len(qvecs):,}")

    grid = sorted({n for n in N_GRID if n <= len(cvecs)} | {len(cvecs)})
    rows = []
    for n in grid:
        sub = np.ascontiguousarray(cvecs[:n])
        exact = exact_top_k(sub, qvecs, TOP_K)
        for ef in EF_SEARCH:
            I, build_s, q_ms = hnsw_top_k(sub, qvecs, TOP_K, ef)
            rec = recall_at_k(I, exact, TOP_K)
            rows.append(
                {
                    "n": n,
                    "ef_search": ef,
                    "ann_recall10": round(rec, 4),
                    "build_s": round(build_s, 2),
                    "query_ms": round(q_ms, 3),
                    "M": HNSW_M,
                    "efc": HNSW_EFC,
                }
            )
            print(f"  N={n:>6} ef={ef:>3}: ANN Recall@10={rec:.4f} (build {build_s:.1f}s, {q_ms:.2f} ms/q)")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["n", "ef_search", "ann_recall10", "build_s", "query_ms", "M", "efc"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[written] {OUT_CSV}")

    # verdict. Distinguish two axes so a tuning artifact isn't mislabeled as a
    # scale threshold: the SCALE-THRESHOLD judgment keys off the HIGHER ef (the
    # "given enough ef, does N alone break recall?"); the lower ef is reported as
    # a separate default-ef sensitivity axis. Tested N maxes at the full FiQA
    # corpus -> anything beyond is EXTRAPOLATION, stated explicitly.
    nmax = max(r["n"] for r in rows)
    for ef in EF_SEARCH:
        series = [(r["n"], r["ann_recall10"]) for r in rows if r["ef_search"] == ef]
        lo = min(v for _, v in series)
        trend = " -> ".join(f"{n // 1000}k:{v:.3f}" for n, v in series)
        print(f"[ef={ef}] {trend}  (min={lo:.3f})")
    hi = max(EF_SEARCH)
    hi_series = sorted([(r["n"], r["ann_recall10"]) for r in rows if r["ef_search"] == hi])
    hi_first, hi_last = hi_series[0][1], hi_series[-1][1]
    hi_lo = min(v for _, v in hi_series)
    slope = hi_first - hi_last
    # A "threshold/cliff" means recall collapses with scale even at good ef. A few
    # tenths of a percent of graceful drift is NOT a cliff — distinguish them.
    if hi_lo >= 0.95:
        print(
            f"[verdict] at ef={hi}: ANN Recall@10 holds {hi_first:.3f}->{hi_last:.3f} "
            f"(slope {slope * 100:.1f}pp) up to N={nmax:,} => NO cliff; gentle, ef-recoverable "
            f"decline. No scale threshold below {nmax:,} (100k = EXTRAPOLATION, untested)."
        )
    elif hi_lo >= 0.90:
        print(
            f"[verdict] at ef={hi}: ANN Recall@10 drifts to {hi_lo:.3f} by N={nmax:,} "
            f"=> soft decline, still no hard cliff (raise ef to recover)."
        )
    else:
        print(
            f"[verdict] at ef={hi}: ANN Recall@10 collapses to {hi_lo:.3f} by N={nmax:,} "
            f"=> SCALE-BOUND cliff (threshold <= {nmax:,})."
        )
    lo_series = sorted([(r["n"], r["ann_recall10"]) for r in rows if r["ef_search"] == min(EF_SEARCH)])
    lo_lo = min(v for _, v in lo_series)
    print(
        f"[note] ef={min(EF_SEARCH)} (low) runs {lo_series[0][1]:.3f}->{lo_series[-1][1]:.3f} "
        f"(min={lo_lo:.3f}) => default-ef sensitivity axis: more approximate, "
        f"ef is the knob (cloud DBs' effective ef sits between these)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
