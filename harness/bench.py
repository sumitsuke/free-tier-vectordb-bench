"""Benchmark orchestrator — runs one or more DB adapters under identical
controlled conditions and writes a results row per run (PLAN §5–§8).

Per DB:
  create -> upsert (TPS) -> warm-up (discarded) -> measured search passes
  (e2e p50/p95 + server-side where available) -> self-match exclude + truncate
  -> save raw run CSV + raw latency CSV -> task metrics (nDCG/Recall vs qrels)
  + ANN Recall@10 (vs the numpy exact top-10) -> raw free-tier stats
  (size/rows/index_bytes via adapter.stats()) -> summary row.

NOTE: only raw capacity stats are recorded today. The consumption-asymmetry
table (CF queried-dims / Turso rows-scanned / capacity-% per DB) — the article's
§7 peak — is NOT computed here yet; it is a TODO for M3 (see AUDIT.md).

Usage:
    python -m harness.bench --db qdrant
    python -m harness.bench --db all --repeats 3
    python -m harness.bench --db supabase --tune --ef 200 --no-teardown
    python -m harness.bench --db qdrant --limit 100        # quick smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone, timedelta

import numpy as np
from dotenv import load_dotenv

from harness import config, dataset, latency, metrics, runio

JST = timezone(timedelta(hours=9))

SUMMARY_FIELDS = [
    "timestamp_jst",
    "db",
    "variant",
    "region",
    "upsert_strategy",
    "n_queries",
    "dim",
    "topk",
    "fetch_k",
    "upsert_s",
    "upsert_copy_s",
    "upsert_index_s",
    "upsert_tps",
    "ndcg10",
    "recall10",
    "ann_recall10",
    "short_queries",
    "e2e_p50_ms",
    "e2e_p95_ms",
    "e2e_mean_ms",
    "e2e_pq_p50_ms",
    "e2e_pq_p95_ms",  # per-query-median view (less jitter)
    "server_p50_ms",
    "server_p95_ms",
    "server_n",
    "rows",
    "total_bytes",
    "index_bytes",
    "ef",
    "note",
]


def make_adapter(name: str):
    if name == "qdrant":
        from harness.adapters.qdrant_store import QdrantStore

        return QdrantStore()
    if name in ("supabase", "pgvector"):
        from harness.adapters.pgvector_store import PgvectorStore

        return PgvectorStore()
    if name in ("turso", "libsql"):
        from harness.adapters.turso_store import TursoStore

        return TursoStore()
    if name in ("cloudflare", "cf", "vectorize"):
        from harness.adapters.cloudflare_store import CloudflareStore

        return CloudflareStore()
    raise ValueError(f"unknown db: {name}")


def load_artifacts():
    corpus_vecs = np.load(config.CORPUS_VECS_NPY)
    corpus_ids = np.load(config.CORPUS_IDS_NPY, allow_pickle=True).tolist()
    query_vecs = np.load(config.QUERY_VECS_NPY)
    query_ids = np.load(config.QUERY_IDS_NPY, allow_pickle=True).tolist()
    qrels = dataset.load_qrels()
    exact = runio.load_run("exact")
    return corpus_vecs, corpus_ids, query_vecs, query_ids, qrels, exact


def measure_search(adapter, query_vecs, query_ids, warmup, repeats):
    """Return (run, e2e_by_query). e2e_by_query maps qid -> [ms per repeat].

    Only `adapter.search` is inside the perf_counter window. The run (for recall)
    is kept from rep 0. Server-side time is taken separately (sampled_server_ms),
    not in this hot loop, to keep e2e clean. The caller derives both the pooled
    p50/p95 (user-felt) and the per-query-median p50/p95 (less network jitter)."""
    for i in range(min(warmup, len(query_ids))):  # warm-up (discarded)
        adapter.search(query_vecs[i], config.FETCH_K)

    e2e_by_query: dict[str, list[float]] = {qid: [] for qid in query_ids}
    run: dict[str, list[tuple[str, float]]] = {}
    for rep in range(repeats):
        for i, qid in enumerate(query_ids):
            t0 = time.perf_counter()
            hits = adapter.search(query_vecs[i], config.FETCH_K)
            e2e_by_query[qid].append((time.perf_counter() - t0) * 1000.0)
            if rep == 0:
                run[qid] = hits
    return run, e2e_by_query


def sampled_server_ms(adapter, query_vecs, query_ids, sample=100):
    """Sampled server-side processing time via adapter.sample_server_ms (separate
    path: Qdrant REST `time` / pgvector EXPLAIN ANALYZE). Empty if unsupported."""
    step = max(1, len(query_ids) // sample)
    out = []
    for i in range(0, len(query_ids), step):
        try:
            ms = adapter.sample_server_ms(query_vecs[i], config.FETCH_K)
        except Exception as e:
            print(f"[server-sample] failed at q{i}: {e}")
            break
        if ms is not None:
            out.append(ms)
    return out


def run_one(name, variant, corpus_vecs, corpus_ids, query_vecs, query_ids, qrels, exact, args):
    print(f"\n=== {name} [{variant}] ===")
    adapter = make_adapter(name)
    if not getattr(adapter, "region", ""):
        # region is an auxiliary structured column (same-region control); warn at
        # run time so a blank cell is visible now, not discovered later in the CSV.
        print(
            f"[warn] {name}: region blank (endpoint carries no AWS region token) -> summary.csv 'region' will be empty"
        )

    # tuning knobs
    ef = None
    if args.tune:
        ef = args.ef
        if name == "qdrant":
            adapter.hnsw_ef = ef
        elif name in ("supabase", "pgvector"):
            adapter.ef_search = ef
        elif name in ("turso", "libsql"):
            # libSQL DiskANN / vector_top_k exposes no query-time ef knob, so Turso
            # is N/A for the tuning layer (treated like the CF black-box). A 'tuned'
            # Turso run is identical to default — flag it so the row isn't misread.
            ef = None
            print("[tune] Turso has no query-time ANN knob -> tuned == default (N/A)")
        elif name in ("cloudflare", "cf", "vectorize"):
            # Vectorize is a managed black-box: no query-time ANN/ef knob exposed.
            ef = None
            print("[tune] Cloudflare Vectorize is managed (no ANN knob) -> tuned == default (N/A)")

    try:
        # create + upsert
        adapter.create(config.DIM, config.METRIC)
        t0 = time.perf_counter()
        adapter.upsert(corpus_ids, corpus_vecs)
        upsert_s = time.perf_counter() - t0
        tps = len(corpus_ids) / upsert_s if upsert_s else float("nan")
        bd = getattr(adapter, "upsert_breakdown", None)
        copy_s = bd.get("copy_s") if bd else ""
        index_s = bd.get("index_s") if bd else ""
        bd_str = f" (insert {copy_s}s + index {index_s}s)" if bd else ""
        print(f"[upsert] {len(corpus_ids):,} vecs in {upsert_s:.1f}s ({tps:.0f}/s){bd_str}")

        try:
            st = adapter.stats()
            print(f"[stats] {st}")
        except Exception as e:
            st = {}
            print(f"[stats] unavailable: {e}")

        # search + latency
        q_ids = query_ids[: args.limit] if args.limit else query_ids
        q_vecs = query_vecs[: args.limit] if args.limit else query_vecs
        run, e2e_by_query = measure_search(adapter, q_vecs, q_ids, args.warmup, args.repeats)
        server_ms = sampled_server_ms(adapter, q_vecs, q_ids)

        # exclude self-match, truncate, save raw run + raw latency samples
        run = metrics.truncate(metrics.exclude_self(run), config.TOP_K)
        runio.save_run(f"{name}_{variant}", run)
        runio.save_latency(f"{name}_{variant}", e2e_by_query)
        short = sum(1 for hits in run.values() if len(hits) < config.TOP_K)
        if short:
            print(f"[warn] {short} queries returned < {config.TOP_K} after self-exclusion")

        # metrics
        tm = metrics.task_metrics(run, qrels, k=config.TOP_K)
        ar = metrics.ann_recall(run, exact, k=config.TOP_K)
        e2e_pooled = [x for v in e2e_by_query.values() for x in v]
        e2e_pq = [statistics.median(v) for v in e2e_by_query.values() if v]
        e2e = latency.summarize(e2e_pooled)
        pq = latency.summarize(e2e_pq)
        sv = latency.summarize(server_ms)
        print(f"[task]  nDCG@10={tm['ndcg']:.4f} Recall@10={tm['recall']:.4f}")
        print(f"[ann ]  ANN Recall@10={ar['ann_recall']:.4f} (vs exact ceiling)")
        print(
            f"[e2e ]  pooled p50={e2e['p50']:.1f} p95={e2e['p95']:.1f} | per-query-median p50={pq['p50']:.1f} p95={pq['p95']:.1f} ms"
        )
        print(
            f"[srv ]  p50={sv['p50']:.1f}ms p95={sv['p95']:.1f}ms (n={sv['n']}, sampled separate path, NOT cross-DB comparable)"
        )

        row = {
            "timestamp_jst": datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S"),
            "db": name,
            "variant": variant,
            "region": getattr(adapter, "region", ""),
            "upsert_strategy": getattr(adapter, "upsert_strategy", ""),
            "n_queries": len(q_ids),
            "dim": config.DIM,
            "topk": config.TOP_K,
            "fetch_k": config.FETCH_K,
            "upsert_s": f"{upsert_s:.2f}",
            "upsert_copy_s": copy_s,
            "upsert_index_s": index_s,
            "upsert_tps": f"{tps:.1f}",
            "ndcg10": f"{tm['ndcg']:.4f}",
            "recall10": f"{tm['recall']:.4f}",
            "ann_recall10": f"{ar['ann_recall']:.4f}",
            "short_queries": short,
            "e2e_p50_ms": f"{e2e['p50']:.2f}",
            "e2e_p95_ms": f"{e2e['p95']:.2f}",
            "e2e_mean_ms": f"{e2e['mean']:.2f}",
            "e2e_pq_p50_ms": f"{pq['p50']:.2f}",
            "e2e_pq_p95_ms": f"{pq['p95']:.2f}",
            "server_p50_ms": f"{sv['p50']:.2f}" if sv["n"] else "",
            "server_p95_ms": f"{sv['p95']:.2f}" if sv["n"] else "",
            "server_n": sv["n"],
            "rows": st.get("rows", st.get("points", "")),
            "total_bytes": st.get("total_bytes", ""),
            "index_bytes": st.get("index_bytes", ""),
            "ef": ef if ef is not None else "",
            "note": args.note,
        }
        return row
    finally:
        # teardown in finally so a mid-run failure never leaks free-tier resources
        if not args.no_teardown:
            try:
                adapter.teardown()
                print("[teardown] dropped")
            except Exception as e:
                print(f"[teardown] failed (manual cleanup may be needed): {e}")
        else:
            print("[teardown] skipped (--no-teardown)")
        if hasattr(adapter, "close"):
            adapter.close()


def write_summary(rows):
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / "summary.csv"
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[summary] appended {len(rows)} row(s) -> {path}")


def write_metadata(args, rows):
    """Record the run environment so latency (region-confounded) can be read
    honestly later. The user passes region/network via --note; the rest is auto."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp_jst": datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S"),
        "note_region_network": args.note,
        "args": {
            "db": args.db,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "limit": args.limit,
            "tune": args.tune,
            "ef": args.ef,
        },
        "config": {
            "dataset": config.DATASET,
            "model": config.MODEL_NAME,
            "dim": config.DIM,
            "max_seq_len": config.MAX_SEQ_LEN,
            "metric": config.METRIC,
            "top_k": config.TOP_K,
            "fetch_k": config.FETCH_K,
        },
        "env": {"platform": platform.platform(), "python": sys.version.split()[0]},
        "dbs_run": [r["db"] for r in rows],
    }
    path = config.RESULTS_DIR / "metadata.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[metadata] -> {path}")


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="all", help="qdrant | supabase | all")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="limit #queries (smoke test)")
    ap.add_argument("--tune", action="store_true", help="raise ef toward the ceiling")
    ap.add_argument("--ef", type=int, default=200)
    ap.add_argument("--no-teardown", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    dbs = ["qdrant", "supabase", "turso"] if args.db == "all" else [args.db]
    variant = "tuned" if args.tune else "default"

    (corpus_vecs, corpus_ids, query_vecs, query_ids, qrels, exact) = load_artifacts()
    print(f"[load] corpus {corpus_vecs.shape} queries {query_vecs.shape} exact-ceiling queries={len(exact)}")

    rows = []
    for name in dbs:
        try:
            rows.append(run_one(name, variant, corpus_vecs, corpus_ids, query_vecs, query_ids, qrels, exact, args))
        except Exception as e:
            import traceback

            print(f"[error] {name} failed: {e}")
            traceback.print_exc()

    if rows:
        write_summary(rows)
        write_metadata(args, rows)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
