"""Generate the §7 consumption meter — the article's main peak.

Thesis (RESEARCH C5 / REPORT §7): for the *same* 8,674 vectors, the binding
constraint of each free tier is a *different unit* — Supabase is capped by
storage, Qdrant by RAM, Turso by index bloat, Cloudflare by stored-dimensions.
There is deliberately **no single "how many months does it last"** number,
because the limiting unit differs per DB (a single figure would be misleading).

The table is *synthesized*, not hand-written:
  - byte sizes (total_bytes / index_bytes) are read from results/summary.csv
    (the full n_queries=1406 row per DB) so the report tracks the real run;
  - free-tier limits, the measured Turso Platform-API usage, and the Cloudflare
    queried-dims arithmetic are encoded below as constants WITH provenance.

Run:
    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m scripts.consumption_report

Writes results/consumption.md and echoes it to stdout.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMARY_CSV = ROOT / "results" / "summary.csv"
OUT_MD = ROOT / "results" / "consumption.md"

# --- the controlled workload (PLAN §5: identical across every DB) ---
N_CORPUS = 8674       # ArguAna corpus = stored vectors
N_QUERIES = 1406      # ArguAna test queries = one full evaluation
DIM = 384             # all-MiniLM-L6-v2
RAW_VEC_BYTES = N_CORPUS * DIM * 4   # 13,323,264 — F32 payload, the honest baseline
STORED_DIMS = N_CORPUS * DIM          # 3,330,816  — Cloudflare's "stored" unit
QUERY_DIMS = N_QUERIES * DIM          # 539,904    — one eval's query side

# Decimal SI for storage quotas (10^6 / 10^9) — the project-wide convention:
# raw=13.3MB, Turso index=710.8MB / total=729MB, Supabase index=17.7MB all match
# 10^6. (An early Supabase note used 1024^2 → 31.3MB/6.3%; that was the lone
# outlier, corrected here to 32.8MB/6.6%. Supabase has huge headroom either way.)
MB = 1_000_000
GB = 1_000_000_000


def pct(x: float, of: float) -> float:
    return 100.0 * x / of


def latest_full_row(db: str) -> dict | None:
    """Return the most-recent summary.csv row for `db` at the full n_queries=1406."""
    found = None
    with SUMMARY_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["db"] == db and row["n_queries"] == str(N_QUERIES):
                found = row  # keep overwriting -> last (most recent) wins
    return found


def as_int(row: dict | None, key: str) -> int | None:
    if not row:
        return None
    v = (row.get(key) or "").strip()
    return int(v) if v else None


# --- Free-tier limits (RESEARCH B, verified on vendor pages 2026-06-25) ---
FREE = {
    "supabase": "500MB DB storage + 7-day inactivity pause",
    "qdrant":   "1GB RAM, 4GB disk + 1-week suspend on inactivity",
    "turso":    "5GB storage, 500M reads/mo, 10M writes/mo",
    "cf":       "5M stored dims/mo, 30M queried dims/mo",
}

# --- Qdrant per-collection size: LIVE probe, measured 2026-06-25 ---
#   scripts/qdrant_size_probe (re-create + upsert 8,674 real vectors -> probe ->
#   teardown). Raw output: results/qdrant_size_probe.txt. CORRECTION (audit
#   2026-06-25): the earlier "structurally unmeasurable" claim was WRONG — it
#   rested on a torn-down (404) collection. Qdrant DOES expose per-collection
#   bytes via /telemetry?details_level=10: shards[].local.vectors_size_bytes.
QDRANT_PROBE = {
    "vectors_size_bytes": 13_323_264,   # == raw 8674*384*4 exactly (no HNSW overhead here)
    "payloads_size_bytes": 1_110_272,
    "node_rss_delta_bytes": 38_117_376,  # node-level RSS delta around the insert (incl HNSW+allocator)
    "ram_free_bytes": 1_000_000_000,     # free tier RAM (binding unit)
    "disk_free_bytes": 4_000_000_000,    # free tier disk
}

# --- Turso real consumption: Platform API (cumulative) ---
# Prefer the live-fetched artifact results/turso_usage.json (scripts/turso_usage_probe);
# fall back to the last recorded snapshot so the report still regenerates offline.
#   GET https://api.turso.tech/v1/organizations/<your-org>/databases/arguana-bench/usage
# storage_bytes is post-DROP-INDEX (reclaimed). rows_* are cumulative monthly counters.
_TURSO_USAGE_FALLBACK = {
    "rows_read": 3_131_435,
    "rows_written": 496_276,
    "storage_bytes_after_drop": 8_192,
    "reads_free_per_mo": 500_000_000,
    "writes_free_per_mo": 10_000_000,
    "fetched_at": "2026-06-25 (snapshot)",
}


def load_turso_usage() -> dict:
    """Read results/turso_usage.json if present, else the built-in fallback."""
    p = OUT_MD.parent / "turso_usage.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return {
            "rows_read": d["rows_read"],
            "rows_written": d["rows_written"],
            "storage_bytes_after_drop": d["storage_bytes"],
            "reads_free_per_mo": d.get("reads_free_per_mo", 500_000_000),
            "writes_free_per_mo": d.get("writes_free_per_mo", 10_000_000),
            "fetched_at": d.get("fetched_at") or "(undated)",
        }
    except (OSError, KeyError, ValueError):
        return dict(_TURSO_USAGE_FALLBACK)


TURSO_USAGE = load_turso_usage()


def build() -> str:
    sup = latest_full_row("supabase")
    qdr = latest_full_row("qdrant")
    tur = latest_full_row("turso")
    cf = latest_full_row("cf")   # measured once CF (M4) has run; None otherwise

    sup_total = as_int(sup, "total_bytes")      # 32,849,920
    sup_index = as_int(sup, "index_bytes")      # 17,678,336
    tur_total = as_int(tur, "total_bytes")      # 729,042,944 (index alive)
    tur_index = as_int(tur, "index_bytes")      # 710,766,592

    # Cloudflare — ARITHMETIC ONLY (Vectorize free tier was never provisioned).
    # Billing model from CF's official worked example: queried dims for a month =
    # (queries_this_month + stored_vectors) * dim  [(30,000+10,000)*768].
    # So stored is subtracted once, then each eval costs N_QUERIES*DIM.
    # stored_dims is MEASURED from the CF row once it has run (rows*dim), else the
    # arithmetic baseline (identical here: all 8,674 unique ids stored -> same value).
    cf_stored = (as_int(cf, "rows") or N_CORPUS) * DIM
    cf_stored_pct = pct(cf_stored, 5_000_000)                 # 66.6%
    cf_first_eval_queried = (N_QUERIES + N_CORPUS) * DIM       # 3,870,720
    cf_monthly_evals = (30_000_000 - cf_stored) // QUERY_DIMS  # 49
    cf_run = cf is not None

    L = []
    w = L.append
    w("# §7 消費メーター — 無料枠の持ちは「律速ユニット」がDBごとに非対称")
    w("")
    w("> 自動生成: `scripts/consumption_report.py`（`results/summary.csv` のフル "
      f"n_queries={N_QUERIES} 行＋無料枠/Turso実測/CF算術を合成）。")
    w("> 同一ワークロード = ArguAna corpus **{:,}件 × {}次元**（生ベクトル {:,} bytes "
      "= {:.1f}MB）を全DBに投入。".format(N_CORPUS, DIM, RAW_VEC_BYTES, RAW_VEC_BYTES / MB))
    w("")
    w("## 核心")
    w("**同じ 8,674 件でも、無料枠を最初に使い切らせる単位（律速）がDBごとに別物**："
      "Supabase=容量 / Qdrant=RAM / Turso=索引肥大 / Cloudflare=stored次元。"
      "→ 「何ヶ月持つ」を単一の数字で出すのは律速が異なるため不正確。**律速ごとに別々に**示す。")
    w("")
    w("## 実数表")
    w("")
    w("| DB | リージョン | 律速ユニット | 使用量 / 無料枠 | 消費率 | 出典 |")
    w("|---|---|---|---|---|---|")

    # Supabase — storage-bound (measured)
    if sup_total is not None:
        w("| **Supabase** | Tokyo | DB容量＋7日休止 | {:.1f}MB / 500MB（うち索引{:.1f}MB） | "
          "**{:.1f}%** | summary.csv 実測 |".format(
              sup_total / MB, (sup_index or 0) / MB, pct(sup_total, 500 * MB)))

    # Qdrant — RAM-bound; per-collection vectors_size IS exposed via telemetry
    qd_vec = QDRANT_PROBE["vectors_size_bytes"]
    qd_rss = QDRANT_PROBE["node_rss_delta_bytes"]
    w("| **Qdrant** | Oregon | RAM/disk＋1週suspend | "
      "vectors {:.1f}MB（生のみ・索引非露出）/ RAM1GB・disk4GB | "
      "**{:.2f}%**(disk) / ~{:.1f}%(RAM・概算) | scripts/qdrant_size_probe 実測 |".format(
          qd_vec / MB, pct(qd_vec, QDRANT_PROBE["disk_free_bytes"]),
          pct(qd_rss, QDRANT_PROBE["ram_free_bytes"])))

    # Turso — index-bloat-bound (measured)
    if tur_total is not None:
        w("| **Turso** | Tokyo | 索引肥大（storage）＋逐次投入の遅さ | "
          "{:.0f}MB / 5GB（索引生存中・生の{:.0f}倍） | **{:.1f}%** | summary.csv 実測（dbstat） |".format(
              tur_total / MB, (tur_index or 0) / RAW_VEC_BYTES, pct(tur_total, 5 * GB)))

    # Cloudflare — stored-bound; measured once CF (M4) has run
    w("| **Cloudflare** | {} | stored次元 | "
      "{:,} / 5,000,000 stored dims | **{:.1f}%** | {} |".format(
          "global(edge)" if cf_run else "（未実行）", cf_stored, cf_stored_pct,
          "summary.csv 実測" if cf_run else "算術のみ（CF未プロビジョン）"))
    w("")

    # --- per-DB narrative with the secondary metrics ---
    w("## 律速の内訳（補助指標）")
    w("")
    w("### Supabase（東京）— 容量律速")
    if sup_total is not None:
        w(f"- DB容量 {sup_total/MB:.1f}MB（索引 {(sup_index or 0)/MB:.1f}MB）= "
          f"**500MB の {pct(sup_total,500*MB):.1f}%**。余裕大。")
    w("- 第2の壁は **7日無操作で自動休止**（容量より先に効くことがある＝時間律速）。")
    w("")
    w("### Qdrant（Oregon）— RAM律速・生ベクトルは測れるが索引サイズは非露出")
    w("- 律速は **RAM 1GB / disk 4GB ＋ 1週 suspend**。")
    w("- 訂正の連鎖（監査2026-06-25）: 当初「サイズは構造的に測定不能」は**誤り**で撤回（空404コレクションを見た早合点）。"
      "ライブ8,674件投入＋再probeで `/telemetry?details_level=10` の `vectors_size_bytes` に"
      "**per-collectionのバイト数が出る**（一次 `results/qdrant_size_probe.txt`）。"
      "ただし**それは生ベクトル分のみで索引(HNSW)は含まない**。")
    w("- 実測（ライブ）: **vectors_size={:,} bytes（{:.1f}MB＝生ベクトルと完全一致）** ＋ payloads={:,} bytes。"
      "★**索引サイズは出ない**: `indexing_threshold=1`でHNSWを**強制構築しても** telemetryの "
      "`ram_usage_bytes`/`disk_usage_bytes`=0（Qdrant Cloudが索引バイト数を露出しない／一次 "
      "`results/qdrant_index_probe.txt`）。⇒ **Supabase 17.7MB(HNSW込) や Turso 710.8MB(DiskANN) とは"
      "直接比較不可**。実RAMの概算は投入前後の **node RSS Δ≈{:.1f}MB**（ノード単位・HNSW＋allocator込み）。".format(
          QDRANT_PROBE["vectors_size_bytes"], QDRANT_PROBE["vectors_size_bytes"] / MB,
          QDRANT_PROBE["payloads_size_bytes"], QDRANT_PROBE["node_rss_delta_bytes"] / MB))
    w("- 容量%: vectors {:.2f}%/disk4GB・{:.1f}%/RAM1GB、node RSSΔ ~{:.1f}%/RAM1GB。"
      "**RAM律速は変わらないが「測れない」は撤回**＝per-collection vectorsは測れる（真の律速=HNSW込み実RAM常駐のみ概算）。".format(
          pct(QDRANT_PROBE["vectors_size_bytes"], QDRANT_PROBE["disk_free_bytes"]),
          pct(QDRANT_PROBE["vectors_size_bytes"], QDRANT_PROBE["ram_free_bytes"]),
          pct(QDRANT_PROBE["node_rss_delta_bytes"], QDRANT_PROBE["ram_free_bytes"])))
    w("")
    w("### Turso（東京）— 索引肥大律速 ＋ 投入律速")
    if tur_total is not None:
        w(f"- storage {tur_total/MB:.0f}MB = **5GB の {pct(tur_total,5*GB):.1f}%**（索引生存中）。"
          f"索引 {(tur_index or 0)/MB:.1f}MB = 生ベクトル {RAW_VEC_BYTES/MB:.1f}MB の "
          f"**{(tur_index or 0)/RAW_VEC_BYTES:.0f}倍**（DiskANN 肥大）。")
    tps_row = tur or {}
    # compute the Supabase-vs-Turso upsert slowdown from the live CSV tps (don't hardcode)
    try:
        slow = round(float(sup["upsert_tps"]) / float(tur["upsert_tps"]))
        slow_s = f"約{slow}倍"
    except (TypeError, ValueError, ZeroDivisionError):
        slow_s = "大幅に"
    w("- **投入が律速**: upsert {}s（{} rows/s・逐次DiskANN）= Supabase（{} rows/s）の{}遅い。".format(
        tps_row.get("upsert_s", "?"), tps_row.get("upsert_tps", "?"),
        (sup or {}).get("upsert_tps", "?"), slow_s))
    w("- **読み出しは安い**: rows_read {:,}（無料 {:,}/月の **{:.1f}%**）／"
      "rows_written {:,}（無料 {:,}/月の **{:.0f}%**）。〔Platform API 実測 {}〕".format(
          TURSO_USAGE["rows_read"], TURSO_USAGE["reads_free_per_mo"],
          pct(TURSO_USAGE["rows_read"], TURSO_USAGE["reads_free_per_mo"]),
          TURSO_USAGE["rows_written"], TURSO_USAGE["writes_free_per_mo"],
          pct(TURSO_USAGE["rows_written"], TURSO_USAGE["writes_free_per_mo"]),
          TURSO_USAGE.get("fetched_at", "")))
    w("- 回収: **VACUUM は Turso Cloud で不可**だが **DROP INDEX/作り直しで課金ストレージは回収**"
      f"（DROP後 storage_bytes = {TURSO_USAGE['storage_bytes_after_drop']:,} ＝ 8KB）。"
      "「物理回収不可」は誤りとして撤回済み。")
    w("")
    w("### Cloudflare（{}）— stored律速".format("global edge・実測M4" if cf_run else "未実行・算術のみ"))
    w("- **stored = {:,} dims / 5,000,000 = {:.1f}%**（4DB中最も窮屈＝最初に詰まる枠）。".format(
        cf_stored, cf_stored_pct))
    w("- queried（CF課金式 =（queries＋stored）× dim）: 初回1評価 = "
      "({:,}＋{:,})×{} = **{:,} dims**。".format(N_QUERIES, N_CORPUS, DIM, cf_first_eval_queried))
    w("- 月あたり評価回数 N ≤ (30,000,000 − {:,}) // ({:,}×{}) = **{}回/月**"
      "（stored を月1回引いた残りを query 分で割る）。".format(cf_stored, N_QUERIES, DIM, cf_monthly_evals))
    if cf_run:
        w("- ★**実測（M4・無料枠で実走）**: 8,674件を Vectorize に投入し stored={:,}（=算術と一致＝全件stored確認）。"
          "Task品質 nDCG@10={}/Recall@10={}、**ANN Recall@10={}＝天井近傍に収束**（4本目も品質収束・実近似で0.9988<1.0）。"
          "e2e p50={}ms（REST×グローバルエッジ＝4DBで最重）／upsert {} rows/s。"
          "※Vectorizeの `vectorCount` は結果整合で非単調にぶれる（実数は8,674・stats()はmax読みで補正）＝§8の落とし穴。".format(
              cf_stored, cf.get("ndcg10"), cf.get("recall10"), cf.get("ann_recall10"),
              cf.get("e2e_p50_ms"), cf.get("upsert_tps")))
    else:
        w("- ※ Vectorize 無料枠は未プロビジョン。上は**算術のみ**で、実測ではない。")
    w("")
    w("## 結論")
    w("無料枠の「持ち」を単一スカラーで語ることはできない。**Supabase=容量、Qdrant=RAM、"
      "Turso=索引肥大、Cloudflare=stored次元** と、同一データ・同一**Task品質**でも"
      "**最初に枯れる単位が違う**。これが本記事 §7 の主張＝「無料枠の非対称は容量数値でなく"
      "*律速の種類* に出る」。（※ANN忠実度は別論点＝近似索引を実際に動かしたのは Supabase HNSW・"
      "Turso DiskANN・CF Vectorize の3つ（CFは0.9988<1.0＝総当たりでない実近似）。Qdrantは8.7kでは索引閾値10,000未満ゆえ総当たりで、その≈1.0は厳密検索由来。"
      "詳細は REPORT §2.5 脚注。）")
    w("")
    return "\n".join(L)


def canonical_tokens() -> list[str]:
    """The load-bearing display tokens REPORT.md §2.6 must contain — emitted so
    scripts/check_consumption_consistency can fail when the hand-typed §2.6 table
    drifts from this generated source (the audit's I1 single-source guard)."""
    sup = latest_full_row("supabase")
    tur = latest_full_row("turso")
    sup_total = as_int(sup, "total_bytes")
    tur_total = as_int(tur, "total_bytes")
    tur_index = as_int(tur, "index_bytes")
    qd = QDRANT_PROBE
    return [
        f"{sup_total / MB:.1f}MB",                         # 32.8MB
        f"{pct(sup_total, 500 * MB):.1f}%",                # 6.6%
        f"{RAW_VEC_BYTES / MB:.1f}MB",                     # 13.3MB
        f"{tur_total / MB:.0f}MB",                         # 729MB
        f"{pct(tur_total, 5 * GB):.1f}%",                  # 14.6%
        f"{tur_index / RAW_VEC_BYTES:.0f}",                # 53 (倍)
        f"{pct(STORED_DIMS, 5_000_000):.1f}%",             # 66.6%
        f"{STORED_DIMS:,}",                                # 3,330,816
        f"{(N_QUERIES + N_CORPUS) * DIM:,}",               # 3,870,720
        str((30_000_000 - STORED_DIMS) // QUERY_DIMS),     # 49
        f"{qd['vectors_size_bytes'] / MB:.1f}MB",          # 13.3MB (Qdrant vectors)
        f"{pct(qd['vectors_size_bytes'], qd['disk_free_bytes']):.2f}%",  # 0.33%
    ]


def main() -> None:
    md = build()
    OUT_MD.write_text(md, encoding="utf-8")
    values_path = OUT_MD.parent / "consumption_values.json"
    values_path.write_text(
        json.dumps({"report_tokens": canonical_tokens()}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(md)
    print(f"\n[written] {OUT_MD}")
    print(f"[written] {values_path}")


if __name__ == "__main__":
    main()
