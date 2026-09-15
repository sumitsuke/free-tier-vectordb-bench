# 無料枠ベクトルDB横断ベンチ (free-tier vector DB cross-benchmark)

無料枠だけで4つのAI対応DB（**Supabase pgvector / Qdrant Cloud / Turso libSQL /
Cloudflare Vectorize**）へ**同一RAGワークロード**を流し、検索品質・レイテンシ・
無料枠の"持ち"・運用ハマりを**統制実験で実測**する Python ハーネス。

差別化軸＝ **無料枠 × DB横断 × 再現可能な実測**。`git clone` → 無料の鍵 → `python -m harness.bench` で同じ表が出る。

> 📝 解説記事（全結果・考察・落とし穴ログ）：[`ARTICLE.md`](ARTICLE.md)（Qiita Tech Festa 2026「AI時代のデータベース」投稿）。

## 結果サマリ（全1,406クエリ・既定設定・cosine・topK10・client=JP）

| DB | リージョン | 品質(nDCG/Recall) | ANN Recall@10 | e2e p50 | upsert | 索引サイズ |
|---|---|---|---|---|---|---|
| 厳密kNN(numpy・ANN参照) | local | 0.5017 / 0.7902 | 1.000 | — | — | — |
| Supabase pgvector | 東京 | 0.5017 / 0.7909 | 0.9991 | 24.7ms | 1442/s | 17.7MB(HNSW) |
| Turso libSQL | 東京 | 0.5017 / 0.7909 | 0.9992 | 49.7ms | 9.8/s | 710.8MB(DiskANN=生の53倍) |
| Qdrant Cloud | Oregon | 0.5019 / 0.7916 | 0.9994※ | 142.9ms | 716/s | 非露出 |
| Cloudflare Vectorize | global edge | 0.5019 / 0.7909 | 0.9988 | 536ms | 184/s | 非露出 |

- **品質はDBで変わらない**（同一埋め込みゆえ）。**ANN忠実度も揃えれば厳密kNN参照にほぼ一致**。
- **本当の差は「無料枠の律速ユニット」**：Supabase=容量 / Qdrant=RAM / Turso=索引肥大 / Cloudflare=stored次元。
- ※Qdrantの0.9994はこの規模では総当たり由来（HNSW近似の忠実度ではない・[`ARTICLE.md`](ARTICLE.md) §6）。
- レイテンシ絶対値は環境（緯度/リージョン）依存＝相対傾向のみ再現対象。

## 計測の核（なぜこの設計か）

- **データセット = ArguAna (8,674 docs / 1,406 queries)**。Cloudflare 無料 stored
  上限 500万次元に収まる唯一クラス（8,674×384 = 333万 < 500万）。
- **埋め込み = all-MiniLM-L6-v2 (384次元・CPU・無料)**。max_seq_len=256 で truncation
  されるため絶対 nDCG は低めだが、全DB共通なので**横断比較は壊れない**。
- **自己一致除外が必須**：ArguAna はクエリ自身がコーパスにあり、生top1の **91%** が
  自分自身。`doc_id == query_id` を除外（正解は常に別ID＝安全）。
- **recall を2系統に分離**：*Task* Recall@10 / nDCG@10（qrels基準）と *ANN* Recall@10
  （numpy厳密kNN top10を参照とした近似忠実度）。

## セットアップ

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows (PowerShell: .venv\Scripts\Activate.ps1)
pip install -r requirements.txt        # cross-OS はこちら推奨
```

> `requirements.lock.txt` は **Windows で生成した厳密版**で、`pywin32` を含み・`torch` は
> CPU版。**Linux/macOS では lock をそのまま使わず** `requirements.txt` を使うこと
> （CPU版torchは `pip install torch --index-url https://download.pytorch.org/whl/cpu`）。

## 実行

```bash
# 1) データ取得（ArguAna zip 直DL）
python -m harness.download_data
# 2) ローカルCPU埋め込み -> datasets/arguana/*.npy（約3分）
python -m harness.embed
# 3) 厳密kNNの"天井"（ANN忠実度の参照）-> results/run_exact.csv
python -m harness.exact_knn
# 4) 鍵を設定（.env.sample を .env にコピーして埋める。.env はコミットしない）
cp .env.sample .env
# 5) ベンチ実行
python -m harness.bench --db all              # Supabase + Qdrant + Turso（3DB）-> results/summary.csv
python -m harness.bench --db cf --no-teardown # Cloudflare（4本目。反映待ち非同期ゆえ分離）
python -m scripts.consumption_report          # §7 消費メーター表
```

結果は `results/summary.csv`（指標の集計）と `results/run_<db>_<variant>.csv`
（生の id/score/rank ＝ 再計算可能）に出力される。

## 構成

- `harness/` … ベンチ本体（`bench.py`・`config.py`・`metrics.py`・`latency.py`・`adapters/`）
- `scripts/` … 消費メーター生成・各種probe・規模スイープ・図生成
- `results/` … 全実測CSV・`consumption.md`（§7消費メーター）・`figures/`
- `assets/` … 記事用の図（PNG）

## ライセンス / 注意

- 認証情報は `.env`（gitignore 済み）にのみ置く。無料枠の数値は時期で改定されるため、各 Pricing の確認日は記事/コードに明記（本計測は 2026-06-26 再確認）。
- レイテンシ絶対値は再現対象外（相対傾向のみ）。品質・消費は同条件で再現可能。

## 筆者

**tauridev** — ソフトウェア開発／AIコード監査（Rust/Tauri＋React/TypeScript・ローカルファースト）。
「AIに本番品質を出させ、AIの誤りに気づく検証規律」が専門。
[ココナラ](https://coconala.com/users/6153961) ／ [getaxiom.dev](https://getaxiom.dev)

## 設計・検証の記録（Sumitsuke Lab）

このリポジトリの背景・検証環境・判定・最終検証日・失敗例は、Sumitsuke Lab の本家記事にまとめています。

- 無料枠の実測シリーズ（ベクトル DB 4 本の横断ベンチ） → https://sumitsuke.jp/lab/
- 受託（生成 AI コード・外注コードの点検と修理・テキスト完結） → https://sumitsuke.jp/works/repair/
