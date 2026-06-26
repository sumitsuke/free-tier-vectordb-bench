# 無料枠ベクトルDB横断ベンチ (free-tier vector DB cross-benchmark)

無料枠だけで複数のAI対応DB（Supabase pgvector / Qdrant Cloud / Turso libSQL /
Cloudflare Vectorize）へ**同一RAGワークロード**を流し、検索品質・レイテンシ・
無料枠の"持ち"・運用ハマりを**統制実験で実測**する Python ハーネス。

差別化軸＝ **無料枠 × DB横断 × 再現可能な実測**。`git clone` → 無料の鍵 → `python -m harness.bench` で同じ表が出ることを目指す。

> Qiita Tech Festa 2026「AI時代のデータベース」投稿予定。設計の詳細は `PLAN.md`、
> 事実と出典は `RESEARCH.md`、手順は `RUNBOOK.md`、過程ログは `BUILD_LOG.md`。

## 計測の核（なぜこの設計か）

- **データセット = ArguAna (8,674 docs / 1,406 queries)**。Cloudflare 無料 stored
  上限 500万次元に収まる唯一クラス（8,674×384 = 333万 < 500万）。
- **埋め込み = all-MiniLM-L6-v2 (384次元・CPU・無料)**。max_seq_len=256 で truncation
  されるため絶対 nDCG は低めだが、全DB共通なので**横断比較は壊れない**。
- **自己一致除外が必須**：ArguAna はクエリ自身がコーパスにあり、生top1の **91%** が
  自分自身。`doc_id == query_id` を除外（正解は常に別ID＝安全）。
- **recall を2系統に分離**：
  - *Task* Recall@10 / nDCG@10（qrels基準・埋め込み依存・DB間でほぼ不変）
  - *ANN* Recall@10（numpy厳密kNN top10を"天井"＝**ANN Recallの定義上の上界**とした近似忠実度。
    なおtask品質ではこのexactは"上界"でなく**参照**で、rank10境界のtieでDBが微小に上回り得る≤0.0014）
- **山場（ArguAna 8,674件規模での話。規模依存はPart2で検証）**：「DBで精度が違う」ではなく
  「**Task品質は同一ベクトルゆえDBで不変、近似索引(Supabase HNSW/Turso DiskANN)も天井に収束**、
  しかし無料の非力資源では到達の **p95 latency** がDBごとに違う」。
  （※Qdrantは8.7kが既定 indexing_threshold=10,000 未満で**総当たり**＝その ANN Recall≈1.0 は厳密検索由来。
  実HNSW×規模は M5 scale_sweep で別途検証＝ef十分なら57kまで≥0.988で崖なし）。
  レイテンシは e2e と サーバ側を分けて報告。**現状の e2e 差は主にリージョン（Qdrant=Oregon /
  Supabase=Tokyo）の交絡で、同一リージョン対照は未実施**（Tursoで東京2点目を取る予定）。

## セットアップ

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows (PowerShell: .venv\Scripts\Activate.ps1)
pip install -r requirements.txt        # cross-OS はこちら推奨
```

> `requirements.lock.txt` は **Windows で生成した厳密版**で、`pywin32` を含み・`torch` は
> CPU版。**Linux/macOS では lock をそのまま使わず** `requirements.txt` を使うこと
> （CPU版torchは `pip install torch --index-url https://download.pytorch.org/whl/cpu`）。
> レイテンシ絶対値は環境（緯度/リージョン）依存で再現対象外＝相対傾向のみ。

## 実行

```bash
# 1) データ取得（ArguAna zip 直DL）
python -m harness.download_data

# 2) ローカルCPU埋め込み -> datasets/arguana/*.npy（約3分）
python -m harness.embed

# 3) 厳密kNNの"天井"を作る -> results/run_exact.csv
python -m harness.exact_knn

# 4) 鍵を設定（.env.sample を .env にコピーして埋める。.env はコミットしない）
cp .env.sample .env

# 5) ベンチ実行
python -m harness.bench --db qdrant            # 単体
python -m harness.bench --db all --repeats 3   # Supabase + Qdrant
python -m harness.bench --db supabase --tune --ef 200   # 天井寄せ tuning run
python -m harness.bench --db qdrant --limit 100         # クイック確認
```

結果は `results/summary.csv`（指標の集計）と `results/run_<db>_<variant>.csv`
（生の id/score/rank ＝ 再計算可能）に出力される。

## 進捗

- [x] M0 環境・ArguAna取得・CPU埋め込み・厳密kNN天井（nDCG@10=0.5017 / Recall@10=0.79・MTEB一致）
- [ ] M1 Supabase + Qdrant 配管（**鍵待ち**）
- [ ] M2 CFゲート合否スパイク（1h・通れば4本目に採用）
- [ ] M3 **Supabase+Qdrant+Turso の3DB**でMin成立ライン＋索引/投入メトリクス採取
- [ ] M4 CFが通れば4本目追加（Full）
- [ ] M5 cold start / 規模スイープのローカル偵察 / 公開・投稿

## ライセンス / 注意

無料枠の数値は時期で改定されるため、公開直前に各 Pricing を再取得し「確認日」を記録する。
認証情報は `.env`（gitignore 済み）にのみ置く。
