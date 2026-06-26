# §7 消費メーター — 無料枠の持ちは「律速ユニット」がDBごとに非対称

> 自動生成: `scripts/consumption_report.py`（`results/summary.csv` のフル n_queries=1406 行＋無料枠/Turso実測/CF算術を合成）。
> 同一ワークロード = ArguAna corpus **8,674件 × 384次元**（生ベクトル 13,323,264 bytes = 13.3MB）を全DBに投入。

## 核心
**同じ 8,674 件でも、無料枠を最初に使い切らせる単位（律速）がDBごとに別物**：Supabase=容量 / Qdrant=RAM / Turso=索引肥大 / Cloudflare=stored次元。→ 「何ヶ月持つ」を単一の数字で出すのは律速が異なるため不正確。**律速ごとに別々に**示す。

## 実数表

| DB | リージョン | 律速ユニット | 使用量 / 無料枠 | 消費率 | 出典 |
|---|---|---|---|---|---|
| **Supabase** | Tokyo | DB容量＋7日休止 | 32.8MB / 500MB（うち索引17.7MB） | **6.6%** | summary.csv 実測 |
| **Qdrant** | Oregon | RAM/disk＋1週suspend | vectors 13.3MB（生のみ・索引非露出）/ RAM1GB・disk4GB | **0.33%**(disk) / ~3.8%(RAM・概算) | scripts/qdrant_size_probe 実測 |
| **Turso** | Tokyo | 索引肥大（storage）＋逐次投入の遅さ | 729MB / 5GB（索引生存中・生の53倍） | **14.6%** | summary.csv 実測（dbstat） |
| **Cloudflare** | global(edge) | stored次元 | 3,330,816 / 5,000,000 stored dims | **66.6%** | summary.csv 実測 |

## 律速の内訳（補助指標）

### Supabase（東京）— 容量律速
- DB容量 32.8MB（索引 17.7MB）= **500MB の 6.6%**。余裕大。
- 第2の壁は **7日無操作で自動休止**（容量より先に効くことがある＝時間律速）。

### Qdrant（Oregon）— RAM律速・生ベクトルは測れるが索引サイズは非露出
- 律速は **RAM 1GB / disk 4GB ＋ 1週 suspend**。
- 訂正の連鎖（監査2026-06-25）: 当初「サイズは構造的に測定不能」は**誤り**で撤回（空404コレクションを見た早合点）。ライブ8,674件投入＋再probeで `/telemetry?details_level=10` の `vectors_size_bytes` に**per-collectionのバイト数が出る**（一次 `results/qdrant_size_probe.txt`）。ただし**それは生ベクトル分のみで索引(HNSW)は含まない**。
- 実測（ライブ）: **vectors_size=13,323,264 bytes（13.3MB＝生ベクトルと完全一致）** ＋ payloads=1,110,272 bytes。★**索引サイズは出ない**: `indexing_threshold=1`でHNSWを**強制構築しても** telemetryの `ram_usage_bytes`/`disk_usage_bytes`=0（Qdrant Cloudが索引バイト数を露出しない／一次 `results/qdrant_index_probe.txt`）。⇒ **Supabase 17.7MB(HNSW込) や Turso 710.8MB(DiskANN) とは直接比較不可**。実RAMの概算は投入前後の **node RSS Δ≈38.1MB**（ノード単位・HNSW＋allocator込み）。
- 容量%: vectors 0.33%/disk4GB・1.3%/RAM1GB、node RSSΔ ~3.8%/RAM1GB。**RAM律速は変わらないが「測れない」は撤回**＝per-collection vectorsは測れる（真の律速=HNSW込み実RAM常駐のみ概算）。

### Turso（東京）— 索引肥大律速 ＋ 投入律速
- storage 729MB = **5GB の 14.6%**（索引生存中）。索引 710.8MB = 生ベクトル 13.3MB の **53倍**（DiskANN 肥大）。
- **投入が律速**: upsert 880.90s（9.8 rows/s・逐次DiskANN）= Supabase（1442.1 rows/s）の約147倍遅い。
- **読み出しは安い**: rows_read 3,131,435（無料 500,000,000/月の **0.6%**）／rows_written 496,276（無料 10,000,000/月の **5%**）。〔Platform API 実測 2026-06-25〕
- 回収: **VACUUM は Turso Cloud で不可**だが **DROP INDEX/作り直しで課金ストレージは回収**（DROP後 storage_bytes = 8,192 ＝ 8KB）。「物理回収不可」は誤りとして撤回済み。

### Cloudflare（global edge・実測M4）— stored律速
- **stored = 3,330,816 dims / 5,000,000 = 66.6%**（4DB中最も窮屈＝最初に詰まる枠）。
- queried（CF課金式 =（queries＋stored）× dim）: 初回1評価 = (1,406＋8,674)×384 = **3,870,720 dims**。
- 月あたり評価回数 N ≤ (30,000,000 − 3,330,816) // (1,406×384) = **49回/月**（stored を月1回引いた残りを query 分で割る）。
- ★**実測（M4・無料枠で実走）**: 8,674件を Vectorize に投入し stored=3,330,816（=算術と一致＝全件stored確認）。Task品質 nDCG@10=0.5019/Recall@10=0.7909、**ANN Recall@10=0.9988＝天井近傍に収束**（4本目も品質収束・実近似で0.9988<1.0）。e2e p50=536.03ms（REST×グローバルエッジ＝4DBで最重）／upsert 184.2 rows/s。※Vectorizeの `vectorCount` は結果整合で非単調にぶれる（実数は8,674・stats()はmax読みで補正）＝§8の落とし穴。

## 結論
無料枠の「持ち」を単一スカラーで語ることはできない。**Supabase=容量、Qdrant=RAM、Turso=索引肥大、Cloudflare=stored次元** と、同一データ・同一**Task品質**でも**最初に枯れる単位が違う**。これが本記事 §7 の主張＝「無料枠の非対称は容量数値でなく*律速の種類* に出る」。（※ANN忠実度は別論点＝近似索引を実際に動かしたのは Supabase HNSW・Turso DiskANN・CF Vectorize の3つ（CFは0.9988<1.0＝総当たりでない実近似）。Qdrantは8.7kでは索引閾値10,000未満ゆえ総当たりで、その≈1.0は厳密検索由来。詳細は REPORT §2.5 脚注。）
