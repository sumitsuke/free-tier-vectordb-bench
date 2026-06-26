# 無料枠ベクトルDB4本、精度は横並び——本当の差は「無料枠がどこで最初に詰まるか」だった（Supabase / Qdrant / Turso / Cloudflare 実測）

> Qiita Tech Festa 2026 のお題「**AI時代のデータベース、何が変わる？**」に、**無料枠 × DB横断 × 再現可能な実測**という角度から答えます。
> 対象: **Supabase(pgvector) / Qdrant Cloud / Turso(libSQL) / Cloudflare Vectorize**。データセット BEIR ArguAna(8,674文書)。ローカルCPUのみ・完全無料。
> **測定器の故障や自分の勘違い（Qdrantが実は総当たりだった等）も含め、訂正の過程を全部公開します**（§8）。

## 30秒でわかる結論（専門用語なし）

無料で使えるベクトルデータベース4つに、**まったく同じ「検索の仕事」**をやらせて比べました。すると——**どれを使っても検索の精度はほぼ同じ**でした。
では何が違うのか。**「無料の上限に、どこが最初にぶつかるか」がDBごとにバラバラ**だったのです。あるDBは保存容量、別のDBはメモリ、別のDBは"検索を速くするための索引"が何十倍にも膨らんで容量を、別のDBは"保存できる量そのものの枠"を——と、最初に音を上げる場所が違いました。
だから無料枠でDBを選ぶコツは「どれが速いか・賢いか」ではなく、**「自分の使い方だと、どの上限に最初にぶつかるか」**を見ることだ、という話です。

*この下は、それを実測で証明していく記事です。技術的な興味がなければ、ここまでで結論は full に伝わっています。*

## TL;DR（結論だけ）

> 観察対象＝**無料枠の4製品・1データセット(ArguAna 8,674件)・無料枠スケール**。以下は"一般法則"でなく**この標本・この条件での観察**です（射程は本文で明示）。

1. **検索品質はDBで変わらない（この条件では）**：4製品とも nDCG@10 = 0.5017〜0.5019 / Recall@10 ≈ 0.79（同一埋め込み・同一統制条件なら、RAGの**検索段(retrieval)**の精度はDB非依存。※n=4・1タスクの観察／品質はLLM生成でなく検索段の指標）。
2. **ANN忠実度も揃えれば厳密kNN参照にほぼ一致**：実際に近似索引を動かした **Supabase(HNSW)・Turso(DiskANN)・Cloudflare(Vectorize) の3エンジン**が ANN Recall@10 = 0.9988〜0.9992 に収束。ローカル実HNSWでも **57,638件まで崖なし**（0.65ptの漸減のみ／100k超は未検証＝外挿）。
3. **本当の差は「無料枠の律速ユニット」がDBごとに別物**：**Supabase=容量 / Qdrant=RAM / Turso=索引肥大 / Cloudflare=stored次元**。同じ8,674件でも「最初に枯れる単位」が違う＝"何ヶ月持つ"を1つの数字で語るのは不正確。
4. **レイテンシ差(e2e)は最大20倍超だが、大半は地理・API経路の交絡**：少なくともSupabase/Qdrantはサーバ側計算<1msで、差の正体は往復RTT（Turso/CFはサーバ側計時N/Aゆえe2e値として読む・後述）。

### サマリ表（全1,406クエリ・既定設定・cosine・topK10・client=JP）

| DB | リージョン | 品質(nDCG/Recall) | ANN Recall@10 | e2e p50 | upsert | 索引サイズ |
|---|---|---|---|---|---|---|
| 厳密kNN(numpy・ANN参照) | local | 0.5017 / 0.7902 | 1.000 | — | — | — |
| **Supabase** pgvector | 東京 | 0.5017 / 0.7909 | 0.9991 | **24.7ms** | 1442/s | 17.7MB(HNSW) |
| **Turso** libSQL | 東京 | 0.5017 / 0.7909 | 0.9992 | 49.7ms | **9.8/s** | **710.8MB(DiskANN=生の53倍)** |
| **Qdrant** Cloud | Oregon | 0.5019 / 0.7916 | 0.9994※ | 142.9ms | 716/s | 非露出 |
| **Cloudflare** Vectorize | global edge | 0.5019 / 0.7909 | 0.9988 | **536ms** | 184/s | 非露出 |

> ※Qdrantの0.9994は**この規模では総当たり(brute-force)由来**で、HNSW近似の忠実度ではない（§6で詳述）。品質列は4DBほぼ同一＝「DBで精度は変わらない」を表で表現。
> ※**律速ユニット・消費率**（無料枠が最初に詰まる単位）は §7 の表を参照。
>
> ⚠ **再現性について**：品質(nDCG/Recall/ANN Recall)と無料枠消費の列は、**同じコード・同じデータ・同じ確認日の公式枠を使えば再生成できます**（マネージドDB側の仕様変更・tie順序・リージョン・ライブラリ版差で微小差はあり得ます）。**レイテンシ列は特にクライアントの地理に依存**します（本記事は client=日本）＝絶対値でなく**相対傾向**で読んでください。

---

## 1. なぜ「無料枠 × DB横断 × 再現可能な実測」なのか

「AI時代のデータベース、何が変わったか」。本記事の答えは、**埋め込みがストレージを"高次元の索引構造"に変えたことで、無料枠で最初に詰まる単位が『行数/バイト』から『索引そのもの』へ移った**——そしてそれを**無料枠だけでローカルCPUから横断実測できる時代になった**、というものだ（この"律速の移動"を実データで示すのが本記事）。

日本語圏の先行記事を見ると、(a)エンタープライズ大規模の多製品比較、(b)単一DB内の手法比較、(c)機能スペック表が多い。とくに **BEIR×nDCG×Qdrant を扱う先行記事には、Qdrant上で検索手法(Dense/BM25/Hybrid/Rerank)を比較するものがある**。本記事は逆で、**手法を固定し、DBと無料枠を変える**。軸が直交するので住み分く。「無料枠を厳密な制約にして、横断で、再現可能に実測した」記事は空白だった。

売りは**再現性**：`git clone` → 無料の鍵を`.env`へ → `python -m harness.bench`。**品質と消費の表はどこでも同じ**（レイテンシだけは地理依存）。埋め込みはローカルCPUで約3分、GPU不要。

## 2. 実験設定（統制条件）

- **データセット = BEIR ArguAna（corpus 8,674 / queries 1,406 / qrels 1,406）**。
  選定理由：**Cloudflare Vectorizeの無料 stored 上限が「500万次元 ÷ 384次元 ≈ 1.3万ベクトル」と市場で最も小さい**。だからコーパスを小さく保てるArguAnaにした（FiQA 57,638件は**このCF無料stored上限(≈1.3万本)の約4.4倍**で無料枠外。間引くと正解集合qrelsが壊れる）。
- **埋め込み = `all-MiniLM-L6-v2`（384次元・Apache-2.0・CPU）**、`normalize_embeddings=True`（cosine=内積）。`max_seq_len=256`で長文は切られ、**絶対nDCGはこの小型モデル相応に低い**（より強い埋め込みなら上がるが、それはDB比較とは直交）。全DB共通なので横断比較は不変。
- **厳密kNNは"ANN忠実度の参照"**：numpyブルートフォースcosine top10を基準に固定（1,406×8,674がわずか0.1秒）。各DBのANN結果はこの参照に対し採点する。※これは**同一埋め込み・同一cosineでの検索順序の参照**であって、qrels基準のTask品質(nDCG/Recall)の"上界"ではない（DBがtie順序の都合で微小に上回り得る）。
- **自己一致除外が必須**：ArguAnaはクエリ自身がコーパスに入り、**生top1の91%(1,281/1,406)が"自分自身"**。除外しないとnDCGが壊滅的に過大評価される。規則は`doc_id == query_id`（正解は常に別ID＝BEIR標準の`ignore_identical_ids`と論理一致で安全）。
- 参照の nDCG@10 = **0.5017** は、HuggingFaceモデルカードのMTEB ArguAna公開値 **50.17（=0.5017・2026-06-26再確認）と一致**＝パイプラインが標準どおりに動いている外部証拠（Task指標ではDBが+0.0002程度上回る0.5019も出るが、これは"参照破り"でなくtie/順序の揺れ）。

## 3. 4つの無料枠の「事実」（公式ページ・2026-06-26 再確認／ドリフトなし）

| DB | 無料枠 | クレカ | 索引 | 休止ポリシー |
|---|---|---|---|---|
| Supabase(pgvector) | DB 500MB・2プロジェクト | 不要 | HNSW(調整可) | **7日でpause** |
| Qdrant Cloud | RAM 1GB / disk 4GB | 不要 | HNSW(ef調整可) | **1週でsuspend・4週で削除** |
| Turso(libSQL) | storage 5GB / 読5億・書1000万 行/月 | 不要 | DiskANN(ノブ無し) | **常時稼働(休止なし)** |
| Cloudflare Vectorize | stored 500万・queried 3000万 dims/月 | 不要 | managed(black-box) | edge(従来型cold startなし) |

- **東京リージョンの可否が地味に効く**：Supabase/Turso は東京を取れたが、**Qdrant無料は確認日(2026-06-26)時点でアジア不可**（最寄りのOregonを選択。無料リージョンは変わり得るので日付付きで）。Cloudflareはグローバルエッジで単一リージョンの概念がない。→ レイテンシは相対傾向で読む。
- 4DBともクレカ不要で開始できた。

> 出典（2026-06-26確認）: [Supabase Pricing](https://supabase.com/pricing)（Free=500MB・2プロジェクト・1週でpause）／ [Qdrant Pricing](https://qdrant.tech/pricing/)（Free=0.5vCPU/1GB RAM/4GB disk）／ [Turso Pricing](https://turso.tech/pricing)（Free=5GB・読5億/書1000万 行/月・CC不要）／ [Cloudflare Vectorize Pricing](https://developers.cloudflare.com/vectorize/platform/pricing/)（Free=stored 500万/queried 3000万 dims/月）。無料枠は改定され得るため、公開直前に再取得し確認日を更新すること。

## 4. ハーネス設計（要点）

統一インターフェース(`create/upsert/search/stats/teardown`)に各DBを実装。横断で公平にするための要点だけ：**全DB `FETCH_K=30`で取得→自己一致除外→top10に統一**（除外で件数が欠けない）／**scoreは全DB「類似度（大きいほど良い）」に正規化**／**IDはQdrantだけ整数必須**（他は文字列そのまま）／**レイテンシは e2e と サーバ側を分離**、ウォームアップ20破棄・各クエリ3反復でp50/p95。詳細はリポジトリ参照。

## 5. 結果①：品質はDBで変わらない（俗説の反証）

4DBとも **nDCG@10 = 0.5017〜0.5019 / Recall@10 ≈ 0.79**（参照=MTEB公開値0.5017とほぼ一致）。
同じ埋め込みベクトルを入れているのだから当然ともいえるが、**「どのベクトルDBを使うかでRAG検索の精度が変わる」は、本記事の統制条件（同一埋め込み・同一前処理・同一metric・フィルタなし・topK固定・ANN Recall≈1）では成立しない**ことを実データで示せた。差が出るとすれば埋め込みモデルや前処理であって、DBの選択ではない。
<details><summary>📐 この結論の射程と正直な留保（実務での差・小規模ゆえの必然性／クリックで展開）</summary>

- 実務では、DBごとの**メタデータ・フィルタ性能／量子化／ハイブリッド検索／更新直後の整合性**で差が出る余地はある。本記事が言えるのは「**素の検索精度**は今回の条件で横並びだった」まで。
- **8,674件という小規模では近似誤差が出にくく、「品質が揃う」のは設計上ある程度予定された面もある**。だから§6でローカル実HNSWを57.6kまで伸ばし崖が出ないか確かめた——が、それでも**実クラウドDBの大規模検証は無く外挿**。よって本結論の射程は厳密には**"無料枠スケール(〜数万件)"に限る**（大規模は Part2 の宿題）。

</details>

## 6. 結果②：ANN忠実度は揃えれば厳密kNN参照にほぼ一致（ただし"誰が本当に近似していたか"に注意）

近似索引を実際に動かした **3エンジン(Supabase/Turso/Cloudflare)で ANN Recall@10 = 0.9988〜0.9992 に収束**。揃えればどのDBも厳密kNN参照にほぼ一致する。

…のだが、ここで**監査中に自分の重大な誤りを見つけた**。
**Qdrantは8,674件では実はHNSW索引を張っておらず、総当たり(brute-force)で検索していた**（一次実測で `indexed_vectors_count=0`）。つまり**Qdrantの0.9994は、HNSW近似の忠実度ではなく、総当たり/未index検索に近い挙動で高く出た値**として扱うべき（1.0でないのはtie順序・float差・FETCH_K処理の揺れ）。表では別枠の※にした。**ANN忠実度を測るには、Qdrantが索引を張る規模で測る必要がある**——だから本記事の規模検証はローカルfaissに委ねた。

<details><summary>📐 なぜ総当たりに？→閾値は"件数"でなく"kB"・しかも2段（移植可能な教訓／クリックで展開）</summary>

鍵は`indexing_threshold`が**「件数」でなく「未索引ベクトルの"サイズ(kB)"」のしきい値**であること（公式：*未索引ベクトルのサイズがこの値(kB)を超えると索引セグメントを構築*）。本コーパスは 8,674本×384次元×4B ≈ **13,000kB** が **`segments_count=2`（collection-info 実測）** に分かれ**各≈6,500kB**で、私のクラスタの `indexing_threshold=10,000kB`（同じくcollection-info実測。※既定値は版により10,000/20,000kBの差があるが本機は実測10,000kB）を**下回る**ため `indexed_vectors_count=0` になった（強制構築すると8,674に到達）。

しかも**閾値は2段**ある：索引を強制構築しても、**`full_scan_threshold`（既定10,000kB）**を下回るクエリは plannerが総当たりを選ぶ。**「しきい値は件数でなくkB・しかも2段」**が移植可能な教訓——*ANN忠実度を測るなら両閾値を超える規模で*。（小規模では総当たりの方がHNSW構築より速く・正確なので、Qdrantが索引を作らないのは設計どおりの正しい挙動。）

</details>

**「8,674件の偶然では？」を潰すため、ローカルでfaiss実HNSWを規模スイープ**した（FiQAを埋めて8.7k→57.6k）。値は`scale_sweep.csv`そのまま：

| N | ANN Recall@10 (ef=64) | (ef=16) |
|---|---|---|
| 8,674 | 0.9945 | 0.9195 |
| 25,000 | 0.9902 | 0.9102 |
| 50,000 | 0.9884 | 0.9020 |
| 57,638 | 0.9880 | 0.9028 |

![図1: 実HNSWでも規模を上げて崩れない（57,638件まで崖なし・ef=64）](https://raw.githubusercontent.com/axiom-pro/free-tier-vectordb-bench/main/assets/fig1_scale_sweep.png)

**ef=64なら57.6kまで単調に0.65ptだけ漸減し、崖は出ない**（efで回復可能・100k超は外挿で未検証）。「収束は小規模の偶然」ではなく、実HNSWでも規模を上げて崩れないことを確認できた。
> ※このスイープのfaiss値(8,674で0.9945)は、本表のクラウドDB(Supabase 0.9991)と**ef/m設定が違う**ので**絶対水準は直接比較しない**。見たいのは「規模を上げても崖が出ない＝傾向」であって、各DBの忠実度の優劣ではない。

**おまけ：e2e差の正体を実測で剥がす。** サーバ側計算p50は **Supabase 0.45ms / Qdrant 0.81ms（いずれも<1ms）**。同一リージョンで比べられるのは東京の Supabase 24.7ms vs Turso 49.7ms だけ（エンジン差で約2倍）。**Supabase東京↔Qdrant Oregon の e2e差(24.7↔142.9ms)は、両者ともサーバ側<1msゆえ大部分が地理・API経路・RTTの交絡**で、エンジンの素の速さの差ではない。Turso/CFはサーバ側計時を取得できないため、ここでは**e2e値としてのみ**扱う（地理だけが原因とは言い切らない）。

## 7. 結果③（主峰）：無料枠の"持ち"は「律速ユニット」がDBごとに非対称

ここが山場。同じ8,674件を入れても、**無料枠を最初に使い切らせる単位がDBごとに別物**だった。

| DB | 律速ユニット | 使用量 / 無料枠 | 消費率 |
|---|---|---|---|
| Supabase | DB容量＋7日休止 | 約33MB / 500MB | **6.6%** |
| Qdrant | RAM/disk＋1週suspend | node RSS Δ≈38MB（索引バイトは非露出） | **~3.8%(概算)** |
| Turso | **索引肥大**(storage)＋逐次投入 | 729MB / 5GB（うち索引710.8MB＝生の53倍） | **14.6%** |
| Cloudflare | **stored次元** | 3,330,816 / 5,000,000 | **66.6%** |

![図2: 同じデータ・同じ品質でも「最初に詰まる枠」がDBごとに別物（律速ユニットの非対称）](https://raw.githubusercontent.com/axiom-pro/free-tier-vectordb-bench/main/assets/fig2_limiting_unit.png)

- **Supabase=容量**：約33MBで余裕。むしろ7日休止が先に効く。
- **Qdrant=RAM**：律速はRAM 1GB。面白いことに**コレクションの索引バイト数はAPIに出ない**（生ベクトル分13.3MBは出るが、HNSWを強制構築してもtelemetryの`ram/disk_usage_bytes`は0＝露出されないだけで、ゼロコストではない）。実RAMは投入前後の**node RSS差で約38MB増を観測**＝1GBの約3.8%（上界推定）。Supabaseの17.7MBやTursoの710.8MBと直接比較できないのが、逆に"測れない律速"の好例。
- **Turso=索引肥大**：屈指の隠れコスト。**Tursoは行数でなくstorageが律速。投入147倍遅・索引53倍肥大は"逐次索引"という同一原因の表裏**だ。**生ベクトル13.3MBが、DiskANN索引で710.8MB（53倍）に膨張**し、**投入は9.8件/秒＝Supabase(1442件/秒)の約147倍遅い**（DBの絶対性能差でなく**"逐次索引 vs 一括後構築"という今回の投入パターン差**）。この2つが同根なのは、TursoがDiskANNグラフを挿入のたびに逐次更新するため、投入(約880秒)のほぼ全部がその書き直しに消えるから。**同じDBでもメーターは複数あり、最初に枯れるのは1つ**——という本記事の核心を体現する。
  <details><summary>📐 Turso運用Tips・行課金の内訳（クリックで展開）</summary>

  - **投入パターンの対比**：Supabaseは**COPY一括(2.8秒)→HNSWを後からまとめて構築(3.2秒)**で済むのに対し、TursoはDiskANNを逐次更新するため約880秒かかる。
  - **行課金はどちらも余裕**：読み0.6%(3,131,435/5億)・書き5%(496,276/1000万)＝Tursoを律速するのは行数でなくstorageだけ。
  - **ストレージ回収**：VACUUMはTurso Cloudで不可だが、DROP/作り直しで課金ストレージは回収できる（DROP後8KB）。

  </details>
- **Cloudflare=stored次元**：**stored枠66.6%＝4DB最窮、Workers Freeでは月約49評価が上限**（stored律速の痛みがいちばん早く来る）。queriedはCF公式の**月間計算式で（月内クエリ数＋stored数）×次元**として表現される（公式例 `(30,000+10,000)×768`＝月間式で stored が一度だけ足される形）。一般化すると **月消費 ≈ (コーパス件数 ＋ クエリ数 × 評価回数) × 次元**。**同じindexを維持・stored数一定・他queryを投げない前提**で本記事に代入すると、N回フル評価で **(8,674＋1,406×N)×384** と近似でき、30M/月では **約49評価/月（≈1.6回/日）** が上限になる（初回1評価387万dimsを30Mで単純割りした7.75回は、storedを毎評価二重計上した誤り）。

**結論：無料枠の"持ち"を単一スカラーで語ることはできない。** 「最初に枯れる単位」がSupabase=容量、Qdrant=RAM、Turso=索引肥大、Cloudflare=stored次元と違うからだ。**埋め込みがストレージを高次元索引に変えたAI時代だからこそ、選定軸は「速い/賢い」でなく"自分のワークロードで最初に枯れる律速ユニットはどれか"に変わった**——価格表でなく律速で選ぶ。

**用途別・律速早見表**（本文の実測値だけで構成・新事実なし）：

| こう使うなら | 最初に枯れる枠 | 無難な選択 | 避けたい |
|---|---|---|---|
| 1万件規模を月数十回評価 | queried枠（CFは月約49評価が上限） | Supabase / Turso | Cloudflare |
| 常時稼働・即応性重視 | cold start／休止 | Turso(常時稼働) / CF(cold startなし) | 休止ありの Supabase / Qdrant |
| 容量・件数を多く積みたい | 索引肥大・stored枠 | Supabase(33MBで余裕) | Turso(索引53倍肥大) / CF(stored 66.6%) |

> ※あくまで本記事の標本(ArguAna 8,674件・無料枠スケール)での傾向。用途と規模が変われば律速も変わる。

## 8. 落とし穴ログ（測ってから語る・自分の誤りも含めて）

この企画でいちばん学びがあったのは、**測ってみて初めて分かった/自分が間違っていた**ことたち。「誤→正」で並べる。

- **Qdrantは8.7kで総当たりだった**（§6）。誤=「天井到達はHNSW忠実度の証拠」→ 正=「indexed_vectors_count=0、近似してるつもりで総当たり」。一次実測で覆して訂正。
- **Tursoの肥大を測る計器自体が壊れていた**。誤=`pragma_page_count`の約695MB → 正=それはTurso Cloudで行数に無関係な定数。dbstatで測り直して53倍を確定。
- **Cloudflare Vectorizeの非同期/結果整合**。誤=「upsert直後にqueryできる」→ 正=**upsertは非同期で反映待ち(バリア)が要る**（実測上の罠であると同時に、公式Client APIにも「upsert後にquery可能になるまで通常数秒」と明記された仕様）。さらに**vectorCountが結果整合で非単調にぶれる**(8,674↔5,000)。これはエッジ分散×結果整合の設計上の帰結で、書込み直後のread-your-writesが保証されない＝**vectorCountは課金照合のソースにできない**。逆に従来型cold startが無いのはこの設計の裏返しの利点。
- **Vectorizeの index は delete→即再作成がレース**：削除済みの名前がしばらく`410 index deleted`を返し初回upsertが落ちた→「再利用or作成（deleteしない）」で回避。
- **管理画面が母国語自動翻訳で死ぬ**：TursoダッシュボードがブラウザのEN→JA翻訳とReactのDOM操作衝突で`removeChild`クラッシュ→**GUIを諦めREST APIで構築**したら一発。「無料DBはGUIよりAPIで作れ」。
- **単位の混在**：容量%をバイナリ(1024)と10進(1000)で混在計算し、同じ値が6.3%と6.6%に割れていた→10進SIに統一。

これらは全部、**新規の上位モデル監査エージェント＋一次裏取り**で公開前に潰した。「鵜呑みにしない」を自分の成果にも適用したのが効いた。

## 9. cold start / 休止ポリシー

「速さ」の裏で効くのが休止だ。**Tursoだけが常時稼働（休止なし）**で、これは地味だが大きい。Supabaseは7日、Qdrantは1週でアイドル休止し初回アクセスで復帰待ちが発生しうる（Qdrantは4週で削除）。Cloudflareはエッジで従来型cold startが無い。
※**真の"休止からの復帰レイテンシ"は、休止条件(数日アイドル)の再現に時間がかかるため本記事では未測定**。ここでは「常時稼働か/休止ありか」のポリシー比較に留める（測っていないものを測ったとは書かない）。
特に**月数回〜数十回の低頻度評価**では、Supabase/Qdrantは毎回休止明けに当たりうる→**低頻度かつ即応性が要るなら Turso(常時稼働)／CF(cold startなし)が有利**（これはポリシー上の含意であって、復帰レイテンシの実測ではない）。

## 10. 再現手順

```bash
git clone https://github.com/axiom-pro/free-tier-vectordb-bench && cd free-tier-vectordb-bench
python -m venv .venv && ./.venv/Scripts/python.exe -m pip install -r requirements.txt
# ↑ Windowsパス。macOS/Linuxは ./.venv/bin/python（以降の python も同様にvenvのものを使う）
# 無料の鍵を .env に設定（変数名は .env.sample 参照）：
#   SUPABASE_DB_URL=postgresql://...                         # Supabase接続文字列
#   QDRANT_URL=...   QDRANT_API_KEY=...                       # Qdrant Cloud
#   TURSO_DATABASE_URL=libsql://...   TURSO_AUTH_TOKEN=...    # Turso
#   CF_ACCOUNT_ID=...   CF_API_TOKEN=...   CF_VECTORIZE_INDEX=arguana-minilm   # Cloudflare
python -m harness.download_data && python -m harness.embed   # ArguAna取得＋CPU埋め込み(約3分)
python -m harness.exact_knn                                   # 厳密kNN（ANN忠実度の参照）
python -m harness.bench --db all                              # 3DB実測 → results/summary.csv
python -m harness.bench --db cf --no-teardown                 # Cloudflare(4本目)。反映待ち(非同期)があるので分離・--no-teardownでindex保持(§8参照)
python -m scripts.consumption_report                          # §7 消費メーター表
```

CPUのみ・GPU不要。Cloudflareは`wrangler login`(ワンクリックOAuth)→`wrangler vectorize create`で無料作成できる。

**検証環境（再現性のため）**：Python 3.12 / sentence-transformers 5.6.0 / numpy 2.5.0 / qdrant-client 1.18.0 / psycopg 3.x / libsql-client 0.3.1 / faiss-cpu 1.14.3（規模スイープ用）。全ピンは `requirements.lock.txt`。データセット＝BEIR ArguAna（TU Darmstadt公式zip）、モデル＝`sentence-transformers/all-MiniLM-L6-v2`。実行OS/CPU等の詳細は `results/metadata.json`（本記事は Windows 11・CPU・client=日本）。

## まとめ：AI時代のDB選定で「何が変わったか」

無料枠だけで、ローカルCPU3分の埋め込みから、4つのベクトルDBに**同一RAG検索ワークロード**を流して横断実測できた。分かったのは：

1. **品質はDBで変わらない**（同一埋め込みなら）。
2. **ANN忠実度も揃えれば厳密kNN参照にほぼ一致**（実HNSWで57.6kまで崖なし・100k超は外挿）。
3. **本当の差は"無料枠の律速ユニット"の非対称**（容量/RAM/索引肥大/stored次元）。

埋め込みがストレージを高次元索引に変えたことで、無料枠の限界は「行/バイト」から「索引そのもの（Turso 53倍肥大・CF stored枠）」へ移った。だから選定の軸は「どれが速い/賢い」でなく、**「自分のワークロードで最初に枯れる単位はどれか」**——価格表でなく律速で選ぶ、それが無料枠時代のDB選定だと思う。

---

*計測コード・全CSV・落とし穴ログはリポジトリに同梱（再現可能）。数値はすべて実測値で、検証できなかったもの（休止復帰レイテンシ・100k超の規模）は「測っていない」と明記した。*

リポジトリ：[github.com/axiom-pro/free-tier-vectordb-bench](https://github.com/axiom-pro/free-tier-vectordb-bench)

---

### 筆者について

**tauridev**（ソフトウェア開発／AIコード監査）。Rust/Tauri＋React/TypeScript でローカルファーストのWindowsデスクトップアプリを開発しつつ、**「AIに本番品質のコードを出させ、AIが間違っているときに気づく検証規律」**を専門にしています。本記事の §8（自分の測定器の故障や "Qdrantが実は総当たりだった" という思い込みを、一次データで突き止めて訂正した過程）は、その検証規律をそのまま実演したものです。

「AIで一気に作ったが本番に出せず詰まった」「測ってみたら想定と違った」——そういうコードやデータの**監査・修正・本番化**を、テキスト完結・固定価格で承っています。

- 🔍 **AIで作ったアプリの不具合を監査・修正します**（テスト/CI整備・実装と表示のズレ検出・本番化）— [ココナラで見る](https://coconala.com/services/4282365)
- 💻 **ローカル完結のWindowsアプリ作ります**（データを外に出さない買い切り型・通話なし・テキスト完結）— [ココナラで見る](https://coconala.com/services/4282483)
- プロフィール：[coconala.com/users/6153961](https://coconala.com/users/6153961) ／ 公式サイト：[getaxiom.dev](https://getaxiom.dev)
