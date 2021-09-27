# Databricks notebook source
# ライブラリのインストール
%pip install pycountry

# COMMAND ----------

# MAGIC %md 
# MAGIC # 2. CORD-19データセットの分析
# MAGIC ### COVID-19 Open Research Dataset Challenge (CORD-19) 作業用ノートブック
# MAGIC 
# MAGIC このノートブックは、CORD-19データセットの分析を容易に始められるようにするための、 [COVID-19 Open Research Dataset Challenge (CORD-19)](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) に対する作業用ノートブックです。  
# MAGIC 
# MAGIC <img src="https://miro.medium.com/max/3648/1*596Ur1UdO-fzQsaiGPrNQg.png" width="900"/>
# MAGIC 
# MAGIC アトリビューション:
# MAGIC * このノートブックで使用されるデータセットのライセンスは、[downloaded dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/download)に含まれるメタデータcsvに記載されています。
# MAGIC * 2020-03-03のデータセットには以下が含まれています。
# MAGIC   * `comm_use_subset`: 商用利用のサブセット (PMCコンテンツを含む) -- 9000 論文(内3論文は空), 186Mb
# MAGIC   * `noncomm_use_subset`: 非商用利用のサブセット (PMCコンテンツを含む) -- 1973 論文(内1論文は空), 36Mb
# MAGIC   * `biorxiov_medrxiv`: bioRxiv/medRxiv サブセット (ピアレビューされていない準備稿) -- 803 論文, 13Mb
# MAGIC * DatabricksあるいはDatabricksコミュニティエディションを使用する際には、`/databricks-datasets/COVID/CORD-19`からデータセットのコピーを利用できます。
# MAGIC * このノートブックは[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/us/)のライセンスの下で共有することができます。
# MAGIC 
# MAGIC > **注意**<br>
# MAGIC > このノートブックを実行する前に「1. JSONデータセットの読み込み」を実行して、ファイルを準備してください。

# COMMAND ----------

# ユーザーごとに一意のパスになるようにユーザー名をパスに含めます
import re
from pyspark.sql.types import * 

# Username を取得
username_raw = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
# Username の英数字以外を除去し、全て小文字化。Username をファイルパスやデータベース名の一部で使用可能にするため。
username = re.sub('[^A-Za-z0-9]+', '', username_raw).lower()

print(username)

# COMMAND ----------

# MAGIC %md ## Parquetパス変数の設定
# MAGIC 
# MAGIC `/tmp/<ユーザー名>/COVID/CORD-19/2020-03-13/`にParquetフォーマットで保存されています。

# COMMAND ----------

# PythonにおけるPathの設定
comm_use_subset_pq_path = f"/tmp/{username}/COVID/CORD-19/2020-03-13/comm_use_subset.parquet"
noncomm_use_subset_pq_path = f"/tmp/{username}/COVID/CORD-19/2020-03-13/noncomm_use_subset.parquet"
biorxiv_medrxiv_pq_path = f"/tmp/{username}/COVID/CORD-19/2020-03-13/biorxiv_medrxiv/biorxiv_medrxiv.parquet"
json_schema_path = "/databricks-datasets/COVID/CORD-19/2020-03-13/json_schema.txt"

# シェル環境変数の設定
import os
os.environ['comm_use_subset_pq_path']=''.join(comm_use_subset_pq_path)
os.environ['noncomm_use_subset_pq_path']=''.join(noncomm_use_subset_pq_path)
os.environ['biorxiv_medrxiv_pq_path']=''.join(biorxiv_medrxiv_pq_path)
os.environ['json_schema_path']=''.join(json_schema_path)

# COMMAND ----------

# MAGIC %md ## Parquetファイルの読み込み
# MAGIC 
# MAGIC これらは適切に整形されたJSONファイルのなので、これらのファイルを読み込むために`spark.read.json`を使用できます。*multiline*オプションを指定する必要があることに注意してください。

# COMMAND ----------

# ファイルの読み込み
comm_use_subset = spark.read.format("parquet").load(comm_use_subset_pq_path)
noncomm_use_subset = spark.read.format("parquet").load(noncomm_use_subset_pq_path)
biorxiv_medrxiv = spark.read.format("parquet").load(biorxiv_medrxiv_pq_path)

# COMMAND ----------

# レコード数のカウント
comm_use_subset_cnt = comm_use_subset.count()
noncomm_use_subset_cnt = noncomm_use_subset.count()
biorxiv_medrxiv_cnt = biorxiv_medrxiv.count()

# 出力
print ("comm_use_subset: %s, noncomm_use_subset: %s, biorxiv_medrxiv: %s" % (comm_use_subset_cnt, noncomm_use_subset_cnt, biorxiv_medrxiv_cnt))

# COMMAND ----------

# MAGIC %sh 
# MAGIC cat /dbfs$json_schema_path

# COMMAND ----------

comm_use_subset.createOrReplaceTempView("comm_use_subset")
comm_use_subset.printSchema()

# COMMAND ----------

# MAGIC %md ## 著者の抽出
# MAGIC 
# MAGIC 論文の地理的位置を特定するために著者のメタデータを抽出し、一時ビュー`paperAuthorLocation`を作成します。

# COMMAND ----------

# MAGIC %sql
# MAGIC select paper_id, metadata.title, metadata.authors, metadata from comm_use_subset limit 10

# COMMAND ----------

paperAuthorLocation = spark.sql("""
select paper_id, 
       title,  
       authors.affiliation.location.addrLine as addrLine, 
       authors.affiliation.location.country as country, 
       authors.affiliation.location.postBox as postBox,
       authors.affiliation.location.postCode as postCode,
       authors.affiliation.location.region as region,
       authors.affiliation.location.settlement as settlement
  from (
    select a.paper_id, a.metadata.title as title, b.authors
      from comm_use_subset a
        left join (
            select paper_id, explode(metadata.authors) as authors from comm_use_subset 
            ) b
           on b.paper_id = a.paper_id  
  ) x
""")
paperAuthorLocation.createOrReplaceTempView("paperAuthorLocation")

# COMMAND ----------

# MAGIC %md ## 著者の国データの問題
# MAGIC 
# MAGIC `authors.affiliation.location.country`には`USA,USA,USA,USA`と言ったデータが含まれている問題があります。

# COMMAND ----------

# MAGIC %sql
# MAGIC select *
# MAGIC   from (
# MAGIC     select paper_id, metadata.title as title, explode(metadata.authors) as authors from comm_use_subset 
# MAGIC   ) a
# MAGIC where authors.affiliation.location.country like '%USA, USA, USA, USA%'

# COMMAND ----------

# MAGIC %md ### データのクレンジング
# MAGIC 
# MAGIC 著者の国データをきれいにしましょう。

# COMMAND ----------

# MAGIC %md ### paperAuthorLocationの確認
# MAGIC 
# MAGIC 一時ビュー`paperAuthorLocation`を確認します。

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from paperAuthorLocation limit 200

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(1), count(distinct paper_id) as papers from paperAuthorLocation

# COMMAND ----------

# MAGIC %md ### 国データの抽出
# MAGIC 
# MAGIC 一時ビュー`paperAuthorLocation`から国データ(`paperCountries`)を抽出します。

# COMMAND ----------

paperCountries = spark.sql("""select distinct country from paperAuthorLocation""")
paperCountries.createOrReplaceTempView("paperCountries")

# COMMAND ----------

# MAGIC %md ### pycountryの活用
# MAGIC 
# MAGIC それぞれの国からalpha_3コードを抽出するために`pycountry`を使用しましす。

# COMMAND ----------

# インポート
import pycountry

# alpha_3の国コードの検索 (pycountryを使用)
def get_alpha_3(country):
    try_alpha_3 = -1
    try:
        try_alpha_3 = pycountry.countries.search_fuzzy(country)[0].alpha_3
    except:
        print("Unknown Country")
    return try_alpha_3

# UDF(ユーザー定義関数)として登録
spark.udf.register("get_alpha_3", get_alpha_3)

# COMMAND ----------

# from pyspark.sql.functions import pandas_udf, PandasUDFType

# # Use pandas_udf to define a Pandas UDF
# @pandas_udf('double', PandasUDFType.SCALAR)
# # Input/output are both a pandas.Series of doubles

# def pandas_plus_one(v):
#     return v + 1

# df.withColumn('v2', pandas_plus_one(df.v))

# COMMAND ----------

# MAGIC %sql
# MAGIC select country, get_alpha_3(country) as alpha_3 from paperCountries

# COMMAND ----------

# MAGIC %md ### 国データのクレンジングのステップ

# COMMAND ----------

# ステップ 1: 容易に識別可能な国の alpha_3 の抽出
paperCountries_s01 = spark.sql("""select country, get_alpha_3(country) as alpha_3 from paperCountries""")
paperCountries_s01.cache()
paperCountries_s01.createOrReplaceTempView("paperCountries_s01")

# COMMAND ----------

# ステップ 2: 分割することで識別可能な国の alpha_3 の抽出 (例 "USA, USA, USA", "Sweden, Norway", etc)
paperCountries_s02 = spark.sql("""
select country, splitCountry as country_cleansed, get_alpha_3(ltrim(rtrim(splitCountry))) as alpha_3
  from (
select country, explode(split(regexp_replace(country, "[^a-zA-Z, ]+", ""), ',')) as splitCountry
  from paperCountries_s01
 where alpha_3 = '-1'
 ) x
""")
paperCountries_s02.cache()
paperCountries_s02.createOrReplaceTempView("paperCountries_s02")

# COMMAND ----------

# ステップ 3: (ステップ1とステップ2の後で)いまだ識別できていない国の抽出  
paperCountries_s03 = spark.sql("""select country, ltrim(rtrim(country_cleansed)) as country_cleansed, get_alpha_3(country_cleansed) from paperCountries_s02 where alpha_3 = -1""")
paperCountries_s03.cache()
paperCountries_s03.createOrReplaceTempView("paperCountries_s03")

# COMMAND ----------

# ステップ 4: settlementから国を識別
paperCountries_s04 = spark.sql("""
select distinct m.country_cleansed, f.settlement, get_alpha_3(f.settlement) as alpha_3
  from paperAuthorLocation f
    inner join paperCountries_s03 m
      on m.country = f.country
""")
paperCountries_s04.cache()
paperCountries_s04.createOrReplaceTempView("paperCountries_s04")

# COMMAND ----------

 # ステップ 5: 新たなマッピングの構築
map_country_cleansed = spark.sql("""select distinct country_cleansed, alpha_3 from paperCountries_s04 where alpha_3 <> '-1'""")
map_country_cleansed.cache()
map_country_cleansed.createOrReplaceTempView("map_country_cleansed")

# COMMAND ----------

# ステップ 6: すてっぷ5のマッピングを用いて paperCountries_s03 を更新
paperCountries_s06 = spark.sql("""
select f.country, f.country_cleansed, m.alpha_3
  from paperCountries_s03 f
    left join map_country_cleansed m
      on m.country_cleansed = f.country_cleansed
 where m.alpha_3 is not null      
""")
paperCountries_s06.cache()
paperCountries_s06.createOrReplaceTempView("paperCountries_s06")

# COMMAND ----------

# MAGIC %md ### map_countryの構築 
# MAGIC 
# MAGIC 上のパイプライン処理に基づきmap_countryを構築します。

# COMMAND ----------

map_country = spark.sql("""
select country, alpha_3 from paperCountries_s01 where alpha_3 <> '-1'
union all
select country, alpha_3 from paperCountries_s02 where alpha_3 <> '-1'
union all
select country, alpha_3 from paperCountries_s06
""")
map_country.cache()
map_country.createOrReplaceTempView("map_country")

# COMMAND ----------

# MAGIC %md ### paperCountryMappedの構築
# MAGIC 
# MAGIC 論文をalpha_3の地理的位置にマップしてすべてをまとめます。

# COMMAND ----------

paperCountryMapped = spark.sql("""
select p.paper_id, p.title, p.addrLine, p.country, p.postBox, p.postCode, p.region, p.settlement, m.alpha_3
 from paperAuthorLocation p
   left outer join map_country m
     on m.country = p.country
""")
paperCountryMapped.cache()
paperCountryMapped.createOrReplaceTempView("paperCountryMapped")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from paperCountryMapped limit 100

# COMMAND ----------

# MAGIC %md ### paperCountryMappedの統計情報

# COMMAND ----------

(ep_no, edp_no) = spark.sql("select count(1), count(distinct paper_id) from paperCountryMapped where country is null and settlement is null").collect()[0]
(ep_geo, edp_geo) = spark.sql("select count(1), count(distinct paper_id) from paperCountryMapped where country is not null or settlement is not null").collect()[0]
(ep_a3, edp_a3) = spark.sql("select count(1), count(distinct paper_id) from paperCountryMapped where alpha_3 is not null").collect()[0]
print("Distinct Papers with No Geographic Information: %s" % edp_no)
print("Distinct Papers with Some Geographic Information: %s" % edp_geo)
print("Distinct Papers with Identified Alpha_3 codes: %s" % edp_a3)

# COMMAND ----------

# MAGIC %md ## 論文と国のマッピングの可視化
# MAGIC 
# MAGIC 論文ごとの著者の国をマッピングします。一つの論文に対して複数の著者がいる場合にはダブルカウントになることに注意してください。

# COMMAND ----------

# MAGIC %sql
# MAGIC select alpha_3, count(distinct paper_id) 
# MAGIC   from paperCountryMapped 
# MAGIC  where alpha_3 is not null
# MAGIC  group by alpha_3

# COMMAND ----------

# MAGIC %md 
# MAGIC # END
