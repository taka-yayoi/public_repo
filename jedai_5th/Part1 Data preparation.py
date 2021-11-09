# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC **DBR: 9.1 LTS ML**
# MAGIC 
# MAGIC クラスターライブラリとして、以下のライブラリをPyPIからインストールしてください。
# MAGIC 
# MAGIC - tldextract
# MAGIC - geoip2
# MAGIC - dnstwist

# COMMAND ----------

# MAGIC %md
# MAGIC 初期化処理

# COMMAND ----------

# MAGIC %run ./utilities/Jedai_mission_impossible_common

# COMMAND ----------

# データベースの準備
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name_silver}")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name_bronze}")

# データ格納場所のクリア
dbutils.fs.rm(work_path, True)
dbutils.fs.mkdirs(work_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ミッションに必要なデータのダウンロード
# MAGIC 
# MAGIC DNSログ：dns_events.json<br>
# MAGIC URLHaus悪質サイトリスト：ThreatDataFeed.txt<br>
# MAGIC DNSTwistの怪しいドメインリスト：domains_dnstwists.csv<br>
# MAGIC Maxmind社地理情報データベース：GeoLite2_City.mmdb<br>
# MAGIC Alexa人気サイトリスト：alexa_100k.txt<br>
# MAGIC ドメイン生成アルゴリズム：dga_domains_header.txt<br>
# MAGIC 辞書データ：words.txt<br>

# COMMAND ----------

# MAGIC %run ./utilities/Jedai_mission_impossible_data_download

# COMMAND ----------

# MAGIC %md
# MAGIC **Thinking Time #1**
# MAGIC 
# MAGIC 最近のサイバー犯罪はDNSを悪用したケースが増えている。これに対処するにはDNSの問い合わせレコードを処理すべきだ。さて、どのようなアプローチをとるべきか。
# MAGIC 
# MAGIC 1. とにかくデータを取り込んで、一気に変換処理を行う！
# MAGIC 2. フェーズごとに異なるテーブルを設け、段階的に処理を行なっていく

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. 事前データ投入
# MAGIC この環境では、すでにAWS S3バケットに保存済みのDNSデータを利用します。我々はS3バケットからデータをインポートします。ステージング、スキーマ適用、分析結果の格納のために複数のテーブルを使用します。以下にテーブル名の説明を示します。
# MAGIC - Bronze: 生データ
# MAGIC - Silver: スキーマが適用され拡張されたデータ
# MAGIC - Gold: 検知およびアラート
# MAGIC 
# MAGIC なぜ、このようなテーブルが必要なのでしょうか？簡単に言えば、いつでもソースに立ち返ることができ、データを失うことなしに分析を繰り返し改善することができるからです。
# MAGIC 
# MAGIC 実際の環境ではリアルタイムにDNSログを監視するため、Databricks AutoLoaderを使用をお勧めします。以下には多くのコードがありますが、多くはAutoLoader向けの定型処理です。AutoLoaderの詳細に関しては、[こちら](https://docs.databricks.com/spark/latest/structured-streaming/auto-loader.html)を参照ください。AutoLoaderの詳細を読む時間がなくても大丈夫です。以下になぜAutoLoaderが重要なのかを示します。
# MAGIC - ファイルの状態管理: 新たなファイルがS3に到着するたびにインクリメントに処理が行われます。どのファイルが到着したのかに関する状態情報を管理する必要がありません。
# MAGIC - スケーラブル: クラウドサービスを活用して新規ファイルの到着を追跡するので、数百万のファイルがディレクトリに到着してもスケールします。
# MAGIC - 使いやすさ: ファイルをインクリメントに処理するために必要な通知、メッセージキューサービスを自動でセットアップします。セットアップは不要です。
# MAGIC 
# MAGIC ここでは、pDNSデータをS3バケットから**Bronzeテーブル**にロードするようにセットアップします。次の**データのロード**セクションまでは実際にはデータはロードしません。

# COMMAND ----------

dbutils.fs.head(f"{work_path}/tables/datasets/dns_events.json")

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *

# pDNSスキーマと詳細はこちらで確認できます: https://tools.ietf.org/id/draft-dulaunoy-dnsop-passive-dns-cof-04.html
# データにはrrnme、rrtype、time_first、time_last、count、bailwick、rdataフィールドが存在します
schema = StructType([ \
    StructField("rrname",StringType(),True), \
    StructField("rrtype",StringType(),True), \
    StructField("time_first",LongType(),True), \
    StructField("time_last", LongType(), True), \
    StructField("count", StringType(), True), \
    StructField("bailiwick", StringType(), True), \
    StructField("rdata", ArrayType(StringType(), True), True) \
  ])

# ここでは、データがどこにあるのかデータのタイプが何であるのかを指定しています
# jsonフォーマットであること、データのパスを確認できます
df = spark.read.format("json").schema(schema).load(f"{work_path}/tables/datasets/dns_events.json")

# COMMAND ----------

df.count()

# COMMAND ----------

display(df)

# COMMAND ----------

# rdataフィールドはarrayとなっています。パーシングや文字列検索を行う際にはこれは有用ではありません。
# このため、rdatastrという新たな列を作成します。以下のサンプルアウトプットで二つのフィールドの違いを確認することができます。
df_enhanced = df.withColumn("rdatastr", concat_ws(",", col("rdata")))
display(df_enhanced)

# COMMAND ----------

# ここで書き込むデータのフォーマットとファイルパスを指定します
# これはまだセットアップ段階です。まだデータはブロンズテーブル(Deltaテーブル)に投入されていません
df_enhanced.write.format("delta").mode("overwrite").option("mergeSchema", "true").save(f"dbfs:{work_path}/tables/tables/bronze/delta/DNS_raw")

# COMMAND ----------

# MAGIC %md
# MAGIC # 1.a 事前データ投入: URLHausの脅威フィードのセットアップ
# MAGIC 
# MAGIC ここでは、pDNSとともにURLHausの脅威フィードを使用します。このセクションではどのようにURLHausのフィードを取り込むのかを説明します。
# MAGIC 
# MAGIC このセットアップにおいては、二つのことを行います。
# MAGIC - URLHausフィードからフィールドを抽出するための関数`registered_domain_extract`、`domain_extract`を定義します。
# MAGIC - 拡張したスキーマをシルバーテーブルとして保存します。

# COMMAND ----------

# URLHausフィードのソース位置、フォーマットがCSVであること、CSVにはヘッダーがあることを指定します
threat_feeds_location = f"dbfs:{work_path}/tables/datasets/ThreatDataFeed.txt"
threat_feeds_raw = spark.read.format("csv").option("header", "true").load(threat_feeds_location)
# 内容が正しいことを確認するためにサンプルを表示します
display(threat_feeds_raw)

# COMMAND ----------

# URLHausフィードからregistered_domain_extractとdomain_extractを抽出します
import tldextract
import numpy as np

def registred_domain_extract(uri):
    ext = tldextract.extract(uri)
    if (not ext.suffix):
        return " "
    else:
        return ext.registered_domain
      
def domain_extract(uri):
    ext = tldextract.extract(uri)
    if (not ext.suffix):
        return " "
    else:
        return ext.domain

#　以下の行ではDatabricksランタイム環境にユーザー定義関数(UDF)を登録しています
spark.udf.register("registred_domain_extract", registred_domain_extract)
spark.udf.register("domain_extract", domain_extract)

# COMMAND ----------

# 上で定義したユーザー定義関数domain_extractorを用いてURLからドメイン名を抽出することで、拡張したビューを作成します
threat_feeds_raw.createOrReplaceTempView("threat_feeds_raw")
threat_feeds_enriched = spark.sql("select *, domain_extract(url) as domain from threat_feeds_raw").filter("char_length(domain) >= 2")
# 表示されるサンプルには新たなフィールド"domain"が追加されます
display(threat_feeds_enriched)

# COMMAND ----------

# 拡張された新たなスキーマで保存します 
threat_feeds_enriched.write.format("delta").mode('overwrite').option("mergeSchema", "true").save(f"dbfs:{work_path}/tables/tables/silver/delta/enriched_threat_feeds")

# COMMAND ----------

# MAGIC %md
# MAGIC # 1.b 紛らわしいドメインを検知するためのDNS Twistのセットアップ
# MAGIC 
# MAGIC 攻撃者があなたにアタックする際に使用する紛らわしいドメインをモニタリングするために<a href="https://github.com/elceef/dnstwist">dnstwist</a>を使用します。<a href="https://github.com/elceef/dnstwist">dnstwist</a>を用いることで、<a href="https://capec.mitre.org/data/definitions/630.html">typosquatters</a>、フィッシング攻撃、詐欺、ブランドの毀損を検知することができます。このノートブックのセクション1.bでは、domains_dnstwists.csvを作成するために<a href="https://github.com/elceef/dnstwist">dnstwistの手順書(このノートブックの外)</a>に従う必要があります。以下の例では、dnstwistを用いてgoogle.comのバリエーションを生成しました。これを自身の企業あるいは興味のある企業向けにカスタマイズすることができます。
# MAGIC 
# MAGIC #### 参考資料
# MAGIC - [タイポスクワッティング \- Wikipedia](https://ja.wikipedia.org/wiki/%E3%82%BF%E3%82%A4%E3%83%9D%E3%82%B9%E3%82%AF%E3%83%AF%E3%83%83%E3%83%86%E3%82%A3%E3%83%B3%E3%82%B0)
# MAGIC 
# MAGIC dnstwistのインストール後に以下を実行します:<br/>
# MAGIC <code>
# MAGIC &nbsp;  
# MAGIC dnstwist --registered google.com >> domains_dnstwists.csv<br/>
# MAGIC addition       googlea.com    184.168.131.241 NS:ns65.domaincontrol.com MX:mailstore1.secureserver.net<br/>
# MAGIC addition       googleb.com    47.254.33.193 NS:ns3.dns.com </code>
# MAGIC 
# MAGIC ここでは、domains_dnstwists.csvをヘッダー: PERMUTATIONTYPE, domain, metaでフォーマットしました。
# MAGIC 
# MAGIC domain_dnstwists.csvを作成したら以下の作業を続けます:
# MAGIC - dnstwistの処理が行われたドメインのロード
# MAGIC - ドメイン名によるテーブルの拡張 (TLD(トップレベルドメイン)は除外)
# MAGIC - dnstwistで拡張された結果をシルバーテーブルにロード
# MAGIC 
# MAGIC これらのテーブルは後ほどのtyposquatting検知のプロダクション化で使用します。

# COMMAND ----------

# 注意: domain_dnstwists.csvは、dnstwistの手順書に従い、このノートブックの外で作成する必要があります
# domain_dnstwists.csvをデータフレームbrand_domains_monitored_rawにロードします。headerのtrueオプションに注意してください
# /dbfs/tmp/brickster/dns/tables/datasets
brand_domains_monitored_raw = spark.read.format("csv").option("header", "true").load(f"dbfs:{work_path}/tables/datasets/domains_dnstwists.csv") 

# COMMAND ----------

# 読み込んだCSVを表示
display(brand_domains_monitored_raw)

# COMMAND ----------

#　brand_domains_monitored_rawをbrand_domains_monitored_rawというローカルテーブルにロード
brand_domains_monitored_raw.createOrReplaceTempView("brand_domains_monitored_raw")

# COMMAND ----------

# このノートブックの前半で作成したUDFを用いてドメイン名を抽出
# dnstwistで抽出したドメインを含む新規テーブルを作成します。新たなカラムはdnstwisted_domainとなります
# ハードコードされた ">=2" は潜在的な空のドメインフィールドに対応するためのものです
brand_domains_monitored_enriched = spark.sql("select *, domain_extract(domain) as dnstwisted_domain  from brand_domains_monitored_raw").filter("char_length(dnstwisted_domain) >= 2")
display(brand_domains_monitored_enriched)

# COMMAND ----------

# シルバーDeltaテーブルの定義
brand_domains_monitored_enriched.write.format("delta").mode('overwrite').option("mergeSchema", "false").save(f"dbfs:{work_path}/tables/datasets/tables/silver/delta/brand_domains_monitored_enriched")

# COMMAND ----------

# エンリッチされたデータによるシルバーDeltaテーブルを作成
spark.sql(f"DROP TABLE IF EXISTS {db_name_silver}.EnrichedTwistedDomainBrand")
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {db_name_silver}.EnrichedTwistedDomainBrand 
USING DELTA LOCATION 'dbfs:{work_path}/tables/datasets/tables/silver/delta/brand_domains_monitored_enriched'
""")

# COMMAND ----------

spark.sql(f"USE {db_name_silver}")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- シルバーDeltaテーブルへのクエリー
# MAGIC Select * from EnrichedTwistedDomainBrand

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. データのロード
# MAGIC 
# MAGIC URLHausとdnstwistの準備に数多くの作業が必要であることは否定できませんが、これでtyposquattingの検知と脅威インテリジェンスのエンリッチメントの準備が整いました。
# MAGIC 
# MAGIC これで、pDNSデータを**ブロンズ**Deltaテーブルにロードすることができます。tldextract、GeoIPルックアップ、DGA分類器、URLHaus、脅威インテリジェンスルックアップを用いてデータのエンリッチを行います。Spark SQLを用いてこれらを実行します。

# COMMAND ----------

spark.sql(f"USE {db_name_bronze}")

# COMMAND ----------

# 新規ブロンズテーブルに事前のステップで保存したpDNSデータをロード
spark.sql(f"DROP TABLE IF EXISTS DNS")
spark.sql(f"""
CREATE TABLE IF NOT EXISTS DNS 
USING DELTA LOCATION 'dbfs:{work_path}/tables/tables/bronze/delta/DNS_raw'
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 何レコードをロードしたのかを確認
# MAGIC Select count(*) from DNS

# COMMAND ----------

# MAGIC %md
# MAGIC **Thinking Time #2**
# MAGIC 
# MAGIC DNSデータをロードしたが、これをいろいろ加工していかないと使い物にならないな。大量データに対してデータ加工処理を行うにはどのようなアプローチを取るべきだろうか？
# MAGIC 
# MAGIC 1. Pandasのapplyだ
# MAGIC 1. Forループを回そう
# MAGIC 1. SparkのUDF(ユーザー定義関数)を使う

# COMMAND ----------

# Geoデータをロードし、操作するためのユーザー定義関数(UDF)を作成
# 以下のコードでは、ブロンズテーブルにおけるrdataフィールドから利用できるIPアドレスを用いてGeo-IPルックアップを実行します
# フリーのgeoデータベースであるMaxmindを使用します: https://dev.maxmind.com/geoip/geoip2/geolite2/ 
import geoip2.database
import geoip2.errors
from pyspark.sql.functions import lit

from geoip2 import database
from pyspark import SparkContext, SparkFiles

# データベースは右からダウンロードできます: https://dev.maxmind.com/geoip/geoip2/geolite2/ 
# Databricks UIを用いてGeoLite2_Cityデータベースファイルをアップロードすることができます
# Databricks Navigator (lefthand bar) -> Data -> Upload File -> Select
city_db = default_file_path + '/GeoLite2_City.mmdb'

def get_country_code(ip):
    if ip is None:
      return None
    
    geocity = database.Reader(SparkFiles.get(city_db))
    try:
      record = geocity.city(ip)
      return record.country.iso_code
    except geoip2.errors.AddressNotFoundError:
      return None
    
def get_country(ip):
    if ip is None:
      return None
    
    geocity = database.Reader(SparkFiles.get(city_db))
    try:
      record = geocity.city(ip)
      return record.country.name
    except geoip2.errors.AddressNotFoundError:
      return None

def get_city(ip):
    if ip is None:
      return None
    
    geocity = database.Reader(SparkFiles.get(city_db))
    try:
      record = geocity.city(ip)
      return record.city.name
    except geoip2.errors.AddressNotFoundError:
      return None
 

spark.udf.register("get_city", get_city)
spark.udf.register("get_country", get_country)
spark.udf.register("get_country_code", get_country_code)

# COMMAND ----------

# DGAモデルをロードします。これは投入されるDNSイベントをエンリッチするために使用する事前学習済みモデルです。後のステップでこのモデルをどのようにトレーニングするのかを説明します
import mlflow
import mlflow.pyfunc

model_path = f'dbfs:{work_path}/tables/model'
loaded_model = mlflow.pyfunc.load_model(model_path)
spark.udf.register("ioc_detect", loaded_model.predict)

# COMMAND ----------

# Aのrrtypeでフィルタリング
# リソースレコードタイプが「A」ということは、ドメインとIPアドレスの対応を示すレコードのみを抽出しています。
dns_table = spark.table(f"{db_name_bronze}.DNS").selectExpr("*", "case when rrtype = 'A' then element_at(rdata, 1) else null end as ip_address ")

# COMMAND ----------

# IPアドレスから位置情報を抽出し、city、country、country codes、ioc、ドメイン名でデータを拡張します
# 事前学習済みDGAモデルを適用します
dns_table_enriched = dns_table.selectExpr("*", "case when ip_address is not null then get_country(ip_address) else null end as country", 
                     "case when ip_address is not null then get_city(ip_address) else null end as city", 
                     "case when ip_address is not null then get_country_code(ip_address) else null end as country_code", 
                     "case when char_length(domain_extract(rrname)) > 5 then ioc_detect(string(domain_extract(rrname))) else null end as ioc",
                     " domain_extract(rrname) as domain_name")

# COMMAND ----------

# 拡張したDNSデータを永続化します
dns_table_enriched.write.format("delta").mode('overwrite').option("mergeSchema", "true").save(f"dbfs:{work_path}/tables/tables/silver/delta/DNS")

# COMMAND ----------

# 拡張されたDNSデータからDeltaテーブルを作成。これは後ほどのDGA分析で使用されます
spark.sql(f"DROP TABLE IF EXISTS {db_name_silver}.DNS")
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {db_name_silver}.DNS 
USING DELTA LOCATION 'dbfs:{work_path}/tables/tables/silver/delta/DNS'
""")

# COMMAND ----------

# 拡張された脅威インテリジェンスをシルバーテーブルにロード
# これは後ほどのDGA分析で使用されます
spark.sql(f"DROP TABLE IF EXISTS {db_name_silver}.EnrichedThreatFeeds")
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {db_name_silver}.EnrichedThreatFeeds 
USING DELTA LOCATION 'dbfs:{work_path}/tables/tables/silver/delta/enriched_threat_feeds'
""")

# COMMAND ----------

spark.sql(f"USE {db_name_silver}")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 何レコードをロードしたのかを確認
# MAGIC Select count(*) from EnrichedThreatFeeds

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. アドホック分析: データの探索
# MAGIC ようやく!!!! データの準備ができました。このセクションはデータへの理解を深めるためのオプションとなります。Spark SQLのいくつかのトリックをご説明します。分析における探索、拡張にこれらの戦略を活用することができます。

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- データセットのユニークなドメインの数を見てみましょう
# MAGIC select count(distinct(domain_name)) from dns

# COMMAND ----------

# MAGIC %sql select count(*) from dns

# COMMAND ----------

# MAGIC %md
# MAGIC - iocはDGA（ドメイン生成アルゴリズム）モデルの実行結果として作成したフィールドです。iocフィールドにiocがある場合、DGAモデルはこのドメインがioc(indicator of compromise:セキュリティ侵害インジケーター)と判断したことを意味します。
# MAGIC - 以下のクエリーはDGAアルゴリズムがiocを検知した件数をカウントしています。

# COMMAND ----------

# MAGIC %sql 
# MAGIC select
# MAGIC   count(*),
# MAGIC   domain_name,
# MAGIC   country
# MAGIC from
# MAGIC   dns
# MAGIC where
# MAGIC   ioc = 'ioc'
# MAGIC   --and domain_name not like '%ip%'
# MAGIC   --and char_length(domain_name) > 8
# MAGIC group by
# MAGIC   domain_name,
# MAGIC   country
# MAGIC order by
# MAGIC   count(*) desc

# COMMAND ----------

# MAGIC %md
# MAGIC 既知の脅威フィードに対してチェックしてみましょう。
# MAGIC 
# MAGIC - iocにマッチしたシルバーテーブルDNS、EnrichedThreatFeedsに対してクエリーを行います。
# MAGIC - かつてあなたは、多くのSIEM/ログ集約システムにおいて、数多くのマッチ/ジョインの計算処理コストが膨大になった経験をお持ちかもしれません。Spark SQLはそれらよりもさらに効率的です。

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct(domain_name))
# MAGIC from dns, EnrichedThreatFeeds where dns.domain_name == EnrichedThreatFeeds.domain

# COMMAND ----------

# MAGIC %md
# MAGIC 複数のテーブルに対してiocマッチのクエリーを行います。上の例と似ていますが、結果テーブルにカラムを追加しています。

# COMMAND ----------

# MAGIC %sql 
# MAGIC select  domain_name, rrname, country, time_first, time_last, ioc, rrtype, rdata, bailiwick, EnrichedThreatFeeds.* 
# MAGIC from dns, EnrichedThreatFeeds where dns.domain_name == EnrichedThreatFeeds.domain and ioc='ioc'

# COMMAND ----------

# MAGIC %md
# MAGIC # To Be Continued
