# Databricks notebook source
# MAGIC %md
# MAGIC ライブラリのインストール

# COMMAND ----------

# MAGIC %pip install tldextract
# MAGIC %pip install geoip2
# MAGIC %pip install dnstwist

# COMMAND ----------

import re

# ログインIDからUsernameを取得
username_raw = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
# Username の英数字以外を除去し、全て小文字化。Username をファイルパスやデータベース名の一部で使用可能にするため。
username = re.sub('[^A-Za-z0-9]+', '', username_raw).lower()

# データベース:SQL Analyticsから参照します
db_name_silver = f"dns_analytics_silver_{username}"
db_name_bronze = f"dns_analytics_bronze_{username}"

# データベースの準備
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name_silver}")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name_bronze}")

# データ格納場所
work_path = f"/tmp/{username}/dns/"

dbutils.fs.rm(work_path, True)
dbutils.fs.mkdirs(work_path)

print("work_path:", work_path)
print("db_name_silver:", db_name_silver)
print("db_name_bronze:", db_name_bronze)

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /tmp/takaakiyayoidatabrickscom/dns/

# COMMAND ----------

# Community Editionで利用する際にはTrue
community_edition = False

# COMMAND ----------

if community_edition == True:
    default_file_path = '/databricks/driver/data'
else:
    default_file_path =  f'/dbfs{work_path}/tables/datasets'

# COMMAND ----------

# MAGIC %scala
# MAGIC displayHTML("""<iframe src="https://drive.google.com/file/d/1ZMu8nFMuCzPZonOJmib8TpFR9JNypS0L/preview" frameborder="0" height="480" width="640"></iframe>
# MAGIC """)

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. 事前データ投入: pDNS AutoLoaderのセットアップ
# MAGIC この環境では<a href ="https://www.farsightsecurity.com/technical/passive-dns/passive-dns-faq/#:~:text=%22Passive%20DNS%22%20or%20%22Passive,can%20be%20indexed%20and%20queried.">passive DNS (pDNS) </a>データが一定間隔でAWS S3バケットに投入されます。我々はS3バケットをモニタリングし、AutoLoaderを用いてデータをインポートします。ステージング、スキーマ適用、分析結果の格納のために複数のテーブルを使用します。以下にテーブル名の説明を示します。
# MAGIC - Bronze: 生データ
# MAGIC - Silver: スキーマが適用され拡張されたデータ
# MAGIC - Gold: 検知およびアラート
# MAGIC 
# MAGIC なぜ、このようなテーブルが必要なのでしょうか？簡単に言えば、いつでもソースに立ち返ることができ、データを失うことなしに分析を繰り返し改善することができるからです。詳細は[こちら](https://databricks.com/blog/2019/08/14/productionizing-machine-learning-with-delta-lake.html)を参照ください。
# MAGIC 
# MAGIC ここでは、Databricks AutoLoaderを使用します。以下には多くのコードがありますが、多くはAutoLoader向けの定型処理です。AutoLoaderの詳細に関しては、[こちら](https://docs.databricks.com/spark/latest/structured-streaming/auto-loader.html)を参照ください。AutoLoaderの詳細を読む時間がなくても大丈夫です。以下になぜAutoLoaderが重要なのかを示します。
# MAGIC - ファイルの状態管理: 新たなファイルがS3に到着するたびにインクリメントに処理が行われます。どのファイルが到着したのかに関する状態情報を管理する必要がありません。
# MAGIC - スケーラブル: クラウドサービスを活用して新規ファイルの到着を追跡するので、数百万のファイルがディレクトリに到着してもスケールします。
# MAGIC - 使いやすさ: ファイルをインクリメントに処理するために必要な通知、メッセージキューサービスを自動でセットアップします。セットアップは不要です。
# MAGIC 
# MAGIC ここでは、pDNSデータをS3バケットから**Bronzeテーブル**にロードするようにセットアップします。次の**データのロード**セクションまでは実際にはデータはロードしません。
# MAGIC 
# MAGIC [DNS分析を通じたサイバー犯罪の検知 \- Qiita](https://qiita.com/taka_yayoi/items/3320a1b26c65385baf75)

# COMMAND ----------

# MAGIC %md
# MAGIC - 以下では、ノートブックの実行に必要なすべてのデータセットをダウンロードします。
# MAGIC - これらのデータセットには、匿名化したDNSデータ、GeoIPルックアップデータベース、我々の補強パイプラインで使用するdnstwistで生成した脅威フィード、ドメイン名が含まれます。
# MAGIC - GDAモデルをトレーニングするためのalexaにおけるトップ100kのドメイン、ディクショナリー単語のリスト、dgaドメインのリストも含めています。
# MAGIC 
# MAGIC #### 参考資料
# MAGIC - [脅威に関する情報: ドメイン生成アルゴリズム\(DGA\)について](https://unit42.paloaltonetworks.jp/threat-brief-understanding-domain-generation-algorithms-dga/)

# COMMAND ----------

# MAGIC %sh mkdir data
# MAGIC mkdir data/latest
# MAGIC mkdir model
# MAGIC curl -o data/dns_events.json https://raw.githubusercontent.com/zaferbil/dns-notebook-datasets/master/data/dns_events.json 
# MAGIC curl -o data/GeoLite2_City.mmdb https://raw.githubusercontent.com/zaferbil/dns-notebook-datasets/master/data/GeoLite2_City.mmdb
# MAGIC curl -o data/ThreatDataFeed.txt https://raw.githubusercontent.com/zaferbil/dns-notebook-datasets/master/data/ThreatDataFeed.txt
# MAGIC curl -o data/alexa_100k.txt https://raw.githubusercontent.com/zaferbil/dns-notebook-datasets/master/data/alexa_100k.txt
# MAGIC curl -o data/dga_domains_header.txt https://raw.githubusercontent.com/zaferbil/dns-notebook-datasets/master/data/dga_domains_header.txt
# MAGIC curl -o data/domains_dnstwists.csv https://raw.githubusercontent.com/zaferbil/dns-notebook-datasets/master/data/domains_dnstwists.csv
# MAGIC curl -o data/words.txt https://raw.githubusercontent.com/zaferbil/dns-notebook-datasets/master/data/words.txt
# MAGIC curl -o data/latest/dns_test_1.json https://raw.githubusercontent.com/zaferbil/dns-notebook-datasets/master/data/dns_test_1.json
# MAGIC curl -o model/MLmodel https://raw.githubusercontent.com/zaferbil/dns-notebook-datasets/master/model/MLmodel
# MAGIC curl -o model/conda.yaml https://raw.githubusercontent.com/zaferbil/dns-notebook-datasets/master/model/conda.yaml
# MAGIC curl -o model/python_model.pkl https://raw.githubusercontent.com/zaferbil/dns-notebook-datasets/master/model/python_model.pkl

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /databricks/driver/data

# COMMAND ----------

# ダウンロードしたデータをワークスペースのDBFSにコピーします
dbutils.fs.cp("file:///databricks/driver/data", f"dbfs:{work_path}/tables/datasets/", True)
dbutils.fs.cp("file:///databricks/driver/model", f"dbfs:{work_path}/tables/model/", True)

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /tmp/takaakiyayoidatabrickscom/dns/tables

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
# これはまだセットアップ段階です。まだデータはブロンズテーブルに投入されていません
df_enhanced.write.format("delta").mode("overwrite").option("mergeSchema", "true").save(f"dbfs:{work_path}/tables/tables/bronze/delta/DNS_raw")

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /tmp/takaakiyayoidatabrickscom/dns/tables/tables

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

# URLHausフィードのソース位置、フォーマットがCSVであること、CSVにはヘッダーがあることを指定します
threat_feeds_location = f"dbfs:{work_path}/tables/datasets/ThreatDataFeed.txt"
threat_feeds_raw = spark.read.format("csv").option("header", "true").load(threat_feeds_location)
# 内容が正しいことを確認するためにサンプルを表示します
display(threat_feeds_raw)

# COMMAND ----------

# 上で定義したユーザー定義関数domain_extractorを用いてURLからドメイン名を抽出することで、拡張したビューを作成します
threat_feeds_raw.createOrReplaceTempView("threat_feeds_raw")
threat_feeds_enriched = spark.sql("select *, domain_extract(url) as domain  from threat_feeds_raw").filter("char_length(domain) >= 2")
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
brand_domains_monitored_raw = spark.read.format("csv").option("header", "true").load(f"dbfs:{work_path}tables/datasets/domains_dnstwists.csv") 

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
brand_domains_monitored_enriched.write.format("delta").mode('overwrite').option("mergeSchema", "false").save(f"dbfs:{work_path}tables/datasets/tables/silver/delta/brand_domains_monitored_enriched")

# COMMAND ----------

# エンリッチされたデータによるシルバーDeltaテーブルを作成
spark.sql(f"DROP TABLE IF EXISTS {db_name_silver}.EnrichedTwistedDomainBrand")
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {db_name_silver}.EnrichedTwistedDomainBrand 
USING DELTA LOCATION 'dbfs:{work_path}tables/datasets/tables/silver/delta/brand_domains_monitored_enriched'
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
USING DELTA LOCATION 'dbfs:{work_path}tables/tables/bronze/delta/DNS_raw'
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 何レコードをロードしたのかを確認
# MAGIC Select count(*) from DNS

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

model_path = f'dbfs:{work_path}tables/model'
loaded_model = mlflow.pyfunc.load_model(model_path)
spark.udf.register("ioc_detect", loaded_model.predict)

# COMMAND ----------

# Aのrrtypeでフィルタリング 
dns_table = spark.table(f"{db_name_bronze}.DNS").selectExpr("*", "case when rrtype = 'A' then element_at(rdata, 1) else null end as ip_address ")

# COMMAND ----------

# city、country、country codes、ioc、ドメイン名でデータを拡張します
dns_table_enriched = dns_table.selectExpr("*", "case when ip_address is not null then get_country(ip_address) else null end as country", 
                     "case when ip_address is not null then get_city(ip_address) else null end as city", 
                     "case when ip_address is not null then get_country_code(ip_address) else null end as country_code", 
                     "case when char_length(domain_extract(rrname)) > 5 then ioc_detect(string(domain_extract(rrname))) else null end as ioc",
                     " domain_extract(rrname) as domain_name")

# COMMAND ----------

# 拡張したDNSデータを永続化します
dns_table_enriched.write.format("delta").mode('overwrite').option("mergeSchema", "true").save(f"dbfs:{work_path}tables/tables/silver/delta/DNS")

# COMMAND ----------

# 拡張されたDNSデータからDeltaテーブルを作成。これは後ほどのDGA分析で使用されます
spark.sql(f"DROP TABLE IF EXISTS {db_name_silver}.DNS")
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {db_name_silver}.DNS 
USING DELTA LOCATION 'dbfs:{work_path}tables/tables/silver/delta/DNS'
""")

# COMMAND ----------

# 拡張された脅威インテリジェンスをシルバーテーブルにロード
# これは後ほどのDGA分析で使用されます
spark.sql(f"DROP TABLE IF EXISTS {db_name_silver}.EnrichedThreatFeeds")
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {db_name_silver}.EnrichedThreatFeeds 
USING DELTA LOCATION 'dbfs:{work_path}tables/tables/silver/delta/enriched_threat_feeds'
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
# MAGIC - iocはDGAモデルの実行結果として作成したフィールドです。iocフィールドにiocがある場合、DGAモデルはこのドメインがioc(indicator of compromise:セキュリティ侵害インジケーター)と判断したことを意味します。
# MAGIC - 以下のクエリーはDGAアルゴリズムがiocを検知した件数をカウントしています。しかし、ドメインに`ip`を含み、10文字以上ドメイン名であるものを除外しています。

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
# MAGIC   and domain_name not like '%ip%'
# MAGIC   and char_length(domain_name) > 8
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
# MAGIC select  domain_name, rrname, country, time_first, time_last, ioc,rrtype,rdata,bailiwick, EnrichedThreatFeeds.* 
# MAGIC from dns, EnrichedThreatFeeds where dns.domain_name == EnrichedThreatFeeds.domain and ioc='ioc'

# COMMAND ----------

# MAGIC %md
# MAGIC 複数のテーブルから特定のrrnames(リソースレコード)を検索します。

# COMMAND ----------

# MAGIC %sql
# MAGIC select  domain_name, rrname, country, time_first, time_last, ioc,rrtype,rdata,bailiwick, EnrichedThreatFeeds.* 
# MAGIC from dns, EnrichedThreatFeeds where dns.domain_name == EnrichedThreatFeeds.domain  and dns.rrname = "ns1.asdklgb.cf." OR dns.rrname LIKE "%cn."

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. MLトレーニングおよび分析
# MAGIC 
# MAGIC このセクションでは、DGAモデルとtyposquattingモデルを構築します。以下のスライドにはDGAに関するハイレベルなディスカッションが含まれています。
# MAGIC 
# MAGIC - DGAに関する詳細な議論に関してはこちら: http://www.covert.io/getting-started-with-dga-research/
# MAGIC - typosquattingに関する詳細な議論はこちら: https://www.mcafee.com/blogs/consumer/what-is-typosquatting/
# MAGIC 
# MAGIC ハイレベルにおいては、以下を行います。
# MAGIC 
# MAGIC - gTLD(例 .com, .org)とccTLD(例 .ru, cn, .uk, .ca)を除外してドメイン名を抽出
# MAGIC - モデルを構築

# COMMAND ----------

# MAGIC %python
# MAGIC displayHTML("""<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vRqDKRAKkXWhcRavKMvJE1BKzpoI4UvofIFQdIpoTV1d7Z3b4XdIsRt6O0iAFV8waBPvrMLVUdHFcND/embed?start=false&loop=false&delayms=3000" frameborder="0" width="960" height="569" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
# MAGIC """)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alexaドメインリストの読み込み
# MAGIC - Alexaは人気でランキングされたインターネットのドメインのリストです
# MAGIC - Alexaはホワイトリストを目的としたものではありません 

# COMMAND ----------

import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
alexa_dataframe = pd.read_csv(default_file_path + "/alexa_100k.txt");
display(alexa_dataframe)

# COMMAND ----------

# MAGIC %md
# MAGIC ## URLからドメイン名を抽出
# MAGIC - URLの登録ドメイン、サブドメインからgTLDやccTLD(generic or country code top-level domain)を除外してドメイン名を抽出します。
# MAGIC - トレーニングにはドメイン名を必要とします。
# MAGIC - tldextractの結果は右のようになります: `ExtractResult(subdomain='forums.news', domain='cnn', suffix='com')`

# COMMAND ----------

import tldextract
import numpy as np
def domain_extract(uri):
    ext = tldextract.extract(uri)
    if (not ext.suffix):
        return np.nan
    else:
        return ext.domain

spark.udf.register("domain_extract", domain_extract)

alexa_dataframe['domain'] = [ domain_extract(uri) for uri in alexa_dataframe['uri']]
del alexa_dataframe['uri']
del alexa_dataframe['rank']
display(alexa_dataframe)

# COMMAND ----------

# MAGIC %md
# MAGIC ## トレーニングデータにAlexaの合法なドメインを追加

# COMMAND ----------

# 空行などからNaNを持つ場合があります
alexa_dataframe = alexa_dataframe.dropna()
alexa_dataframe = alexa_dataframe.drop_duplicates()

# クラス(legit)の設定
alexa_dataframe['class'] = 'legit'

# データのシャッフル (トレーニング/テストにおいては重要です)
alexa_dataframe = alexa_dataframe.reindex(np.random.permutation(alexa_dataframe.index))
alexa_total = alexa_dataframe.shape[0]
print('Total Alexa domains %d' % alexa_total)
display(alexa_dataframe)

# COMMAND ----------

file_location = default_file_path + "/dga_domains_header.txt"

dga_dataframe = pd.read_csv(file_location, header=0);
# ブラックリストの値が大文字小文字の違いや.com/.org/.infoの違いであることに気づきました
dga_dataframe['domain'] = dga_dataframe.applymap(lambda x: x.split('.')[0].strip().lower())

# 空行などからNaNを持つ場合があります
dga_dataframe = dga_dataframe.dropna()
dga_dataframe = dga_dataframe.drop_duplicates()
dga_total = dga_dataframe.shape[0]
print('Total DGA domains %d' % dga_total)

# クラス(ioc)の設定
dga_dataframe['class'] = 'ioc'

print('Number of DGA domains: %d' % dga_dataframe.shape[0])
all_domains = pd.concat([alexa_dataframe, dga_dataframe], ignore_index=True)

# COMMAND ----------

# データセットからGDA検知結果をアウトプット
display(dga_dataframe)

# COMMAND ----------

# MAGIC %md
# MAGIC - いくつかの特徴量エンジニアリングを行い、データセットのエントロピー、長さの計算処理を追加します。
# MAGIC - 文字列長に対するユニークな文字列の数を比較することでエントロピー(文字列長に対してユニークな文字の数が多いほど高くなる)を計算します。

# COMMAND ----------

all_domains['length'] = [len(x) for x in all_domains['domain']]
all_domains = all_domains[all_domains['length'] > 6]

# COMMAND ----------

all_domains

# COMMAND ----------

import math
from collections import Counter
 
def entropy(s):
    p, lns = Counter(s), float(len(s))
    
    return -math.fsum( count/lns * math.log(count/lns, 2) for count in p.values()  )
  
all_domains['entropy'] = [entropy(x) for x in all_domains['domain']]

# COMMAND ----------

# 結果を表示します。エントロピーが高いほどDGAの可能性が高くなりますが、まだ十分ではありません。
display(all_domains)

# COMMAND ----------

# ここでは適切なドメインに対するn-gramの頻度分析を行うために追加の特徴量エンジニアリングを行います

y = np.array(all_domains['class'].tolist()) # 不思議ですがこれが必要となります 

import sklearn.ensemble
from sklearn import feature_extraction

alexa_vc = sklearn.feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(3,5), min_df=1e-4, max_df=1.0)
counts_matrix = alexa_vc.fit_transform(alexa_dataframe['domain'])
alexa_counts = np.log10(counts_matrix.sum(axis=0).getA1())
ngrams_list = alexa_vc.get_feature_names()

counts_matrix = alexa_vc.fit_transform(alexa_dataframe['domain'])
alexa_counts = np.log10(counts_matrix.sum(axis=0).getA1())
ngrams_list = alexa_vc.get_feature_names()

# COMMAND ----------

# ディクショナリーの単語をデータフレームにロードします
file_location = default_file_path + "/words.txt"
word_dataframe = pd.read_csv(file_location, header=0, sep=';');
word_dataframe = word_dataframe[word_dataframe['words'].map(lambda x: str(x).isalpha())]
word_dataframe = word_dataframe.applymap(lambda x: str(x).strip().lower())
word_dataframe = word_dataframe.dropna()
word_dataframe = word_dataframe.drop_duplicates()

# COMMAND ----------

word_dataframe

# COMMAND ----------

# ワードリストからディクショナリーを作成します
dict_vc = sklearn.feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(3,5), min_df=1e-5, max_df=1.0)
counts_matrix = dict_vc.fit_transform(word_dataframe['words'])
dict_counts = np.log10(counts_matrix.sum(axis=0).getA1())
ngrams_list = dict_vc.get_feature_names()

def ngram_count(domain):
    alexa_match = alexa_counts * alexa_vc.transform([domain]).T  # ベクトルの掛け算と転置です！
    dict_match = dict_counts * dict_vc.transform([domain]).T
    print('%s Alexa match:%d Dict match: %d' % (domain, alexa_match, dict_match))

# Examples:
ngram_count('beyonce')
ngram_count('dominos')
ngram_count('1cb8a5f36f')
ngram_count('zfjknuh38231')
ngram_count('bey6o4ce')
ngram_count('washington')

# COMMAND ----------

# MAGIC %md
# MAGIC - Alexaトップ100kリストとディクショナリーからn-gramを作成し、マッチング関数を作成して、テストサンプルを実行します。
# MAGIC - n-gramの詳細についてはこちら: https://blog.xrds.acm.org/2017/10/introduction-n-grams-need/

# COMMAND ----------

all_domains['alexa_grams']= alexa_counts * alexa_vc.transform(all_domains['domain']).T 
all_domains['word_grams']= dict_counts * dict_vc.transform(all_domains['domain']).T 

# COMMAND ----------

# MAGIC %md
# MAGIC ## n-gramのベクトル化モデルを構築
# MAGIC 
# MAGIC モデル構築にはベクトルが必要となります。

# COMMAND ----------

weird_cond = (all_domains['class']=='legit') & (all_domains['word_grams']<3) & (all_domains['alexa_grams']<2)
weird = all_domains[weird_cond]
print(weird.shape[0])
all_domains.loc[weird_cond, 'class'] = 'weird'
print(all_domains['class'].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルのトレーニング
# MAGIC 
# MAGIC - 不自然さに基づいてドメインのラベリングを行います。
# MAGIC - MLランタイムを使用していれば、パッケージは事前インストールされています。
# MAGIC - MLflowを用いることで繰り返し過程でのエクスペリメントをトラッキングできます。

# COMMAND ----------

from sklearn.model_selection import train_test_split
clf = sklearn.ensemble.RandomForestClassifier(n_estimators=20) # フォレストの木の数

not_weird = all_domains[all_domains['class'] != 'weird']
X = not_weird[['length', 'entropy', 'alexa_grams', 'word_grams']].values

# ラベル (scikit learnは分類ラベルに 'y' を使います)
y = np.array(not_weird['class'].tolist())

with mlflow.start_run():
  # 80/20 スプリットでトレーニングします
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  # 予測パフォーマンスを見る前にまず全体に対してトレーニングを行います
  clf.fit(X, y)

# COMMAND ----------

# MLflowを用いて我々のコンテンツライブラリにおけるモデルの位置を特定します
run_id = mlflow.search_runs()['run_id'][0]
model_uri = 'runs:/' + run_id + '/model'

# COMMAND ----------

# このランのランIDを取得します
run_id

# COMMAND ----------

# MAGIC %md
# MAGIC - 後でDFA予測を行うために使う予測関数を構築します。
# MAGIC - 予測関数に対して、事前/事後処理を追加します。

# COMMAND ----------

import mlflow.pyfunc

class vc_transform(mlflow.pyfunc.PythonModel):
    def __init__(self, alexa_vc, dict_vc, ctx):
        self.alexa_vc = alexa_vc
        self.dict_vc = dict_vc
        self.ctx = ctx

    def predict(self, context, model_input):
        _alexa_match = alexa_counts * self.alexa_vc.transform([model_input]).T  
        _dict_match = dict_counts * self.dict_vc.transform([model_input]).T
        _X = [len(model_input), entropy(model_input), _alexa_match, _dict_match]
        return str(self.ctx.predict([_X])[0])

# COMMAND ----------

# MAGIC %md
# MAGIC - 予測関数を保存します
# MAGIC - 注意 - 既知のバグ: 以下のコマンドは一度のみ実行してください。二回実行した際のエラーは無視して構いません。

# COMMAND ----------

from mlflow.exceptions import MlflowException
model_path = f'dbfs:{work_path}tables/new_model/dga_model'

vc_model = vc_transform(alexa_vc, dict_vc, clf)
mlflow.pyfunc.save_model(path=model_path.replace("dbfs:", "/dbfs"), python_model=vc_model)

# COMMAND ----------

# トレーニングしたモデルを使用します 
vc_model = vc_transform(alexa_vc, dict_vc, clf)
vc_model.predict(mlflow.pyfunc.PythonModel, '7ydbdehaaz')

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. ニアリアルタイムのストリーミング分析
# MAGIC 
# MAGIC 脅威インテリジェンスでデータを補強し、分析と補強を用いて脅威のあるアクティビティを検知します。

# COMMAND ----------

# pDNSのスキーマ定義 
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType, StringType, ArrayType

schema = StructType([
    StructField("rrname", StringType(), True),
    StructField("rrtype", StringType(), True),
    StructField("time_first", LongType(), True),
    StructField("time_last", LongType(), True),
    StructField("count", LongType(), True),
    StructField("bailiwick", StringType(), True),
    StructField("rdata", ArrayType(StringType(), True), True)
])


# COMMAND ----------

# テストデータセットのロード
df = spark.readStream.format("json").schema(schema).load(f"dbfs:{work_path}tables/datasets/latest")

# COMMAND ----------

# テストデータセットに対する一時テーブルの作成 
df.createOrReplaceTempView("dns_latest")

# COMMAND ----------

#　ビジュアライズを通じた調査、1行目、3行目が怪しいことがわかります
display(df)

# COMMAND ----------

# MAGIC %sql -- ここでDGA検知を行います
# MAGIC -- DNSデータをスコアリングするためのビューを作成します
# MAGIC CREATE
# MAGIC OR REPLACE TEMPORARY VIEW dns_latest_scoring AS
# MAGIC SELECT
# MAGIC   rdata,
# MAGIC   count,
# MAGIC   rrname,
# MAGIC   bailiwick,
# MAGIC   rrtype,
# MAGIC   time_last,
# MAGIC   time_first,
# MAGIC   ioc_detect(domain_extract(rrname)) as isioc,
# MAGIC   domain_extract(dns_latest.rrname) domain
# MAGIC from
# MAGIC   dns_latest

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.1 DNSイベントストリームにおける脅威の発見

# COMMAND ----------

#spark.conf.set("spark.sql.view.maxNestedViewDepth", "1000") 

# COMMAND ----------

# MAGIC %sql
# MAGIC Select * from dns_latest_scoring where isioc = 'ioc'

# COMMAND ----------

# MAGIC %md
# MAGIC - PhishingあるいはTyposquating?
# MAGIC - ここでtyposquattingの検知を行います
# MAGIC - dnstwistを使うことで、怪しいgoogleeのようなドメインを検知できます

# COMMAND ----------

# MAGIC %sql
# MAGIC Select
# MAGIC   EnrichedTwistedDomainBrand.*
# MAGIC FROM
# MAGIC   dns_latest_scoring,
# MAGIC   EnrichedTwistedDomainBrand
# MAGIC Where
# MAGIC   EnrichedTwistedDomainBrand.dnstwisted_domain = dns_latest_scoring.domain

# COMMAND ----------

# MAGIC %md
# MAGIC 以下のラインではモデルを適用しています。
# MAGIC - 不正なドメインの検知
# MAGIC - アラート用テーブルの作成

# COMMAND ----------

dns_stream_iocs = spark.sql("Select * from dns_latest_scoring where isioc = 'ioc'")
dbutils.fs.rm(f'dbfs:{work_path}datasets/gold/delta/DNS_IOC_Latest', True)
dns_stream_iocs.writeStream.format("delta").outputMode("append").option("checkpointLocation", f"dbfs:{work_path}datasets/gold/delta/_checkpoints/DNS_IOC_Latest").start(f"dbfs:{work_path}datasets/gold/delta/DNS_IOC_Latest")

# COMMAND ----------

res = spark.sql(f"SELECT * FROM delta.`{work_path}datasets/gold/delta/DNS_IOC_Latest`")
display(res)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.2 Agent Tesla
# MAGIC ミッション成功です!!! 
# MAGIC - ストリーミンDNSイベントに対してDGA検知モデルを適用しました。 
# MAGIC - DNSログから怪しいドメイン(ioc)を特定しました。
# MAGIC - URLHausを用いてiocデータを補強しました
# MAGIC - このDGAドメインがAgent Teslaに貢献していることを確認できます。

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 不正なドメインを特定したので、我々が補強した脅威インテリジェンスフィードがこのドメインで活動しているのかをみてみましょう
# MAGIC select * from EnrichedThreatFeeds where EnrichedThreatFeeds.domain = domain_extract('ns1.asdklgb.cf.')

# COMMAND ----------

# MAGIC %md
# MAGIC # END
