# Databricks notebook source
# MAGIC %md
# MAGIC # Delta Sharing - 外部チーム/パートナーとセキュアにデータ共有
# MAGIC 
# MAGIC このノートブックではOSS版Delta Sharaingの使い方をデモします。
# MAGIC 
# MAGIC - データレイク/レイクハウスにあるデータをライブで共有(コピーする必要はありません)
# MAGIC - 既存のオープンデータフォーマットを用いることで幅広いクライアントをサポート(pandas、spark、Tableau)
# MAGIC - 強力なセキュリティ、監査、ガバナンス
# MAGIC - 大規模データに対して効率的にスケール
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/delta-sharing/resources/images/delta-sharing-flow.png" width="900px"/>

# COMMAND ----------

# MAGIC %sh sudo apt-get install jq

# COMMAND ----------

# MAGIC %md ## Databricks、OSSのDelta Sharingサーバー用いたREST APIの探索
# MAGIC 
# MAGIC Databricksではテストのためにシェアリングサーバーをホストしています: https://sharing.delta.io/ 
# MAGIC 
# MAGIC *注意: こちらは認証を必要としませんが、実運用で利用するシナリオにおいては、呼び出しにBearerトークンが必要になります*

# COMMAND ----------

# DBTITLE 1,Sharesのリスト、一つのshareはトップレベルのコンテナになります
# MAGIC %sh curl https://sharing.delta.io/delta-sharing/shares -s | jq '.'

# COMMAND ----------

# DBTITLE 1,delta_sharingのshareのスキーマ一覧
# MAGIC %sh curl https://sharing.delta.io/delta-sharing/shares/delta_sharing/schemas -s | jq '.'

# COMMAND ----------

# DBTITLE 1,shareに含まれるテーブルの一覧
# MAGIC %sh curl https://sharing.delta.io/delta-sharing/shares/delta_sharing/schemas/default/tables -s | jq '.'

# COMMAND ----------

# MAGIC %md ### "boston-housing"テーブルからメタデータを取得

# COMMAND ----------

# MAGIC %sh curl https://sharing.delta.io/delta-sharing/shares/delta_sharing/schemas/default/tables/boston-housing/metadata -s | jq '.'

# COMMAND ----------

# MAGIC %md ### データの取得
# MAGIC 
# MAGIC Delta Sharingにおいては、背後のデータをダウンロードするために、一時的な自己サイン済みリンクを作成します。クエリーをプッシュダウンし、ファイルのサブセットのみを取得できるようにDelta Lakeの統計情報を活用します。
# MAGIC 
# MAGIC REST APIを用いることで、これらのリンクを取得しデータをダウンロードすることができます。

# COMMAND ----------

# DBTITLE 1,boston-housingデータへのアクセス
# MAGIC %sh curl -X POST https://sharing.delta.io/delta-sharing/shares/delta_sharing/schemas/default/tables/boston-housing/query -s -H 'Content-Type: application/json' -d @- << EOF
# MAGIC {
# MAGIC    "predicateHints" : [
# MAGIC       "date >= '2021-01-01'",
# MAGIC       "date <= '2021-01-31'"
# MAGIC    ],
# MAGIC    "limitHint": 1000
# MAGIC }
# MAGIC EOF

# COMMAND ----------

# MAGIC %md ## Python APIを使う

# COMMAND ----------

# MAGIC %pip install delta-sharing

# COMMAND ----------

dbutils.fs.mkdirs("/tmp/takaakiyayoi")

# COMMAND ----------

# DBTITLE 1,シークレットを用いてshare設定ファイルを作成
# MAGIC %sh 
# MAGIC cat <<EOT > /dbfs/tmp/takaakiyayoi/open-datasets.share
# MAGIC {
# MAGIC   "shareCredentialsVersion": 1,
# MAGIC   "endpoint": "https://sharing.delta.io/delta-sharing/",
# MAGIC   "bearerToken": "faaie590d541265bcab1f2de9813274bf233"
# MAGIC }
# MAGIC EOT

# COMMAND ----------

import os
import delta_sharing

# 上で作成したファイルをポイントします。ファイルはローカルファイルシステム、あるいは、リモートストレージに格納することができます。
profile_file = "/dbfs/tmp/takaakiyayoi/open-datasets.share"

# SharingClientの作成
client = delta_sharing.SharingClient(profile_file)

# 全ての共有テーブルのリスト
print("########### All Available Tables #############")
print(client.list_all_tables())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sparkを用いて大規模データを処理する
# MAGIC 
# MAGIC 大規模データセットを並列で処理する必要がありますか？Delta Sharing sparkコネクターを使ってデータをロードします。(クラスターライブラリとして`io.delta:delta-sharing-spark_2.12:0.2.0`(Maven)をインストールする必要があります。)

# COMMAND ----------

import os
import delta_sharing
from pyspark.sql import SparkSession

# フォーマット"deltaSharing"を指定してデータを読み込み
spark.read.format("deltaSharing").load("/tmp/takaakiyayoi/open-datasets.share" + "#delta_sharing.default.boston-housing") \
	 .where("age > 18") \
	 .display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### これで全てです。お使いのデータをあなたのデータメッシュや、他の企業と共有する準備ができました！

# COMMAND ----------

# MAGIC %md Extra: Databricksを用いて外部サプライヤーに共有、アクセス許可を行うことができます。
# MAGIC 
# MAGIC *こちらのコマンドはDatabricksでホストされるDelta Sharingでサポートされます*

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SHARE retail;
# MAGIC 
# MAGIC ALTER SHARE retail ADD TABLE sales;
# MAGIC 
# MAGIC GRANT SELECT ON SHARE retail TO supplier1;
