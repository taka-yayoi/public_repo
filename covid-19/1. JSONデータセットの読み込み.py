# Databricks notebook source
# MAGIC %md 
# MAGIC # 1. JSONデータセットの読み込み
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

# COMMAND ----------

# MAGIC %md ## パスの設定
# MAGIC 
# MAGIC `/databricks-datasets/COVID/CORD-19/2020-03-13/`からCORD-19 (2020-03-13)データセットを利用できます。

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

# Pythonにおけるパスの設定
comm_use_subset_path = "/databricks-datasets/COVID/CORD-19/2020-03-13/comm_use_subset/comm_use_subset/"
noncomm_use_subset_path = "/databricks-datasets/COVID/CORD-19/2020-03-13/noncomm_use_subset/noncomm_use_subset/"
biorxiv_medrxiv_path = "/databricks-datasets/COVID/CORD-19/2020-03-13/biorxiv_medrxiv/biorxiv_medrxiv/"
json_schema_path = "/databricks-datasets/COVID/CORD-19/2020-03-13/json_schema.txt"

# COMMAND ----------

display(dbutils.fs.ls(comm_use_subset_path))

# COMMAND ----------

display(dbutils.fs.ls(noncomm_use_subset_path))

# COMMAND ----------

# シェル環境変数におけるパスの設定
import os
os.environ['comm_use_subset_path']=''.join(comm_use_subset_path)
os.environ['noncomm_use_subset_path']=''.join(noncomm_use_subset_path)
os.environ['biorxiv_medrxiv_path']=''.join(biorxiv_medrxiv_path)
os.environ['json_schema_path']=''.join(json_schema_path)
os.environ['username']=''.join(username)

# COMMAND ----------

# MAGIC %md ## JSONスキーマの確認
# MAGIC 
# MAGIC 以下のセルで表示しているように、これらのデータセットのスキーマは `json_schema.txt` で定義されています。

# COMMAND ----------

# MAGIC %sh 
# MAGIC cat /dbfs$json_schema_path

# COMMAND ----------

# MAGIC %md ## Parquetパス変数の設定
# MAGIC 
# MAGIC `/tmp/<ユーザー名>/COVID/CORD-19/2020-03-13/`にParquetフォーマットで保存します。

# COMMAND ----------

# PythonにおけるParquetパスの設定
comm_use_subset_pq_path = f"/tmp/{username}/COVID/CORD-19/2020-03-13/comm_use_subset.parquet"
noncomm_use_subset_pq_path = f"/tmp/{username}/COVID/CORD-19/2020-03-13/noncomm_use_subset.parquet"
biorxiv_medrxiv_pq_path = f"/tmp/{username}/COVID/CORD-19/2020-03-13/biorxiv_medrxiv/biorxiv_medrxiv.parquet"

# シェル環境変数におけるパスの設定
os.environ['comm_use_subset_pq_path']=''.join(comm_use_subset_pq_path)
os.environ['noncomm_use_subset_pq_path']=''.join(noncomm_use_subset_pq_path)
os.environ['biorxiv_medrxiv_pq_path']=''.join(biorxiv_medrxiv_pq_path)

# COMMAND ----------

# クリーンアップ
print(f"/tmp/{username}/COVID/")
dbutils.fs.rm(f"/tmp/{username}/COVID/", True)

# COMMAND ----------

# MAGIC %md ## comm_use_subset JSONファイルの読み込み
# MAGIC 
# MAGIC これらは適切に整形されたJSONファイルのなので、これらのファイルを読み込むために`spark.read.json`を使用できます。*multiline*オプションを指定する必要があることに注意してください。

# COMMAND ----------

# comm_use_subset
comm_use_subset = spark.read.option("multiLine", True)\
  .json(comm_use_subset_path)

comm_use_subset.printSchema()

# COMMAND ----------

# レコード数 (JSONドキュメントの本来の数)
comm_use_subset.count()

# COMMAND ----------

# MAGIC %md ### ファイル数の検証

# COMMAND ----------

# MAGIC %sh 
# MAGIC ls /dbfs$comm_use_subset_path | wc -l

# COMMAND ----------

# MAGIC %md ### comm_use_subset JSONファイルの保存
# MAGIC 
# MAGIC ファイル読み込みに時間を要するので、クエリー性能を改善するためにParquetファイルで保存しましょう。

# COMMAND ----------

# パーティション数の取得
comm_use_subset.rdd.getNumPartitions()

# COMMAND ----------

print(comm_use_subset_pq_path)

# COMMAND ----------

# 4パーティションでParquetフォーマットで書き出します 
# 今回のクラスターは4ノードであることを想定しています 
comm_use_subset.repartition(4).write.format("parquet").mode("overwrite").save(comm_use_subset_pq_path)

# COMMAND ----------

# MAGIC %sh 
# MAGIC ls -lsgA /dbfs/tmp/$username/COVID/CORD-19/2020-03-13/comm_use_subset.parquet

# COMMAND ----------

# 再度ファイルを読み込みます
comm_use_subset = spark.read.format("parquet").load(comm_use_subset_pq_path)

# COMMAND ----------

# レコード数 (JSONドキュメントの本来の数)
comm_use_subset.count()

# COMMAND ----------

# MAGIC %md ## noncomm_use_subset JSONファイルの読み込み
# MAGIC 
# MAGIC これらは適切に整形されたJSONファイルのなので、これらのファイルを読み込むために`spark.read.json`を使用できます。*multiline*オプションを指定する必要があることに注意してください。

# COMMAND ----------

# noncomm_use_subset
noncomm_use_subset = spark.read.option("multiLine", True).json(noncomm_use_subset_path)
noncomm_use_subset.printSchema()

# COMMAND ----------

# レコード数 (JSONドキュメントの本来の数)
noncomm_use_subset.count()

# COMMAND ----------

# MAGIC %sh 
# MAGIC ls /dbfs$noncomm_use_subset_path | wc -l

# COMMAND ----------

# MAGIC %md ### noncomm_use_subset JSONファイルの保存
# MAGIC 
# MAGIC ファイル読み込みに時間を要するので、クエリー性能を改善するためにParquetファイルで保存しましょう。

# COMMAND ----------

# パーティション数の取得
noncomm_use_subset.rdd.getNumPartitions()

# COMMAND ----------

# 4パーティションでPqarquetフォーマットで書き出します 
# ここではクラスターは4ノードであることを想定しています  
noncomm_use_subset.repartition(4).write.format("parquet").mode("overwrite").save(noncomm_use_subset_pq_path)

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -lsgA /dbfs$noncomm_use_subset_pq_path

# COMMAND ----------

# ファイルの再読み込み
noncomm_use_subset = spark.read.format("parquet").load(noncomm_use_subset_pq_path)

# COMMAND ----------

# レコード数 (JSONドキュメントの本来の数)
noncomm_use_subset.count()

# COMMAND ----------

# MAGIC %md ## biorxiv_medrxiv JSONファイルの読み込み
# MAGIC 
# MAGIC これらは適切に整形されたJSONファイルのなので、これらのファイルを読み込むために`spark.read.json`を使用できます。*multiline*オプションを指定する必要があることに注意してください。

# COMMAND ----------

# biorxiv_medrxiv
biorxiv_medrxiv = spark.read.option("multiLine", True).json(biorxiv_medrxiv_path)
biorxiv_medrxiv.count()

# COMMAND ----------

# 4パーティションでPqarquetフォーマットで書き出します 
# ここではクラスターは4ノードであることを想定しています   
biorxiv_medrxiv.repartition(4).write.format("parquet").mode("overwrite").save(biorxiv_medrxiv_pq_path)

# COMMAND ----------

# ファイルの再読み込み
biorxiv_medrxiv = spark.read.format("parquet").load(biorxiv_medrxiv_pq_path)
biorxiv_medrxiv.count()

# COMMAND ----------

# MAGIC %md クエリー性能改善のため、以降ではオリジナルのJSONファイルではなくParquetファイルを読み込みます。

# COMMAND ----------

# MAGIC %md
# MAGIC # END
