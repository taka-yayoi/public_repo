# Databricks notebook source
# MAGIC %md
# MAGIC ## セットアップ

# COMMAND ----------

# ファイルのアップロード先を作成
user_dir = 'takaaki.yayoi@databricks.com'
upload_path = "/FileStore/shared_uploads/" + user_dir + "/parquet_data_upload"
dbutils.fs.mkdirs(upload_path)

# COMMAND ----------

# チェックポイントのパス
checkpoint_path = '/tmp/delta/parquet_data/_checkpoints'
# 書き込み先のパス
write_path = '/tmp/delta/parquet_data'

# COMMAND ----------

# MAGIC %md
# MAGIC ## ファイルのアップロード
# MAGIC 
# MAGIC 1つのParquetファイルを上記`upload_path`にアップロードしておきます。ここでは、以下のParquetファイルを使用しています。
# MAGIC 
# MAGIC [parquet\-dotnet/postcodes\.plain\.parquet at master · elastacloud/parquet\-dotnet](https://github.com/elastacloud/parquet-dotnet/blob/master/src/Parquet.Test/data/postcodes.plain.parquet)

# COMMAND ----------

# Parquetファイルのスキーマを取得するために一旦読み込みます
parquetFile = spark.read.parquet(f"{upload_path}/postcodes_plain_1.parquet")
parquetFile.schema

# COMMAND ----------

# MAGIC %md
# MAGIC ## Auto Loaderの起動

# COMMAND ----------

# upload_pathの場所に到着するファイルを読み込むストリームのセットアップ
df = spark.readStream.format('cloudFiles') \
.option('cloudFiles.format', 'parquet') \
.option('header', 'true') \
.schema(parquetFile.schema) \
.load(upload_path)

# ストリームを起動します。
# upload_pathにアップロード済みの全てのファイルの記録を保持するためにcheckpoint_pathを使用します。
# 最後のチェック以降のファイルに対して、新規にアップロードされたファイルのデータをwrite_pathに書き込みます
df.writeStream.format('delta') \
.option('checkpointLocation', checkpoint_path) \
.start(write_path)

# COMMAND ----------

df_parquet = spark.read.format('delta').load(write_path)

display(df_parquet)

# COMMAND ----------

# MAGIC %md
# MAGIC ## クラスターの停止・起動
# MAGIC 
# MAGIC 一度、クラスターを停止します。クラスターを再起動後、上のストリームを起動し、追加のParquetファイル(上のファイルのコピー)を`upload_path`にアップロードします。上でAuto Loaderのチェックポイントを作成しているので、Auto Loaderは新たにアップロードされたファイルのみを処理します。

# COMMAND ----------

# MAGIC %md
# MAGIC 追加されたファイルのみが処理されます。

# COMMAND ----------

df_parquet = spark.read.format('delta').load(write_path)

display(df_parquet)

# COMMAND ----------

# クリーンアップ
dbutils.fs.rm(write_path, True)
dbutils.fs.rm(upload_path, True)

# COMMAND ----------

# チェックポイントのリセット
dbutils.fs.rm(checkpoint_path, True)

# COMMAND ----------


