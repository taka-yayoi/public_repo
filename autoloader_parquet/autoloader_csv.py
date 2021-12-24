# Databricks notebook source
# MAGIC %md
# MAGIC ## セットアップ

# COMMAND ----------

# ファイルのアップロード先を作成
user_dir = 'takaaki.yayoi@databricks.com'
upload_path = "/FileStore/shared_uploads/" + user_dir + "/csv_data_upload"
dbutils.fs.mkdirs(upload_path)

# COMMAND ----------

# チェックポイントのパス
checkpoint_path = '/tmp/delta/population_data/_checkpoints'
# 書き込み先のパス
write_path = '/tmp/delta/population_data'

# COMMAND ----------

# MAGIC %md
# MAGIC ## ファイルのアップロード
# MAGIC 
# MAGIC 以下の2つのファイルを上記`upload_path`にアップロードしておきます。ファイルがない状態でAuto Loaderを起動すると`java.util.NoSuchElementException`となります。
# MAGIC 
# MAGIC WA.csv:
# MAGIC 
# MAGIC ```
# MAGIC city,year,population
# MAGIC Seattle metro,2019,3406000
# MAGIC Seattle metro,2020,3433000
# MAGIC ```
# MAGIC 
# MAGIC OR.csv:
# MAGIC 
# MAGIC ```
# MAGIC city,year,population
# MAGIC Portland metro,2019,2127000
# MAGIC Portland metro,2020,2151000
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Auto Loaderの起動

# COMMAND ----------

# upload_pathの場所に到着するファイルを読み込むストリームのセットアップ
df = spark.readStream.format('cloudFiles') \
.option('cloudFiles.format', 'csv') \
.option('header', 'true') \
.schema('city string, year int, population long') \
.load(upload_path)

# ストリームを起動します。
# upload_pathにアップロード済みの全てのファイルの記録を保持するためにcheckpoint_pathを使用します。
# 最後のチェック以降のファイルに対して、新規にアップロードされたファイルのデータをwrite_pathに書き込みます
df.writeStream.format('delta') \
.option('checkpointLocation', checkpoint_path) \
.start(write_path)

# COMMAND ----------

df_population = spark.read.format('delta').load(write_path)

display(df_population)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 一度、クラスターを停止します。クラスターを再起動後、上のストリームを起動し。追加で以下のファイルを`upload_path`にアップロードします。
# MAGIC 
# MAGIC ID.csv:
# MAGIC 
# MAGIC ```
# MAGIC city,year,population
# MAGIC Boise,2019,438000
# MAGIC Boise,2020,447000
# MAGIC ```
# MAGIC 
# MAGIC MT.csv:
# MAGIC 
# MAGIC ```
# MAGIC city,year,population
# MAGIC Helena,2019,81653
# MAGIC Helena,2020,82590
# MAGIC ```
# MAGIC 
# MAGIC Misc.csv:
# MAGIC 
# MAGIC ```
# MAGIC city,year,population
# MAGIC Seattle metro,2021,3461000
# MAGIC Portland metro,2021,2174000
# MAGIC Boise,2021,455000
# MAGIC Helena,2021,81653
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC 追加されたファイルのみが処理されます。

# COMMAND ----------

df_population = spark.read.format('delta').load(write_path)

display(df_population)

# COMMAND ----------

# クリーンアップ
dbutils.fs.rm(write_path, True)
dbutils.fs.rm(upload_path, True)

# COMMAND ----------

# チェックポイントのリセット
dbutils.fs.rm(checkpoint_path, True)

# COMMAND ----------


