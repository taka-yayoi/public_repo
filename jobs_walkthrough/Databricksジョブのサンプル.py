# Databricks notebook source
# MAGIC %md
# MAGIC # ジョブ用ノートブック

# COMMAND ----------

# MAGIC %md
# MAGIC ## パラメーターの受け取り
# MAGIC 
# MAGIC ジョブのパラメーターを受け取るには`dbutils.widgets.get`を使用します。

# COMMAND ----------

# ノートブックでパラメーターを指定するにはウィジェットを使うと便利です
#dbutils.widgets.text("flower", "sunflowers")
#dbutils.widgets.dropdown("flower", "sunflowers", ["dandelion", "tulips", "sunflowers", "roses", "daisy"])

# COMMAND ----------

# ノートブック上のウィジェットを削除
#dbutils.widgets.removeAll()

# COMMAND ----------

flower = dbutils.widgets.get('flower')
print(flower)

# COMMAND ----------

# MAGIC %md
# MAGIC ## データの読み取り

# COMMAND ----------

# 花の種類を絞り込ります
flowers = spark.read.format("delta").load("dbfs:/databricks-datasets/flowers/delta/").filter(f"label='{flower}'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## データの書き込み

# COMMAND ----------

import uuid

# 画像データは圧縮済みなのでParquetの圧縮をオフにします
spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")
path = "/tmp/flowers/" + str(uuid.uuid4()) # ユニークなIDでDBFS上の保存場所のパスを作成します
flowers.write.format("delta").mode("overwrite").save(path)

# COMMAND ----------

saved_flowers = spark.read.format("delta").load(path)
display(saved_flowers)

# COMMAND ----------

# MAGIC %md
# MAGIC ## クリーンアップ
# MAGIC 
# MAGIC このノートブックはデモ目的のものなので、上で保存したDeltaファイルを削除しておきます。

# COMMAND ----------

dbutils.fs.rm(path, True)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
