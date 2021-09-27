# Databricks notebook source
# MAGIC %md ## Sparkクイックスタート(Python)
# MAGIC * Pythonを用いたデータフレーム操作をDatabricksノートブックで実行します。
# MAGIC * 参考 http://spark.apache.org/docs/latest/quick-start.html

# COMMAND ----------

# ファイルシステムを参照します
display(dbutils.fs.ls("/databricks-datasets/samples/docs/"))

# COMMAND ----------

# MAGIC %md データフレームには、新たなデータフレームへのポインタを返却する ***変換処理(transformations)*** と値を返却する ***アクション(actions)*** があります。

# COMMAND ----------

# 変換(transformation)
textFile = spark.read.text("/databricks-datasets/samples/docs/README.md")

# COMMAND ----------

# アクション(action)
textFile.count()

# COMMAND ----------

# テキストファイルの最初の行を出力
textFile.first()

# COMMAND ----------

# MAGIC %md 
# MAGIC ここでは、ファイルの一部を新たなデータフレームとして返却するフィルター ***変換処理(transformation)*** を用います

# COMMAND ----------

# データフレームの全ての行をフィルタリングします
linesWithSpark = textFile.filter(textFile.value.contains("Spark"))

# COMMAND ----------

# MAGIC %md 上の処理がすぐに完了したことに注意してください。上は変換処理であり、アクションがまだ呼ばれていません。
# MAGIC * しかし、下のセルでアクション(例: count、take)を実行することで、処理が実行される様子を見ることになります。

# COMMAND ----------

# count(アクション)を実行します 
linesWithSpark.count()

# COMMAND ----------

# 最初の5行をアウトプットします
linesWithSpark.take(5)
