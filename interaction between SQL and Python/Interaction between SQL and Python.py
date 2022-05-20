# Databricks notebook source
# MAGIC %md
# MAGIC # SQLセルとPythonセルのデータのやり取り

# COMMAND ----------

# MAGIC %md
# MAGIC ## データロード
# MAGIC 
# MAGIC Sparkデータフレームとしてデータをロードし、一時ビューとして登録します。

# COMMAND ----------

path = "/databricks-datasets/learning-spark-v2/flights/departuredelays.csv"

df = spark.read.option("inferSchema", True).option("header", True).csv(path)
df.createOrReplaceTempView("departure_delays")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SQLからのアクセス
# MAGIC 
# MAGIC 一時ビューとして登録されているのでSQLでアクセスできます。SQLで集計を行います。
# MAGIC 
# MAGIC 以下のように表示されます。SQLの実行結果は`_sqldf`というSparkデータフレームに格納されます。
# MAGIC 
# MAGIC > SQL cell result stored as PySpark data frame `_sqldf`

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT origin, destination, count(*) as num_flights
# MAGIC FROM departure_delays
# MAGIC GROUP BY origin, destination
# MAGIC ORDER BY num_flights DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pythonからのアクセス
# MAGIC 
# MAGIC Pythonセルからデータフレーム`_sqldf`にアクセスします。

# COMMAND ----------

display(_sqldf)

# COMMAND ----------

# MAGIC %md
# MAGIC pandasデータフレームに変換して、matplotlibを用いた可視化を行います。

# COMMAND ----------

pdf = _sqldf.toPandas()

# COMMAND ----------

import matplotlib.pyplot as plt

# 出発地がLAXの上位5件に限定
lax_pdf = pdf[pdf["origin"]=="LAX"][0:5]

x = lax_pdf['destination']
y = lax_pdf['num_flights']
label_x = lax_pdf['destination']

# 中央寄せで棒グラフ作成
plt.bar(x, y, align="center")           
plt.xticks(x, label_x)  # X軸のラベル
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # END
