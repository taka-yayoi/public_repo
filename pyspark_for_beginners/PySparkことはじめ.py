# Databricks notebook source
# MAGIC %md
# MAGIC # PySparkことはじめ
# MAGIC 
# MAGIC PythonからApache Sparkを操作する際に使用するAPIであるPySparkの基本的な使い方を説明します。
# MAGIC 
# MAGIC **参考資料**
# MAGIC - [About Spark – Databricks](https://databricks.com/jp/spark/about)
# MAGIC - [Databricks Apache Sparkクイックスタート \- Qiita](https://qiita.com/taka_yayoi/items/bf5fb09a0108aa14770b)
# MAGIC - [Databricks Apache Sparkデータフレームチュートリアル \- Qiita](https://qiita.com/taka_yayoi/items/2a7e9bb792eba316de4b)
# MAGIC - [PySpark Documentation — PySpark 3\.2\.1 documentation](https://spark.apache.org/docs/latest/api/python/)
# MAGIC - [Beginner’s Guide on Databricks: Spark Using Python & PySpark \| by Christopher Lewis \| Analytics Vidhya \| Medium](https://medium.com/analytics-vidhya/beginners-guide-on-databricks-spark-using-python-pyspark-de74d92e4885)
# MAGIC - [【PySpark入門】第１弾 PySparkとは？ \- サーバーワークスエンジニアブログ](https://blog.serverworks.co.jp/introducing-pyspark-1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ライブラリのインポート
# MAGIC 
# MAGIC 処理に必要なモジュールをインポートします。

# COMMAND ----------

from pyspark.sql.functions import col, avg
from pyspark.sql.types import IntegerType, FloatType

# COMMAND ----------

# MAGIC %md
# MAGIC ## データのロード
# MAGIC 
# MAGIC PySparkでデータをロードする際には`spark.read`を使用します。`format`の引数に読み込むデータのフォーマットを指定します。`json`、`parquet`、`delta`などが指定できます。読み込んだデータはSparkデータフレームとなります。
# MAGIC 
# MAGIC その前に、読み込むデータを以下のコマンドで確認します。

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/samples/population-vs-price/

# COMMAND ----------

# データフレームにサンプルデータをロードします
df = spark.read.format("csv").option("header", True).load("/databricks-datasets/samples/population-vs-price/data_geo.csv")

# COMMAND ----------

df.show(20)

# COMMAND ----------

# Databricksでデータフレームを表示するにはdisplay関数を使うと便利です
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### カラムの確認

# COMMAND ----------

df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ### スキーマの確認

# COMMAND ----------

# データフレームのスキーマを表示
df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## カラム名の変更
# MAGIC 
# MAGIC - `withColumnRenamed`を使ってカラム名を変更します。

# COMMAND ----------

df2 = df.withColumnRenamed('2014 rank', '2014_rank')\
.withColumnRenamed('State Code', 'state_code')\
.withColumnRenamed('2014 Population estimate', '2014_pop_estimate')\
.withColumnRenamed('2015 median sales price', '2015_median_sales_price')

# COMMAND ----------

display(df2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## データ型の変換
# MAGIC 
# MAGIC - 既に存在しているデータフレームのカラムを指定するには、`col`関数の引数にカラム名を指定します。
# MAGIC - `cast`にデータ型を指定してキャストします。
# MAGIC - `withColumn`を用いて、キャストした後の値を持つカラムで更新します。
# MAGIC 
# MAGIC [Data Types \- Spark 3\.2\.1 Documentation](https://spark.apache.org/docs/latest/sql-ref-datatypes.html)

# COMMAND ----------

df3 = df2.withColumn("2014_rank", col("2014_rank").cast(IntegerType()))\
 .withColumn("2014_pop_estimate", col("2014_pop_estimate").cast(IntegerType()))\
 .withColumn("2015_median_sales_price", col("2015_median_sales_price").cast(FloatType()))

display(df3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## データの操作

# COMMAND ----------

# MAGIC %md
# MAGIC ### フィルタリング、ソート
# MAGIC 
# MAGIC 以下の例では、`df3`で`2015_median_sales_price`が100より大きいレコードを`2015_median_sales_price`の降順でソートし、カラム`2014_rank`, `City`, `2015_median_sales_price`を取得しています。

# COMMAND ----------

display(df3.select("2014_rank", "City", "2015_median_sales_price")\
        .where("2015_median_sales_price > 100")\
        .orderBy(col("2015_median_sales_price").desc()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 集計
# MAGIC 
# MAGIC 以下の処理を行なっています。
# MAGIC 
# MAGIC 1. `state_code`でレコードをグルーピング
# MAGIC 1. グループごとの`2015_median_sales_price`の平均値を計算
# MAGIC 1. 平均値降順でレコードを取得

# COMMAND ----------

display(df3.groupBy("state_code")\
        .agg(avg("2015_median_sales_price").alias("2015_median_sales_price_avg"))
        .orderBy(col("2015_median_sales_price_avg").desc()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## データの書き込み
# MAGIC 
# MAGIC ファイルシステムにデータフレームを永続化するには、`spark.write`を使用します。

# COMMAND ----------

# Databricksユーザー名の取得
username_raw = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
print(username)

# データの書き込み先、あとでクリーンアップします
tmp_file_path = f"/tmp/{username}/datacsv"
print(tmp_file_path)

# formatを指定しない場合、デフォルトのDelta形式で保存されます
df3.write.mode("overwrite").save(tmp_file_path)

# COMMAND ----------

# データフレームに保存したデータをロードします
df_new = spark.read.format("delta").load(tmp_file_path)
display(df_new)

# COMMAND ----------

# MAGIC %md
# MAGIC ### クリーンアップ

# COMMAND ----------

# 上のセルで保存したファイルを削除しておきます
dbutils.fs.rm(tmp_file_path, True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## pandasとのやりとり
# MAGIC 
# MAGIC matplotlibで可視化したいなどpandas前提の処理を行う場合には、Sparkデータフレームをpandasデータフレームに変換します。

# COMMAND ----------

df4 = df3.groupBy("state_code")\
        .agg(avg("2015_median_sales_price").alias("2015_median_sales_price_avg"))\
        .orderBy(col("2015_median_sales_price_avg").desc()).limit(10)

# COMMAND ----------

import matplotlib.pyplot as plt

# pandasデータフレームに変換します
pdf = df4.toPandas()

# 棒グラフを描画します
plt.bar(pdf['state_code'], pdf['2015_median_sales_price_avg'], align="center")           
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC pandasデータフレームをSparkデータフレームに変換することもできます。

# COMMAND ----------

# Sparkデータフレームへの変換
sdf = spark.createDataFrame(pdf)
display(sdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## その他のAPI

# COMMAND ----------

# MAGIC %md
# MAGIC ### Spark SQL
# MAGIC 
# MAGIC データフレームをテーブルあるいは一時ビューに登録することで、SQLを使用してデータを操作することができるようになります。
# MAGIC 
# MAGIC テーブルは永続化されますが、一時ビューは永続化されず、クラスターが稼働している間のみ一時ビューを作成したセッションでのみ利用することができます。

# COMMAND ----------

# データフレームを一時ビューに登録します
df3.createOrReplaceTempView("pop_price")

# COMMAND ----------

# '2014_rank' カラムに基づいて上位10位の市を参照します
top_10_results = spark.sql("""SELECT * FROM pop_price 
                              WHERE 2014_rank <= 10
                              SORT BY 2014_rank ASC""")
display(top_10_results)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   *
# MAGIC FROM
# MAGIC   pop_price
# MAGIC WHERE
# MAGIC   2014_rank <= 10 SORT BY 2014_rank ASC

# COMMAND ----------

# MAGIC %md
# MAGIC ### pandas API on Spark
# MAGIC 
# MAGIC pandas APIに慣れ親しんでいる方は、pandas API on Spark(旧Koalas)を活用することもできます。
# MAGIC 
# MAGIC [Apache Spark™ 3\.2におけるPandas APIのサポート \- Qiita](https://qiita.com/taka_yayoi/items/63a21a0e5113e33ad6a3)

# COMMAND ----------

import pyspark.pandas as ps

# COMMAND ----------

psdf = sdf.to_pandas_on_spark()  # pandas-on-Sparkデータフレーム

# COMMAND ----------

# pandasのお作法でカラムにアクセスします
psdf['state_code']

# COMMAND ----------

# MAGIC %md
# MAGIC # END
