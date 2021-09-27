# Databricks notebook source
# MAGIC %md
# MAGIC # pandasとSpark(PySpark)の比較
# MAGIC 
# MAGIC 本ノートブックでは、取り扱うデータ量の観点からpandasとPySparkの比較を行います。また、SparkでpandasのAPIを活用できるライブラリKoalasをご紹介します。
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2021/06/23</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>クラスター</td><td>8.3ML</td></tr>
# MAGIC </table>
# MAGIC <img style="margin-top:25px;" src="https://jixjiadatabricks.blob.core.windows.net/images/databricks-logo-small-new.png" width="140">
# MAGIC 
# MAGIC **注意**
# MAGIC - Sparkの並列処理を体験いただくためには、Single noteクラスターよりもStandardクラスターをご利用いただくことをお勧めします。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/handson20210629/standard_cluster.png)

# COMMAND ----------

# MAGIC %md
# MAGIC PySparkはPythonからSparkを利用するためのAPIを提供します。Sparkの分散処理機能を活用することで、ローカルマシンのメモリーに乗り切らないような大量データであっても、容易に集計、分析、さらには機械学習が可能となります。
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>pandas</th><th>Apache Spark(PySpark)</th></tr>
# MAGIC <tr>
# MAGIC     
# MAGIC <td>
# MAGIC   データセットが小さい場合はpandasが正しい選択となります。
# MAGIC </td>
# MAGIC 
# MAGIC <td>
# MAGIC   大きなデータに対する「フィルタリング」「クリーニング」「集計」などの処理が必要な場合は、Apache Sparkのような並列データフレームを使用することで線形の高速化が期待できます。
# MAGIC </td>
# MAGIC </tr>
# MAGIC   
# MAGIC </table>  
# MAGIC 
# MAGIC 参考資料
# MAGIC - [PySparkとは \- Databricks](https://databricks.com/jp/glossary/pyspark)
# MAGIC - [Databricks Apache Sparkクイックスタート \- Qiita](https://qiita.com/taka_yayoi/items/bf5fb09a0108aa14770b)
# MAGIC - [Databricks Apache Sparkデータフレームチュートリアル \- Qiita](https://qiita.com/taka_yayoi/items/2a7e9bb792eba316de4b)
# MAGIC - [オープンソースのPandasとApache Sparkを比較](https://www.ossnews.jp/compare/Pandas/Apache_Spark)
# MAGIC - [最新のApache Spark v2\.4にふれてみよう: 概要と新機能の紹介 \| by Takeshi Yamamuro \| nttlabs \| Medium](https://medium.com/nttlabs/apache-spark-v24-159ab8983ead)

# COMMAND ----------

# MAGIC %md ## 処理対象データの確認
# MAGIC ここではデータブリックス環境に格納されているサンプルデータ`databricks-datasets`を使用します。
# MAGIC 
# MAGIC **参考資料**
# MAGIC - [Databricks datasets \| Databricks on AWS](https://docs.databricks.com/data/databricks-datasets.html)

# COMMAND ----------

# MAGIC %md
# MAGIC セルに`%fs`マジックコマンドを記述すると、ファイルシステム(DBFS)に対する操作を行えます。
# MAGIC 
# MAGIC **参考資料**
# MAGIC - [Databricksにおけるファイルシステム \- Qiita](https://qiita.com/taka_yayoi/items/e16c7272a7feb5ec9a92)

# COMMAND ----------

# MAGIC %md
# MAGIC データセットの一覧を表示

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /databricks-datasets/

# COMMAND ----------

# MAGIC %md
# MAGIC ここでは、ニューヨークにおけるタクシーの乗車履歴データを使用します。

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /databricks-datasets/nyctaxi/tripdata/green/

# COMMAND ----------

# MAGIC %md
# MAGIC ファイルサイズを確認します。

# COMMAND ----------

def dbfs_tree(folder, files_to_show=10, indent=0):
  import os
  bytes_to_string = sc._jvm.org.apache.spark.util.Utils.bytesToString
  name = lambda p: os.path.basename(os.path.abspath(p))
  files = dbutils.fs.ls(folder)
  total_size = 0
  for f in files:
    total_size += f.size
  total_size_readable = bytes_to_string(total_size)
  print(f"{'  ' * indent}`{name(folder)}`: total files: {len(files)}, total size: {total_size_readable}. Last {files_to_show} files:")
  for f in files[-files_to_show:]:
    if '/' == f.path[-1]:
      #print(f" * {f.path.split('/')[-2]}")
      dbfs_tree(f.path, files_to_show=files_to_show, indent=indent+1)
    else:
      print(f"{'  ' * indent} * {name(f.path)} ({bytes_to_string(f.size)})")

# COMMAND ----------

dbfs_tree("/databricks-datasets/nyctaxi/tripdata/green/")

# COMMAND ----------

# MAGIC %md ## pandasによる処理
# MAGIC 
# MAGIC 上の通りこのデータは圧縮された状態で約2.2Gバイトあります。
# MAGIC 
# MAGIC まずは、このデータをpandasデータフレームに読み込んでみます。処理に6分程度かかりますので、以下では実行結果のスクリーンショットをお見せします。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/handson20210629/OOM.png)

# COMMAND ----------

import pandas as pd
import glob

# COMMAND ----------

pd_full_df = pd.concat([pd.read_csv(f) for f in glob.glob('/dbfs/databricks-datasets/nyctaxi/tripdata/green/*.gz')])
pd_full_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC 全てのデータがメモリに乗り切らないため、上のようにOut-of-memory(OOM)エラーとなってしまいます。このため、通常は一部のデータやサンプリングデータを使用して分析を進めなくてはなりません。

# COMMAND ----------

# 一部のデータのみを読み込み
pd_sample_df = pd.read_csv("/dbfs/databricks-datasets/nyctaxi/tripdata/green/green_tripdata_2013-08.csv.gz")
len(pd_sample_df.index)

# COMMAND ----------

# MAGIC %md ## PySparkによる処理
# MAGIC 
# MAGIC Sparkの分散処理機能を活用すれば、膨大なデータであっても容易に処理を行うことができます。
# MAGIC 
# MAGIC 以下の例では、ディレクトリに格納されている全ての圧縮CSVファイルを読み込んで、Sparkデータフレームに読み込んでいます。

# COMMAND ----------

s_df = spark.read.option("header", "true").csv("/databricks-datasets/nyctaxi/tripdata/green/")
display(s_df)

# COMMAND ----------

# データフレームの行数を確認
s_df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC このようにSparkを活用することで、大規模データを容易に分析できるようになります。以下では、簡単な集計処理と可視化を行います。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 可視化
# MAGIC 
# MAGIC データの傾向を容易に把握できるように、データブリックスには可視化機構がビルトインされています。可視化ライブラリを使用しなくても柔軟にグラフを表示することができます。
# MAGIC 
# MAGIC ここでは、乗車距離と料金の散布図を作成します。
# MAGIC 
# MAGIC 1. ![](https://docs.databricks.com/_images/chart-button.png)をクリック
# MAGIC 1. Plot Optionsをクリック<br>
# MAGIC   - **Values**に`Trip_distance`と`Total_amount`をドラッグアンドドロップ
# MAGIC   - **Display Type**でScatter plotを選択
# MAGIC   - **Apply**をクリック。確認メッセージには**Confirm**をクリック
# MAGIC ![](https://sajpstorage.blob.core.windows.net/handson20210629/plot_options.png)
# MAGIC 
# MAGIC 
# MAGIC **参考資料**
# MAGIC - [Databricksにおけるデータの可視化 \- Qiita](https://qiita.com/taka_yayoi/items/36a307e79e9433121c38)

# COMMAND ----------

from pyspark.sql.functions import unix_timestamp, col

# 集計できるようにデータ型を変換します
s_df2 = (s_df
  .withColumn("lpep_pickup_datetime", (col("lpep_pickup_datetime").cast("timestamp")))
  .withColumn("Lpep_dropoff_datetime", (col("Lpep_dropoff_datetime").cast("timestamp")))
  .withColumn("Payment_type", (col("Payment_type").cast("integer")))
  .withColumn("Passenger_count", (col("Passenger_count").cast("integer")))
  .withColumn("Pickup_longitude", (col("Pickup_longitude").cast("double")))
  .withColumn("Pickup_latitude", (col("Pickup_latitude").cast("double")))
  .withColumn("Dropoff_longitude", (col("Dropoff_longitude").cast("double")))
  .withColumn("Dropoff_latitude", (col("Dropoff_latitude").cast("double")))
  .withColumn("Trip_distance", (col("Trip_distance").cast("double")))
  .withColumn("Total_amount", (col("Total_amount").cast("double")))
        )

display(s_df2)

# COMMAND ----------

# MAGIC %md ### 集計
# MAGIC 
# MAGIC PySparkのAPIを利用して、SQLと同様の集計を行うことができます。以下では乗車時間ごとの件数をカウントしています。

# COMMAND ----------

from pyspark.sql.functions import from_utc_timestamp, hour, col

s_df3 = (s_df2
  .select(hour(col("lpep_pickup_datetime")).alias("hour"))
  .groupBy("hour")
  .count()
  .orderBy("hour")
)
display(s_df3)

# COMMAND ----------

# MAGIC %md ## Koalas
# MAGIC 
# MAGIC ![](https://koalas.readthedocs.io/en/v1.6.0/_static/koalas-logo-docs.png)
# MAGIC 
# MAGIC [Koalas](https://github.com/databricks/koalas)は、[pandas](https://pandas.pydata.org/)の補完材を提供するオープンソースプロジェクトです。主にデータサイエンティストによって用いられるpandasは、簡単に使えるデータ構造とPython言語向けのデータ分析ツールを提供するPythonのパッケージです。しかし、pandasは大量データに対してスケールしません。KoalasはApache Sparkで動作するpandasと同等のAPIを提供することでこのギャップを埋めます。Koalasはpandasユーザーにとって有益であるだけではなく、例えばSparkデータフレームから直接データをプロットするなど、PySparkで実行するのが困難なタスクをサポートするので、KoalasはPySparkユーザーにも役立ちます。
# MAGIC 
# MAGIC **参考資料**
# MAGIC - [Koalasのご紹介 \- Qiita](https://qiita.com/taka_yayoi/items/5bbb3280940e73395bf5)
# MAGIC - [Koalas \| Databricks on AWS](https://docs.databricks.com/languages/koalas.html)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
