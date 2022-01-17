# Databricks notebook source
# MAGIC %md
# MAGIC # PythonによるSparkのZipファイル読み込み
# MAGIC 
# MAGIC Databricksにおいて、zipファイルの解凍を行う際には、ドライバーノードのファイルシステムとDBFS(Databricksファイルシステム)の両方を操作することになるので、それぞれにタイルする理解が必要となります。詳細は以下の記事をご覧ください。
# MAGIC 
# MAGIC - [Databricksにおけるzipファイルの取り扱い \- Qiita](https://qiita.com/taka_yayoi/items/0197d5c985089255f16a)
# MAGIC - [Databricksにおけるファイルシステム \- Qiita](https://qiita.com/taka_yayoi/items/e16c7272a7feb5ec9a92)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## サンプルzipファイルの取得

# COMMAND ----------

import urllib 
urllib.request.urlretrieve("https://resources.lendingclub.com/LoanStats3a.csv.zip", "/tmp/LoanStats3a.csv.zip")

# COMMAND ----------

# MAGIC %md ## ディレクトリの空き容量を確認

# COMMAND ----------

# MAGIC %md
# MAGIC ファイルに対する操作を行う際に`%sh`を使用するときには、結果は`/databricks/driver`に格納されます。

# COMMAND ----------

# MAGIC %sh pwd

# COMMAND ----------

# MAGIC %md このため、当該ディレクトリに解凍後のファイルが格納できるだけの空き容量があることを確認する必要があります。

# COMMAND ----------

# MAGIC %sh
# MAGIC df -h

# COMMAND ----------

# MAGIC %md ## ファイルの解凍、クリーンアップ
# MAGIC 1. ファイルの解凍
# MAGIC 1. 最初のコメント行の削除
# MAGIC 1. 解凍ファイルの削除

# COMMAND ----------

# MAGIC %sh
# MAGIC unzip /tmp/LoanStats3a.csv.zip
# MAGIC tail -n +2 LoanStats3a.csv > temp.csv
# MAGIC rm LoanStats3a.csv

# COMMAND ----------

# MAGIC %md ### 一時ファイルをDBFSに移動

# COMMAND ----------

dbutils.fs.mv("file:/databricks/driver/temp.csv", "dbfs:/tmp/zip_sample/LoanStats3a.csv")  

# COMMAND ----------

# MAGIC %md
# MAGIC ## 大規模ファイルの解凍

# COMMAND ----------

# MAGIC %md
# MAGIC 大規模データを解凍する際には、解凍先をDBFS(Databricksファイルシステム)にします。`%sh`でファイルパスの先頭に`/dbfs`をつけることで、ドライバーノードにマウントされたFUSEパスを使用してDBFSにアクセスすることができます。

# COMMAND ----------

# MAGIC %sh
# MAGIC unzip /tmp/LoanStats3a.csv.zip -d /dbfs/tmp/zip_sample/unzipped

# COMMAND ----------

# MAGIC %md
# MAGIC 解凍ファイルを格納しているDBFS上のディレクトリを確認します。

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/tmp/zip_sample/"))

# COMMAND ----------

# MAGIC %md
# MAGIC ドライバーノードのストレージを介さずにファイルが解凍されていることが確認できます。

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/tmp/zip_sample/unzipped"))

# COMMAND ----------

# MAGIC %md ## データフレームとしてファイルをロード
# MAGIC クリーンアップしたファイルをSparkでロードします。

# COMMAND ----------

df = spark.read.format("csv").option("inferSchema", "true").option("header","true").load("dbfs:/tmp/zip_sample/LoanStats3a.csv")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## クリーンアップ

# COMMAND ----------

dbutils.fs.rm("/tmp/zip_sample", True)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
