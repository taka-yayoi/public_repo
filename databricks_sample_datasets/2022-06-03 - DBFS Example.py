# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## 概要
# MAGIC 
# MAGIC このノートブックでは、DBFSのファイルからテーブルを作成・検索したり、データフレームを作成する方法を説明します。[DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html)はDatabricksファイルシステムであり、Databricks内でデータを格納、クエリーを行うことができます。このノートブックでは、すでにDBFSに格納されているファイルを読み込むことを前提としています。
# MAGIC 
# MAGIC このノートブックは**Python**で記述されており、デフォルトのセルの言語はPythonとなっています。しかし、`%LANGUAGE`構文、あるいはセルの右上のボタンを用いることで、言語を切り替えることができます。Python、Scala、SQL、Rがサポートされています。

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /databricks-datasets/COVID/covid-19-data

# COMMAND ----------

# MAGIC %fs
# MAGIC head /databricks-datasets/COVID/covid-19-data/us-states.csv

# COMMAND ----------

# ファイルの格納場所とタイプ
file_location = "dbfs:/databricks-datasets/COVID/covid-19-data/us-states.csv"
file_type = "csv"

# CSVのオプション
infer_schema = "false" # スキーマの推定は行わない
first_row_is_header = "true" # 先頭行はヘッダー
delimiter = "," # 区切り文字

# CSVファイルのオプションが適用されます。他のファイルタイプの場合、これらは無視されます。
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

# 読み込んだデータフレームを表示します
display(df)

# COMMAND ----------

# ビューの作成

temp_table_name = "usstates"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC /* 作成した一時ビューをSQLセルでクエリーします */
# MAGIC 
# MAGIC select * from `usstates`

# COMMAND ----------

# これは一時ビューとして登録されているので、このノートブックでしか利用できません。他のユーザーもテーブルにクエリーできるようにするには、上のデータフレームからテーブルを作成することもできます。
# 保存することで、テーブルは永続化され、クラスターを再起動してもデータにアクセスでき、他のユーザーのノートブックからもアクセスできるようになります。
# このためには、テーブル名を指定し、最後の行のコメントを外します。

permanent_table_name = "taka_usstates"

df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- クリーンアップ
# MAGIC DROP TABLE taka_usstates;

# COMMAND ----------

# MAGIC %md
# MAGIC # END
