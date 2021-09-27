# Databricks notebook source
# MAGIC %md # Delta Live Tables quickstart (Python)
# MAGIC 
# MAGIC このノートブックは以下の処理を行うDelta Live Tablesパイプラインのサンプルです：
# MAGIC 
# MAGIC - 生のJSONクリックストリームデータをテーブルに読み込みます。
# MAGIC - 生データテーブルからレコードを読み込み、Delta Live Tablesのエクスペクテーションを用いて、クレンジングされたデータを格納する新たなテーブルを作成します。
# MAGIC - Delta Live Tablesのクエリーを行い、クレンジングデータテーブルのレコードから新たなデータセットを作成します。
# MAGIC 
# MAGIC > **重要**
# MAGIC このノートブックはJobのユーザーインタフェースからパイプラインを実行します。ノートブック上では実行できません。
# MAGIC 
# MAGIC 1. サイドバーの![](https://docs.databricks.com/_images/jobs-icon.png)をクリックし、**Pipelines**タブをクリック、**Create Pipeline**をクリックします。
# MAGIC 1. パイプライン名を指定し、ノートブックを選択します。
# MAGIC 1. 任意で、パイプラインの出力データのストレージを指定します。**Storage Location**を空にした場合、システムはデフォルトの位置を使用します。
# MAGIC 1. **Pipeline Mode**に対して**Triggered**を選択します。
# MAGIC 1. **Create**を作成します。
# MAGIC 
# MAGIC ![](https://docs.databricks.com/_images/dlt-create-notebook-pipeline.png)
# MAGIC 
# MAGIC **Create**をクリック後、**Pipeline Details**ページが表示されます。** Pipelines**タブでパイプライン名をクリックすることでパイプラインの詳細を表示することができます。
# MAGIC 
# MAGIC 新たなパイプラインのアップデートを実行するには、トップパネルの![](https://docs.databricks.com/_images/dlt-start-button.png)ボタンをクリックします。パイプラインの開始を示すメッセージが表示されます。
# MAGIC 
# MAGIC ![](https://docs.databricks.com/_images/dlt-notebook-pipeline-successful-start.png)

# COMMAND ----------

# MAGIC %md ## ライブラリのインポート

# COMMAND ----------

import dlt
from pyspark.sql.functions import *
from pyspark.sql.types import *

# COMMAND ----------

# MAGIC %md ## 生のクリックストリームデータの投入

# COMMAND ----------

json_path = "/databricks-datasets/wikipedia-datasets/data-001/clickstream/raw-uncompressed-json/2015_2_clickstream.json"
@dlt.create_table(
  comment="The raw wikipedia click stream dataset, ingested from /databricks-datasets.",
  table_properties={
    "quality": "bronze"
  }
)
def clickstream_raw():          
  return (
    spark.read.json(json_path)
  )

# COMMAND ----------

# MAGIC %md ## データのクレンジングと準備

# COMMAND ----------

@dlt.create_table(
  comment="Wikipedia clickstream dataset with cleaned-up datatypes / column names and quality expectations.",
  table_properties={
    "quality": "silver"  
  }
)
@dlt.expect("valid_current_page", "current_page_id IS NOT NULL AND current_page_title IS NOT NULL")
@dlt.expect_or_fail("valid_count", "click_count > 0")
def clickstream_clean():
  return (
    dlt.read("clickstream_raw")
      .withColumn("current_page_id", expr("CAST(curr_id AS INT)"))
      .withColumn("click_count", expr("CAST(n AS INT)"))
      .withColumn("previous_page_id", expr("CAST(prev_id AS INT)"))
      .withColumnRenamed("curr_title", "current_page_title")
      .withColumnRenamed("prev_title", "previous_page_title")
      .select("current_page_id", "current_page_title", "click_count", "previous_page_id", "previous_page_title")      
  )

# COMMAND ----------

# MAGIC %md ## 参照数の多いページ

# COMMAND ----------

@dlt.create_table(
  comment="A table of the most common pages that link to the Apache Spark page.",
  table_properties={
    "quality": "gold"  
  }  
)
def top_spark_referrers():
  return (
    dlt.read("clickstream_clean")
      .filter(expr("current_page_title == 'Apache_Spark'"))
      .withColumnRenamed("previous_page_title", "referrer")
      .sort(desc("click_count"))
      .select("referrer", "click_count")
      .limit(10)
  )

# COMMAND ----------

# MAGIC %md ## クリック数の多いページ

# COMMAND ----------

@dlt.create_table(
  comment="A list of the top 50 pages by number of clicks.",
  table_properties={
    "quality": "gold"  
  }  
)
def top_pages():
  return (
    dlt.read("clickstream_clean")
      .groupBy("current_page_title")
      .agg(sum("click_count").alias("total_clicks"))
      .sort(desc("total_clicks"))
      .limit(50)
  )
