-- Databricks notebook source
-- MAGIC %md # Delta Live Tablesクイックスタート(SQL)
-- MAGIC 
-- MAGIC このノートブックは以下の処理を行うDelta Live Tablesパイプラインのサンプルです：
-- MAGIC 
-- MAGIC - 生のJSONクリックストリームデータをテーブルに読み込みます。
-- MAGIC - 生データテーブルからレコードを読み込み、Delta Live Tablesのエクスペクテーションを用いて、クレンジングされたデータを格納する新たなテーブルを作成します。
-- MAGIC - Delta Live Tablesのクエリーを行い、クレンジングデータテーブルのレコードから新たなデータセットを作成します。
-- MAGIC 
-- MAGIC > **重要**
-- MAGIC このノートブックはJobのユーザーインタフェースからパイプラインを実行します。ノートブック上では実行できません。
-- MAGIC 
-- MAGIC 1. サイドバーの![](https://docs.databricks.com/_images/jobs-icon.png)をクリックし、**Pipelines**タブをクリック、**Create Pipeline**をクリックします。
-- MAGIC 1. パイプライン名を指定し、ノートブックを選択します。
-- MAGIC 1. 任意で、パイプラインの出力データのストレージを指定します。**Storage Location**を空にした場合、システムはデフォルトの位置を使用します。
-- MAGIC 1. **Pipeline Mode**に対して**Triggered**を選択します。
-- MAGIC 1. **Create**を作成します。
-- MAGIC 
-- MAGIC ![](https://docs.databricks.com/_images/dlt-create-notebook-pipeline.png)
-- MAGIC 
-- MAGIC **Create**をクリック後、**Pipeline Details**ページが表示されます。** Pipelines**タブでパイプライン名をクリックすることでパイプラインの詳細を表示することができます。
-- MAGIC 
-- MAGIC 新たなパイプラインのアップデートを実行するには、トップパネルの![](https://docs.databricks.com/_images/dlt-start-button.png)ボタンをクリックします。パイプラインの開始を示すメッセージが表示されます。
-- MAGIC 
-- MAGIC ![](https://docs.databricks.com/_images/dlt-notebook-pipeline-successful-start.png)

-- COMMAND ----------

-- MAGIC %md ## 生のクリックストリームデータの投入

-- COMMAND ----------

CREATE LIVE TABLE clickstream_raw
COMMENT "The raw wikipedia click stream dataset, ingested from /databricks-datasets."
TBLPROPERTIES ("quality" = "bronze")
AS SELECT * FROM json.`/databricks-datasets/wikipedia-datasets/data-001/clickstream/raw-uncompressed-json/2015_2_clickstream.json`

-- COMMAND ----------

-- MAGIC %md ## データのクレンジングと準備

-- COMMAND ----------

CREATE LIVE TABLE clickstream_clean(
  CONSTRAINT valid_current_page EXPECT (current_page_id IS NOT NULL and current_page_title IS NOT NULL),
  CONSTRAINT valid_count EXPECT (click_count > 0) ON VIOLATION FAIL UPDATE
)
COMMENT "Wikipedia clickstream dataset with cleaned-up datatypes / column names and quality expectations."
TBLPROPERTIES ("quality" = "silver")
AS SELECT
  CAST (curr_id AS INT) AS current_page_id,
  curr_title AS current_page_title,
  CAST(n AS INT) AS click_count,
  CAST (prev_id AS INT) AS previous_page_id,
  prev_title AS previous_page_title
FROM live.clickstream_raw

-- COMMAND ----------

-- MAGIC %md ## 参照数の多いページ

-- COMMAND ----------

CREATE LIVE TABLE top_spark_referers
COMMENT "A table of the most common pages that link to the Apache Spark page."
TBLPROPERTIES ("quality" = "gold")
AS SELECT
  previous_page_title as referrer,
  click_count
FROM live.clickstream_clean
WHERE current_page_title = 'Apache_Spark'
ORDER BY click_count DESC
LIMIT 10

-- COMMAND ----------

-- MAGIC %md ## クリック数の多いページ

-- COMMAND ----------

CREATE LIVE TABLE top_pages
COMMENT "A list of the top 50 pages by number of clicks."
TBLPROPERTIES ("quality" = "gold")
AS SELECT
  current_page_title,
  SUM(click_count) as total_clicks
FROM live.clickstream_clean
GROUP BY current_page_title
ORDER BY 2 DESC
LIMIT 50
