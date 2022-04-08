-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Databricksを5分でウォークスルー

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## クイックスタートのクラスターの作成
-- MAGIC 
-- MAGIC 1. サイドバーで**Compute**ボタンをクリくし、リンクを新規ウィンドウで開きます。
-- MAGIC 1. クラスターページで **Create Cluster** をクリックします。
-- MAGIC 1. 名前に **Quickstart** を入力します。
-- MAGIC 1. Databricks Runtime Versionのドロップダウンで **7.3 LTS (Scala 2.12, Spark 3.0.1)** を選択します。
-- MAGIC 1. **Create Cluster** をクリックします。

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## ノートブックをクラスターにアタッチ、全てのコマンドを実行
-- MAGIC 
-- MAGIC 1. ノートブックに戻ります。
-- MAGIC 1. ノートブックのメニューバーから **<img src="http://docs.databricks.com/_static/images/notebooks/detached.png"/></a> > Quickstart**を選択します。
-- MAGIC 1. クラスターが <img src="http://docs.databricks.com/_static/images/clusters/cluster-starting.png"/></a> から <img src="http://docs.databricks.com/_static/images/clusters/cluster-running.png"/></a> に変わったら、 **<img src="http://docs.databricks.com/_static/images/notebooks/run-all.png"/></a> Run All** をクリックします。

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## テーブルの作成
-- MAGIC 
-- MAGIC 次のコマンドはDatabricksデータセットからテーブルを作成します。

-- COMMAND ----------

DROP TABLE IF EXISTS diamonds;

CREATE TABLE diamonds
USING csv
OPTIONS (path "/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv", header "true")

-- COMMAND ----------

SELECT * from diamonds

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 以下のコマンドは読み込んだデータをDelta形式で保存します。

-- COMMAND ----------

-- MAGIC %python
-- MAGIC diamonds = spark.read.csv("/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv", header="true", inferSchema="true")
-- MAGIC diamonds.write.format("delta").mode("overwrite").save("/delta/diamonds")

-- COMMAND ----------

DROP TABLE IF EXISTS diamonds;

CREATE TABLE diamonds USING DELTA LOCATION '/delta/diamonds/'

-- COMMAND ----------

SELECT * from diamonds

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## テーブルにクエリーを実行
-- MAGIC 
-- MAGIC 以下のコマンドを実行します。
-- MAGIC 
-- MAGIC 1. 色ごとにグルーピングし、価格の平均を計算し、色と価格を選択し並び替える。
-- MAGIC 1. 結果のテーブルを表示する。
-- MAGIC 
-- MAGIC Databricksのノートブックには可視化機能がビルトインされていますので、結果をその場でグラフにすることができます。

-- COMMAND ----------

SELECT color, avg(price) AS price FROM diamonds GROUP BY color ORDER BY color

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## テーブルをチャートに切り替える
-- MAGIC 
-- MAGIC テーブルの下にある棒グラフアイコン<img src="http://docs.databricks.com/_static/images/notebooks/chart-button.png"/></a>をクリックします。

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ## Python Dataframe APIの活用
-- MAGIC 
-- MAGIC 同じオペレーションをPython Dataframe APIで繰り返します。
-- MAGIC 
-- MAGIC これはSQLノートブックです。デフォルトではコマンド文はSQLインタプリタに渡されます。コマンド文をPythonインタプリタに渡すには、`%python`マジックコマンドを使用します。

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 次のコマンドはDatabricksデータセットからデータフレームを作成します。

-- COMMAND ----------

-- MAGIC %python
-- MAGIC diamonds = spark.read.csv("/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv", header="true", inferSchema="true")

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 次のコマンドはデータの操作を行い結果を表示します。

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import avg
-- MAGIC 
-- MAGIC display(diamonds.select("color","price").groupBy("color").agg(avg("price")).sort("color"))

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # END
