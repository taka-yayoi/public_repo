-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Adaptive Query Executionのデモ
-- MAGIC 
-- MAGIC Adaptive Query Execution (AQE)は実行時の統計情報に基づき、クエリー実行時に再度の最適化を行います。Spark 3.0におけるAQEには以下の3つの機能が含まれています。
-- MAGIC 
-- MAGIC - シャッフルパーティションの動的結合
-- MAGIC - join戦略の動的切り替え
-- MAGIC - skew joinの動的最適化

-- COMMAND ----------

-- MAGIC %md ## AQEの有効化

-- COMMAND ----------

SET -v

-- COMMAND ----------

-- In Databricks Runtime 7.3 LTS, AQE is enabled by default
set spark.sql.adaptive.enabled = true;

-- COMMAND ----------

-- デモ目的なので以下を設定
-- 実際の使用では設定不要です

set spark.sql.adaptive.coalescePartitions.minPartitionNum = 1;

-- COMMAND ----------

-- MAGIC %md ## テーブルの作成

-- COMMAND ----------

-- MAGIC %scala
-- MAGIC dbutils.fs.rm("dbfs:/user/hive/warehouse/aqe_demo_db", true)

-- COMMAND ----------

CREATE DATABASE IF NOT EXISTS aqe_demo_db;
USE aqe_demo_db;

DROP TABLE IF EXISTS items;
DROP TABLE IF EXISTS sales;

-- "items"テーブルの作成

CREATE TABLE items
USING parquet
AS
SELECT id AS i_item_id,
CAST(rand() * 1000 AS INT) AS i_price
FROM RANGE(30000000);

-- 偏りがある"sales"テーブルを作成
-- 全ての売り上げの80%がItem id 100となっています

CREATE TABLE sales
USING parquet
AS
SELECT CASE WHEN rand() < 0.8 THEN 100 ELSE CAST(rand() * 30000000 AS INT) END AS s_item_id,
CAST(rand() * 100 AS INT) AS s_quantity,
DATE_ADD(current_date(), - CAST(rand() * 360 AS INT)) AS s_date
FROM RANGE(1000000000);

-- COMMAND ----------

-- MAGIC %md ## シャッフルパーティションの動的結合

-- COMMAND ----------

-- 売り上げの日付でグルーピングし、売り上げ個数の合計を取得
-- このグルーピング結果は非常に小さいものになります

SELECT s_date, sum(s_quantity) AS q
FROM sales
GROUP BY s_date
ORDER BY q DESC;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC * 集計後のパーティションサイズは非常に小さいものになります。平均13KB、トータルで431KBとなります(ハイライトされたボックス**shuffle bytes written**をご覧ください)。
-- MAGIC * AQEはこれらの小さいパーティションを1つの新たなパーティションに結合します(ハイライトされたボックス**CustomShuffleReader**をご覧ください)。
-- MAGIC 
-- MAGIC ![screenshot_coalesce](https://docs.databricks.com/_static/images/spark/aqe/coalesce_partitions.png)

-- COMMAND ----------

-- MAGIC %md ## join戦略の動的切り替え

-- COMMAND ----------

-- 価格が10未満のアイテムに対して、売り上げ日ごとにグルーピングした売り上げ総額を計算
-- 価格によるフィルターの選択度は、静的な計画時には不明であるため、初期の計画時にはsort merge joinを選択します
-- しかし実際には、フィルタリング後の"items"テーブルは非常に小さいため、クエリーは代わりにbroadcast hash joinを実行することができます

-- 静的なexplainは初期計画時にsort merge joinであることを示しています

EXPLAIN FORMATTED
SELECT s_date, sum(s_quantity * i_price) AS total_sales
FROM sales
JOIN items ON s_item_id = i_item_id
WHERE i_price < 10
GROUP BY s_date
ORDER BY total_sales DESC;

-- COMMAND ----------

-- 実行時のjoin戦略はbroadcast hash joinに変更されています

SELECT s_date, sum(s_quantity * i_price) AS total_sales
FROM sales
JOIN items ON s_item_id = i_item_id
WHERE i_price < 10
GROUP BY s_date
ORDER BY total_sales DESC;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC * フィルタリング後の"items"テーブルのデータサイズは非常に小さく6.9 MB(ハイライトされたボックス**data size**)となっています。
-- MAGIC * AQEは実行時にsort merge joinからbroadcast hash joinに変更(ハイライトされたボックス**BroadcastHashJoin**)しています。
-- MAGIC 
-- MAGIC ![screenshot_strategy](https://docs.databricks.com/_static/images/spark/aqe/join_strategy.png)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## skew joinの動的最適化

-- COMMAND ----------

-- 売り上げ日でグルーピングし、トータルの売り上げを取得
-- "s_item_id"の値が"100"である"sales"テーブルのパーティションは他のパーティションよりはるかに大きいものとなります
-- AQEは"sales"テーブルと"items"テーブルのjoinの前に偏りのあるパーティションをより小さいパーティションに分割します

SELECT s_date, sum(s_quantity * i_price) AS total_sales
FROM sales
JOIN items ON i_item_id = s_item_id
GROUP BY s_date
ORDER BY total_sales DESC;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC * "sales"テーブルには偏りのあるパーティション(ハイライトされているボックス**number of skewed partitions**)が存在してます。
-- MAGIC * AQEは偏りのあるパーティションを小さいパーティションに分割(ハイライトされているボックス**number of skewed partition splits**)します。
-- MAGIC * sort merge joinオペレーターにはskew joinフラグがつけられています(ハイライトされているボックス**SortMergeJoin(isSkew=true)**)。
-- MAGIC 
-- MAGIC ![screenshot_skew](https://docs.databricks.com/_static/images/spark/aqe/skew_join.png)
