# Databricks notebook source
# MAGIC %md
# MAGIC # Delta Lakeのチェンジデータフィードのデモ

# COMMAND ----------

# MAGIC %md
# MAGIC ## テーブルの準備

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 他のユーザーと重複しないデータベースを選択
# MAGIC USE 20210712_demo_takaakiyayoi;

# COMMAND ----------

# MAGIC %md ### シルバーテーブル: 国ごとのワクチン接種量と利用可能なワクチンの絶対値を追跡するシルバーテーブルを作成

# COMMAND ----------

countries = [("USA", 10000, 20000), ("India", 1000, 1500), ("UK", 7000, 10000), ("Canada", 500, 700) ]
columns = ["Country","NumVaccinated","AvailableDoses"]
spark.createDataFrame(data=countries, schema = columns).write.format("delta").mode("overwrite").saveAsTable("silverTable")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM silverTable

# COMMAND ----------

# MAGIC %md ### ゴールドテーブル: 国ごとのワクチン接種率を示すゴールドテーブルを作成

# COMMAND ----------

import pyspark.sql.functions as F
spark.read.format("delta").table("silverTable").withColumn("VaccinationRate", F.col("NumVaccinated") / F.col("AvailableDoses")) \
  .drop("NumVaccinated").drop("AvailableDoses") \
  .write.format("delta").mode("overwrite").saveAsTable("goldTable")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM goldTable

# COMMAND ----------

# MAGIC %md
# MAGIC ## シルバーテーブルでチェンジデータフィードを有効化

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE silverTable SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

# MAGIC %md
# MAGIC ## デイリーでシルバーレコードを更新

# COMMAND ----------

# 新規レコードをinsert
new_countries = [("Australia", 100, 3000)]
spark.createDataFrame(data=new_countries, schema = columns).write.format("delta").mode("append").saveAsTable("silverTable")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- レコードのupdate
# MAGIC UPDATE silverTable SET NumVaccinated = '11000' WHERE Country = 'USA'

# COMMAND ----------

# MAGIC %sql
# MAGIC -- レコードの削除
# MAGIC DELETE from silverTable WHERE Country = 'UK'

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM silverTable

# COMMAND ----------

# MAGIC %md
# MAGIC ## SQLとPySparkでチェンジデータを探索する

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- 変更の参照
# MAGIC SELECT * FROM table_changes('silverTable', 2, 5) order by _commit_timestamp

# COMMAND ----------

changes_df = spark.read.format("delta").option("readChangeData", True).option("startingVersion", 2).table('silverTable')
display(changes_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## シルバーからゴールドテーブルに変更を伝播

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 国ごとの最新バージョンのみを収集
# MAGIC CREATE OR REPLACE TEMPORARY VIEW silverTable_latest_version as
# MAGIC SELECT * 
# MAGIC     FROM 
# MAGIC          (SELECT *, rank() over (partition by Country order by _commit_version desc) as rank
# MAGIC           FROM table_changes('silverTable', 2, 5)
# MAGIC           WHERE _change_type !='update_preimage')
# MAGIC     WHERE rank=1

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ゴールドに変更をマージ
# MAGIC MERGE INTO goldTable t USING silverTable_latest_version s ON s.Country = t.Country
# MAGIC         WHEN MATCHED AND s._change_type='update_postimage' THEN UPDATE SET VaccinationRate = s.NumVaccinated/s.AvailableDoses
# MAGIC         WHEN NOT MATCHED THEN INSERT (Country, VaccinationRate) VALUES (s.Country, s.NumVaccinated/s.AvailableDoses)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM goldTable

# COMMAND ----------

# MAGIC %md
# MAGIC ## テーブルのクリーンアップ

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE silverTable;
# MAGIC DROP TABLE goldTable;

# COMMAND ----------

# MAGIC %md
# MAGIC # END
