# Databricks notebook source
# DBTITLE 0,High Performance Spark Jobs and Queries with Databricks Delta
# MAGIC %md
# MAGIC # Databricks Deltaによる高性能Sparkクエリー
# MAGIC 
# MAGIC Databricks Deltaはデータの信頼性をシンプルに実現し、Sparkのパフォーマンスを改善するようにApache Sparkを拡張します。
# MAGIC 
# MAGIC 堅牢かつ高性能なデータパイプラインの構築は以下の理由で困難なものとなっています。
# MAGIC 
# MAGIC - _インデックス、統計情報の欠如_
# MAGIC - _スキーマ変更によるデータの一貫性の欠如_
# MAGIC - _パイプラインの失敗_
# MAGIC - _バッチ、ストリーミング処理のトレードオフ_
# MAGIC 
# MAGIC Databricks Deltaを用いることで、データエンジニアは信頼性があり高速なデータパイプラインを構築することができます。Databricks Deltaは以下を含む数多くのメリットを提供します。
# MAGIC 
# MAGIC - インデックス、統計情報を用いた高速なクエリー実行、自動キャッシュのサポート
# MAGIC - 豊富なスキーマ検証、トランザクション保証によるデータの信頼性
# MAGIC - 柔軟性のあるUPSERTのサポート、単一のデータソースに対する構造化ストリーミング + バッチ処理の統合によるシンプル化されたデータパイプライン
# MAGIC 
# MAGIC ### Databricks DeltaがどのようにSparkクエリーを高速化するのか見ていきましょう！
# MAGIC 
# MAGIC この例では、どのようにDatabricks Deltaがクエリー性能を最適化するのかを見ていきます。Parquetフォーマットを用いた標準的なテーブルを作成し、レーテンシーを観察するために簡単なクエリーを実行します。次に、標準的なテーブルとDatabricks Deltaテーブルの間での性能の違いを見るために、同じテーブルのDatabricks Deltaバージョンに対して二つ目のクエリーを実行します。
# MAGIC 
# MAGIC シンプルに以下の4ステップを踏んでいきます。
# MAGIC 
# MAGIC * __Step 1__ : USのフライトスケジュールデータを用いて標準的なParquetベーステーブルを作成します。
# MAGIC * __Step 2__ : 年間を通じて出発空港、月毎のフライト数を計算するクエリーを実行します。
# MAGIC * __Step 3__ : Databricks Deltaを用いてフライトテーブルを作成し、テーブルの最適化を行います。
# MAGIC * __Step 4__ : ステップ２のクエリーを再実行し、レーテンシーを観察します。
# MAGIC 
# MAGIC __注意:__ _この例では、1000万行のテーブルをいくつか構築します。お使いのクラスター設定によりますが、いくつかのオペレーションは数分要します。_

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parquetテーブルのクリーンアップ

# COMMAND ----------

# DBTITLE 0,Clean up Parquet tables
# MAGIC %fs rm -r /tmp/flights_parquet 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Databricks Deltaテーブルのクリーンアップ

# COMMAND ----------

# DBTITLE 0,Clean up Databricks Delta tables
# MAGIC %fs rm -r /tmp/flights_delta

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ0: フライトデータの読み込み

# COMMAND ----------

# DBTITLE 0,Step 0: Read flights data
flights = spark.read.format("csv") \
  .option("header", "true") \
  .option("inferSchema", "true") \
  .load("/databricks-datasets/asa/airlines/2008.csv")

# COMMAND ----------

flights.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ1: フライトデータを用いたParquetテーブルの書き込み

# COMMAND ----------

# DBTITLE 0,Step 1: Write a Parquet based table using flights data
flights.write.format("parquet").mode("overwrite").partitionBy("Origin").save("/tmp/flights_parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC ステップ1が完了すると、"flights"テーブルには年を通じたUSのフライト詳細が含まれます。
# MAGIC 
# MAGIC 次にステップ2では、週の初日のフライト数の月間合計に基づくトップ20の都市を取得するクエリーを実行します。

# COMMAND ----------

# MAGIC %md ## ステップ2: クエリー実行

# COMMAND ----------

# DBTITLE 0,Step 2: Run a query
from pyspark.sql.functions import count

flights_parquet = spark.read.format("parquet").load("/tmp/flights_parquet")

display(flights_parquet.filter("DayOfWeek = 1").groupBy("Month","Origin").agg(count("*").alias("TotalFlights")).orderBy("TotalFlights", ascending=False).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ステップ2が完了すると、標準的な"flights_parquet"テーブルにおけるレーテンシーを観測することができます。
# MAGIC 
# MAGIC ステップ3とステップ4においては、Databricks Deltaテーブルで同じことを行います。今回はクエリーを実行する前に、検索を高速化するためにデータを最適化するために`OPTIMIZE`と`ZORDER`を実行します。

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ3: フライトデータを用いたDatabricks Deltaテーブルの書き込み

# COMMAND ----------

# DBTITLE 0,Step 3: Write a Databricks Delta based table using flights data
flights.write.format("delta").mode("overwrite").partitionBy("Origin").save("/tmp/flights_delta")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ3(続き): Databricks DeltaテーブルのOPTIMIZE

# COMMAND ----------

# DBTITLE 0,Step 3 Continued: OPTIMIZE the Databricks Delta table
display(spark.sql("DROP TABLE  IF EXISTS flights"))

display(spark.sql("CREATE TABLE flights USING DELTA LOCATION '/tmp/flights_delta'"))
                  
display(spark.sql("OPTIMIZE flights ZORDER BY (DayofWeek)"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ4: Step2のクエリーを実行してレーテンシーを測定

# COMMAND ----------

# DBTITLE 0,Step 4 : Rerun the query from Step 2 and observe the latency
flights_delta = spark.read.format("delta").load("/tmp/flights_delta")

display(flights_delta.filter("DayOfWeek = 1").groupBy("Month","Origin").agg(count("*").alias("TotalFlights")).orderBy("TotalFlights", ascending=False).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC `OPTIMIZE`の実行後は、Databricks Deltaテーブルに対するクエリーが非常に高速になりました。クエリーをどのくらい高速化できるかは、処理を実行するクラスターに依存しますが、標準的なテーブルと比較して**5-10X**の高速化を実現することができます。
