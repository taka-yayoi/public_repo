# Databricks notebook source
# DBTITLE 0,High Performance Spark Jobs and Queries with Databricks Delta
# MAGIC %md
# MAGIC # Delta LakeによるSparkクエリーの高速化
# MAGIC 
# MAGIC Delta Lakeはデータの信頼性をシンプルに実現し、Sparkのパフォーマンスを改善するようにApache Sparkを拡張します。
# MAGIC 
# MAGIC 堅牢かつ高性能なデータパイプラインの構築は以下の理由で困難なものとなっています。
# MAGIC 
# MAGIC - _インデックス、統計情報の欠如_
# MAGIC - _スキーマ変更によるデータの一貫性の欠如_
# MAGIC - _パイプラインの失敗_
# MAGIC - _バッチ、ストリーミング処理のトレードオフ_
# MAGIC 
# MAGIC Delta Lakeを用いることで、データエンジニアは信頼性があり高速なデータパイプラインを構築することができます。Delta Lakeは以下を含む数多くのメリットを提供します。
# MAGIC 
# MAGIC - インデックス、統計情報を用いた高速なクエリー実行、自動キャッシュのサポート
# MAGIC - 豊富なスキーマ検証、トランザクション保証によるデータの信頼性
# MAGIC - 柔軟性のあるUPSERTのサポート、単一のデータソースに対する構造化ストリーミング + バッチ処理の統合によるシンプル化されたデータパイプライン
# MAGIC 
# MAGIC [Databricksにおけるデータファイル管理によるパフォーマンスの最適化 \- Qiita](https://qiita.com/taka_yayoi/items/a82b0b8fbdc74e6f9f01)
# MAGIC 
# MAGIC ### Delta LakeがどのようにSparkクエリーを高速化するのか見ていきましょう！
# MAGIC 
# MAGIC この例では、どのようにDelta Lakeがクエリー性能を最適化するのかを見ていきます。Parquetフォーマットを用いた標準的なテーブルを作成し、レーテンシーを観察するために簡単なクエリーを実行します。次に、標準的なテーブルとDelta Lakeテーブルの間での性能の違いを見るために、同じテーブルのDelta Lakeバージョンに対して二つ目のクエリーを実行します。
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

import re
from pyspark.sql.types import * 

# Username を取得
username_raw = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
# Username の英数字以外を除去し、全て小文字化
username = re.sub('[^A-Za-z0-9]+', '', username_raw).lower()

# ユーザー固有のデータベース名を生成します
db_name = f"databricks_handson_{username}"

# ファイル格納パス
work_path = f"dbfs:/tmp/databricks_handson/{username}/delta_optimization"

# データベースの準備
spark.sql(f"DROP DATABASE IF EXISTS {db_name} CASCADE")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
spark.sql(f"USE {db_name}")

# データベースを表示。
print(f"database_name: {db_name}")
print(f"path_name: {work_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parquet/Deltaテーブルのクリーンアップ

# COMMAND ----------

# DBTITLE 0,Clean up Parquet tables
dbutils.fs.rm(work_path, True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ0: フライトデータの読み込み

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /databricks-datasets/asa/airlines/2008.csv

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
# MAGIC Sparkパーティションの数を確認します。

# COMMAND ----------

flights.rdd.getNumPartitions()

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ1: フライトデータを用いたParquetテーブルの書き込み
# MAGIC 
# MAGIC カラム`Origin`でParquetのパーティション(Sparkのパーティションとは異なります)を作成し、ディスクに書き込みます。

# COMMAND ----------

# DBTITLE 0,Step 1: Write a Parquet based table using flights data
flights.write.format("parquet").mode("overwrite").partitionBy("Origin").save(f"{work_path}/flights_parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC ステップ1が完了すると、"flights"テーブルには年を通じたUSのフライト詳細が含まれます。
# MAGIC 
# MAGIC 次にステップ2では、週の初日のフライト数の月間合計に基づくトップ20の都市を取得するクエリーを実行します。

# COMMAND ----------

display(dbutils.fs.ls(f"{work_path}/flights_parquet"))

# COMMAND ----------

# MAGIC %md ## ステップ2: クエリー実行

# COMMAND ----------

# DBTITLE 0,Step 2: Run a query
from pyspark.sql.functions import count

flights_parquet = spark.read.format("parquet").load(f"{work_path}/flights_parquet")

display(flights_parquet.filter("DayOfWeek = 1").groupBy("Month","Origin").agg(count("*").alias("TotalFlights")).orderBy("TotalFlights", ascending=False).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ステップ2が完了すると、標準的な"flights_parquet"テーブルにおけるレーテンシーを観測することができます。
# MAGIC 
# MAGIC ステップ3とステップ4においては、Deltaテーブルで同じことを行います。今回はクエリーを実行する前に、検索を高速化するためにデータを最適化するために`OPTIMIZE`と`ZORDER`を実行します。

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ3: フライトデータを用いたDeltaテーブルの書き込み

# COMMAND ----------

# DBTITLE 0,Step 3: Write a Databricks Delta based table using flights data
flights.write.format("delta").mode("overwrite").partitionBy("Origin").save(f"{work_path}/flights_delta")

# COMMAND ----------

display(dbutils.fs.ls(f"{work_path}/flights_delta"))

# COMMAND ----------

display(dbutils.fs.ls(f"{work_path}/flights_delta/Origin=ABE/"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ3(続き): Databricks DeltaテーブルのOPTIMIZE
# MAGIC 
# MAGIC Z-Orderingは関連する情報を同じファイルセットに配置するパフォーマンスチューニングの[テクニック](https://en.wikipedia.org/wiki/Z-order_curve)です。Delta Lakeのデータスキッピングアルゴリズムは読み取る必要があるデータ量を劇的に削減するために、この局所性(locality)を自動で活用します。データをZ-Orderするには、`ZORDER BY`句に並び替えを行うカラムを指定します。
# MAGIC 
# MAGIC [Databricksにおけるデータファイル管理によるパフォーマンスの最適化 \- Qiita](https://qiita.com/taka_yayoi/items/a82b0b8fbdc74e6f9f01#z-ordering%E5%A4%9A%E6%AC%A1%E5%85%83%E3%82%AF%E3%83%A9%E3%82%B9%E3%82%BF%E3%83%AA%E3%83%B3%E3%82%B0)
# MAGIC 
# MAGIC 以下の処理には3分程度時間を要します。通常は夜間バッチなどで最適化を行います。

# COMMAND ----------

# DBTITLE 0,Step 3 Continued: OPTIMIZE the Databricks Delta table
display(spark.sql("DROP TABLE IF EXISTS flights"))

display(spark.sql(f"CREATE TABLE flights USING DELTA LOCATION '{work_path}/flights_delta'"))
                  
display(spark.sql("OPTIMIZE flights ZORDER BY (DayofWeek)"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ4: ステップ2のクエリーを実行してレーテンシーを測定

# COMMAND ----------

# DBTITLE 0,Step 4 : Rerun the query from Step 2 and observe the latency
flights_delta = spark.read.format("delta").load(f"{work_path}/flights_delta")

display(flights_delta.filter("DayOfWeek = 1").groupBy("Month","Origin").agg(count("*").alias("TotalFlights")).orderBy("TotalFlights", ascending=False).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC `OPTIMIZE`の実行後は、Deltaテーブルに対するクエリーが非常に高速になりました。クエリーをどのくらい高速化できるかは、処理を実行するクラスターに依存しますが、標準的なテーブルと比較して**5-10X**の高速化を実現することができます。

# COMMAND ----------

# MAGIC %md
# MAGIC ## クリーンアップ

# COMMAND ----------

display(spark.sql("DROP TABLE IF EXISTS flights"))
dbutils.fs.rm(work_path, True)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
