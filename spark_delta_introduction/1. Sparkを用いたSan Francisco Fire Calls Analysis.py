# Databricks notebook source
# MAGIC %md
# MAGIC # サンフランシスコ消防署の通報データの分析
# MAGIC 
# MAGIC このノートブックでは書籍**Learning Spark 2nEd**の第3章のエンドツーエンドのサンプルを示すものであり、[San Francisco Fire Department Calls ](https://data.sfgov.org/Public-Safety/Fire-Department-Calls-for-Service/nuek-vuh3)データセットに対する一般的な分析パターンとオペレーションのために、どのようにデータフレームとSpark SQLを用いるのかを説明します。また、分析のためにどのようにETLやデータの検証、データのクエリーを行うのかをデモします。さらに、インメモリのSparkデータフレームをどのようにParquetファイルとして保存するのか、SparkがサポートするParquetデータソースとしてどのように読み込むのかを説明します。
# MAGIC 
# MAGIC ## 前提条件
# MAGIC 
# MAGIC このノートブックはDBR8.1以降で動作します。
# MAGIC 
# MAGIC ノートブック: https://databricks.com/notebooks/gallery/SanFranciscoFireCallsAnalysis.html

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
work_path = f"dbfs:/tmp/databricks_handson/{username}/fireServiceParquet"

# データベースの準備
spark.sql(f"DROP DATABASE IF EXISTS {db_name} CASCADE")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
spark.sql(f"USE {db_name}")

# データベースを表示。
print(f"database_name: {db_name}")
print(f"path_name: {work_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## データの確認
# MAGIC 
# MAGIC S3バケットに格納されているSF Fire Department Fire callsデータセットの場所を確認します。

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/learning-spark-v2/sf-fire/sf-fire-calls.csv

# COMMAND ----------

# MAGIC %md
# MAGIC S3バケット上のパブリックデータセットの位置を定義します。

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *

sf_fire_file = "/databricks-datasets/learning-spark-v2/sf-fire/sf-fire-calls.csv"

# COMMAND ----------

# MAGIC %md
# MAGIC スキーマを定義する前にデータの中身を調査します。

# COMMAND ----------

# MAGIC %fs head dbfs:/databricks-datasets/learning-spark-v2/sf-fire/README-sf-fire-calls.md

# COMMAND ----------

# MAGIC %fs head /databricks-datasets/learning-spark-v2/sf-fire/sf-fire-calls.csv

# COMMAND ----------

# MAGIC %md
# MAGIC ## スキーマ定義
# MAGIC 
# MAGIC ファイルには400万レコードが含まれているのでスキーマを定義します。大規模ファイルにおけるスキーマ推定には大きなコストが必要となります。

# COMMAND ----------

fire_schema = StructType([StructField('CallNumber', IntegerType(), True),
                     StructField('UnitID', StringType(), True),
                     StructField('IncidentNumber', IntegerType(), True),
                     StructField('CallType', StringType(), True),                  
                     StructField('CallDate', StringType(), True),      
                     StructField('WatchDate', StringType(), True),
                     StructField('CallFinalDisposition', StringType(), True),
                     StructField('AvailableDtTm', StringType(), True),
                     StructField('Address', StringType(), True),       
                     StructField('City', StringType(), True),       
                     StructField('Zipcode', IntegerType(), True),       
                     StructField('Battalion', StringType(), True),                 
                     StructField('StationArea', StringType(), True),       
                     StructField('Box', StringType(), True),       
                     StructField('OriginalPriority', StringType(), True),       
                     StructField('Priority', StringType(), True),       
                     StructField('FinalPriority', IntegerType(), True),       
                     StructField('ALSUnit', BooleanType(), True),       
                     StructField('CallTypeGroup', StringType(), True),
                     StructField('NumAlarms', IntegerType(), True),
                     StructField('UnitType', StringType(), True),
                     StructField('UnitSequenceInCallDispatch', IntegerType(), True),
                     StructField('FirePreventionDistrict', StringType(), True),
                     StructField('SupervisorDistrict', StringType(), True),
                     StructField('Neighborhood', StringType(), True),
                     StructField('Location', StringType(), True),
                     StructField('RowID', StringType(), True),
                     StructField('Delay', FloatType(), True)])

# COMMAND ----------

# MAGIC %md
# MAGIC ## データの読み込み

# COMMAND ----------

fire_df = spark.read.csv(sf_fire_file, header=True, schema=fire_schema)

# COMMAND ----------

# MAGIC %md
# MAGIC データに対して複数のオペレーションを実行するので、データフレームをキャッシュします。

# COMMAND ----------

fire_df.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC アクション`count()`を実行するのと同時に、データがキャッシュされます。

# COMMAND ----------

fire_df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC データフレームのスキーマを確認します。

# COMMAND ----------

fire_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC データフレームの最初の50レコードを表示します。以下ではグラフが設定されているので、呼び出しごとの遅延時間が棒グラフで表示されます。

# COMMAND ----------

display(fire_df.limit(50))

# COMMAND ----------

# MAGIC %md
# MAGIC ## フィルタリング
# MAGIC 
# MAGIC Call Typeの`"Medical Incident"`を除外します。
# MAGIC 
# MAGIC データフレームに対する`filter()`と`where()`メソッドは同じものであることに注意してください。引数のタイプに関しては[ドキュメント](https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.DataFrame.filter.html)をチェックしてください。

# COMMAND ----------

few_fire_df = (fire_df.select("IncidentNumber", "AvailableDtTm", "CallType") 
              .where(col("CallType") != "Medical Incident"))

few_fire_df.show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q-1) 消防署に対して何種類の呼び出しの電話がありましたか？
# MAGIC 
# MAGIC 念のために、カラムに含まれる"null"文字列はカウントしないようにしましょう。

# COMMAND ----------

fire_df.select("CallType").where(col("CallType").isNotNull()).distinct().count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q-2) 消防署に対してどのようなタイプの呼び出しがありましたか？
# MAGIC 
# MAGIC これらがSF Fire Departmentに対する全ての呼び出しタイプとなります。

# COMMAND ----------

fire_df.select("CallType").where(col("CallType").isNotNull()).distinct().show(10, False)

# COMMAND ----------

# MAGIC %md
# MAGIC `display()`メソッドを使用することもできます。

# COMMAND ----------

fire_df.select("CallType").where(col("CallType").isNotNull()).distinct().display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q-3) 遅延時間が5分以上の全ての応答を見つけ出してください
# MAGIC 
# MAGIC 1. カラム名`Delay`を`ReponseDelayedinMins`に変更します。
# MAGIC 2. 新たなデータフレームを返却します。
# MAGIC 3. 出火地点への反応時間が5分以上遅延した全ての電話を探し出します。

# COMMAND ----------

new_fire_df = fire_df.withColumnRenamed("Delay", "ResponseDelayedinMins")
new_fire_df.select("ResponseDelayedinMins").where(col("ResponseDelayedinMins") > 5).show(5, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ETL(Extract/Transform/Load)
# MAGIC 
# MAGIC いくつかのETL処理をやってみましょう:
# MAGIC 
# MAGIC 1. 後で時間に基づくクエリーを行えるように、文字列の日付をSparkのTimestampデータ型に変換します。
# MAGIC 1. 変換された結果が返却されます。
# MAGIC 1. 新たなデータフレームをキャッシュします。

# COMMAND ----------

fire_ts_df = (new_fire_df
              .withColumn("IncidentDate", to_timestamp(col("CallDate"), "MM/dd/yyyy")).drop("CallDate") 
              .withColumn("OnWatchDate",   to_timestamp(col("WatchDate"), "MM/dd/yyyy")).drop("WatchDate")
              .withColumn("AvailableDtTS", to_timestamp(col("AvailableDtTm"), "MM/dd/yyyy hh:mm:ss a")).drop("AvailableDtTm"))          

# COMMAND ----------

fire_ts_df.cache()
fire_ts_df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC 変換されたカラムがSpark Timestamp型であることを確認します。

# COMMAND ----------

fire_ts_df.select("IncidentDate", "OnWatchDate", "AvailableDtTS").show(5, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q-4) 最も多い呼び出しタイプは何ですか？
# MAGIC 
# MAGIC 降順で並び替えをしましょう。

# COMMAND ----------

(fire_ts_df
 .select("CallType").where(col("CallType").isNotNull())
 .groupBy("CallType")
 .count()
 .orderBy("count", ascending=False)
 .show(n=10, truncate=False))

# COMMAND ----------

display(fire_ts_df
 .select("CallType").where(col("CallType").isNotNull())
 .groupBy("CallType")
 .count()
 .orderBy("count", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q-4a) 呼び出しの大部分を占めているzipコードはなんですか？
# MAGIC 
# MAGIC サンフランシスコ消防署への通報においてどのzipコードが多いのか、どの場所でどのタイプが多いのかを調査してみましょう。
# MAGIC 
# MAGIC 1. `CallType`でフィルタリングします。
# MAGIC 2. `CallType`と`Zip code`でグルーピングします。
# MAGIC 3. カウントを行い、降順で表示します。
# MAGIC 
# MAGIC 最も共通する電話は`Medical Incident`に関連するものであり、多いzipコードは`94102`と`94103`です。

# COMMAND ----------

display(fire_ts_df
 .select("CallType", "ZipCode")
 .where(col("CallType").isNotNull())
 .groupBy("CallType", "Zipcode")
 .count()
 .orderBy("count", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q-4b) Zipコード94102と94103はサンフランシスコのどの地域ですか？
# MAGIC 
# MAGIC これらの2つのZipコードに関連づけられる地域を見つけ出しましょう。おそらく、通報率が高い地域は隣接しているケースがあるのでしょう。

# COMMAND ----------

display(fire_ts_df.select("Neighborhood", "Zipcode").where((col("Zipcode") == 94102) | (col("Zipcode") == 94103)).distinct())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q-5) 全ての呼び出しの合計、呼び出しに対する反応時間の平均値、最小値、最大値は何ですか？
# MAGIC 
# MAGIC いくつかのカラムに対して合計、平均値、最小値、最大値を計算するためにビルトインのSpark SQL関数を使いましょう。
# MAGIC 
# MAGIC * 通報の合計数
# MAGIC * 通報地点に消防隊員が到着するまでの反応時間の平均値、最小値、最大値

# COMMAND ----------

fire_ts_df.select(sum("NumAlarms"), avg("ResponseDelayedinMins"), min("ResponseDelayedinMins"), max("ResponseDelayedinMins")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q-6a) CSVファイルには何年分のデータが含まれていますか？
# MAGIC 
# MAGIC Timestamp型の`IncidentDate`から年を取り出すためにSpark SQL関数`year()`を使うことができます。
# MAGIC 
# MAGIC 全体的には2000-2018のデータが含まれていることがわかります。

# COMMAND ----------

fire_ts_df.select(year('IncidentDate')).distinct().orderBy(year('IncidentDate')).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q-6b) 2018年のどの週が最も通報が多かったですか？
# MAGIC 
# MAGIC **注意**: Week 1は新年の週で、Week 25は7/4の週となります。花火の季節を考えると、この週の通報が多いのは納得できます。

# COMMAND ----------

fire_ts_df.filter(year('IncidentDate') == 2018).groupBy(weekofyear('IncidentDate')).count().orderBy('count', ascending=False).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q-7) 2018年で最も反応が悪かったサンフランシスコの地域はどこですか？
# MAGIC 
# MAGIC Presidio Heightsに住んでいると消防隊員は3分以内に到着し、Mission Bayに住んでいるのであれば6分以上かかるようです。

# COMMAND ----------

fire_ts_df.select("Neighborhood", "ResponseDelayedinMins").filter(year("IncidentDate") == 2018).show(10, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q-8a) どのようにしてデータフレームをParquetファイルに保存し、読み戻すことができますか？

# COMMAND ----------

fire_ts_df.write.format("parquet").mode("overwrite").save(work_path)

# COMMAND ----------

display(dbutils.fs.ls(work_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q-8b) データを保存し、読み戻せるようにするためにどのようにSQLテーブルを使用できますか？

# COMMAND ----------

fire_ts_df.write.format("parquet").mode("overwrite").saveAsTable("FireServiceCalls")

# COMMAND ----------

# MAGIC %sql
# MAGIC CACHE TABLE FireServiceCalls

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM FireServiceCalls LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q-8c) どのようにParquetファイルを読み込むことができますか？
# MAGIC 
# MAGIC Parquetメタデータの一部にスキーマが格納されているので、スキーマを指定する必要がないことに注意してください。

# COMMAND ----------

file_parquet_df = spark.read.format("parquet").load(work_path)

# COMMAND ----------

display(file_parquet_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## クリーンアップ

# COMMAND ----------

dbutils.fs.rm(work_path, True)
spark.sql("DROP TABLE FireServiceCalls")

# COMMAND ----------

# MAGIC %md
# MAGIC # END
