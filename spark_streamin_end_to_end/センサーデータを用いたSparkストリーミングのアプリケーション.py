# Databricks notebook source
# MAGIC %md # Apache Spark構造化ストリーミングPython APIを用いた連続アプリケーションによる書き込み
# MAGIC 
# MAGIC #### <img src="https://databricks.com/wp-content/uploads/2019/03/SAISE2019-OG-1.jpg" alt="" width="3%"/> San Francisco向けチュートリアル
# MAGIC 
# MAGIC パッとみると、分散ストリーミングエンジンの構築は、一連のサーバーを起動し、これらの間でデータをプッシュすれば良いというようにシンプルに見えるかもしれません。残念なことですが、分散ストリーミング処理は、バッチジョブのようなシンプルな計算処理では影響がでないような複数の複雑性に行き当たります。幸運なことに、PySpark 2.4やDatabricksを用いることで、これらをシンプルにします！
# MAGIC 
# MAGIC **オリジナルの著者**: Michael John
# MAGIC 
# MAGIC このチュートリアルのために、Jules S. Damjiによって変更され、PySparkに移植されました

# COMMAND ----------

# MAGIC %md ## フェイクセンサーデータの生成

# COMMAND ----------

# MAGIC %scala
# MAGIC 
# MAGIC // --- データを生成するパスを設定・適宜変更してください ---
# MAGIC val path = "/tmp/takaakiyayoidatabrickscom/Streaming/sdevices/"
# MAGIC dbutils.fs.mkdirs(path)
# MAGIC 
# MAGIC val numFiles = 100
# MAGIC val numDataPerFile = 100
# MAGIC 
# MAGIC import scala.util.Random
# MAGIC 
# MAGIC val deviceTypes = Seq("SensorTypeA", "SensorTypeB", "SensorTypeC", "SensorTypeD")
# MAGIC val startTime = System.currentTimeMillis
# MAGIC dbutils.fs.rm(path, true)
# MAGIC 
# MAGIC (1 to numFiles).par.foreach { fileId =>
# MAGIC   val file = s"$path/file-$fileId.json"
# MAGIC   val data = (1 to numDataPerFile).map { x => 
# MAGIC     val timestamp = new java.sql.Timestamp(startTime + (fileId * 60000) + (Random.nextInt() % 10000))
# MAGIC     val deviceId = Random.nextInt(100)
# MAGIC     val deviceType = deviceTypes(Random.nextInt(deviceTypes.size))
# MAGIC     val signalStrength = math.abs(Random.nextDouble % 100)
# MAGIC     s"""{"timestamp":"$timestamp","deviceId":$deviceId,"deviceType":"$deviceType","signalStrength":$signalStrength}"""
# MAGIC   }.mkString("\n")
# MAGIC   dbutils.fs.put(file, data)
# MAGIC }
# MAGIC dbutils.fs.head(dbutils.fs.ls(s"$path/file-1.json").head.path)

# COMMAND ----------

# MAGIC %md ### PySparkのドキュメント: [Click here](https://spark.apache.org/docs/latest/api/python/index.html)
# MAGIC  * [DataFrame](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=dataframe#pyspark.sql.DataFrame)
# MAGIC  * [Spark SQL](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#module-pyspark.sql.types)
# MAGIC  * [Spark SQL Functions](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#module-pyspark.sql.functions)

# COMMAND ----------

# MAGIC %md # ストリーミングセンサーデータの処理
# MAGIC 
# MAGIC エンドツーエンドの連続アプリケーションを構築する目的においては、構造化ストリーミングはパワフルな機能となります。ハイレベルにおいては、以下の機能を提供します:
# MAGIC 
# MAGIC 1. データのすべてのレコードはプレフィックス(パーティション)に保持され、順序を守って処理・カウントされるので、**出力テーブルは常に一貫性があります**
# MAGIC 1. **耐障害性**は、出力シンクとのやり取りを含み、構造化ストリーミングによって全体的に取り扱われます。
# MAGIC 1. **遅延データ、順序を守らないデータ**を取り扱う能力
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2019/04/streaming_continuous-apps-1024x366.png" alt="" width="40%"/>
# MAGIC 
# MAGIC <sub>リファレンス [Structured Streaming Blog](https://databricks.com/blog/2016/07/28/structured-streaming-in-apache-spark.html)</sub>

# COMMAND ----------

# MAGIC %md ## 1-ソース
# MAGIC 
# MAGIC 入力データソースを「入力テーブル」と考えましょう。ストリームに到着するすべてのデータアイテムは、入力テーブルに追加される新たな行なようなものとなります。

# COMMAND ----------

# MAGIC %md ## 2-連続処理 & クエリー
# MAGIC 
# MAGIC 次に、開発者は出力シンクに書き込まれる最終の結果テーブルを計算するために、*静的なテーブル*であるかのように、このソース、あるいは入力テーブルに対するクエリーを定義します。Sparkはこのバッチのようなクエリーをストリーミング実行プランに自動で変換します。これはインクリメンタル化とよばれるものです: Sparkは、レコードが到着するたびに結果をアップデートするために、どのような状態を維持する必要があるのかを特定します。最後に、開発者はいつ結果をアップデートするのかをコントロールするためのトリガーを指定します。トリガーが実行されるたびに、Sparkは新規データ(入力テーブルの新規行)をチェックし、結果をインクリメンタルにアップデートを行います。
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2019/04/streaming_cloudtrail-unbounded-tables.png" alt="" width="40%"/>

# COMMAND ----------

# MAGIC %md ## 3-シンク
# MAGIC 
# MAGIC このモデルの最後の部分は、出力モードとなります。結果テーブルがアップデートされる都度、開発者はS3、HDFS、データベースのような外部システムに対する変更を書き込みを行いたいと考えます。通常、出力をインクリメンタルに書き込みたいと考えます。このためには、構造化ストリーミングでは3つの出力モードを提供します:
# MAGIC </p>
# MAGIC 
# MAGIC * __Append__: 最後のトリガー以降に結果テーブルに追加された新規行のみが、外部ストレージに書き込まれます。
# MAGIC * __Complete__: 集計のようにアップデートされた結果テーブル全体が外部ストレージに書き込まれます。
# MAGIC * __Update__: 最後のトリガー以降に結果テーブル更新された行のみが、外部ストレージ上で変更されます。
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2019/04/treaming_stream-example1-phone-updated.png" width="40%"/>

# COMMAND ----------

# MAGIC %md ## PySpark構造化ストリーミングAPIを用いた連続処理アプリケーションのサンプル

# COMMAND ----------

# DBTITLE 0,Setup File Paths
# MAGIC %md 
# MAGIC 出力、チェックポイント、不正レコードのためのファイルパスをセットアップします。

# COMMAND ----------

# Cmd3の4行目のパスから /sdevices/ を除外してください
base_path = "/tmp/takaakiyayoidatabrickscom/Streaming"

output_path = f"{base_path}/out/iot-stream/"
checkpoint_path = f"{base_path}/out/iot-stream-checkpoint"
#
# チェックポイントパスの作成
#
dbutils.fs.rm(checkpoint_path,True) # チェックポイントの上書き
dbutils.fs.mkdirs(checkpoint_path)
#
#
bad_records_path = f"{base_path}/badRecordsPath/streaming-sensor/"
dbutils.fs.rm(bad_records_path, True) # ディレクトリを空に
dbutils.fs.mkdirs(bad_records_path)

# COMMAND ----------

# MAGIC %md ### センサーからのデータはのどうなものでしょうか？

# COMMAND ----------

sensor_path = f"{base_path}/sdevices/"
sensor_file_name= sensor_path + "file-1.json"
dbutils.fs.head(sensor_file_name, 233)

# COMMAND ----------

# MAGIC %md ### 入力ストリームと出力ストリームのスキーマを定義
# MAGIC 
# MAGIC 良いベストプラクティスはパフォーマンス上の理由からSparkにスキーマを推定させるのではなく、スキーマを定義するというものです。スキーマが無い場合、Sparkはいくつかのジョブを起動します: ヘッダーを読み込むためのジョブ、データが合致するようにスキーマを検証するためにパーティションの一部を読み込むジョブです。
# MAGIC 
# MAGIC 問題があれば即座にエラーを発生させ、欠損値やデータ型のミスマッチがあった際に、許容するか、NaNやnullで置き換えるように、スキーマを設定するオプションが存在しています。

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *

# オリジナルの入力スキーマ
jsonSchema = (
  StructType()
  .add("timestamp", TimestampType()) # ソースのイベント時間
  .add("deviceId", LongType())
  .add("deviceType", StringType())
  .add("signalStrength", DoubleType())
)
# いくつかのETL(変換およびカラムの追加)を行うのでカラムを追加してスキーマを変更します。
# この変換データは、処理やレポート生成に使用できるようにSQLテーブルを作成元としてのParquetファイルに格納されます。
parquetSchema = (
  StructType()
  .add("timestamp", TimestampType()) # ソースのイベント時間
  .add("deviceId", LongType())
  .add("deviceType", StringType())
  .add("signalStrength", DoubleType())
  .add("INPUT_FILE_NAME", StringType()) # このデータアイテムを読み込んだファイル名
  .add("PROCESSED_TIME", TimestampType())) # 処理中のエグゼキューターの時間

# COMMAND ----------

# MAGIC %md ### オブジェクトストアソースからのストリームの読み込み
# MAGIC 
# MAGIC このケースでは、ファイルから一度に読み込むことでKafkaライブストリームをシミュレートします。しかし、これをApache Kafkaのトピックにすることもできます。
# MAGIC 
# MAGIC __注意: チュートリアルのために意図的に遅くしています。__

# COMMAND ----------

inputDF = ( spark 
          .readStream 
          .schema(jsonSchema) 
          .option("maxFilesPerTrigger", 1)  # チュートリアルのために処理を遅くしています
          .option("badRecordsPath", bad_records_path) # いかなる不正レコードはこちらに格納されます
          .json(sensor_path) # ソース
          .withColumn("INPUT_FILE_NAME", input_file_name()) # ファイルパスを保持
          .withColumn("PROCESSED_TIME", current_timestamp()) # 処理時刻のタイムスタンプを追加
          .withWatermark("PROCESSED_TIME", "1 minute") # オプション: 順序が遅れたデータに対するウィンドウ
         )

# COMMAND ----------

# MAGIC %md ### Parquetファイルシンクへのストリームの書き込み

# COMMAND ----------

query = (inputDF
         .writeStream
         .format("parquet") # 後段処理あるいは必要に応じてバッチクエリーのために保存を行うシンク
         .option("path", output_path)
         .option("checkpointLocation", checkpoint_path) # 障害復旧のためのチェックポイントの追加
         .outputMode("append")
         .queryName("devices") # オプションとして、クエリーを実行する際に指定するクエリー名を指定
         .trigger(processingTime='5 seconds')
         .start() 
        )

# COMMAND ----------

# MAGIC %md
# MAGIC 以下のパスも適宜変更してください。

# COMMAND ----------

# MAGIC %fs ls "/tmp/takaakiyayoidatabrickscom/Streaming/out/iot-stream/"

# COMMAND ----------

# MAGIC %md #### クイックにSQLクエリーを実行できるように入力ストリームから一時テーブルを作成

# COMMAND ----------

inputDF.createOrReplaceTempView("parquet_sensors")

# COMMAND ----------

# MAGIC %md ### 入力ストリームから作成された一時テーブルに対してクエリーを実行

# COMMAND ----------

# MAGIC %sql select * from parquet_sensors where deviceType = 'SensorTypeD' or deviceType = 'SensorTypeA'

# COMMAND ----------

# MAGIC %md (上のセルで、**上記をすべて実行**をクリックしてください)
# MAGIC 
# MAGIC (その後で以下のセルを個々に実行してください)

# COMMAND ----------

# MAGIC %md ### 入力ストリームから保存されたParquetファイルに対して追加の処理を行い、クエリーを実行

# COMMAND ----------

spark.conf.set("spark.sql.shuffle.partitions", "1") # 優れたクエリー性能のためにシャッフルサイズを小さく維持
devices = (spark.readStream
           .schema(parquetSchema)
           .format("parquet")
           .option("maxFilesPerTrigger", 1) # デモのために遅くしています
           .load(output_path)
           .withWatermark("PROCESSED_TIME", "1 minute") # 順序を守らないデータに対するウィンドウ
          )

 # より複雑な集計クエリーを行うために一時テーブルを作成
devices.createOrReplaceTempView("sensors")

# COMMAND ----------

# MAGIC %md #### 何ファイルが処理されましたか？

# COMMAND ----------

display(
  devices.
  select("INPUT_FILE_NAME", "PROCESSED_TIME")
  .groupBy("INPUT_FILE_NAME", "PROCESSED_TIME")
  .count()
  .orderBy("PROCESSED_TIME", ascending=False)
)

# COMMAND ----------

# MAGIC %md #### どれだけのデータが通過しましたか？

# COMMAND ----------

# MAGIC %sql select count(*) from sensors

# COMMAND ----------

# MAGIC %md #### それぞれのセンサータイプにおける強度の最小値、最大値、平均値はどのようになっていますか？
# MAGIC 
# MAGIC Spark SQLの関数であるmin()、max()、avg()を使います。
# MAGIC 
# MAGIC __Note: Pythonノートブックで`%sql`マジックコマンドを使うことでSQLを使うことができます。__

# COMMAND ----------

# MAGIC %sql 
# MAGIC 
# MAGIC select count(*), deviceType, min(signalStrength), max(signalStrength), avg(signalStrength) 
# MAGIC   from sensors 
# MAGIC     group by deviceType 
# MAGIC     order by deviceType asc

# COMMAND ----------

# MAGIC %md #### デバイスと5秒ウィンドウごとのシグナルのカウントを集計するストリームを作成してみましょう
# MAGIC 
# MAGIC **注意: これはタンブリングウィンドウであり、スライディングウィンドウではありません。サイズは5秒です。**
# MAGIC 
# MAGIC 例えば、サイズ5の日、時間、分のタンブリングウィンドウは以下のようになります。
# MAGIC 
# MAGIC *[(00:00 - 00:05), (00:05: 00:10), (00:10: 00:15)]*
# MAGIC 
# MAGIC イベントは、これらのタンブリングウィンドウのいずれかに属することになります。

# COMMAND ----------

(devices
 .groupBy(
   window("timestamp", "5 seconds"),
   "deviceId"
 )
 .count()
 .createOrReplaceTempView("sensor_counts")) # データフレームを用いて一時ビューを作成

# COMMAND ----------

# MAGIC %md #### これらの5秒ウィンドウにおいて、どのデバイスがシグナルのロスを経験しているのか？

# COMMAND ----------

# MAGIC %sql select * from sensor_counts where count < 5 order by window.start desc

# COMMAND ----------

# MAGIC %md ### シグナルを送信していないダウンしている可能性があるセンサーのアラートの送信
# MAGIC 
# MAGIC 一時テーブル`sensor_counts`からデータフレームを作成しましょう。

# COMMAND ----------

lost_sensor_signals = (spark.table("sensor_counts")
         .filter(col("count") < 5)
         .select("window.start", "window.end", "deviceId", "count")
         )

# データフレームの表示
display(lost_sensor_signals)

# COMMAND ----------

# MAGIC %md ##### ワーカーのログに書き込むために、`foreach`メカニズムを活用
# MAGIC 
# MAGIC これは、モニタリングの目的で使用することができます。別のジョブがアラートのためにログをスキャンし、Kafkaのトピックに公開するジョブや、Gangliaにポストすることができます。利用できるKafkaサーバーや、REST API経由で利用できるGangliaサービスがあるのであればトライしてみるといい練習になります。

# COMMAND ----------

def processRow(row):
  # 今時点ではログファイルに書き込みを行いますが、このロジックは容易にKafkaのトピックや、GangliaやPagerDutyのようなモニタリング、ページングサービスにアラートを発行するように拡張することができます
  print("ALERT from Sensors: Between {} and {}, device {} reported only {} times".format(row.start, row.end, row.deviceId, row[3]))
  
(lost_sensor_signals
 .writeStream
 .outputMode("complete") # モニタリングのためにKafkaの"alerts"トピックにすることもできます
 .foreach(processRow)
 .start()
)

# COMMAND ----------

# MAGIC %md _こちらが書き出された結果のサンプルとなります。_
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2018/12/Screen-Shot-2018-12-18-at-5.53.39-PM.png" alt="" width="50%"/>

# COMMAND ----------

# MAGIC %md ## データのクリーンアップ

# COMMAND ----------

# MAGIC %md ### 稼働中のすべてのクエリーを取得

# COMMAND ----------

sqm = spark.streams
[q.name for q in sqm.active]

# COMMAND ----------

# MAGIC %md ### 停止

# COMMAND ----------

[q.stop() for q in sqm.active]

# COMMAND ----------

# 残っているファイルの削除
dbutils.fs.rm("/tmp/takaakiyayoidatabrickscom/Streaming", True)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
