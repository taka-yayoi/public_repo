# Databricks notebook source
# MAGIC %md # Python DataFrames APIを用いた構造化ストリーミング
# MAGIC 
# MAGIC Apache Sparkには、高レベルのストリーミング処理APIである[構造化ストリーミング](http://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)が含まれています。このノートブックでは、構造化ストリーミングアプリケーションを構築するためにどのようにデータフレームAPIを活用するのかをクイックに見てみます。ここでは、ストリームにおけるタイムスタンプを伴うアクション(Open、Closeなど)に対するウィンドウ内のカウントやランニングカウントのようなリアルタイムのメトリクスを計算します。
# MAGIC 
# MAGIC このノートブックを実行するには、ノートブックをインポートしてSparkクラスターにアタッチしてください。

# COMMAND ----------

# MAGIC %md ## サンプルデータ
# MAGIC 
# MAGIC ここでアプリケーションを構築するために用いるサンプルのアクションデータは、`/databricks-datasets/structured-streaming/events/`にファイルとして格納されています。ディレクトリの中身を見てみましょう。

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/structured-streaming/events/

# COMMAND ----------

# MAGIC %md 
# MAGIC ディレクトリには約50個のJSONファイルが格納されています。個々のJSONファイルの中身を見てみましょう。

# COMMAND ----------

# MAGIC %fs head /databricks-datasets/structured-streaming/events/file-0.json

# COMMAND ----------

# MAGIC %md 
# MAGIC ファイル内のそれぞれの行には、2つのフィールド`time`と`action`が含まれています。インタラクティブにこれらのファイルを解析してみましょう。

# COMMAND ----------

# MAGIC %md ## バッチ/インタラクティブ処理
# MAGIC 
# MAGIC データを処理する最初のステップは、通常はデータに対するインタラクティブなクエリーの実行です。ファイルに対する静的なデータフレームを定義し、テーブル名をつけましょう。

# COMMAND ----------

from pyspark.sql.types import *

inputPath = "/databricks-datasets/structured-streaming/events/"

# すでにデータのフォーマットを知っているので、処理を高速化するためにスキーマを定義しましょう(Sparkがスキーマを推定する必要がなくなります)
jsonSchema = StructType([ StructField("time", TimestampType(), True), StructField("action", StringType(), True) ])

# JSONファイルのデータを表現する静的なデータフレーム
staticInputDF = (
  spark
    .read
    .schema(jsonSchema)
    .json(inputPath)
)

display(staticInputDF)

# COMMAND ----------

# MAGIC %md 
# MAGIC これで、1時間のタイムウィンドウにおける`open`と`close`アクションの数を計算することができます。このためには、`action`カラムと`time`カラムに対する1時間のウィンドウでグルーピングを行います。

# COMMAND ----------

from pyspark.sql.functions import *      # window()関数を使用するために必要

staticCountsDF = (
  staticInputDF
    .groupBy(
       staticInputDF.action, 
       window(staticInputDF.time, "1 hour"))    
    .count()
)
staticCountsDF.cache()

# データフレームをテーブル'static_counts'として登録します 
staticCountsDF.createOrReplaceTempView("static_counts")

# COMMAND ----------

# MAGIC %md
# MAGIC これで、テーブルに対してSQLを用いて直接クエリーを実行できます。例えば、以下のように全ての時間におけるアクションの総数を計算します。

# COMMAND ----------

# MAGIC %sql select action, sum(count) as total_count from static_counts group by action

# COMMAND ----------

# MAGIC %md ウィンドウ内のカウントはどうでしょうか？

# COMMAND ----------

# MAGIC %sql select action, date_format(window.end, "MMM-dd HH:mm") as time, count from static_counts order by time, action

# COMMAND ----------

# MAGIC %md
# MAGIC グラフの最後の2つに注意してください。対応するOpenアクションの後にCloseアクションが生じるように生成されており、最初に多くの"Open"が存在し、最後に多くの"Close"が存在しています。

# COMMAND ----------

# MAGIC %md ## ストリーム処理
# MAGIC 
# MAGIC ここまではデータをインタラクティブに分析しましたが、データの到着に伴って連続的に更新するストリーミングクエリーに切り替えましょう。今回は一連の静的なファイルがあるのみなので、一度に一つのファイルを時系列に読み込むことでストリームをシミュレートします。ここで記述すべきクエリーは、上述のインタラクティブなクエリーと非常に似たものとなります。

# COMMAND ----------

from pyspark.sql.functions import *

# 上で定義したstaticInputDFの定義と似ていますが、`read`ではなく`readStream`を使用します
streamingInputDF = (
  spark
    .readStream                       
    .schema(jsonSchema)               # JSONデータのスキーマを設定
    .option("maxFilesPerTrigger", 1)  # 一度に一つのファイルを取り込むことで一連のファイルをストリームとして取り扱います
    .json(inputPath)
)

# staticInputDFと同じクエリー
streamingCountsDF = (                 
  streamingInputDF
    .groupBy(
      streamingInputDF.action, 
      window(streamingInputDF.time, "1 hour"))
    .count()
)

# このデータフレームはストリーミングデータフレームでしょうか？
streamingCountsDF.isStreaming

# COMMAND ----------

# MAGIC %md 
# MAGIC 上でわかるように、`streamingCountsDF`はストリーミングデータフレームです(`streamingCountsDF.isStreaming`が`true`でした)。sinkを定義し、起動することで、ストリーミング処理をスタートすることができます。
# MAGIC 
# MAGIC 今回のケースでは、カウントをインタラクティブに取得(上と同じクエリー)したいので、インメモリのテーブルに1時間ごとのカウントをセットします。

# COMMAND ----------

spark.conf.set("spark.sql.shuffle.partitions", "2")  # シャッフルのサイズを小さくします

query = (
  streamingCountsDF
    .writeStream
    .format("memory")        # memory = インメモリテーブルに格納 
    .queryName("counts")     # counts = インメモリテーブルの名称
    .outputMode("complete")  # complete = 全てのカウントをテーブルに保持
    .start()
)

# COMMAND ----------

# MAGIC %md 
# MAGIC `query`はバックグラウンドで実行されるストリーミングクエリーのハンドルとなります。このクエリーは継続的にファイルを取得し、ウィンドウ内のカウントを更新します。
# MAGIC 
# MAGIC 上のセルのクエリーのステータスに注意してください。プログレスバーはクエリーがアクティブであることを示しています。さらに、上の`> counts`を展開すると、すでに処理されたファイルの数を確認することができます。
# MAGIC 
# MAGIC いくつかのファイルが処理されるまで待ち、インメモリの`counts`テーブルに対してインタラクティブにクエリーを実行してみましょう。

# COMMAND ----------

from time import sleep
sleep(5)  # 計算がスタートするまで少し待ちます

# COMMAND ----------

# MAGIC %sql select action, date_format(window.end, "MMM-dd HH:mm") as time, count from counts order by time, action

# COMMAND ----------

# MAGIC %md 
# MAGIC (上の静的な例と同様に)ウィンドウのカウントのタイムラインが構築されていく様子を確認することができます。このインタラクティブなクエリーを繰り返し実行すると、ストリーミングクエリーがバックグラウンドで更新を行なっているカウントの最新の値を確認することができます。

# COMMAND ----------

sleep(5)  # より多くのデータが処理されるまでもう少し待ちます

# COMMAND ----------

# MAGIC %sql select action, date_format(window.end, "MMM-dd HH:mm") as time, count from counts order by time, action

# COMMAND ----------

sleep(5)  # より多くのデータが処理されるまでもう少し待ちます

# COMMAND ----------

# MAGIC %sql select action, date_format(window.end, "MMM-dd HH:mm") as time, count from counts order by time, action

# COMMAND ----------

# MAGIC %md それでは"Open"と"Close"の総数を見てみましょう。

# COMMAND ----------

# MAGIC %sql select action, sum(count) as total_count from counts group by action order by action

# COMMAND ----------

# MAGIC %md 
# MAGIC 上のクエリーを繰り返し実行し続けることで、"Close"が常に対応する"Open"の後に出現するデータストリームで期待されたように、常に"Close"の数より"Open"の数が多いことを確認することができます。これは、構造化ストリーミングが**prefix integrity**を保証していることを示しています。prefix integrityの詳細を知りたい方は以下のブログ記事を参照ください。
# MAGIC 
# MAGIC [Spark Structured Streaming \- The Databricks Blog](https://databricks.com/blog/2016/07/28/structured-streaming-in-apache-spark.html)
# MAGIC 
# MAGIC ここでは少数のファイルしか取り扱っていないので、全てのファイルを処理してしまうとカウントは更新されなくなります。再度ストリーミングクエリーを操作したい場合には、クエリーを際実行してください。
# MAGIC 
# MAGIC 最後に、クエリーのセルの**Cancel**リンクをクリックするか、`query.stop()`を実行することで、バックグラウンドで実行しているクエリーを停止します。いずれの方法でも、クエリーが停止されると対応するセルのステータスは自動的に`TERMINATED`となります。
