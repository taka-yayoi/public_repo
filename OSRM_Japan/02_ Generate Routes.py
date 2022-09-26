# Databricks notebook source
# MAGIC %md 
# MAGIC このノートブックの目的は、ルーティングデータを生成するために、どのようにDatabricksクラスターで動作しているOSRMソフトウェアを使用するのかをデモンストレーションすることです。
# MAGIC 
# MAGIC **注意** このノートブックを実行する際に使用するクラスターにおいても十分なメモリーを搭載したインスタンスタイプ(128GBのRAM以上)を使用する様にしてください。そうしないと、initスクリプトでOSRMサーバーが起動しません。OSRMサーバーに接続できない場合には、[クラスターログ](https://qiita.com/taka_yayoi/items/8d951b660cd87c6c5f18#%E3%82%AF%E3%83%A9%E3%82%B9%E3%82%BF%E3%83%BC%E3%83%AD%E3%82%B0%E3%83%87%E3%83%AA%E3%83%90%E3%83%AA%E3%83%BC)や[Webターミナル](https://qiita.com/taka_yayoi/items/b3be567839a6fcb84136)でデバッグすることをお勧めします。

# COMMAND ----------

# MAGIC %md ## イントロダクション
# MAGIC 
# MAGIC ソフトウェアと地図ファイルを準備することで、それぞれのワーカーにOSRMバックエンドサーバーのインスタンスがデプロイされた(複数のワーカーノードを持つ)クラスターを起動することができました。
# MAGIC </p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/osrm_scaled_deployment2.png' width=500>
# MAGIC </p>
# MAGIC デフォルトでワーカーノードに分散されるSparkデータフレームにあるポイントデータを用いることで、スケーラブルな方法でルーティングデータを生成するために、これらのサーバーインスタンスに対してローカルの呼び出しを行う一連の関数を定義することができます。

# COMMAND ----------

# DBTITLE 1,必要なライブラリのインストール
# MAGIC %pip install tabulate databricks-mosaic

# COMMAND ----------

# DBTITLE 1,必要なライブラリのインポート
import requests

import pandas as pd
import numpy as np
import json

import itertools

import subprocess

import pyspark.sql.functions as fn
from pyspark.sql.types import *

# mosaicのインポートと設定
import mosaic as mos
spark.conf.set('spark.databricks.labs.mosaic.geometry.api', 'ESRI')
spark.conf.set('spark.databricks.labs.mosaic.index.system', 'H3')
mos.enable_mosaic(spark, dbutils)

from tabulate import tabulate

# COMMAND ----------

# MAGIC %md ## Step 1: それぞれのワーカーで稼働するサーバーの検証
# MAGIC 
# MAGIC 最初のステップは、それぞれのワーカーノードで実行されるOSRMバックエンドサーバーが期待通りに動作していることを確認することになります。このためには、クラスターのワーカーに対して小規模なデータセットの分散を強制するいにしえのSpark RDDを用いて、クラスターのそれぞれのIPアドレスを明らかにする必要があります。
# MAGIC 
# MAGIC さらにこれを理解するためには、それぞれのワーカーノードにおいて利用できるメモリーとプロセッサーのリソースが、Java Virtual Machines (JVM)でどのように分割されるのかを知ることが助けとなります。これらのJVMは *Executors* として参照され、Spark RDDやSparkデータフレームのデータのサブセットを保持します。多くの場合、エグゼキューターとワーカーノードには1対1の関係がありますが、常にそうと言う訳ではありません。
# MAGIC 
# MAGIC [*sc.defaultParallelism*](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.SparkContext.defaultParallelism.html#pyspark.SparkContext.defaultParallelism)プロパティは、クラスターのワーカーノードで利用できるプロセッサーの数をトラックし、この数に等しい値のレンジの並列度を用いてSpark RDDを用いることで、それぞれの仮想コアに1つの整数値を関連づけます。そして、[*sc.runJob*](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.SparkContext.runJob.html)メソッドが、RDDのそれぞれの値が存在するマシンのパブリックIPアドレスを収集する *hostname -I* コマンドのローカルインスタンスを実行するPythonの[*subprocess.run*](https://docs.python.org/3/library/subprocess.html#subprocess.run)メソッドを強制します。出力はコマンドによって識別されるユニークなIPの値を返却するために、Pythonのsetに変換されるlistとして返却されます。
# MAGIC 
# MAGIC このようなシンプルなタスクに対して多くの説明があるように聞こえますが、このノートブックの後半で別の関数呼び出しを行う際に同じパターンを用いることに注意してください。

# COMMAND ----------

# DBTITLE 1,ワーカーノードのIPアドレスの取得
# それぞれのワーカーのエグゼキューターに分散させるRDDの生成
myRDD = sc.parallelize(range(sc.defaultParallelism))

# IPアドレスの一覧を取得
ip_addresses = set( # 出力の重複を排除するためにsetに変換
  sc.runJob(
    myRDD, 
    lambda _: [subprocess.run(['hostname','-I'], capture_output=True).stdout.decode('utf-8').strip()] # それぞれのエグゼキューターで hostname -I を実行
    )
  )

ip_addresses

# COMMAND ----------

# MAGIC %md 
# MAGIC ワーカーノードのIPアドレスがわかったので、それぞれからルーティングのレスポンスをリクエストすることで、デフォルトポート5000でリッスンしているそれぞれのOSRMバックエンドサーバーのレスポンスをクイックにテストすることができます。

# COMMAND ----------

# DBTITLE 1,ルーティングレスポンスに対するそれぞれのワーカーのテスト
responses = []

# それぞれのワーカーのIPアドレスに対するループ
for ip in ip_addresses:
  print(ip)
  
  # OSRMバックエンドサーバーからレスポンスを取得
  resp = requests.get(f'http://{ip}:5000/route/v1/driving/139.6503,35.6762;139.6380,35.4437').text
  responses += [(ip, resp)]
  
# それぞれのワーカーで生成されたレスポンスを表示
display(
  pd.DataFrame(responses, columns=['ip','response'])
  )

# COMMAND ----------

# MAGIC %md ## Step 2: ルート生成データの取得
# MAGIC 
# MAGIC どのようにクラスターでルーティング能力を活用できるのかをデモンストレーションするためには、ルートを生成するためのデータを取得する必要があります。
# MAGIC 
# MAGIC 日本のルート生成をデモンストレーションするために、駅の緯度経度を使用します。
# MAGIC 
# MAGIC [東京都の駅\-路線の最新リストデータ 鉄道 \| オープンポータル](https://opendata-web.site/station/13/eki/)
# MAGIC 
# MAGIC GUIからCSVをアップロードして`csv_eki_13_csv`というテーブルを作成します。

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS takaakiyayoi_osrm;

# COMMAND ----------

# MAGIC %md
# MAGIC UIを使うなどしてテーブルを作成します。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/create_eki_table.png)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM hive_metastore.takaakiyayoi_osrm.csv_eki_13_csv;

# COMMAND ----------

# MAGIC %md
# MAGIC 出発地点と行き先をランダムに組み合わせます。

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS hive_metastore.takaakiyayoi_osrm.eki_movement;

# COMMAND ----------

# MAGIC %sql CREATE TEMPORARY VIEW from_table AS (
# MAGIC   SELECT
# MAGIC     station_name AS from_station_name,
# MAGIC     station_lat AS from_station_lat,
# MAGIC     station_lon AS from_station_lon,
# MAGIC     floor(RAND() * 700) AS random_id,
# MAGIC     floor(RAND() * 50) AS random_table_id -- あとでドライビングテーブルを作成する際に使用します
# MAGIC   FROM
# MAGIC     hive_metastore.takaakiyayoi_osrm.csv_eki_13_csv
# MAGIC )

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT * FROM from_table;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE hive_metastore.takaakiyayoi_osrm.eki_movement AS 
# MAGIC 
# MAGIC SELECT
# MAGIC   from_station_name,
# MAGIC   from_station_lat,
# MAGIC   from_station_lon,
# MAGIC   to_station_name,
# MAGIC   to_station_lat,
# MAGIC   to_station_lon,
# MAGIC   random_table_id
# MAGIC FROM
# MAGIC   from_table A
# MAGIC   INNER JOIN (
# MAGIC     SELECT
# MAGIC       station_name AS to_station_name,
# MAGIC       station_lat AS to_station_lat,
# MAGIC       station_lon AS to_station_lon,
# MAGIC       floor(RAND() * 700) AS random_id
# MAGIC     FROM
# MAGIC       hive_metastore.takaakiyayoi_osrm.csv_eki_13_csv
# MAGIC     ORDER BY
# MAGIC       RAND()
# MAGIC   ) B ON A.random_id = B.random_id

# COMMAND ----------

eki_movement_df = spark.read.table("hive_metastore.takaakiyayoi_osrm.eki_movement")
display(eki_movement_df)

# COMMAND ----------

# DBTITLE 1,データセットの行のカウント
eki_movement_df.count()

# COMMAND ----------

# MAGIC %md ## Step 3: 移動ルートの取得
# MAGIC 
# MAGIC 上で準備したデートセットには開始地点と終了地点が記録されています。我々は正確なルートを知らないので、本日移動する際にベストなルートを特定するために、OSRMバックエンドサーバーの[*route* メソッド](http://project-osrm.org/docs/v5.5.1/api/#route-service)を使用することができます。これを有効化するために、それぞれの移動のスタート地点、到着地点の緯度経度を引き渡す関数を記述します。この関数は、OSRMバックエンドサーバーからのルートをリクエストし、結果のJSONドキュメントを返却するためにこのデータを使用します。

# COMMAND ----------

# DBTITLE 1,ルートを取得すための関数の定義
@fn.pandas_udf(StringType())
def get_osrm_route(
  start_longitudes: pd.Series, 
  start_latitudes:pd.Series, 
  end_longitudes: pd.Series, 
  end_latitudes: pd.Series
  ) -> pd.Series:
   
  # データフレームを構成するために入力を組み合わせます
  df = pd.concat([start_longitudes, start_latitudes, end_longitudes, end_latitudes], axis=1)
  df.columns = ['start_lon','start_lat','end_lon','end_lat']

  # 特定の行に対するルートを取得するための内部関数
  def _route(row):
    r = requests.get(
      f'http://127.0.0.1:5000/route/v1/driving/{row.start_lon},{row.start_lat};{row.end_lon},{row.end_lat}?alternatives=true&steps=false&geometries=geojson&overview=simplified&annotations=false'
    )
    return r.text
  
  # 行ごとにルーティング関数を適用
  return df.apply(_route, axis=1)

# COMMAND ----------

# MAGIC %md 
# MAGIC この関数を理解するためには、データフレームのデータがクラスターのワーカーノードの仮想コアに割り当てられたエグゼキューターに対して分散されるサブセット(パーティション)に分割されることに注意してください。(エグゼキューターの概念に関しては、IPアドレスを取得しているセクションの説明をご覧ください) この関数をSparkデータフレームに適用すると、データフレーム自身の並列性に基づき、それぞれのパーティションに並列で適用されます。
# MAGIC 
# MAGIC この関数に指定される引数を通じて、それぞれのパーティションから値を受け取ります。それぞれの引数はカラムにマッピングされ、それぞれのカラムから、値に対応する複数の行がそれぞれの引数のpandasのSeriesとして受け取ります。Seriesによって受け取られる値の数は、パーティションのサイズと設定 *spark.databricks.execution.pandasUDF.maxBatchesToPrefetch* に依存します。
# MAGIC 
# MAGIC それぞれのseriesの値は同じ順番でソートされます。これらのseriesを結合すると、パーティションにあるデータの行を際作成することができます。結果のpandasデータフレームのそれぞれの行に対して、OSRMバックエンドサーバーのローカルインスタンスにリクエストを行う内部で定義した関数を適用します。バックエンドサーバーはJSON文字列としてルーティング情報を返却します。このJSON文字列は、pandasデータフレームのそれぞれに対して返却され、返却された値の結果の文字列は、Sparkデータフレームに組み込むために外部関数からSparkエンジンに送信されます。
# MAGIC 
# MAGIC ユーザー定義関数(UDF)で定義されたそれぞれの引数に対するpandasのSeriesとして値のセットを受け取り、pandasのシリーズとして対応する結果セットを返却する、この全体的なパターンによって、この関数をpandasのSeries-to-Seriesユーザー定義関数にします。このタイプのpandas UDFに関しては[こちら](https://docs.databricks.com/spark/latest/spark-sql/udf-python-pandas.html)を参照ください。
# MAGIC 
# MAGIC 我々のデータにpandas UDFを適用するには、以下の様にシンプルに[*withColumn*](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.withColumn.html)メソッド呼び出しのコンテキストでUDFを使用するだけです。

# COMMAND ----------

# DBTITLE 1,ルートの取得
display(
  eki_movement_df
    .withColumn(
      'osrm_route',
      get_osrm_route('from_station_lon','from_station_lat','to_station_lon','to_station_lat')
      )
    .selectExpr(
      'from_station_lon',
      'from_station_lat',
      'to_station_lon',
      'to_station_lat',
      'osrm_route',
      'from_station_name',
      'to_station_name'
      )
  )

# COMMAND ----------

# MAGIC %md
# MAGIC 関数呼び出しの結果はJSON文字列となります。pandas UDFはpandas UDFとSparkエンジンの間ですべての複雑な型をまとめる能力を持たないので、複雑なデータ型ではなく文字列を返却する様にしました。このため、文字列を複雑なデータ表現に変換する必要がある場合には、関数がこの処理を終えた後に行う必要があります。
# MAGIC 
# MAGIC 例えば、戻り値を希望の複雑な型に変換するために、文字列ベースのJSONスキーマの表現を指定し、このスキーマを引数として[*from_json*](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.from_json.html#pyspark.sql.functions.from_json)を適用します。
# MAGIC 
# MAGIC **注意** 従来のPySparkデータ型表現を用いてスキーマを定義したいのであれば、同様に動作する様子をこのノートブックの後半で説明しています。

# COMMAND ----------

# DBTITLE 1,ルートのJSONを複雑なデータ型表現に変換
# JSONドキュメントのスキーマ
response_schema = '''
  STRUCT<
    code: STRING, 
    routes: 
      ARRAY<
        STRUCT<
          distance: DOUBLE, 
          duration: DOUBLE, 
          geometry: STRUCT<
            coordinates: ARRAY<ARRAY<DOUBLE>>, 
            type: STRING
            >, 
          legs: ARRAY<
            STRUCT<
              distance: DOUBLE, 
              duration: DOUBLE, 
              steps: ARRAY<STRING>, 
              summary: STRING, 
              weight: DOUBLE
              >
            >, 
          weight: DOUBLE, 
          weight_name: STRING
          >
        >,
      waypoints: ARRAY<
        STRUCT<
          distance: DOUBLE, 
          hint: STRING, 
          location: ARRAY<DOUBLE>, 
          name: STRING
          >
        >
      >
  '''

# ルートの取得、JSONをstructに変換
eki_routes = (
  eki_movement_df
  .withColumn(
    'osrm_route',
    get_osrm_route('from_station_lon','from_station_lat','to_station_lon','to_station_lat')
    )
  .withColumn(
    'osrm_route',
    fn.from_json('osrm_route',response_schema)
    )
  .selectExpr(
    'osrm_route',
    'from_station_name',
    'to_station_name'
    )
  )


display(
  eki_routes
  )

# COMMAND ----------

# MAGIC %md 
# MAGIC JSONドキュメントの構造はOSRMバックエンドサーバーによって定義されます。これらの要素はシンプルなドット表記参照で抽出することができます。
# MAGIC 
# MAGIC **注意** JSONドキュメント内では、ドキュメントあたり一つのルートのみ存在したとしても、ルートは配列として表現されます。[*explode*](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.explode.html#pyspark.sql.functions.explode)関数は配列を展開し、配列のそれぞれの要素のフィールドの重複を発生させます。ルートの配列に1つのルートしかないので、この関数呼び出しはデータセットのサイズは増加しません。

# COMMAND ----------

# DBTITLE 1,ルートから距離と時間を取得
display(
   eki_routes
    .withColumn('route', fn.explode('osrm_route.routes'))
    .withColumn('route_meters', fn.col('route.distance'))
    .withColumn('route_seconds', fn.col('route.duration'))
    .selectExpr(
      'from_station_name',
      'to_station_name',
      'route_meters',
      'route_seconds'
      )
  )

# COMMAND ----------

# MAGIC %md 
# MAGIC 距離や時間のような属性に加え、OSRMバックエンドサーバーによって返却されるルート情報には[GeoJSON format](https://datatracker.ietf.org/doc/html/rfc7946)に準拠した*geometry*要素が含まれています。[Databricks Mosaic](https://databricks.com/blog/2022/05/02/high-scale-geospatial-processing-with-mosaic.html)ライブラリの[*st_geomfromgeojson*](https://databrickslabs.github.io/mosaic/api/geometry-constructors.html#st-geomfromgeojson) と[*st_aswkb*](https://databrickslabs.github.io/mosaic/api/geometry-accessors.html#st-aswkb)メソッドを用いることで、この要素を標準的な表現に変換します。

# COMMAND ----------

# DBTITLE 1,ルートジオメトリの取得
eki_geometry = (
  eki_routes
    .withColumn('route', fn.explode('osrm_route.routes')) # ルートの配列をexplode
    .withColumn('geojson', fn.to_json(fn.col('route.geometry')))
    .withColumn('geom', mos.st_aswkb(mos.st_geomfromgeojson('geojson')))
    .drop('osrm_route')
  )

display(eki_geometry)

# COMMAND ----------

# MAGIC %md 
# MAGIC そして、我々のルートデータの情報の検証に役立てるために、[Kepler visualization](https://databrickslabs.github.io/mosaic/usage/kepler.html)を用いてこの標準ジオメトリを可視化することができます。
# MAGIC 
# MAGIC **注意** [*mosaic_kepler magic* command](https://databrickslabs.github.io/mosaic/usage/kepler.html)の構文は、*dataset* *column_name* *feature_type* \[*row_limit*\]です。表示結果の右上のトグルを用いて、可視化の調整を行うことができます。

# COMMAND ----------

# DBTITLE 1,ルートの可視化
# MAGIC %%mosaic_kepler
# MAGIC eki_geometry geom "geometry" 1000

# COMMAND ----------

# MAGIC %md 
# MAGIC しかし、もちろんですが、我々はOSRMバックエンドサーバーからルーティングデータを取得することに限定されません。我々のゴールがポイント間の移動の最適化であるのならば、移動時間のテーブルを作成する必要があるかもしれません。このためには、OSRMバックエンドサーバーの[*table* method](http://project-osrm.org/docs/v5.5.1/api/#table-service)を呼び出す関数を記述することができます。

# COMMAND ----------

# DBTITLE 1,ドライブ時間テーブルの取得
@fn.pandas_udf(StringType())
def get_driving_table(
  points_arrays: pd.Series
  ) -> pd.Series:

  # 配列に含まれるポイントのテーブルを取得する内部関数
  def _table(points_array):
    
    points = ';'.join(points_array)
    
    r = requests.get(
      f'http://127.0.0.1:5000/table/v1/driving/{points}'
    )
    
    return r.text
  
  # テーブル関数を行ごとに適用
  return points_arrays.apply(_table)

# COMMAND ----------

# MAGIC %md 
# MAGIC この関数を呼び出すためにはポイントのコレクションを提供する必要があります。使っているデータはこれに適した良い方法を提供していませんので、上で準備した`random_table_id`でグルーピングされる地点を移動することにします。

# COMMAND ----------

# DBTITLE 1,ドライビング時間テーブルの取得
# ドライビングテーブルのスキーマ
response_schema = StructType([
  StructField('code',StringType()),
  StructField('destinations',ArrayType(
    StructType([
      StructField('hint',StringType()),
      StructField('distance',FloatType()),
      StructField('name',StringType()),
      StructField('location',ArrayType(FloatType()))
      ])
     )
   ),
  StructField('durations',ArrayType(ArrayType(FloatType()))),
  StructField('sources',ArrayType(
    StructType([
      StructField('hint',StringType()),
      StructField('distance',FloatType()),
      StructField('name',StringType()),
      StructField('location',ArrayType(FloatType()))
      ])
    ))
  ])

# COMMAND ----------

# ドライビングテーブルを取得しマトリクスを抽出
driving_tables = (
  eki_movement_df
  .withColumn('start_point', fn.expr("concat(from_station_lon,',',from_station_lat)"))
  .groupBy('random_table_id')
    .agg(fn.collect_set('start_point').alias('start_points'))
  .filter(fn.expr('size(start_points) > 1')) # テーブルには1つ以上のポイントが必要です
  .withColumn('driving_table', get_driving_table('start_points'))
  .withColumn('driving_table', fn.from_json('driving_table', response_schema))
  .withColumn('driving_table_durations', fn.col('driving_table.durations'))
  )  

display(driving_tables)

# COMMAND ----------

# MAGIC %md 
# MAGIC ドライビングテーブルから抽出したマトリクスを検証する際には、ルーティングが異なる場合があるため、方向が異なる際は推定されたドライビング時間に対称性がないことに注意することが重要です。データセットから取得したマトリクスの一つを以下に示します。

# COMMAND ----------

# DBTITLE 1,単一のドライビングテーブルの表示
# 単一のテーブルの生成
driving_table = driving_tables.limit(1).collect()[0]['driving_table_durations']

# ドライビングテーブルの表示
print(
    tabulate(
      np.array(driving_table),
      tablefmt='grid'
      )
    )

# COMMAND ----------

# MAGIC %md 
# MAGIC OSRMバックエンドサーバーの*route* と *table*メソッドはサーバーのREST API経由で利用できるメソッドの2つです。利用できるメソッドの完全なリストを以下に示します。
# MAGIC </p>
# MAGIC 
# MAGIC * [route](http://project-osrm.org/docs/v5.5.1/api/#route-service) - 指定された順序の座標間で最も早いルートを見つけ出します
# MAGIC * [nearest](http://project-osrm.org/docs/v5.5.1/api/#nearest-service) - 座標をストリートネットワークにスナップし、近傍のnマッチを返却します
# MAGIC * [table](http://project-osrm.org/docs/v5.5.1/api/#table-service) - 指定された座標のすべてのペア間で最速のルートの時間を計算します
# MAGIC * [match](http://project-osrm.org/docs/v5.5.1/api/#match-service) - 指定されたGPS地点を最も納得いく方法で道路のネットワークにスナップします
# MAGIC * [trip](http://project-osrm.org/docs/v5.5.1/api/#trip-service) - 貪欲なヒューリスティック(farthest-insertion algorithm)を用いてセールスマン巡回問題を解きます
# MAGIC * [tile](http://project-osrm.org/docs/v5.5.1/api/#tile-service) - ベクタータイルの能力を持つslippy-mapビューアーで参照可能なMapbox Vector Tileを生成します
# MAGIC 
# MAGIC Sparkデータフレームの処理過程でこれらにアクセスできる様にするには、上述した様にシンプルにHTTP REST API呼び出しを行うpandas UDFを構築し、文字列として結果のJSONを返却し、上の例で示した様に結果に対して適切なスキーマを適用するだけです。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | OSRM Backend Server                                  | High performance routing engine written in C++14 designed to run on OpenStreetMap data | BSD 2-Clause "Simplified" License    | https://github.com/Project-OSRM/osrm-backend                   |
# MAGIC | Mosaic | An extension to the Apache Spark framework that allows easy and fast processing of very large geospatial datasets | Databricks License| https://github.com/databrickslabs/mosaic | 
# MAGIC | Tabulate | pretty-print tabular data in Python | MIT License | https://pypi.org/project/tabulate/ |
