# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Store タクシーデータサンプルノートブック
# MAGIC 
# MAGIC このノートブックでは、NYCのイエロータクシー料金の予測モデルを構築するためにFeature Storeの使用方法を説明します。このノートブックには以下のステップが含まれます。
# MAGIC 
# MAGIC - 特徴量の計算、書き出し
# MAGIC - 料金を予測するために特徴量を用いてモデルをトレーニング
# MAGIC - 新たなバッチデータに対し、既存の特徴量を用いてモデルを評価し、Feature Storeに保存
# MAGIC 
# MAGIC **要件**
# MAGIC - Databricks Runtime for Machine Learning 8.3以降
# MAGIC - このノートブックを実行する前に「Feature Store向けタクシーデータセットの作成」ノートブックの実行が必要です。([AWS](https://docs.databricks.com/_static/notebooks/machine-learning/feature-store-taxi-example-dataset.html)|[Azure](https://docs.microsoft.com/azure/databricks/_static/notebooks/machine-learning/feature-store-taxi-example-dataset.html)|[GCP](https://docs.gcp.databricks.com/_static/notebooks/machine-learning/feature-store-taxi-example-dataset.html))
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2021/06/22</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>クラスター</td><td>8.3ML</td></tr>
# MAGIC </table>
# MAGIC <img style="margin-top:25px;" src="https://jixjiadatabricks.blob.core.windows.net/images/databricks-logo-small-new.png" width="140">

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/taxi_example_flow_v3.png"/>

# COMMAND ----------

import re
from pyspark.sql.types import * 

# ログインIDからUsernameを取得
username_raw = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
# Username の英数字以外を除去し、全て小文字化。Username をファイルパスやデータベース名の一部で使用可能にするため。
username = re.sub('[^A-Za-z0-9]+', '', username_raw).lower()

# データベース名
db_name = f"nyc_yellow_taxi_with_zips_{username}"

# Hiveメタストアのデータベースの準備:データベースの作成
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
# Hiveメタストアのデータベースの選択
spark.sql(f"USE {db_name}")

print("database name: " + db_name)

# COMMAND ----------

# MAGIC %md ## 特徴量の計算

# COMMAND ----------

# MAGIC %md ### 特徴量を計算するために生のデータを読み込み
# MAGIC 
# MAGIC このステップを実行する前に、「Feature Store向けタクシーデータセットの作成」ノートブックを実行してDeltaテーブルを作成しておく必要があります。([AWS](https://docs.databricks.com/_static/notebooks/machine-learning/feature-store-taxi-example-dataset.html)|[Azure](https://docs.microsoft.com/azure/databricks/_static/notebooks/machine-learning/feature-store-taxi-example-dataset.html)|[GCP](https://docs.gcp.databricks.com/_static/notebooks/machine-learning/feature-store-taxi-example-dataset.html))

# COMMAND ----------

raw_data = spark.read.table(f"{db_name}.nyc_yellow_taxi_with_zips")
display(raw_data)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC タクシー料金のトランザクションデータから、乗車地点のZIPコードと降車時点のZIPコードに基づいて2つのグループの特徴量を計算します。
# MAGIC 
# MAGIC ### 乗車の特徴量
# MAGIC 1. 乗車回数 (時間ウィンドウ = 1時間、スライディングウィンドウ = 15分)
# MAGIC 1. 平均料金 (時間ウィンドウ = 1時間、スライディングウィンドウ = 15分)
# MAGIC 
# MAGIC ### 降車の特徴量
# MAGIC 1. 降車回数 (時間ウィンドウ = 30分)
# MAGIC 1. 移動は週末か否か (Pythonコードによるカスタム特徴量)
# MAGIC 
# MAGIC <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/taxi_example_computation_v5.png"/>

# COMMAND ----------

# MAGIC %md ### ヘルパー関数

# COMMAND ----------

from databricks import feature_store
from pyspark.sql.functions import *
from pyspark.sql.types import FloatType, IntegerType, StringType
from pytz import timezone


@udf(returnType=IntegerType())
def is_weekend(dt):
    tz = "America/New_York"
    return int(dt.astimezone(timezone(tz)).weekday() >= 5)  # 5 = 土曜日, 6 = 日曜日
  
@udf(returnType=StringType())  
def partition_id(dt):
    # datetime -> "YYYY-MM"
    return f"{dt.year:04d}-{dt.month:02d}"


def filter_df_by_ts(df, ts_column, start_date, end_date):
    if ts_column and start_date:
        df = df.filter(col(ts_column) >= start_date)
    if ts_column and end_date:
        df = df.filter(col(ts_column) < end_date)
    return df

# COMMAND ----------

# MAGIC %md ### 特徴量計算のためのデータサイエンティストカスタムコード

# COMMAND ----------

def pickup_features_fn(df, ts_column, start_date, end_date):
    """
    pickup_features特徴量グループを計算
    特徴量を特定の期間に限定するために、ts_column, start_date, (and/or) end_dateをkwargsとして渡します
    """
    df = filter_df_by_ts(
        df, ts_column, start_date, end_date
    )
    pickupzip_features = (
        df.groupBy(
            "pickup_zip", window("tpep_pickup_datetime", "1 hour", "15 minutes")
        )  # 1時間のタイムウィンドウ、15分のスライディングウィンドウ
        .agg(
            mean("fare_amount").alias("mean_fare_window_1h_pickup_zip"),
            count("*").alias("count_trips_window_1h_pickup_zip"),
        )
        .select(
            col("pickup_zip").alias("zip"),
            unix_timestamp(col("window.end")).alias("ts").cast(IntegerType()),
            partition_id(to_timestamp(col("window.end"))).alias("yyyy_mm"),
            col("mean_fare_window_1h_pickup_zip").cast(FloatType()),
            col("count_trips_window_1h_pickup_zip").cast(IntegerType()),
        )
    )
    return pickupzip_features
  
def dropoff_features_fn(df, ts_column, start_date, end_date):
    """
    dropoff_features特徴量グループを計算
    特徴量を特定の期間に限定するために、ts_column, start_date, (and/or) end_dateをkwargsとして渡します    
    """
    df = filter_df_by_ts(
        df,  ts_column, start_date, end_date
    )
    dropoffzip_features = (
        df.groupBy("dropoff_zip", window("tpep_dropoff_datetime", "30 minute"))
        .agg(count("*").alias("count_trips_window_30m_dropoff_zip"))
        .select(
            col("dropoff_zip").alias("zip"),
            unix_timestamp(col("window.end")).alias("ts").cast(IntegerType()),
            partition_id(to_timestamp(col("window.end"))).alias("yyyy_mm"),
            col("count_trips_window_30m_dropoff_zip").cast(IntegerType()),
            is_weekend(col("window.end")).alias("dropoff_is_weekend"),
        )
    )
    return dropoffzip_features  

# COMMAND ----------

# MAGIC %md 1月のデータから特徴量を生成

# COMMAND ----------

from datetime import datetime

pickup_features = pickup_features_fn(
    raw_data, ts_column="tpep_pickup_datetime", start_date=datetime(2016, 1, 1), end_date=datetime(2016, 1, 31)
)
dropoff_features = dropoff_features_fn(
    raw_data, ts_column="tpep_dropoff_datetime", start_date=datetime(2016, 1, 1), end_date=datetime(2016, 1, 31)
)

# COMMAND ----------

display(pickup_features)

# COMMAND ----------

# MAGIC %md
# MAGIC 新たな特徴量テーブルを作成するためにFeature Storeライブラリを使用します。

# COMMAND ----------

fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# MAGIC %md
# MAGIC スキーマとユニークなIDキーを定義するために、`create_feature_table`APIを使用します。オプションの引数`features_df`が指定された場合には、APIはFeature Storeにデータを書き込みます。

# COMMAND ----------

sqlContext.setConf("spark.sql.shuffle.partitions", "5")

fs.create_feature_table(
    name=f"{db_name}.trip_pickup_features",
    keys=["zip", "ts"],
    features_df=pickup_features,
    partition_columns="yyyy_mm",
    description="タクシー料金、乗車特徴量",
)
fs.create_feature_table(
    name=f"{db_name}.trip_dropoff_features",
    keys=["zip", "ts"],
    features_df=dropoff_features,
    partition_columns="yyyy_mm",
    description="タクシー料金、降車特徴量",
)

# COMMAND ----------

# MAGIC %md ## 特徴量の更新
# MAGIC 
# MAGIC 特徴量テーブルの値を更新するには`compute_and_write`関数を使用します。このFeature Store関数は`@feature_store.feature_table`でデコレートされたユーザー定義関数の属性を持ちます。
# MAGIC 
# MAGIC <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/taxi_example_compute_and_write.png"/>

# COMMAND ----------

display(raw_data)

# COMMAND ----------

# MAGIC %md
# MAGIC 2月のデータから特徴量を生成し、特徴量テーブルを更新

# COMMAND ----------

# pickup_features特徴量グループの計算
pickup_features_df = pickup_features_fn(
  df=raw_data,
  ts_column="tpep_pickup_datetime",
  start_date=datetime(2016, 2, 1),
  end_date=datetime(2016, 2, 29),
)

# 乗車時の特徴量データフレームをfeature storeテーブルに書き込み
fs.write_table(
  name=f"{db_name}.trip_pickup_features",
  df=pickup_features_df,
  mode="merge",
)

# dropoff_features特徴量グループの計算
dropoff_features_df = dropoff_features_fn(
  df=raw_data,
  ts_column="tpep_dropoff_datetime",
  start_date=datetime(2016, 2, 1),
  end_date=datetime(2016, 2, 29),
)

# 降車時の特徴量データフレームをfeature storeテーブルに書き込み
fs.write_table(
  name=f"{db_name}.trip_dropoff_features",
  df=dropoff_features_df,
  mode="merge",
)

# COMMAND ----------

# MAGIC %md 
# MAGIC 書き込みの際、`merge`と`overwrite`がサポートされています。
# MAGIC 
# MAGIC     dropoff_features_fn.compute_and_write(
# MAGIC         input={
# MAGIC           'df': raw_data, 
# MAGIC           'ts_column': "tpep_dropoff_datetime", 
# MAGIC           'start_date': datetime(2016, 2, 1),
# MAGIC           'end_date': datetime(2016, 2, 29),
# MAGIC         },
# MAGIC         feature_table_name="feature_store_taxi_example.trip_dropoff_features",
# MAGIC         mode="overwrite"
# MAGIC     )
# MAGIC 
# MAGIC `compute_and_write_streaming`を用いることで、ストリーミングとしてデータをFeature Storeに流し込むことができます。例えば、
# MAGIC 
# MAGIC     dropoff_features_fn.compute_and_write_streaming(
# MAGIC         input={
# MAGIC           'df': streaming_input, 
# MAGIC           'ts_column': "tpep_dropoff_datetime", 
# MAGIC           'start_date': datetime(2016, 2, 1),
# MAGIC           'end_date': datetime(2016, 2, 29),
# MAGIC         },
# MAGIC         feature_table_name="feature_store_taxi_example.trip_dropoff_features",
# MAGIC     )
# MAGIC 
# MAGIC Databricksのジョブを使用することで定期的にノートブックの処理を実行することができます。([AWS](https://docs.databricks.com/jobs.html)|[Azure](https://docs.microsoft.com/azure/databricks/jobs)|[GCP](https://docs.gcp.databricks.com/jobs.html))

# COMMAND ----------

# MAGIC %md 
# MAGIC 分析者はSQLを用いてFeature Storeを操作することができます。

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT SUM(count_trips_window_30m_dropoff_zip) AS num_rides,
# MAGIC        dropoff_is_weekend
# MAGIC FROM   trip_dropoff_features
# MAGIC WHERE  dropoff_is_weekend IS NOT NULL
# MAGIC GROUP  BY dropoff_is_weekend;

# COMMAND ----------

# MAGIC %md ## 特徴量の検索、発見

# COMMAND ----------

# MAGIC %md
# MAGIC ここまで実行することで、<a href="#feature-store/" target="_blank">Feature Store UI</a>から特徴量テーブルを検索することができます。
# MAGIC 
# MAGIC <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/taxi_example_flow_v3.png"/>
# MAGIC 
# MAGIC "trip_pickup_features"や"trip_dropoff_features"で検索することで、テーブルのスキーマ、メタデータ、データソース、作成者、オンラインストアなどを参照できます。
# MAGIC 
# MAGIC また、特徴量テーブルの説明文を編集したり、特徴量テーブル名の隣にある下向き矢印アイコンをクリックすることで特徴量テーブルに対するアクセス権を設定できます。
# MAGIC 
# MAGIC 詳細は[Use the Feature Store UI
# MAGIC ](https://docs.databricks.com/applications/machine-learning/feature-store.html#use-the-feature-store-ui)をご覧ください。

# COMMAND ----------

# MAGIC %md ## モデルのトレーニング
# MAGIC 
# MAGIC このセクションでは、Feature Storeに格納された乗車時、降車時の特徴量を用いてどのようにモデルをトレーニングするのかを説明します。タクシー料金を予測するためにLightGBMモデルをトレーニングします。

# COMMAND ----------

# MAGIC %md ### 　ヘルパー関数

# COMMAND ----------

from pyspark.sql import *
from pyspark.sql.functions import current_timestamp
from pyspark.sql.types import IntegerType
import math
from datetime import timedelta
import mlflow.pyfunc


def rounded_unix_timestamp(dt, num_minutes=15):
    """
    datetimeのdtをnum_minutes間隔に切り上げ、unixタイムスタンプを返却します
    """
    nsecs = dt.minute * 60 + dt.second + dt.microsecond * 1e-6
    delta = math.ceil(nsecs / (60 * num_minutes)) * (60 * num_minutes) - nsecs
    return int((dt + timedelta(seconds=delta)).timestamp())


rounded_unix_timestamp_udf = udf(rounded_unix_timestamp, IntegerType())


def rounded_taxi_data(taxi_data_df):
    # タクシーデータのタイムスタンプを15分、30分間隔に丸めることで、乗車時の特徴量、降車時の特徴量を結合できるようにします
    taxi_data_df = (
        taxi_data_df.withColumn(
            "rounded_pickup_datetime",
            rounded_unix_timestamp_udf(taxi_data_df["tpep_pickup_datetime"], lit(15)),
        )
        .withColumn(
            "rounded_dropoff_datetime",
            rounded_unix_timestamp_udf(taxi_data_df["tpep_dropoff_datetime"], lit(30)),
        )
        .drop("tpep_pickup_datetime")
        .drop("tpep_dropoff_datetime")
    )
    taxi_data_df.createOrReplaceTempView("taxi_data")
    return taxi_data_df
  
def get_latest_model_version(model_name):
  latest_version = 1
  mlflow_client = MlflowClient()
  for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
    version_int = int(mv.version)
    if version_int > latest_version:
      latest_version = version_int
  return latest_version

# COMMAND ----------

# MAGIC %md ### トーレニング用タクシーデータの読み込み

# COMMAND ----------

# トレーニングデータのロード
raw_taxi_data = spark.read.table(
    f"{db_name}.nyc_yellow_taxi_with_zips"
)

taxi_data = rounded_taxi_data(raw_taxi_data)

# COMMAND ----------

# MAGIC %md ### どのようにトレーニングデータセットが作成されるのかを理解する
# MAGIC 
# MAGIC モデルをトレーニングするためには、モデルをトレーニングするのに必要なトレーニングデータセットを作成する必要があります。トレーニングデータセットは以下から構成されます：
# MAGIC 
# MAGIC 1. 生の入力データ
# MAGIC 1. Feature Storeから得られる特徴量
# MAGIC 
# MAGIC 生のデータが必要なのは、生データには以下が含まれているためです。
# MAGIC 
# MAGIC 1. 特徴量を結合するための主キー
# MAGIC 1. Feature Storeに格納されていない`trip_distance`のような生データ
# MAGIC 1. モデルトレーニングに必要となる`fare`のような予測対象
# MAGIC 
# MAGIC トレーニングデータセットを生成するためにFeature Storeの特徴量と組み合わされた生データを以下の図で示します。
# MAGIC 
# MAGIC <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/taxi_example_feature_lookup.png"/>
# MAGIC 
# MAGIC これらのコンセプトはトレーニングデータセット生成に関するドキュメントで詳細が説明されています。([AWS](https://docs.databricks.com/applications/machine-learning/feature-store.html#create-a-training-dataset)|[Azure](https://docs.microsoft.com/en-us/azure/databricks/applications/machine-learning/feature-store#create-a-training-dataset)|[GCP](https://docs.gcp.databricks.com/applications/machine-learning/feature-store.html#create-a-training-dataset))
# MAGIC 
# MAGIC 次のセルでは、必要となる特徴量それぞれに対して`FeatureLookup`を作成し、モデルトレーニングのためにFeature Storeから特徴量をロードしています。

# COMMAND ----------

from databricks.feature_store import FeatureLookup
import mlflow

pickup_features_table = f"{db_name}.trip_pickup_features"
dropoff_features_table = f"{db_name}.trip_dropoff_features"

pickup_feature_lookups = [
    FeatureLookup( 
      table_name = pickup_features_table,
      feature_name = "mean_fare_window_1h_pickup_zip",
      lookup_key = ["pickup_zip", "rounded_pickup_datetime"],
    ),
    FeatureLookup( 
      table_name = pickup_features_table,
      feature_name = "count_trips_window_1h_pickup_zip",
      lookup_key = ["pickup_zip", "rounded_pickup_datetime"],
    ),
]

dropoff_feature_lookups = [
    FeatureLookup( 
      table_name = dropoff_features_table,
      feature_name = "count_trips_window_30m_dropoff_zip",
      lookup_key = ["dropoff_zip", "rounded_dropoff_datetime"],
    ),
    FeatureLookup( 
      table_name = dropoff_features_table,
      feature_name = "dropoff_is_weekend",
      lookup_key = ["dropoff_zip", "rounded_dropoff_datetime"],
    ),
]

# COMMAND ----------

# MAGIC %md ### トレーニングデータセットの作成
# MAGIC 
# MAGIC 下のセルで`fs.create_training_set(..)`が呼び出されると、以下のステップが実行されます。
# MAGIC 
# MAGIC 1. `TrainingSet`オブジェクトが作成され、あなたのモデルをトレーニングするのに使われる特定の特徴量をFeature Storeから取得します。それぞれの特徴量は、上で作成された`FeatureLookup`で指定されます。
# MAGIC 1. それぞれの`FeatureLookup`の`lookup_key`に基づいて、生データと特徴量が結合されます。
# MAGIC 
# MAGIC そして、`TrainingSet`はトレーニング対象となるデータフレームに変換されます。このデータフレームにはtaxi_dataのカラムと`FeatureLookups`で指定された特徴量が含まれます。

# COMMAND ----------

# (2回目以降のノートブックの実行の場合)実行中のrunを終了します
mlflow.end_run()

# MLflowのrunをスタートします。これはFeature Storeがモデルを記録するために必要となります
mlflow.start_run() 

# タイムスタンプのカラムは丸められているので、さらなる特徴量エンジニアリングを行ない、それらに対するトレーニングを回避しない限り、モデルが過学習する可能性があります
exclude_columns = ["rounded_pickup_datetime", "rounded_dropoff_datetime"]

# 生データ両方の特徴量テーブルとマージされた生データを含むトレーニングセットを作成します
training_set = fs.create_training_set(
  taxi_data,
  feature_lookups = pickup_feature_lookups + dropoff_feature_lookups,
  label = "fare_amount",
  exclude_columns = exclude_columns
)

# モデルをトレーニングするためにsklearnに引き渡せるように、TrainingSetをデータフレームに読み込みます
training_df = training_set.load_df()

# COMMAND ----------

# トレーニングデータフレームを表示します。生データと`dropoff_is_weekend`のようにFeature Storeから得られる特徴量が含まれていることに注意してください。
display(training_df)

# COMMAND ----------

# MAGIC %md
# MAGIC `TrainingSet.load_df`から得られたデータに対してLightGBMモデルのトレーニングを行います。そして、`FeatureStoreClient.log_model`を用いてモデルをロギングします。モデルは特徴量メタデータとともにパッケージングされます。

# COMMAND ----------

from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient
import lightgbm as lgb
import mlflow.lightgbm
from mlflow.models.signature import infer_signature

features_and_label = training_df.columns

# トレーニングのためにPandas arrayにデータをロードします
data = training_df.toPandas()[features_and_label]

train, test = train_test_split(data, random_state=123)
X_train = train.drop(["fare_amount"], axis=1)
X_test = test.drop(["fare_amount"], axis=1)
y_train = train.fare_amount
y_test = test.fare_amount

mlflow.lightgbm.autolog()
train_lgb_dataset = lgb.Dataset(X_train, label=y_train.values)
test_lgb_dataset = lgb.Dataset(X_test, label=y_test.values)

param = {"num_leaves": 32, "objective": "regression", "metric": "rmse"}
num_rounds = 100

# lightGBMモデルのトレーニング
model = lgb.train(
  param, train_lgb_dataset, num_rounds
)

# COMMAND ----------

# トレーニングしたモデルをMLflowでロギングし、特徴量検索情報と共にパッケージングします
fs.log_model(
  model,
  artifact_path="model_packaged",
  flavor=mlflow.lightgbm,
  training_set=training_set,
  registered_model_name=f"{db_name}_taxi_example_fare_packaged"
)

# COMMAND ----------

# MAGIC %md ## スコアリング: バッチ推論

# COMMAND ----------

# MAGIC %md 
# MAGIC 別のデータサイエンティストがこのモデルを別のデータバッチに適用するものとします。

# COMMAND ----------

raw_new_taxi_data = spark.read.table(
    f"{db_name}.nyc_yellow_taxi_with_zips"
)

new_taxi_data = rounded_taxi_data(raw_new_taxi_data)

# COMMAND ----------

# MAGIC %md 
# MAGIC 推論するデータを表示します。推論対象である`fare_amount`カラムを強調するためにカラムの並び替えを行います。

# COMMAND ----------

cols = ['fare_amount', 'trip_distance', 'pickup_zip', 'dropoff_zip', 'rounded_pickup_datetime', 'rounded_dropoff_datetime']
new_taxi_data_reordered = new_taxi_data.select(cols)
display(new_taxi_data_reordered)

# COMMAND ----------

# MAGIC %md
# MAGIC このバッチデータに対して推論を行うために、Feature Storeから必要な特徴量を収集するために`score_batch` APIを呼び出します。

# COMMAND ----------

# モデルのURIの取得
latest_model_version = get_latest_model_version(f"{db_name}_taxi_example_fare_packaged")
model_uri = f"models:/{db_name}_taxi_example_fare_packaged/{latest_model_version}"

# モデルによる推論を行うためにscore_batchを呼び出します
with_predictions = fs.score_batch(model_uri, new_taxi_data)

# COMMAND ----------

# MAGIC %md <img src="https://docs.databricks.com/_static/images/machine-learning/feature-store/taxi_example_score_batch.png"/>

# COMMAND ----------

# MAGIC %md ### タクシー料金予測結果の参照
# MAGIC 
# MAGIC このコードはタクシー料金の予測値が最初に来るように列の並び替えを行なっています。モデルの精度を高めるにはさらなるデータと特徴量エンジニアリングが必要かもしれませんが、`predicted_fare_amount`が実際の`fare_amount`に近い傾向を示していることに注意してください。

# COMMAND ----------

import pyspark.sql.functions as func

cols = ['prediction', 'fare_amount', 'trip_distance', 'pickup_zip', 'dropoff_zip', 
        'rounded_pickup_datetime', 'rounded_dropoff_datetime', 'mean_fare_window_1h_pickup_zip', 
        'count_trips_window_1h_pickup_zip', 'count_trips_window_30m_dropoff_zip', 'dropoff_is_weekend']

with_predictions_reordered = (
    with_predictions.select(
        cols,
    )
    .withColumnRenamed(
        "prediction",
        "predicted_fare_amount",
    )
    .withColumn(
      "predicted_fare_amount",
      func.round("predicted_fare_amount", 2),
    )
)

display(with_predictions_reordered)

# COMMAND ----------

# MAGIC %md ## 次のステップ
# MAGIC 
# MAGIC 1. このサンプルで作成された特徴量テーブルを<a href="#feature-store">Feature Store UI</a>で特徴量テーブルを調査する
# MAGIC 1. このノートブックにご自身のデータを適用して特徴量テーブルを作成する
