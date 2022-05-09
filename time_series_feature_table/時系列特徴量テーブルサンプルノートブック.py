# Databricks notebook source
# MAGIC %md # Feature Storeにおける時系列特徴量テーブル
# MAGIC 
# MAGIC このノートブックでは、シミュレートされたInternet of Things (IoT)センサーデータに基づく時系列特徴量テーブルを作成します。その後に以下のことを行います:
# MAGIC - 時系列特徴量テーブルに対するポイントインタイム検索を行うことでトレーニングセットを生成
# MAGIC - モデルトレーニングのためにトレーニングセットを使用
# MAGIC - モデルの登録
# MAGIC - 新たなセンサーデータに対するバッチ推論の実行
# MAGIC 
# MAGIC ## 要件
# MAGIC - Databricks機械学習ラインタイム10.3以降

# COMMAND ----------

# MAGIC %md ## バックグラウンド
# MAGIC 
# MAGIC このノートブックで使用するデータは、以下のシチュエーションを表現するためにシミュレートされています。
# MAGIC 
# MAGIC - 倉庫の異なる部屋に設置された一連のIoTセンサーから読み取ったデータがあります。
# MAGIC - このデータを用いて、誰かが部屋に入ったことを検知モデルをトレーニングしたいと考えています。
# MAGIC - それぞれの部屋には温度センサー、光センサー、CO2センサーがあり、異なる周期でデータが記録されています。

# COMMAND ----------

import uuid
run_id = str(uuid.uuid4()).replace('-', '')

database_name = f"point_in_time_demo_{run_id}"
model_name = f"pit_demo_model_{run_id}"

print(f"Database name: {database_name}")
print(f"Model name: {model_name}")

# データベースの作成
spark.sql(f"CREATE DATABASE IF NOT EXISTS {database_name}")

# COMMAND ----------

# MAGIC %md ## シミュレートデータの生成
# MAGIC 
# MAGIC このステップでは、シミュレートされたデータセットを生成し、光センサー、温度センサー、CO2センサー、正解データから構成される4つのSparkデータフレームを作成します。

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql.functions import *

wavelength_lo, wavelength_hi = 209.291, 213.111
ppm_lo, ppm_hi = 35, 623.99
temp_lo, temp_hi = 15.01, 25.99
humidity_lo, humidity_hi = 35.16, 43.07

def is_person_in_the_room(wavelength, ppm, temp, humidity):
  return (
    (wavelength < (wavelength_lo + (wavelength_hi - wavelength_lo) * .45)) &
    (ppm > (.9 * ppm_hi)) &
    (temp > ((temp_hi + temp_lo) / 2)) &
    (humidity > (humidity_hi * .6))
  )

def generate_dataset(start, end):
  def generate_sensor_df(features):
    return pd.DataFrame({
      'room': np.random.choice(3, end-start),
      'ts': start + np.random.choice(end-start, end-start, replace=False) + np.random.uniform(-0.99, 0.99, end-start),
      **features
    }).sort_values(by=['ts'])    
  
  wavelength_df = generate_sensor_df({
    'wavelength': np.random.normal(np.mean([wavelength_lo, wavelength_hi]), 2, end-start),
  })
  temp_df = generate_sensor_df({
    'temp': np.random.normal(np.mean([temp_lo, temp_hi]), 4, end-start),
    'humidity': np.random.normal(np.mean([humidity_lo, humidity_hi]), 2, end-start), 
  })
  
  ppm_bern = np.random.binomial(1, 0.3, end-start)
  ppm_normal_1 = np.random.normal(ppm_lo, 8, end-start)
  ppm_normal_2 = np.random.normal(ppm_hi, 3, end-start)
  ppm_df = generate_sensor_df({
    'ppm': ppm_bern*ppm_normal_1+(1-ppm_bern)*ppm_normal_2
  })
  
  df = pd.DataFrame({
    'room': np.random.choice(3, end-start),    
    'ts': np.random.uniform(start, end, end-start)
  }).sort_values(by=['ts'])
  for right_df in [wavelength_df, ppm_df, temp_df]:
    df = pd.merge_asof(
      df, 
      right_df, 
      on='ts', 
      by='room'
    )
  df['person'] = is_person_in_the_room(df['wavelength'], df['ppm'], df['temp'], df['humidity'])
  
  wavelength_df['wavelength'] += np.random.uniform(-1, 1, end-start) * 0.2
  ppm_df['ppm'] += np.random.uniform(-1, 1, end-start) * 2
  temp_df['temp'] += np.random.uniform(-1, 1, end-start) 
  temp_df['humidity'] += np.random.uniform(-1, 1, end-start)
  
  light_sensors = spark.createDataFrame(wavelength_df) \
    .withColumn("ts", col("ts").cast('timestamp')) \
    .select(col("room").alias("r"), col("ts").alias("light_ts"), col("wavelength"))
  temp_sensors = spark.createDataFrame(temp_df) \
    .withColumn("ts", col("ts").cast('timestamp')) \
    .select("room", "ts", "temp", "humidity")
  co2_sensors = spark.createDataFrame(ppm_df) \
    .withColumn("ts", col("ts").cast('timestamp')) \
    .select(col("room").alias("r"), col("ts").alias("co2_ts"), col("ppm"))
  ground_truth = spark.createDataFrame(df[['room', 'ts', 'person']]) \
    .withColumn("ts", col("ts").cast('timestamp'))  

  return temp_sensors, light_sensors, co2_sensors, ground_truth  

temp_sensors, light_sensors, co2_sensors, ground_truth = generate_dataset(1458031648, 1458089824)
fixed_temps = temp_sensors.select("room", "ts", "temp").sample(False, 0.01).withColumn("temp", temp_sensors.temp + 0.25)

# COMMAND ----------

# 生成したデータフレームの確認

# 部屋ごとの温度/湿度センサーの値
display(temp_sensors.limit(3))
# 部屋ごとの光センサーの値
display(light_sensors.limit(3))
# 部屋ごとのCO2センサーの値
display(co2_sensors.limit(3))
# 部屋に人がいる時に関する正解データ
display(ground_truth.limit(3))

# COMMAND ----------

# MAGIC %md ## 時系列特徴量テーブルの作成
# MAGIC 
# MAGIC このステップでは、時系列特徴量テーブルを作成します。それぞれのテーブルでは、部屋番号をプライマリーキーとします。

# COMMAND ----------

from databricks.feature_store.client import FeatureStoreClient
from databricks.feature_store.entities.feature_lookup import FeatureLookup

fs = FeatureStoreClient()

# COMMAND ----------

# 部屋番号をプライマリーキー、時刻をタイムスタンプキーとして使用して温度センサーに対する時系列特徴量テーブルを作成します
fs.create_table(
    f"{database_name}.temp_sensors",
    primary_keys=["room"],
    timestamp_keys=["ts"],
    df=temp_sensors,
    description="Readings from temperature and humidity sensors",
)

# 部屋番号をプライマリーキー、時刻をタイムスタンプキーとして使用して光センサーに対する時系列特徴量テーブルを作成します
fs.create_table(
    f"{database_name}.light_sensors",
    primary_keys=["r"],
    timestamp_keys=["light_ts"],
    df=light_sensors,
    description="Readings from light sensors",
)

# 部屋番号をプライマリーキー、時刻をタイムスタンプキーとして使用してCO2センサーに対する時系列特徴量テーブルを作成します 
fs.create_table(
    f"{database_name}.co2_sensors",
    primary_keys=["r"],
    timestamp_keys=["co2_ts"],
    df=co2_sensors,
    description="Readings from CO2 sensors",
)

# COMMAND ----------

# MAGIC %md 
# MAGIC これで、<a href="#feature-store" target="_blank">Feature StoreのUI</a>に時系列特徴量テーブルが表示されます。これらの特徴量テーブルでは`Timestamp Keys`フィールドが指定されています。

# COMMAND ----------

# MAGIC %md ## 時系列特徴量テーブルのの更新
# MAGIC 
# MAGIC 特徴量テーブルを作成した後には、更新された値を受け取る場合があります。例えば、いくつかの気温データが不適切に前処理されていたので、気温の時系列特徴量テーブルを更新する必要があるかもしれません。

# COMMAND ----------

display(fixed_temps.limit(3))

# COMMAND ----------

# MAGIC %md 
# MAGIC 時系列特徴量テーブルにデータフレームを書き込む際には、データフレームでは特徴量テーブルの全ての特徴量を指定しなくてはなりません。時系列特徴量テーブルの単一の特徴量カラムをアップデートするためには、最初にプライマリーキーとタイムスタンプキーを指定して、更新された特徴量カラムをテーブルの他のカラムとjoinしなくてはなりません。その後で、特徴量テーブルを更新することができます。

# COMMAND ----------

temp_ft = fs.read_table(f"{database_name}.temp_sensors").drop('temp')
temp_update_df = fixed_temps.join(temp_ft, ["room", "ts"])
fs.write_table(f"{database_name}.temp_sensors", temp_update_df, mode="merge")

# COMMAND ----------

# MAGIC %md ## 時系列特徴量テーブルにおけるポイントインタイム検索を用いたトレーニングセットの作成
# MAGIC 
# MAGIC このステップでは、時系列特徴量テーブルにおけるセンサーデータに対してポイントインタイム検索を行うことで、正解データを含むトレーニングセットを作成します。
# MAGIC 
# MAGIC ポイントインタイム検索は、正解データで指定される部屋番号に対応するタイムスタンプ時点での最新のセンサーデータを取得します。

# COMMAND ----------

training_labels, test_labels = ground_truth.randomSplit([0.75, 0.25])

display(training_labels.limit(5))

# COMMAND ----------

# トレーニングセットのための特徴量を定義するポイントインタイム検索を作成します。それぞれのポイントインタイム検索には`lookup_key`と`timestamp_lookup_key`が含まれる必要があります。
feature_lookups = [
    FeatureLookup(
        table_name=f"{database_name}.temp_sensors",
        feature_names=["temp", "humidity"],
        rename_outputs={
          "temp": "room_temperature",
          "humidity": "room_humidity"
        },
        lookup_key="room",
        timestamp_lookup_key="ts"
    ),
    FeatureLookup(
        table_name=f"{database_name}.light_sensors",
        feature_names=["wavelength"],
        rename_outputs={"wavelength": "room_light"},
        lookup_key="room",
        timestamp_lookup_key="ts",      
    ),
    FeatureLookup(
        table_name=f"{database_name}.co2_sensors",
        feature_names=["ppm"],
        rename_outputs={"ppm": "room_co2"},
        lookup_key="room",
        timestamp_lookup_key="ts",      
    ),  
]

training_set = fs.create_training_set(
    training_labels,
    feature_lookups=feature_lookups,
    exclude_columns=["room", "ts"],
    label="person",
)
training_df = training_set.load_df()

# COMMAND ----------

display(training_df.limit(5))

# COMMAND ----------

# MAGIC %md ## モデルのトレーニング

# COMMAND ----------

features_and_label = training_df.columns
training_data = training_df.toPandas()[features_and_label]

X_train = training_data.drop(["person"], axis=1)
y_train = training_data.person.astype(int)

import lightgbm as lgb
import mlflow.lightgbm
from mlflow.models.signature import infer_signature

mlflow.lightgbm.autolog()

model = lgb.train(
  {"num_leaves": 32, "objective": "binary"}, 
  lgb.Dataset(X_train, label=y_train.values),
  5
)

# COMMAND ----------

# モデルレジストリにモデルを登録します。
# `log_model`を使用する際、モデルには特徴量のメタデータもパッケージングされるので、推論時には自動で特徴量を検索します。
fs.log_model(
  model,
  artifact_path="model_packaged",
  flavor=mlflow.lightgbm,
  training_set=training_set,
  registered_model_name=model_name
)

# COMMAND ----------

# MAGIC %md ## 時系列特徴量テーブルにおけるポイントインタイム検索を用いたデータのスコアリング
# MAGIC 
# MAGIC トレーニングセットを作成するために指定されたポイントインタイム検索のメタデータがモデルにパッケージングされているので、スコアリングの際にも同じ検索処理を実行することができます。

# COMMAND ----------

from mlflow.tracking import MlflowClient
def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
      version_int = int(mv.version)
      if version_int > latest_version:
        latest_version = version_int
    return latest_version

# COMMAND ----------

scored = fs.score_batch(
  f"models:/{model_name}/{get_latest_model_version(model_name)}",
  test_labels,
  result_type="float",
)

# COMMAND ----------

from pyspark.sql.types import BooleanType

classify_udf = udf(lambda pred: pred > 0.5, BooleanType())
class_scored = scored.withColumn("person_prediction", classify_udf(scored.prediction))

display(class_scored.limit(5))

# COMMAND ----------

from pyspark.sql.functions import avg, round
display(class_scored.select(round(avg((class_scored.person_prediction == class_scored.person).cast("int")), 3).alias("accuracy")))

# COMMAND ----------

# MAGIC %md
# MAGIC # END
