# Databricks notebook source
# MAGIC %md # IoTデバイスに対する機械学習モデルの管理およびスケーラビリティの確保
# MAGIC 
# MAGIC このノートブックでは、pandas UDF(ユーザー定義関数)を用いて、シングルマシンの機械学習ソリューションをどのようにスケールさせるのかをデモンストレーションします。
# MAGIC 
# MAGIC **要件：** Databricksランタイム5.5以降

# COMMAND ----------

# MAGIC %md 
# MAGIC ## ダミーデータの作成
# MAGIC 
# MAGIC - `device_id`: 10個のデバイス
# MAGIC - `record_id`: 10kのユニークなレコード
# MAGIC - `feature_1`: モデルトレーニングのための特徴量
# MAGIC - `feature_2`: モデルトレーニングのための特徴量
# MAGIC - `feature_3`: モデルトレーニングのための特徴量
# MAGIC - `label`: 予測する変数

# COMMAND ----------

import pyspark.sql.functions as f

df = (spark.range(1000*1000)
  .select(f.col("id").alias("record_id"), (f.col("id")%10).alias("device_id"))
  .withColumn("feature_1", f.rand() * 1)
  .withColumn("feature_2", f.rand() * 2)
  .withColumn("feature_3", f.rand() * 3)
  .withColumn("label", (f.col("feature_1") + f.col("feature_2") + f.col("feature_3")) + f.rand())
)

display(df)

# COMMAND ----------

# MAGIC %md 戻り値のスキーマの定義

# COMMAND ----------

import pyspark.sql.types as t

trainReturnSchema = t.StructType([
  t.StructField('device_id', t.IntegerType()), # ユニークなデバイスID
  t.StructField('n_used', t.IntegerType()),    # トレーニングで使用するレコード数
  t.StructField('model_path', t.StringType()), # 特定のデバイスに対応するモデルのパス
  t.StructField('mse', t.FloatType())          # モデルパフォーマンスを示すメトリック
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## トレーニング用UDFの定義
# MAGIC 
# MAGIC 特定のデバイスの全データを受け取り、トレーニングを行い、ネストされたランとして保存し、上述のスキーマのSparkオブジェクトを返却するpandas UDFを定義します。

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

@f.pandas_udf(trainReturnSchema, functionType=f.PandasUDFType.GROUPED_MAP)
def train_model(df_pandas):
  '''
  Trains an sklearn model on grouped instances
  '''
  # メタデータの抽出
  device_id = df_pandas['device_id'].iloc[0]
  n_used = df_pandas.shape[0]
  run_id = df_pandas['run_id'].iloc[0] # ネストされたランにするためのランIDを抽出
  
  # モデルのトレーニング
  X = df_pandas[['feature_1', 'feature_2', 'feature_3']]
  y = df_pandas['label']
  rf = RandomForestRegressor()
  rf.fit(X, y)

  # モデルの評価
  predictions = rf.predict(X)
  mse = mean_squared_error(y, predictions) # トレーニング/テストのスプリットを追加できることに注意してください
 
  # トップレベルのトレーニングの再開
  with mlflow.start_run(run_id=run_id):
    # 特定デバイスに対応するネストされたランの作成
    with mlflow.start_run(run_name=str(device_id), nested=True) as run:
      mlflow.sklearn.log_model(rf, str(device_id))
      mlflow.log_metric("mse", mse)
      
      artifact_uri = f"runs:/{run.info.run_id}/{device_id}"
      # 上述のスキーマに合致する返却用pandasデータフレームの作成
      returnDF = pd.DataFrame([[device_id, n_used, artifact_uri, mse]], 
        columns=["device_id", "n_used", "model_path", "mse"])

  return returnDF 


# COMMAND ----------

# MAGIC %md 
# MAGIC ## グルーピングされたデータに対するpandas UDFの適用

# COMMAND ----------

with mlflow.start_run(run_name="Training session for all devices") as run:
  run_id = run.info.run_uuid
  
  modelDirectoriesDF = (df
    .withColumn("run_id", f.lit(run_id)) # run_idの追加
    .groupby("device_id")
    .apply(train_model)
  )
  
combinedDF = (df
  .join(modelDirectoriesDF, on="device_id", how="left")
)

display(combinedDF)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## スコアリングUDFの定義
# MAGIC 
# MAGIC モデルを適用するpandas UDFを定義します。
# MAGIC 
# MAGIC *デバイスごとに一度のDBFS読み込みが必要となります*

# COMMAND ----------

applyReturnSchema = t.StructType([
  t.StructField('record_id', t.IntegerType()),
  t.StructField('prediction', t.FloatType())
])

@f.pandas_udf(applyReturnSchema, functionType=f.PandasUDFType.GROUPED_MAP)
def apply_model(df_pandas):
  '''
  pandasデータフレームで表現される特定デバイスのデータにモデルを適用します
  '''
  model_path = df_pandas['model_path'].iloc[0]
  
  input_columns = ['feature_1', 'feature_2', 'feature_3']
  X = df_pandas[input_columns]
  
  model = mlflow.sklearn.load_model(model_path)
  prediction = model.predict(X)
  
  returnDF = pd.DataFrame({
    "record_id": df_pandas['record_id'],
    "prediction": prediction
  })
  return returnDF

predictionDF = combinedDF.groupby("device_id").apply(apply_model)
display(predictionDF)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
