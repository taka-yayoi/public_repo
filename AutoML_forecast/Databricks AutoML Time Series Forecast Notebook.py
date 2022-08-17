# Databricks notebook source
# MAGIC %md
# MAGIC # AutoMLによる予測サンプル
# MAGIC 
# MAGIC **注意**
# MAGIC ノートブックから時系列予測のAutoMLを行う際、フォルダ名、ノートブック名にマルチバイト文字(日本語)を含めないでください。エラーになります。
# MAGIC 
# MAGIC ## 要件
# MAGIC - Databricks Runtime for Machine Learning 10.0以降

# COMMAND ----------

# MAGIC %md
# MAGIC ## COVID-19データセット
# MAGIC 
# MAGIC USにおける日毎のCOVID-19の感染者数と追加の位置情報のレコードを含むデータセットです。ここでのゴールは、USにおいて向こう30日のウィルス感染者数の予測を行うことです。

# COMMAND ----------

import pyspark.pandas as ps
df = ps.read_csv("/databricks-datasets/COVID/covid-19-data")
df["date"] = ps.to_datetime(df['date'], errors='coerce')
df["cases"] = df["cases"].astype(int)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## AutoMLトレーニング
# MAGIC 
# MAGIC 以下のコマンドはAutoMLのランを起動します。予測すべきカラムを`target_col`引数と時間のカラムを指定する必要があります。ランが完了すると、トレーニングコードを検証するためにベストなトライアルのノートブックへのリンクにアクセスすることができます。
# MAGIC 
# MAGIC このサンプルでは以下の指定も行っています。
# MAGIC - `horizon=30`, AutoMLが未来の30日を予測するように指定 
# MAGIC - `frequency="d"`, 日毎の予測を行うことを指定 
# MAGIC - `primary_metric="mdape"`, トレーニングの際に最適化すべきメトリックを指定

# COMMAND ----------

import databricks.automl
import logging

# fbprophetの情報レベルのメッセージを無効化
logging.getLogger("py4j").setLevel(logging.WARNING)

# ハンズオンのため、timeout_minutesを5分にしています。通常は30分など十分な時間を指定するようにしてください
summary = databricks.automl.forecast(df, target_col="cases", time_col="date", horizon=30, frequency="d",  primary_metric="mdape", timeout_minutes=5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 次のステップ
# MAGIC 
# MAGIC * 上のノートブックとエクスペリメントを探索
# MAGIC * ベストなトライアルのノートブックのメトリックが好適であれば、次のセルに進むことができます。
# MAGIC * ベストトライアルによるモデルを改善したいのであれば、以下をトライします。
# MAGIC   * ベストトライアルのノートブックに移動し、クローンします。
# MAGIC   * モデルを改善するためにノートブックに必要な修正を加えます。
# MAGIC   * モデルに満足したら、トレーニングされたモデルが記録されているアーティファクトのURIをメモします。そのURIを次のセルの`model_uri`に指定します。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 予測のためにモデルを使用する
# MAGIC 
# MAGIC このセクションのコマンドはDatabricks機械学習ランタイム10.0以降で使用できます。

# COMMAND ----------

# MAGIC %md ### MLflowを用いてモデルをロードする
# MAGIC 
# MAGIC MLflowを用いることで、AutoMLの`trial_id`を用いてモデルを容易にPythonにインポートすることができます。

# COMMAND ----------

import mlflow.pyfunc
from mlflow.tracking import MlflowClient

run_id = MlflowClient()
trial_id = summary.best_trial.mlflow_run_id

model_uri = "runs:/{run_id}/model".format(run_id=trial_id)
pyfunc_model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

# MAGIC %md ### 予測を行うためのモデルを使う
# MAGIC 
# MAGIC 予測を行うために`predict_timeseries`のモデルメソッドを呼び出します。詳細は[Prophet documentation](https://facebook.github.io/prophet/docs/quick_start.html#python-api)をご覧ください。

# COMMAND ----------

forecast = pyfunc_model._model_impl.python_model.predict_timeseries()
display(forecast)

# COMMAND ----------

# MAGIC %md ### 予測の変化点とトレンドをプロット
# MAGIC 
# MAGIC 以下のプロットでは、太い黒線は時系列データセットを示しており、青い線がモデルによる予測値を示しています。

# COMMAND ----------

df_true = df.groupby("date").agg(y=("cases", "avg")).reset_index().to_pandas()

# COMMAND ----------

import matplotlib.pyplot as plt
 
fig = plt.figure(facecolor='w', figsize=(10, 6))
ax = fig.add_subplot(111)
forecasts = pyfunc_model._model_impl.python_model.predict_timeseries(include_history=True)
fcst_t = forecasts['ds'].dt.to_pydatetime()
ax.plot(df_true['date'].dt.to_pydatetime(), df_true['y'], 'k.', label='Observed data points')
ax.plot(fcst_t, forecasts['yhat'], ls='-', c='#0072B2', label='Forecasts')
ax.fill_between(fcst_t, forecasts['yhat_lower'], forecasts['yhat_upper'],
                color='#0072B2', alpha=0.2, label='Uncertainty interval')
ax.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # END
