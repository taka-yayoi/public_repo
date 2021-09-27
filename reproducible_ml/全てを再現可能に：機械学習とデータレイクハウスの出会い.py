# Databricks notebook source
# MAGIC %md
# MAGIC # 全てを再現可能に：機械学習とデータレイクハウスの出会い
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2021/07/10</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>クラスター</td><td>8.3ML</td></tr>
# MAGIC </table>
# MAGIC 
# MAGIC <img style="margin-top:25px;" src="https://sajpstorage.blob.core.windows.net/workshop20210205/databricks-logo-small-new.png" width="140">
# MAGIC 
# MAGIC **参考資料**
# MAGIC - [全てを再現可能に：機械学習とデータレイクハウスの出会い \- Qiita](https://qiita.com/taka_yayoi/items/cb1fafc96e7337d1fa58)

# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pyspark.sql.functions as F
from delta.tables import *
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## DBFSからワインデータを読み込み

# COMMAND ----------

# データの読み込み 
path = '/databricks-datasets/wine-quality/winequality-white.csv'
wine_df = (spark.read
           .option('header', 'true')
           .option('inferSchema', 'true')
           .option('sep', ';')
           .csv(path))
wine_df_clean = wine_df.select([F.col(col).alias(col.replace(' ', '_')) for col in wine_df.columns])
display(wine_df_clean)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ディレクトリの作成

# COMMAND ----------

# MAGIC %fs mkdirs /tmp/takaakiyayoidatabrickscom/reproducible_ml_blog

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deltaとしてデータを書き出し

# COMMAND ----------

# Deltaテーブルにデータを書き出し 
write_path = 'dbfs:/tmp/takaakiyayoidatabrickscom/reproducible_ml_blog/wine_quality_white.delta'
wine_df_clean.write.format('delta').mode('overwrite').save(write_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 新規行の追加

# COMMAND ----------

new_row = spark.createDataFrame([[7, 0.27, 0.36, 1.6, 0.045, 45, 170, 1.001, 3, 0.45, 8.8, 6]])
wine_df_extra_row = wine_df_clean.union(new_row)
display(wine_df_extra_row)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deltaテーブルの上書き、スキーマの更新

# COMMAND ----------

# Deltaの格納場所に上書き 
wine_df_extra_row.write.format('delta').mode('overwrite').option('overwriteSchema', 'true').save(write_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deltaのテーブル履歴

# COMMAND ----------

from delta.tables import *

deltaTable = DeltaTable.forPath(spark, write_path)
fullHistoryDF = deltaTable.history()    # バージョンを選択するためにテーブルの完全な履歴を取得

display(fullHistoryDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## トレーニングのためのデータのバージョンを指定

# COMMAND ----------

# モデルトレーニングのためのデータバージョンを指定
version = 1 
wine_df_delta = spark.read.format('delta').option('versionAsOf', version).load(write_path).toPandas()
display(wine_df_delta)

# COMMAND ----------

# MAGIC %md
# MAGIC ## データの分割

# COMMAND ----------

# データをトレーニングデータセットとテストデータセットに分割 (0.75, 0.25)の割合で分割
seed = 1111
train, test = train_test_split(wine_df_delta, train_size=0.75, random_state=seed)

# スカラー[3, 9]の値を取る目標変数"quality"カラム 
X_train = train.drop(['quality'], axis=1)
X_test = test.drop(['quality'], axis=1)
y_train = train[['quality']]
y_test = test[['quality']]

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflowを用いてモデルを構築

# COMMAND ----------

with mlflow.start_run() as run:
  # パラメーターのロギング
  n_estimators = 1000
  max_features = 'sqrt'
  params = {'data_version': version,
           'n_estimators': n_estimators,
           'max_features': max_features}
  mlflow.log_params(params)
  # モデルのトレーニング 
  rf = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, random_state=seed)
  rf.fit(X_train, y_train)

  # テストデータを用いた予測
  preds = rf.predict(X_test)

  # メトリクスの生成 
  rmse = np.sqrt(mean_squared_error(y_test, preds))
  mae = mean_absolute_error(y_test, preds)
  r2 = r2_score(y_test, preds)
  metrics = {'rmse': rmse,
             'mae': mae,
             'r2' : r2}

  # メトリクスのロギング
  mlflow.log_metrics(metrics)

  # モデルのロギング 
  mlflow.sklearn.log_model(rf, 'model')  

# COMMAND ----------

# MAGIC %md
# MAGIC # END
