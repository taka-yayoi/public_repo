# Databricks notebook source
# MAGIC %md 
# MAGIC # Databricksでディープラーニングを始めてみる：TensorFlow Keras、Hyperopt、MLflowを用いたエンドツーエンドのサンプル
# MAGIC 
# MAGIC [DatabricksでTensorFlow Kerasを使ってみる](https://qiita.com/taka_yayoi/items/0f28fd854cbe2f5728e4)
# MAGIC 
# MAGIC このチュートリアルでは、Databricksにおけるディープラーニング開発のために、どのようにTensorFlow Keras、Hyperopt、MLflowを使うのかを説明するために小規模データセットを使用します。
# MAGIC 
# MAGIC 以下のステップが含まれます。
# MAGIC 
# MAGIC - データのロード、準備
# MAGIC - Part 1. TensorFlow Kerasによるニューラルネットワークモデルを作成し、インラインのTensorBoardでトレーニングを参照
# MAGIC - Part 2. HyperoptとMLflowを用いて自動ハイパーパラメーターチューニングを実行し、オートロギングを用いて結果を保存
# MAGIC - Part 3. 最終的なモデルを構築するために最適なハイパーパラメータのセットを使用
# MAGIC - Part 4. MLflowでモデルを登録し、予測を行うためにモデルを使用
# MAGIC 
# MAGIC ### セットアップ
# MAGIC - Databricks機械学習ランタイム7.0以降が必要です。このノートブックではニューラルネットワークのトレーニング結果を表示するためにTensorBoardを使用します。お使いのDatabricksランタイムのバージョンに応じて、TensorBoardの使用法が異なります。

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import mlflow
import mlflow.keras
import mlflow.tensorflow

# COMMAND ----------

# MAGIC %md ## データのロード、前処理
# MAGIC 
# MAGIC このサンプルでは、`scikit-learn`から提供されるCalifornia Housingデータセットを使用します。

# COMMAND ----------

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

cal_housing = fetch_california_housing()

# 80/20でトレーニングセット、テストデータセットに分割
X_train, X_test, y_train, y_test = train_test_split(cal_housing.data,
                                                    cal_housing.target,
                                                    test_size=0.2)

# COMMAND ----------

# MAGIC %md ### 特徴量のスケーリング
# MAGIC 
# MAGIC ニューラルネットワークを使用する際には特徴量のスケーリングが重要になります。このノートブックでは、`scikit-learn`の`StandardScaler`関数を使用します。

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# COMMAND ----------

# MAGIC %md ## Part 1. モデルの作成、ノートブック上のTensorBoardでの参照

# COMMAND ----------

# MAGIC %md ### ニューラルネットワークの構築

# COMMAND ----------

def create_model():
  model = Sequential()
  model.add(Dense(20, input_dim=8, activation="relu"))
  model.add(Dense(20, activation="relu"))
  model.add(Dense(1, activation="linear"))
  return model

# COMMAND ----------

# MAGIC %md ### モデルのコンパイル

# COMMAND ----------

model = create_model()

model.compile(loss="mse",
              optimizer="Adam",
              metrics=["mse"])

# COMMAND ----------

# MAGIC %md ### コールバックの作成

# COMMAND ----------

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# 以下の行では、<username>をご自身のユーザー名で置き換えてください。
experiment_log_dir = "/dbfs/<username>/tb"
checkpoint_path = "/dbfs/<username>/keras_checkpoint_weights.ckpt"

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=experiment_log_dir)
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor="loss", mode="min", patience=3)

history = model.fit(X_train, y_train, validation_split=.2, epochs=35, callbacks=[tensorboard_callback, model_checkpoint, early_stopping])

# COMMAND ----------

# MAGIC %md ### Databricks機械学習ランタイム7.2以降でのTensorBoardコマンド
# MAGIC 
# MAGIC 以下のコマンドでTensorBoardを起動すると、ノートブックをクラスターからデタッチするまで実行し続けます。
# MAGIC 
# MAGIC **注意**： TensorBoardの実行中にTensorBoardをクリアしたい場合には右のコマンドを実行してください。`dbutils.fs.rm(experiment_log_dir.replace("/dbfs",""), recurse=True)`

# COMMAND ----------

# MAGIC %load_ext tensorboard
# MAGIC %tensorboard --logdir $experiment_log_dir

# COMMAND ----------

# MAGIC %md ### Databricks機械学習ランタイム7.1以前でのTensorBoardコマンド
# MAGIC 
# MAGIC 以下のセルのコマンドを実行するとリンクが表示されます。リンクをクリックすると新規タブでTensorBoardが開きます。
# MAGIC 
# MAGIC この方法でTensorBoardを起動すると、`dbutils.tensorboard.stop()`で停止するか、クラスターを停止するまで実行し続けます。

# COMMAND ----------

#dbutils.tensorboard.start(experiment_log_dir)

# COMMAND ----------

# MAGIC %md ### テストデータセットを用いたモデルの評価

# COMMAND ----------

model.evaluate(X_test, y_test)

# COMMAND ----------

# MAGIC %md ## Part 2. HyperoptとMLflowを用いたハイパーパラメーターチューニング
# MAGIC 
# MAGIC [Hyperopt](https://github.com/hyperopt/hyperopt)はハイパーパラメーターチューニングのためのPythonライブラリです。Databricks機械学習ランタイムには、最適化、エンハンスされたバージョンのHyperoptと自動化されたMLflowトラッキングが含まれています。Hyperoptの使用法の詳細に関しては、[Hyperopt documentation](https://github.com/hyperopt/hyperopt/wiki/FMin)を参照ください。

# COMMAND ----------

# MAGIC %md ### 隠しレイヤーのノード数の変数を含むニューラルネットワークの構築

# COMMAND ----------

def create_model(n):
  model = Sequential()
  model.add(Dense(int(n["dense_l1"]), input_dim=8, activation="relu"))
  model.add(Dense(int(n["dense_l2"]), activation="relu"))
  model.add(Dense(1, activation="linear"))
  return model

# COMMAND ----------

# MAGIC %md ### Hyperoptの目的関数の作成

# COMMAND ----------

from hyperopt import fmin, hp, tpe, STATUS_OK, SparkTrials

def runNN(n):
  # tensorflowのインポート 
  import tensorflow as tf
  
  # mlflow.tensorflow.autolog()によるラン情報の記録
  mlflow.tensorflow.autolog()
  
  model = create_model(n)

  # optimizerの選択
  optimizer_call = getattr(tf.keras.optimizers, n["optimizer"])
  optimizer = optimizer_call(learning_rate=n["learning_rate"])
 
  # モデルのコンパイル
  model.compile(loss="mse",
                optimizer=optimizer,
                metrics=["mse"])

  history = model.fit(X_train, y_train, validation_split=.2, epochs=10, verbose=2)

  # モデルの評価
  score = model.evaluate(X_test, y_test, verbose=0)
  obj_metric = score[0]  
  return {"loss": obj_metric, "status": STATUS_OK}

# COMMAND ----------

# MAGIC %md ### Hyperopt探索空間の定義

# COMMAND ----------

space = {
  "dense_l1": hp.quniform("dense_l1", 10, 30, 1),
  "dense_l2": hp.quniform("dense_l2", 10, 30, 1),
  "learning_rate": hp.loguniform("learning_rate", -5, 0),
  "optimizer": hp.choice("optimizer", ["Adadelta", "Adam"])
 }

# COMMAND ----------

# MAGIC %md ### `SparkTrials`オブジェクトの作成
# MAGIC 
# MAGIC `SparkTrials`オブジェクトは、Sparkクラスターにおけるチューニングジョブを分散するために、`fmin()`に通知を行います。`SparkTrials`オブジェクトを作成すると、同時実行で評価を行うトライアルの最大数を設定するために、引数`parallelism`を使用することができます。デフォルトの設定値は、利用可能なSparkのエグゼキューターの数となります。
# MAGIC 
# MAGIC 大きな値を設定することで、より多くのハイパーパラメーターの設定のテストをスケールアウトさせることができます。Hyperoptは過去の結果に基づいて新たなトライアルを処理するので、ここには並列度と適応性のトレードオフが存在します。固定の`max_evals`においては、並列度を増やせば計算を高速化することができますが、並列度を減らすことでそれぞれのイテレーションがより多くの過去の結果にアクセスできることで、より優れた結果が得られる場合があります。

# COMMAND ----------

# 引数parallelismを指定しない場合、デフォルト値は利用できるSparkのエグゼキューターの数となります 
spark_trials = SparkTrials()

# COMMAND ----------

# MAGIC %md ### ハイパーパラメーターチューニングの実行
# MAGIC 
# MAGIC MLflowに結果を保存するために、MLflowのランの中に`fmin()`の呼び出しを配置します。MLflwoはそれぞれのランのパラメーターとパフォーマンスメトリクスをトラッキングします。
# MAGIC 
# MAGIC 以下のセルの実行後に、MLflowで結果を参照することができます。右上の**Experiment**をクリックすることで、エクスペリメントのランのサイドバーが表示されます。MLflowのランのテーブルを表示するためには、**Experiment Runs**の右のにあるアイコン![](https://docs.databricks.com/_images/external-link.png)をクリックします。
# MAGIC 
# MAGIC MLflowを使用してどのようにランを分析するのかに関しては、右のドキュメントを参照ください。([AWS](https://qiita.com/taka_yayoi/items/1a4e82f7e20c56ba4f72)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/)|[GCP](https://docs.gcp.databricks.com/applications/mlflow/index.html))

# COMMAND ----------

with mlflow.start_run():
  best_hyperparam = fmin(fn=runNN, 
                         space=space, 
                         algo=tpe.suggest, 
                         max_evals=30, 
                         trials=spark_trials)

# COMMAND ----------

# MAGIC %md ## Part 3. 最終的なモデルを構築するために最適なハイパーパラメーターのセットを使用

# COMMAND ----------

import hyperopt

print(hyperopt.space_eval(space, best_hyperparam))

# COMMAND ----------

first_layer = hyperopt.space_eval(space, best_hyperparam)["dense_l1"]
second_layer = hyperopt.space_eval(space, best_hyperparam)["dense_l2"]
learning_rate = hyperopt.space_eval(space, best_hyperparam)["learning_rate"]
optimizer = hyperopt.space_eval(space, best_hyperparam)["optimizer"]

# COMMAND ----------

# optimizerを取得しlearning_rateの値を更新します
optimizer_call = getattr(tf.keras.optimizers, optimizer)
optimizer = optimizer_call(learning_rate=learning_rate)

# COMMAND ----------

def create_new_model():
  model = Sequential()
  model.add(Dense(first_layer, input_dim=8, activation="relu"))
  model.add(Dense(second_layer, activation="relu"))
  model.add(Dense(1, activation="linear"))
  return model

# COMMAND ----------

new_model = create_new_model()
  
new_model.compile(loss="mse",
                optimizer=optimizer,
                metrics=["mse"])

# COMMAND ----------

# MAGIC %md 
# MAGIC `autolog()`がアクティブの場合、MLflowは自動でランを終了しません。新規のランのオートロギングを開始する前に、Cmd 30でスタートしたランを終了させる必要があります。
# MAGIC 
# MAGIC 詳細に関しては[こちら](https://www.mlflow.org/docs/latest/tracking.html#automatic-logging)を参照ください。

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

import matplotlib.pyplot as plt

mlflow.tensorflow.autolog()

with mlflow.start_run() as run:
  
  history = new_model.fit(X_train, y_train, epochs=35, callbacks=[early_stopping])
  
  # 後でモデルを登録するためにランの情報を保存します
  kerasURI = run.info.artifact_uri
  kerasRunId = run.info.run_id
  
  # テストデータセットでモデルを評価し、結果を記録します
  mlflow.log_param("eval_result", new_model.evaluate(X_test, y_test)[0])
  
  # モデルをクイックに可視化でチェックするために予測値と既知の値を対比するプロットを行い、プロットをアーティファクトとして記録します
  keras_pred = new_model.predict(X_test)
  plt.plot(y_test, keras_pred, "o", markersize=2)
  plt.xlabel("observed value")
  plt.ylabel("predicted value")
  plt.savefig("kplot.png")
  mlflow.log_artifact("kplot.png") 

# COMMAND ----------

# MAGIC %md ## Part 4. MLflowにモデルを登録し、予測にモデルを使用
# MAGIC 
# MAGIC モデルレジストリの詳細に関しては、([AWS](https://qiita.com/taka_yayoi/items/e7a4bec6420eb7069995)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/model-registry)|[GCP](https://docs.gcp.databricks.com/applications/mlflow/model-registry.html))を参照ください。

# COMMAND ----------

import time

# モデルレジストリ上の登録モデル名を指定してください
model_name = "cal_housing_keras_taka"
model_uri = f"runs:/{kerasRunId}/model"
new_model_version = mlflow.register_model(model_uri, model_name)

# モデル登録には数秒要しますので、次のセルを続ける前に待ち処理を追加しています
time.sleep(5)

# COMMAND ----------

# MAGIC %md ### 推論のためにモデルをロードし、予測を実施

# COMMAND ----------

keras_model = mlflow.keras.load_model(f"models:/{model_name}/{new_model_version.version}")

keras_pred = keras_model.predict(X_test)
keras_pred

# COMMAND ----------

# MAGIC %md ## クリーンアップ
# MAGIC 
# MAGIC TensorBoardを停止するには:
# MAGIC - Databricks機械学習ランタイム7.1以下を使用しているのであれば、以下のセルのコメントを解除して実行してください。
# MAGIC - Databricks機械学習ランタイム7.2以降を使用しているのであれば、このノートブックをクラスターからデタッチしてください。

# COMMAND ----------

#dbutils.tensorboard.stop()
