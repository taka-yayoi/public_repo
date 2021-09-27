# Databricks notebook source
# MAGIC %md
# MAGIC # 波形データに対するLSTMのトレーニング、<img width="90px" src="https://mlflow.org/docs/0.7.0/_static/MLflow-logo-final-black.png">によるトラッキング
# MAGIC 
# MAGIC このノートブックでは、前回のノートブックで準備した波形データを用いてLSTM(Long short-term memory)をトレーニングします。Condaサポートがある[MLランタイム](https://databricks.com/product/databricks-runtime-for-machine-learning)を使用します。ベストのパフォーマンスを得るために、GPUクラスターを使いますが、このノートブックはCPUクラスターでも動作します。このノートブックで行われる分析は[Luc Niesのブログ記事](https://blog.orikami.nl/diagnosing-myocardial-infarction-using-long-short-term-memory-networks-lstms-cedf5770a257)から移植されたものです。
# MAGIC 
# MAGIC まず初めに、必要なライブラリをインポートします。
# MAGIC 
# MAGIC **注意**
# MAGIC トレーニングの際にアウトオブメモリーエラー(OOM)が発生する際にはクラスターのスペックを大きなものに変更してみてください。
# MAGIC 
# MAGIC **参考資料**
# MAGIC - [Delta Lake、Keras、MLflowを用いた機械学習による医療機器データのモニタリング \- Qiita](https://qiita.com/taka_yayoi/items/65e463a3eab84d4e2ce7)
# MAGIC - [Monitoring patient medical device data with ML \+ Delta Lake, Keras, and MLflow](https://databricks.com/blog/2019/09/12/monitor-medical-device-data-with-machine-learning-using-delta-lake-keras-and-mlflow-on-demand-webinar-and-faqs-now-available.html)
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2021/7/9</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>クラスター</td><td>8.3ML</td></tr>
# MAGIC </table>
# MAGIC <img style="margin-top:25px;" src="https://jixjiadatabricks.blob.core.windows.net/images/databricks-logo-small-new.png" width="140">

# COMMAND ----------

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import TensorBoard

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report

import math
import time

# COMMAND ----------

import mlflow
import mlflow.keras
import os

from datetime import datetime
from hyperopt import fmin, hp, tpe, STATUS_OK

print(f"MLflow Version: {mlflow.__version__}.")

# COMMAND ----------

# MAGIC %md
# MAGIC モデルのトレーニングを始める前に、少しデータセットの中身を見てみましょう。最初に、入院の理由の観点でデータセットにおける患者の分布を見てみます。多くの患者が心臓疾患であり、次いで健常者である事がわかります。
# MAGIC 
# MAGIC **注意** 以下の`/tmp/takaaki.yayoi@databricks.com...`のパスは前のノートブックと揃えてください。

# COMMAND ----------

ecg_df_raw = spark.read.format('delta').load('/tmp/takaaki.yayoi@databricks.com/hls/ecg/staged/')

ecg_df = ecg_df_raw.select(
  ecg_df_raw.patient_id,
  ecg_df_raw.comments["Reason for admission"].alias('label'),
  ecg_df_raw.signals)

display(ecg_df.groupBy(ecg_df.label).count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 健常対象群と心筋梗塞患者の可視化
# MAGIC 
# MAGIC 波形を可視化するために[display関数](https://docs.databricks.com/user-guide/visualizations/index.html#display-function)を使用します。まず、三つのシグナルコンポーネントを抽出して、配列における位置でタグづけを行うヘルパー関数を定義します。

# COMMAND ----------

from pyspark.sql.window import Window
import pyspark.sql.functions as F

def make_waveform_df(df):

  return (df.limit(1)
      .select(F.arrays_zip('signals.v1','signals.v2','signals.v3').alias('leads'))
      .select(F.explode('leads').alias('leads'))
      .select(F.col('leads.0').alias('v1'), F.col('leads.1').alias('v2'), F.col('leads.2').alias('v3'))
      .withColumn('index', F.row_number().over(Window.orderBy(F.monotonically_increasing_id()))))

# COMMAND ----------

# MAGIC %md
# MAGIC 次にこの関数から返却されるデータフレームをdisplay関数に渡してラインプロットを作成します。こちらが健常対象群の波形となります。

# COMMAND ----------

display(make_waveform_df(ecg_df.where(ecg_df.label == ' Healthy control')))

# COMMAND ----------

# MAGIC %md
# MAGIC そして、こちらが心臓疾患を持つ患者に対応する波形となり、上とは形状が大きく異なります。

# COMMAND ----------

display(make_waveform_df(ecg_df.where(ecg_df.label == ' Myocardial infarction')))

# COMMAND ----------

# MAGIC %md
# MAGIC ## トレーニングデータセット、テストデータセットの作成
# MAGIC 
# MAGIC モデルをトレーニングする前に、独立したトレーニングデータセットとテストデータセットを作成します。20%のデータをテストデータセットとしています。

# COMMAND ----------

selected_labels = [
    ' Healthy control',
    ' Myocardial infarction'
    ]
# ラベルに対するone-hotエンコーディングでラベルのマッピングを使用します
label_map = {label: value for label, value in zip(selected_labels, range(len(selected_labels)))}

test_patients = []
train_patients = []
test_size = 0.2

# 対象をランダムにトレーニングデータセットとテストデータセットに分割します 
for label in selected_labels:
    patients_df = ecg_df.where(ecg_df.label == (label)).dropDuplicates(['patient_id']).cache()
    test_patients_df = patients_df.sample(test_size)
    test_patients += test_patients_df.collect()
    train_patients += patients_df.join(test_patients_df, on='patient_id', how='left_anti').collect()
    patients_df.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC データセットを準備する過程で、ニューラルネットワークで処理できるように形状を変更する必要があります。

# COMMAND ----------

def make_set(data, label_map, record_id, window_size=2048, n_channels=15):
    """
    1. df_dataで指定されたレコードからECGデータをロード
    2. window_sizeのサイズのウィンドウでシグナルデータを分割 (デフォルトは3つの鼓動をキャプチャできるのに十分な2048)
    
    returns:
        dataX: ウィンドウ化されたECGデータ (shape = n_windwows, n_channels, window_size)
        dataY: それぞれのウィンドウに対するラベル
        record_list: 必要な場合にはそれぞれのウィンドウに対するレコード名を指定するリストを返却、そうでない場合には空のリスト
    """
    n_windows = 0
    
    for record in data:
        n_windows += len(record.signals['v6']) // window_size

    dataX = np.zeros((n_windows, n_channels, window_size))
    dataY = np.zeros((n_windows, len(label_map)))
    
    record_list = []
    
    nth_window = 0
    for record in data:
        # read the record, get the signal data and transpose it
        signal_data = list(record.signals.values())
        n_rows = len(signal_data[0])
        signal_data = np.array(signal_data)
        n_windows = n_rows // window_size
        dataX[nth_window:nth_window+n_windows] = np.array([signal_data[:,i*window_size:(i+1)*window_size] for i in range(n_windows)])
        dataY[nth_window:nth_window+n_windows][:, label_map[record.label]] = 1
        nth_window+=n_windows
        
        if record_id:
            record_list += n_windows * [record.patient_id]
        
    return dataX, dataY, record_list

# COMMAND ----------

# MAGIC %md
# MAGIC 上の関数をデータに適用してトレーニングデータセットとテストデータセットを作成します。

# COMMAND ----------

window_size = 2048
trainX, trainY, _ = make_set(train_patients, label_map, False, window_size, 15)
testX, testY, record_list = make_set(test_patients, label_map, True, window_size, 15)

num_classes = 2

train_x_path = "/dbfs/ml/hls/train_x.npy"
train_y_path = "/dbfs/ml/hls/train_y.npy"
val_x_path = "/dbfs/ml/hls/val_x.npy"
val_y_path = "/dbfs/ml/hls/val_y.npy"

weights_path = "/dbfs/ml/hls/weights.npy"

# COMMAND ----------

# MAGIC %md
# MAGIC データを書き出します。この際にはローカルファイルAPIを利用可能にする、Amazon S3、Azure Blob Storage上で稼働する高性能ファイルシステムである[Databricks ML FUSE mount](https://docs.databricks.com/applications/deep-learning/data-prep/ddl-storage.html)を使用します。

# COMMAND ----------

dbutils.fs.rm('/ml/hls/', True)
dbutils.fs.mkdirs('/ml/hls/')

# データのシャッフル
trainX, trainY = shuffle(trainX, trainY)
trainX = trainX.astype('float32') # トレーニングの高速化のためにデータボリュームを限定
trainY = trainY
np.save(train_x_path, trainX)
np.save(train_y_path, trainY)

testX, testY = shuffle(testX, testY)
testX = testX
testY = testY
np.save(val_x_path, testX)
np.save(val_y_path, testY)

record_list = record_list

# COMMAND ----------

# MAGIC %md
# MAGIC 大きなクラスの不均衡があるので、トレーニングにおける重みを調整します。

# COMMAND ----------

def get_weights(trainY):
  """
  それぞれのクラスの割合を取得して重みを返却
  """
  fractions = 1-trainY.sum(axis=0)/len(trainY)
  for i,a in enumerate(fractions):
    print(f'class {i} represents {a} of the data')
  weights = fractions[trainY.argmax(axis=1)]
  return weights

# COMMAND ----------

weights = get_weights(trainY)
np.save(weights_path, weights)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Long Short Term Memory (LSTM) Recurrent Neural Networkのトレーニング
# MAGIC 
# MAGIC <img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png" width="900" />
# MAGIC 
# MAGIC これでLSTMモデルをトレーニングできます。我々のアーキテクチャをチューニングするために、モデルで利用可能なハイパーパラメータを探索する[hyperopt](http://hyperopt.github.io/hyperopt)ライブラリを使用します。最初にモデルを定義します。トレーニングの過程で生成されるモデル、ハイパーパラメータ、検証結果をトラッキングするために[統合されているMLflow](https://docs.azuredatabricks.net/applications/mlflow/index.html)を使用します。

# COMMAND ----------

def make_model(input_shape, output_dim, dropout):
  print(f"model dim: {input_shape} {output_dim}")
  model = Sequential()
  model.add(LSTM(256, input_shape=input_shape, batch_size=None, return_sequences=True))
  model.add(Dropout(dropout))
  model.add(LSTM(128, return_sequences=True))
  model.add(Dropout(dropout))
  model.add(LSTM(64))
  model.add(Dropout(dropout))
  model.add(Dense(output_dim, activation='softmax'))
  return model

def select_sgd_algorithm(hpo):
  """
  オプティマイザと学習率に基づき最急降下法アルゴリズムを選択
  """
  optimizer_call = getattr(tensorflow.keras.optimizers, hpo['optimizer'])
  optimizer = optimizer_call(math.pow(10, hpo['learning_rate']))
  return optimizer

def compile_model(model, optimizer):
  model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                optimizer=optimizer,
                metrics=['accuracy'])

def fit_model(model, trainX, trainY, weights, epochs=50, batch_size=512, verbose=1):
  """
  kerasモデルのフィット、モデルトラッキングのためにhistoryオブジェクトを返却
  """
  history = model.fit(trainX, trainY, 
                      verbose=verbose,
                      epochs=epochs, 
                      batch_size=batch_size, 
                      sample_weight=weights
                      )
  return history

def log_mlflow_params(hpo):
  for key, value in hpo.items():
    mlflow.log_param(key, value)

def log_mlflow_runtime(start_train_time, complete_train_time, complete_eval_time):
  total_train_time = complete_train_time - start_train_time
  total_eval_time = complete_eval_time - complete_train_time
  mlflow.log_metric("train_duration", total_train_time)
  mlflow.log_metric("eval_duration", total_eval_time)

def get_summed_predictions(record_list, predictions, labels):
  """
  全ての予測の平均を取るために同じ対象のシーケンスをグルーピング
  ラベルはone-hotエンコードされており、[1,0]が健康、[0,1]が疾患
  アウトプットに対してargmaxを取り、ラベルとして0 (健康) or 1 (疾患)を取得
  予測結果のpandasデータフレームを返却
  """
  summed2 = pd.DataFrame({'record':record_list, 'predictions':output, 'label':[1 if x[0] == 0 else 0 for x in testY]})
  summed = summed2.groupby('record').mean()
  summed["label"] = summed['label'] > 0.5
  summed["predicted label"] = summed['predictions'] > 0.5
  return summed
  
def get_confusion_metrics(record_list, predictions, labels):
  """
  sklearnのclassification_reportを用いて混合メトリクスを取得
  """
  summed = get_summed_predictions(record_list, predictions, labels)
  confusion_metrics = classification_report(summed['label'], summed["predicted label"])
  return confusion_metrics

def log_history_mlflow(history):
  """
  model.fit()からhistoryオブジェクトを取得しmlflowにメトリクスをロギング
  """
  final_loss = history.history["loss"][-1]
  final_acc = history.history["accuracy"][-1]
  mlflow.log_metric("loss", final_loss)
  mlflow.log_metric("accuracy", final_acc)
  
def log_confusion_metrics_mlflow(confusion_metrics):
  """
  (それぞれのクラスに対する混合メトリクスを含む)分類レポートを取得
  全体的な精度、再現率、f1スコアをロギング
  (ハイパーパラメータチューニングにおける最大の)f1スコアを返却
  """
  classification_array = confusion_metrics.split()
  precision = float(classification_array[17])
  recall = float(classification_array[18])
  f1_score = float(classification_array[19])
  mlflow.log_metric("precision", precision)
  mlflow.log_metric("recall", recall)
  mlflow.log_metric("f1_score", f1_score)

def runCNN(hpo):
  """
  LSTM recurrent convolutional neural networkの実行、MLflowによるトラッキング
  """ 
  train_x = np.load(train_x_path)
  train_y = np.load(train_y_path)
  model = make_model((train_x.shape[1], train_x.shape[2]), train_y.shape[-1], hpo['dropout'])
  optimizer = select_sgd_algorithm(hpo)
  
  # MLflowトラッキング
  with mlflow.start_run() as run:
    log_mlflow_params(hpo)
    
    compile_model(model, optimizer)
    
    # モデルのフィッテイング、トレーニング時間の記録
    start_train_time = time.time()
    history = fit_model(model, train_x, train_y, np.load(weights_path), 1) # epoch=1
    #print(history.history)
    
    complete_train_time = time.time()

    # モデルの評価、評価時間の記録
    output = model.predict_classes(np.load(val_x_path))
    # confusion_metrics = get_confusion_metrics(record_list=record_list, predictions=output, labels=np.load(val_y_path))
    
    complete_eval_time = time.time()
    log_mlflow_runtime(start_train_time, complete_train_time, complete_eval_time)
    log_history_mlflow(history)
    # log_confusion_metrics_mlflow(confusion_metrics)
    mlflow.keras.log_model(model, "model")
    
    obj_metric = history.history["loss"][-1]
    mlflow.end_run()  
    return {'loss': obj_metric, 'status': STATUS_OK}


# COMMAND ----------

# MAGIC %md
# MAGIC 次にアーキテクチャの探索空間を定義し、hyperoptオプティマイザに渡します。これはシーケンシャルにそれぞれのモデルアーキテクチャを実行し、MLflowでロギングします。

# COMMAND ----------

space = {'dropout': hp.uniform('dropout', .1, .8),
         'learning_rate': hp.uniform('learning_rate', -10, 0),
         'optimizer': hp.choice('optimizer', ['Adadelta', 'Adam', 'RMSprop'])
        }

fmin(runCNN, space, algo=tpe.suggest, max_evals=12)

# COMMAND ----------

# MAGIC %md
# MAGIC 探索すべきアーキテクチャ、ハイパーパラメータの探索空間が大きい場合には、Databricksに統合されている[SparkTrials](https://docs.databricks.com/spark/latest/mllib/hyperopt-spark-mlflow-integration.html)フレームワークを用いて、ハイパーパラメータ探索をSparkクラスターに分散させることができます。ここでは、並列度を4に設定しており、2倍以上のトレーニングの高速化を実現しています。

# COMMAND ----------

def runCNN_for_trial(hpo):
  """
  LSTM recurrent convolutional neural networkの実行及びMLflowによるトラッキング
  """ 
  train_x = np.load(train_x_path)
  train_y = np.load(train_y_path)
  model = make_model((train_x.shape[1], train_x.shape[2]), train_y.shape[-1], hpo['dropout'])
  optimizer = select_sgd_algorithm(hpo)
  
  # MLflowによるトラッキング
  with mlflow.start_run() as run:
    #log_mlflow_params(hpo)
    #mlflow.log_param("param_from_worker", "test")
    
    # SparkTrialによるoptimizerのロギングと競合しないための処理
    for key, value in hpo.items():
      if key != "optimizer":
        mlflow.log_param(key, value)
      else:
        mlflow.log_param("optimizer_name", value)
    
    compile_model(model, optimizer)
    
    # モデルのフィッテイング、トレーニング時間の記録
    start_train_time = time.time()
    history = fit_model(model, train_x, train_y, np.load(weights_path), 1) # epoch=1
        
    complete_train_time = time.time()

    # モデルの評価、評価時間の記録
    output = model.predict_classes(np.load(val_x_path))
    # confusion_metrics = get_confusion_metrics(record_list=record_list, predictions=output, labels=np.load(val_y_path))
    
    complete_eval_time = time.time()
    log_mlflow_runtime(start_train_time, complete_train_time, complete_eval_time)
    log_history_mlflow(history)
    # log_confusion_metrics_mlflow(confusion_metrics)
    
    mlflow.keras.log_model(model, "model")
    #mlflow.pyfunc.log_model(model, "model")
    
    obj_metric = history.history["loss"][-1]
    mlflow.end_run()  
    return {'loss': obj_metric, 'status': STATUS_OK}

# COMMAND ----------

from hyperopt import SparkTrials

space = {'dropout': hp.uniform('dropout', .1, .8),
         'learning_rate': hp.uniform('learning_rate', -10, 0),
         'optimizer': hp.choice('optimizer', ['Adadelta', 'Adam', 'RMSprop'])
        }

fmin(runCNN_for_trial, space, algo=tpe.suggest, max_evals=12, trials=SparkTrials(parallelism=4))

# COMMAND ----------

# MAGIC %md
# MAGIC このケースでは、シングルノードによるデータ処理で事足りました。より大きなデータセットを取り扱う場合には、[HorovodRunner](https://docs.databricks.com/applications/deep-learning/distributed-training/horovod-runner.html#horovodrunner)を用いて複数のGPUマシンにまたがってモデルをトレーニングすることもできます。

# COMMAND ----------

# MAGIC %md
# MAGIC # END
