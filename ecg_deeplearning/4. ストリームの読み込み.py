# Databricks notebook source
# MAGIC %md
# MAGIC # ストリーミング波形データの読み込み
# MAGIC 
# MAGIC `read`を`readStream`に変更するだけで、Delta Lake Tableをライブで更新されるストリーミングとして読み込むことができます。
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

# 前のノートブック「3. データのストリーミング」とパスを揃えてください
stream_path = '/tmp/takaaki.yayoi@databricks.com/hls/ecg/streaming/'

df = spark.readStream.format('delta').load(stream_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 到着データの確認
# MAGIC 
# MAGIC 従来のデータフレームと同様にこのデータフレームを操作することができます。テーブルに対するクエリーの結果は更新され続けます。これは[`display`関数](https://docs.databricks.com/user-guide/visualizations/index.html#display-function)にも適用されます。データフレームをテーブルとして参照するたびに新たなレコードが表示されます。

# COMMAND ----------

display(df.drop('signals'))

# COMMAND ----------

# MAGIC %md
# MAGIC しかし、この機能はグラフにしたときにより興味深いものになります。ここでは、これまでに処理したレコード数をプロットしています。このグラフはライブで更新され、レコード数が一定の割合で増加する様子を確認できます。

# COMMAND ----------

display(df.groupBy(df.time_interval).count())

# COMMAND ----------

# MAGIC %md
# MAGIC これは全ての`display`のチャートタイプに適用されます。特に興味深いには、入院理由に基づくパイチャートです。患者が到着するたびに、分布が変化する様子を見て取れます。

# COMMAND ----------

display(df.groupBy(df.comments["Reason for admission"]).count())

# COMMAND ----------

# MAGIC %md
# MAGIC 患者の年齢に関しても同様に動作します。

# COMMAND ----------

from pyspark.sql.types import IntegerType
from pyspark.sql.functions import trim

display(df.select(trim(df.comments["age"]).cast(IntegerType()).alias('age')))

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflowとストリーミングを組み合わせる
# MAGIC 
# MAGIC MLflowと構造化ストリーミングを用いることで、データの到着に合わせてモデルを適用でき、リアルタイムでのレポートを実現できます。まず最初に、トラッキングされたKerasモデルをMLflowのランからロードするために、[mlflow.keras](https://www.mlflow.org/docs/latest/python_api/mlflow.keras.html)ライブラリを使用します。以下のセルでは、使用したいモデルのランIDで更新する必要があります。

# COMMAND ----------

import mlflow.keras

run_id = "5060b23ff2fd4faa985c547d65042776" # ご自身のMLFlowランIDで更新してください
model_uri = "runs:/" + run_id + "/model"
model = mlflow.keras.load_model(model_uri=model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC 新たなデータの到着に合わせてスコアリングを行う簡単な方法は、UDFからKerasモデルを呼び出すというものです。以下のセルでは、それぞれの行がKerasモデルの入力に合致するように変換を行い、データをKerasっモデルに渡し、推論結果を返却するUDFを定義しています。効率を最大にするために、Spark、Python間の効率的な中間メモリである[Apache Arrow](https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html)を利用できるように、UDFを[Pandas UDF](https://docs.databricks.com/spark/latest/spark-sql/udf-python-pandas.html)として定義しています。
# MAGIC 
# MAGIC 以下のセルで定義している`ModelWrapperPickable`はpysparkのUDF作成の際のエラーを回避するためのものです。
# MAGIC 
# MAGIC **参考資料**
# MAGIC - [apache spark \- Using tensorflow\.keras model in pyspark UDF generates a pickle error \- Stack Overflow](https://stackoverflow.com/questions/61096573/using-tensorflow-keras-model-in-pyspark-udf-generates-a-pickle-error)
# MAGIC - [Pickling Keras Models](http://zachmoshe.com/2017/04/03/pickling-keras-models.html)

# COMMAND ----------

class ModelWrapperPickable:

  def __init__(self, model):
    self.model = model

  def __getstate__(self):
    import tempfile
    import tensorflow
    
    model_str = ''
    with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
      tensorflow.keras.models.save_model(self.model, fd.name, overwrite=True)
      model_str = fd.read()
      d = { 'model_str': model_str }
      return d

  def __setstate__(self, state):
    import tempfile
    import tensorflow
    
    with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
      fd.write(state['model_str'])
      fd.flush()
      self.model = tensorflow.keras.models.load_model(fd.name)

# COMMAND ----------

model_wrapper = ModelWrapperPickable(model)

# COMMAND ----------

import numpy as np
import pandas as pd
import os
import mlflow.keras

def predict_using_model(signals_array):
    """
    1. Loads the ECG data from the records specified in df_data
    2. Divide the signal data in windows of size window_size (default of 2048 which is enough to capture 3 heart beats.)
    
    returns:
        dataX: contains windowed ecg data (shape = n_windwows, n_channels, window_size)
        dataY: containts label for each window
        record_list: If required also returns a list specifying the record name for each window, else is empty list.
    """
    
    window_size = 2048
    n_channels = 15
    n_windows = 0
    
    preds = []
    
    for signals in signals_array:
    
      n_windows = len(signals[0]) // window_size

      dataX = np.zeros((n_windows, n_channels, window_size))
    
      record_list = []
    
      # レコードの読み込み、シグナルデータの取得および転置
      l = signals.tolist()
      signal_data = np.array(l)
      n_rows = len(list(signal_data[0]))
      n_windows = n_rows // window_size
      dataX[0:n_windows] = np.array([signal_data[:,i*window_size:(i+1)*window_size] for i in range(n_windows)])
    
      predictions = model_wrapper.model.predict(dataX)
    
      class0 = 0
      class1 = 1
    
      for x in predictions:
        if x[0] > x[1]:
          class0 += 1
        else:
          class1 += 1
        
      preds.append(0 if class0 > class1 else 1)
    
    return pd.Series(preds)

# COMMAND ----------

# MAGIC %md
# MAGIC 関数を使う前には、UDFを登録する必要があります。

# COMMAND ----------

from pyspark.sql.types import ArrayType, IntegerType
from pyspark.sql import functions as F

predict_pudf = F.pandas_udf(predict_using_model, IntegerType())

# COMMAND ----------

predict_pudf(F.map_values(df.signals))

# COMMAND ----------

# MAGIC %md
# MAGIC これで、ライブストリーミングのデータセットに対して継続的にモデルを適用して、患者が心臓疾患を持っているかどうかを予測できるようになりました！
# MAGIC 
# MAGIC 0が(健康)、1が(疾患)となります。

# COMMAND ----------

display(df.select(predict_pudf(F.map_values(df.signals)).alias('prediction')))

# COMMAND ----------

# MAGIC %md
# MAGIC # END
