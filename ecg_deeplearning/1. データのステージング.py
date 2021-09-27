# Databricks notebook source
# MAGIC %md
# MAGIC # Databricksにおける心電図(ECG)データ処理のためのデータのステージング
# MAGIC 
# MAGIC このデモでは、PTB Diagnostic [ECG Database](https://physionet.org/physiobank/database/ptbdb/)からのデータを用いて、ECGデータをDelta Lakeへ格納する際にDatabricksレイクハウスプラットフォームが提供する機能をご説明します。これは4つあるノートブックの1つめとなります。このノートブックでは、PTBの研究で収集された人々のECGデータを示す[WFDB](https://wfdb.readthedocs.io/en/latest/)ファイルをダウンロードします。そして、データに対する機械学習、ストリーミング分析を行えるように、このデータを[Delta Lake](https://delta.io/)にロードします。
# MAGIC 
# MAGIC 最初に、AWS S3、Azure Blob Storage上に乗るメタデータ管理レイヤーの[DBFS](https://qiita.com/taka_yayoi/items/e16c7272a7feb5ec9a92)にデータをダウンロードします。マジックコマンド`%sh`を用いて、[Databricksノートブック上でシェルコマンドを実行](https://qiita.com/taka_yayoi/items/dfb53f63aed2fbd344fc#%E6%B7%B7%E6%88%90%E8%A8%80%E8%AA%9E)することができます。Physionetのウェブサイトからデータをダウンロードするためにシェルで`wget`コマンドを使用します。
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

# MAGIC %md
# MAGIC ## データのダウンロード

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC wget https://physionet.org/static/published-projects/ptbdb/ptb-diagnostic-ecg-database-1.0.0.zip
# MAGIC 
# MAGIC unzip ptb-diagnostic-ecg-database-1.0.0.zip

# COMMAND ----------

# MAGIC %md
# MAGIC ## DBFSへのデータコピー
# MAGIC 
# MAGIC データのダウンロードが完了したら、マジックコマンド`%fs`を用いて、クラスターのドライバーノードのローカルファイルシステムからDBFSにデータをコピーします。これは[ファイルシステムオペレーションのためのDatabricksユーティリティ](https://docs.databricks.com/user-guide/databricks-file-system.html#dbutils)へのショートカットとなります。
# MAGIC 
# MAGIC **注意** 以下のコピー先のパス`/tmp/takaaki.yayoi@databricks.com...`は適宜変更してください。

# COMMAND ----------

# MAGIC %fs cp -r file:///databricks/driver/ptb-diagnostic-ecg-database-1.0.0 /tmp/takaaki.yayoi@databricks.com/hls/ptb-diagnostic-ecg

# COMMAND ----------

# MAGIC %md
# MAGIC ## 必要なライブラリのインストール
# MAGIC 
# MAGIC 分析を行う前に必要なライブラリをインストールする必要があります。ここでは、CondaとMLランタイムを使用しており、ライブラリを永続的にインストールする簡単な方法は、[init scripts](https://docs.databricks.com/user-guide/clusters/conda.html#id10)を用いることです。以下のセルを実行してスクリプトをDBFSに書き出した後で、このスクリプトを[クラスタースコープinit script](https://docs.databricks.com/user-guide/clusters/init-scripts.html#cluster-scoped-init-scripts)としてアタッチし、クラスターを再起動する必要があります。
# MAGIC 
# MAGIC **注意** 
# MAGIC - 以下のセルの実行は初回のみとなります。作成されたinit scriptをクラスターに設定してください。
# MAGIC - `/tmp/takaaki.yayoi@databricks.com...`のパスは適宜変更してください。

# COMMAND ----------

dbutils.fs.rm("/tmp/takaaki.yayoi@databricks.com/hls/wfdb.sh")
dbutils.fs.put("/tmp/takaaki.yayoi@databricks.com/hls/wfdb.sh", '''#!/bin/bash
set -ex
/databricks/python/bin/python -V
. /databricks/conda/etc/profile.d/conda.sh
conda activate /databricks/python
conda install -c conda-forge -y wfdb mlflow tqdm''')

# COMMAND ----------

# MAGIC %md
# MAGIC ## UDF(ユーザー定義関数)を用いてWFDBファイルをDelta Lakeにロードする
# MAGIC 
# MAGIC PhysionetのWFDBファイルをDelta Lakeにロードするには、変換するためのコードを記述する必要があります。Apache SparkにはビルトインのWFDBローダーがないため、WFDBレコードをSparkデータフレームにパースするためのカスタムコードを記述する必要があります。このためには、指定されたファイルパスからWFDBレコードを読み込み、ファイルごとにSparkデータフレームのレコードを生成する[ユーザー定義関数](https://docs.databricks.com/spark/latest/spark-sql/udf-python.html)を定義します。
# MAGIC 
# MAGIC 最初のステップとして、UDFのために必要なライブラリをインポートします。

# COMMAND ----------

from wfdb import io
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import time

# COMMAND ----------

# MAGIC %md
# MAGIC 次にDBFS上に格納したファイルパス、および書き出し先のパスを指定します。
# MAGIC 
# MAGIC **注意**
# MAGIC `/tmp/takaaki.yayoi@databricks.com...`のパスは適宜変更してください。

# COMMAND ----------

LOCAL_DATA_PATH = '/tmp/takaaki.yayoi@databricks.com/hls/ptb-diagnostic-ecg'
DELTA_PATH = '/tmp/takaaki.yayoi@databricks.com/hls/ecg/staged/'

# COMMAND ----------

# MAGIC %md
# MAGIC ### UDFの定義及び登録
# MAGIC 
# MAGIC これでUDFを記述できるようになりました。このUDFでは、クラウドストレージにアクセスするために[Databricks FUSE mount](https://docs.databricks.com/applications/deep-learning/data-prep/ddl-storage.html)を用いて、標準的なPOSIXファイルAPIを介してWFDBにアクセスします。これによって、変更を加えることなしにwfdbライブラリを利用できます。我々のUDFはPythonのdictを返却し、Sparkはこれをrowにエンコードします。それぞれのWFDBファイルに対して、患者のID、ファイルに添付されたコメント、ファイルでエンコードされたすべてのシグナルからなるrowを生成します。

# COMMAND ----------

def extract_signals(record_name):
  
  pname = record_name.split('/')[0]
  
  record_id = record_name.replace('/','_')
  record_path = '/dbfs'+LOCAL_DATA_PATH+'/'+record_name
  record = io.rdrecord(record_name=record_path)
  record_comments = {m.split(':')[0]:m.split(':')[1] for m in record.comments}
  record_sig_name = record.sig_name
  record_signals = {}
  
  for i in range(len(record_sig_name)):
    record_signals[record_sig_name[i]] = (record.p_signal[:,i]).tolist()
    
  return {'patient_id': pname, 'comments': record_comments, 'signals': record_signals}

# COMMAND ----------

# MAGIC %md
# MAGIC 関数を記述したら、Sparkに関数を登録します。このためには、関数から返却されるスキーマを定義する必要があります。

# COMMAND ----------

from pyspark.sql.types import MapType, StringType, IntegerType, StructType, StructField, ArrayType, FloatType
from pyspark.sql import functions as F

udf_schema = StructType([ 
  StructField("patient_id", StringType(), True),
  StructField("comments", MapType(StringType(), StringType()), True),
  StructField("signals", MapType(StringType(), ArrayType(FloatType())), True),
])

extract_signals_udf = F.udf(extract_signals, udf_schema)

# COMMAND ----------

# MAGIC %md
# MAGIC ## WFDBファイルをSparkデータフレームにロード
# MAGIC 
# MAGIC ファイルパスを受け取り、患者に関するシグナル、メタデータからなるSparkデータフレームの行を生成するUDFを登録しました。ファイルをロードするには、最初にWFDBファイルが格納されているパスの一覧を作成する必要があります。ここでは、事前に準備したサンプルIDのリストを用いますが、同じことを[dbutils.fs APIs](https://docs.databricks.com/user-guide/dev-tools/dbutils.html#file-system-utilities)で行えます。

# COMMAND ----------

from pyspark.sql import Row
record_names = io.get_record_list('ptbdb')
df_record_names = spark.createDataFrame(list(map(lambda x: Row(record_name=x,record_id=x.replace('/','-')), record_names)))
display(df_record_names)

# COMMAND ----------

# MAGIC %md
# MAGIC これでデータをロードする準備が整いました。最初に並列性を高めるためにデータフレームを再パーティションします。そして、事前定義SQL関数のように、Spark SQLのselect句でUDFを使用します。さらにデータを操作する前に、行がどのようなものかを確認するために [display](https://docs.databricks.com/user-guide/visualizations/index.html#display-function)関数を使用します。

# COMMAND ----------

df_signals = df_record_names.repartition('record_name').select(
  df_record_names.record_id,
  extract_signals_udf(df_record_names.record_name).alias('signal_info'))

df_signals = df_signals.select(df_signals.record_id,
                               df_signals.signal_info.patient_id.alias('patient_id'),
                               df_signals.signal_info.comments.alias('comments'),
                               df_signals.signal_info.signals.alias('signals'))

# the signal arrays are too large to display, so we'll drop them first
display(df_signals.drop('signals'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 波形データをDelta Lakeに書き込む
# MAGIC 
# MAGIC 今回の作業の難しい部分は終わりました！以下のセルではデータをDelta Lakeテーブルに保存しています。

# COMMAND ----------

df_signals.write.format("delta").mode("overwrite").save(DELTA_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
