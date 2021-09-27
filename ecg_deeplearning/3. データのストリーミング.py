# Databricks notebook source
# MAGIC %md
# MAGIC # 既存Deltaテーブルからのストリーミングデータの生成
# MAGIC 
# MAGIC このノートブックでは、波形データを保持する既存のDelta Lakeテーブルからストリーミングデータセットを生成します。このためには、サンプルを一定間隔で到着するものとして取り扱います。患者が到着した際には、それぞれの観察期間における患者に対応するレコードを生成します。
# MAGIC 
# MAGIC **注意**
# MAGIC このノートブックは明示的に停止しない限り、処理を継続します。
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

# 以下のパスを前のノートブックと揃えてください
LOCAL_DATA_PATH = '/tmp/takaaki.yayoi@databricks.com/hls/ptb-diagnostic-ecg'
DELTA_PATH = '/tmp/takaaki.yayoi@databricks.com/hls/ecg/staged/'

# COMMAND ----------

import random
import time
from pyspark.sql.functions import lit

base_table = sqlContext.read.format("delta").load(DELTA_PATH)
records = base_table.count()

# 適宜変更してください
stream_path = '/tmp/takaaki.yayoi@databricks.com/hls/ecg/streaming/'

dbutils.fs.rm(stream_path, True)

ts = 0

base_table.limit(1).withColumn('time_interval', lit(0)).write.format('delta').save(stream_path)

for i in range(records - 1):
  derived_table = base_table.limit(i)
  
  sleep_time = random.randint(1, 5)
  
  for j in range(sleep_time):
    ts += 1
    
    base_table.limit(i).withColumn('time_interval', lit(ts)).write.format('delta').mode('append').save(stream_path)
    
while True:
  
  ts += 1
  base_table.withColumn('time_interval', lit(ts)).write.format('delta').mode('append').save(stream_path)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
