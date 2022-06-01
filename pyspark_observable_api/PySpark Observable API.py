# Databricks notebook source
# MAGIC %md
# MAGIC # PySparkにおけるObservable APIのサポート
# MAGIC 
# MAGIC [PySparkにおけるストリーミングクエリーのモニタリング方法 \- Qiita](https://qiita.com/taka_yayoi/items/185199fe176e7904a734)
# MAGIC 
# MAGIC ### 要件
# MAGIC 
# MAGIC - Databricks Runtime 11 Beta

# COMMAND ----------

# MAGIC %md
# MAGIC ## ライブラリのインポート

# COMMAND ----------

import os
import shutil
import time
from pathlib import Path

from pyspark.sql.functions import count, col, lit
from pyspark.sql.streaming import StreamingQueryListener

# 作業ディレクトリ
basedir = "/dbfs/tmp/takaakiyayoidatabrickscom" # 記号を含めると後でエラーになります

# COMMAND ----------

# このディレクトリにCSVファイルが作成されます。
# 本ノートブックを維持度実行した場合、'my_csv_dir'をクリーンアップします。
my_csv_dir = os.path.join(basedir, "my_csv_dir")
shutil.rmtree(my_csv_dir, ignore_errors=True)
os.makedirs(my_csv_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ストリーミングクエリーリスナーの定義

# COMMAND ----------

# リスナーの定義
class MyListener(StreamingQueryListener):
    def onQueryStarted(self, event):
        print(f"ストリーミング '{event.name}' [{event.id}] がスタートしました！")
    def onQueryProgress(self, event):
        row = event.progress.observedMetrics.get("metric")
        if row is not None:
            if row.malformed / row.cnt > 0.5:
                print(f"アラート！　なんということでしょう！ {row.cnt} 行中 {row.malformed} 行の不正なレコードが存在しています！")
            else:
                print(f"{row.cnt} 行が処理されました！")
    def onQueryTerminated(self, event):
        print(f"ストリーミング {event.id} が停止されました！")


# リスナーの追加
my_listener = MyListener()
spark.streams.addListener(my_listener)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ストリーミングクエリーのスタート

# COMMAND ----------

# 'my_csv_dir'ディレクトリをモニタリングするストリーミングクエリーをスタートします
# ここに新規CSVファイルが到着するたびに、これらを処理します
my_csv = spark.readStream.schema(
    "my_key INT, my_val DOUBLE, _corrupt_record STRING"
).csv(Path(my_csv_dir).as_uri())

# `DataFrame.observe`は処理されたレコードと不正なレコードの数を計算し、
# リスナーにイベントを送信します
my_observed_csv = my_csv.observe(
    "metric",
    count(lit(1)).alias("cnt"),  # 処理した行数
    count(col("_corrupt_record")).alias("malformed"))  # 不正なレコードの数
my_query = my_observed_csv.writeStream.format(
    "console").queryName("My observer").start()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 動作確認

# COMMAND ----------

# 次に、ストリーミングで処理されるCSVデータを書き込みます
# このCSVファイルは適切なデータです
with open(os.path.join(my_csv_dir, "my_csv_1.csv"), "w") as f:
    _ = f.write("1,1.1\n")
    _ = f.write("123,123.123\n")

time.sleep(5)  # 別のCSVファイルが5秒後に到着するものとします

# なんということでしょう！ 3行のうち2行が不正です。私のクエリーオブザーバーはアラートあげなくてなりません！
with open(os.path.join(my_csv_dir, "my_csv_error.csv"), "w") as f:
    _ = f.write("1,1.123\n")
    _ = f.write("なんてこった! 不正なレコードだ!\n")
    _ = f.write("ぎゃー!\n")

time.sleep(5)  # OK、全て終わりました。5秒後にクエリーを停止しましょう
my_query.stop()
spark.streams.removeListener(my_listener)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
