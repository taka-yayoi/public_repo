# Databricks notebook source
# MAGIC %md
# MAGIC # Delta LakeとApache Spark™を用いた高信頼データレイクの構築
# MAGIC 
# MAGIC **Delta Lake**: Apache Spark™とビッグデータワークロードにACIDトランザクションを提供するオープンソースストレージフォーマット
# MAGIC 
# MAGIC これは、いくつかのDelta Lakeの機能を説明する入門ノートブックです。
# MAGIC 
# MAGIC <img src="https://docs.delta.io/latest/_static/delta-lake-logo.png" width=300/>
# MAGIC 
# MAGIC * **オープンフォーマット**: blobストレージ上のParquetフォーマットとして格納されます。
# MAGIC * **ACIDトランザクション**: 複雑かつ同時実行されるデータパイプラインでデータの完全性と読み取りの一貫性を保証します。
# MAGIC * **スキーマ強制、スキーマ進化**: 予期しない書き込みをブロックすることでデータの綺麗さを保持します。
# MAGIC * **監査履歴**: テーブルに生じた全てのオペレーションの履歴。
# MAGIC * **タイムトラベル**: 時間あるいはバージョン番号による以前のバージョンのテーブルへのクエリー。
# MAGIC * **deleteとupsert**: プログラミングAPIによるテーブルのdeleteとupsert(update + insert)のサポート。
# MAGIC * **スケーラブルなメタデータ管理**: Sparkを用いることでメタデータのオペレーションをスケールさせ、数百万のファイルを取り扱うことができます。
# MAGIC * **バッチ、ストリーミングソース、ストリーミングシンクの統合**: Delta Lakeのテーブルはバッチテーブルでもあり、ストリーミングのソースとシンクでもあります。ストリーミングデータの取り込み、バッチによる過去のバックフィル、インタラクティブなクエリーすべてをすぐに活用することができます。
# MAGIC 
# MAGIC ### 前提条件
# MAGIC * DBR 7.6以降が必要です。
# MAGIC 
# MAGIC ### ソース
# MAGIC 
# MAGIC このノートブックは[SAIS EU 2019 Delta Lake Tutorial](https://github.com/delta-io/delta/tree/master/examples/tutorials/saiseu19)の修正バージョンです。使用データは、[Lending Club](https://www.kaggle.com/wendykan/lending-club-loan-data)の公開データの修正バージョンです。2012年から2017年の間にファンディングされた全てのローンが含まれています。それぞれのローンには、申請者の情報、現在のローンのステータス(Current, Late, Fully Paid, etc.)、最新の支払い情報が含まれています。データの完全なビューに関しては、[こちら](https://resources.lendingclub.com/LCDataDictionary.xlsx)のデータ辞書を参照してください。
# MAGIC 
# MAGIC ノートブック: https://databricks.com/notebooks/gallery/IntroductionDeltaLake.html

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) Delta Lakeテーブルへのデータのロード
# MAGIC 
# MAGIC 最初にデータを読み込み、Delta Lakeテーブルとして保存しましょう。

# COMMAND ----------

import re
from pyspark.sql.types import * 

# Username を取得
username_raw = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
# Username の英数字以外を除去し、全て小文字化
username = re.sub('[^A-Za-z0-9]+', '', username_raw).lower()

# ファイル格納パス
work_path = f"dbfs:/tmp/databricks_handson/{username}/delta_introduction"

# パスを表示
print(f"path_name: {work_path}")

# COMMAND ----------

spark.sql("set spark.sql.shuffle.partitions = 1")

# 読み込むデータ
sourcePath = "/databricks-datasets/learning-spark-v2/loans/loan-risks.snappy.parquet"

# Delta Lakeのパスの設定
deltaPath = f"{work_path}/loans_delta"

# フォルダーが存在する場合には削除
dbutils.fs.rm(deltaPath, recurse=True)

# 同じローンデータを用いたDeltaテーブルの作成
(spark.read.format("parquet").load(sourcePath) 
  .write.format("delta").save(deltaPath))

spark.read.format("delta").load(deltaPath).createOrReplaceTempView("loans_delta")
print("Defined view 'loans_delta'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) データの探索
# MAGIC 
# MAGIC データを探索してみましょう。

# COMMAND ----------

spark.sql("SELECT count(*) FROM loans_delta").show()

# COMMAND ----------

display(spark.sql("SELECT * FROM loans_delta LIMIT 5"))

# COMMAND ----------

# MAGIC %md
# MAGIC マジックコマンド`%sql`を用いても同様の処理が可能です。

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM loans_delta LIMIT 5

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) Delta Lakeテーブルへのデータストリームのロード
# MAGIC 
# MAGIC ランダムに生成されるローンIDと金額によるデータストリームを作成します。さらに、有用なユーティリティ関数をいくつか定義します。

# COMMAND ----------

import random
import os
from pyspark.sql.functions import *
from pyspark.sql.types import *

def random_checkpoint_dir(): 
  return f"{work_path}/tmp/chkpt/%s" % str(random.randint(0, 10000))


# 州をランダムに生成するユーザー定義関数
states = ["CA", "TX", "NY", "WA"]

@udf(returnType=StringType())
def random_state():
  return str(random.choice(states))

# ランダムに生成されたデータのストリームによるストリーミングクエリーを起動し、Deltaテーブルにデータを追加する関数
def generate_and_append_data_stream():

  newLoanStreamDF = (spark.readStream.format("rate").option("rowsPerSecond", 5).load() 
    .withColumn("loan_id", 10000 + col("value")) 
    .withColumn("funded_amnt", (rand() * 5000 + 5000).cast("integer")) 
    .withColumn("paid_amnt", col("funded_amnt") - (rand() * 2000)) 
    .withColumn("addr_state", random_state())
    .select("loan_id", "funded_amnt", "paid_amnt", "addr_state"))
    
  checkpointDir = f"{work_path}/chkpt"

  streamingQuery = (newLoanStreamDF.writeStream 
    .format("delta") 
    .option("checkpointLocation", checkpointDir) 
    .trigger(processingTime = "10 seconds") 
    .start(deltaPath))

  return streamingQuery

# 全てのストリーミングクエリーを停止する関数 
def stop_all_streams():
  # 全てのストリームを停止
  print("Stopping all streams")
  for s in spark.streams.active:
    s.stop()
  print("Stopped all streams")
  print("Deleting checkpoints")  
  dbutils.fs.rm(f"{work_path}/chkpt", True)
  print("Deleted checkpoints")

# COMMAND ----------

streamingQuery = generate_and_append_data_stream()

# COMMAND ----------

# MAGIC %md
# MAGIC テーブルのレコード数をカウントすることで、ストリーミングクエリーがテーブルにデータが追加されていることを確認することができます。以下のセルを複数回実行してください。

# COMMAND ----------

display(spark.sql("SELECT count(*) FROM loans_delta"))

# COMMAND ----------

# MAGIC %md
# MAGIC **全てのストリーミングクエリーを停止することを忘れないでください。**

# COMMAND ----------

stop_all_streams()

# COMMAND ----------

# MAGIC %md
# MAGIC ##  ![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) データ破損を防ぐために書き込み時のスキーマを強制
# MAGIC 
# MAGIC ローンが停止されたかどうかを示す追加のカラム`closed`を含むデータを書き込むことで、この機能をテストしてみましょう。このカラムはテーブルには存在しないことに注意してください。

# COMMAND ----------

cols = ['loan_id', 'funded_amnt', 'paid_amnt', 'addr_state', 'closed']

items = [
  (1111111, 1000, 1000.0, 'TX', True), 
  (2222222, 2000, 0.0, 'CA', False)
]

from pyspark.sql.functions import *

loanUpdates = (spark
                .createDataFrame(items, cols)
                .withColumn("funded_amnt", col("funded_amnt").cast("int")))

# COMMAND ----------

# 以下の行のコメントを解除して実行ししてください。エラーになるはずです。
#loanUpdates.write.format("delta").mode("append").save(deltaPath)

# COMMAND ----------

# MAGIC %md
# MAGIC ##  ![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) 変化するデータに対応するためにスキーマを進化
# MAGIC 
# MAGIC スキーマ強制は意図しないスキーマの汚染を防御するためのものですが、ビジネス要件の変化に合わせて明示的にスキーマを変更しなくてはならないケースがあります。この場合、オプション`mergeSchema`に`true`を設定します。

# COMMAND ----------

(loanUpdates.write.format("delta").mode("append")
  .option("mergeSchema", "true")
  .save(deltaPath))

# COMMAND ----------

# MAGIC %md
# MAGIC スキーマを確認するために再度テーブルにクエリーを実行してみましょう。

# COMMAND ----------

display(spark.read.format("delta").load(deltaPath).filter("loan_id = 1111111"))

# COMMAND ----------

# MAGIC %md
# MAGIC 既存のレコードを読み込んだ際、新規のカラムはNULLと見做されます。

# COMMAND ----------

display(spark.read.format("delta").load(deltaPath))

# COMMAND ----------

# MAGIC %md
# MAGIC ##  ![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) 既存データの変換
# MAGIC 
# MAGIC 既存データをどのように変換できるのかをみていきましょう。でも最初に、スキーマが変化したのでビューが変更に追従できるように、ビューを再定義することでテーブルに対するビューを更新しましょう。

# COMMAND ----------

spark.read.format("delta").load(deltaPath).createOrReplaceTempView("loans_delta")
print("Defined view 'loans_delta'")

# COMMAND ----------

# MAGIC %md
# MAGIC #### エラーを修正するためにローンデータを更新
# MAGIC 
# MAGIC * データをレビューしてみると、`addr_state = 'OR'`に割り当てられた全てのローンは`addr_state = 'WA'`に割り当てられるべきであることがわかりました。
# MAGIC * Parquetで`update`を行うには、以下のことが必要となります。
# MAGIC   * 更新されない全ての行を新規テーブルにコピー
# MAGIC   * 更新される全ての行をデータフレームにコピーし、データを更新
# MAGIC   * 新規テーブルに上述のデータフレームをinsert
# MAGIC   * 古いテーブルを削除
# MAGIC   * 新規テーブルの名称を古いものに更新

# COMMAND ----------

display(spark.sql("""SELECT addr_state, count(1) FROM loans_delta WHERE addr_state IN ('OR', 'WA', 'CA', 'TX', 'NY') GROUP BY addr_state"""))

# COMMAND ----------

# MAGIC %md
# MAGIC データを修正しましょう。

# COMMAND ----------

from delta.tables import *

deltaTable = DeltaTable.forPath(spark, deltaPath)
deltaTable.update("addr_state = 'OR'",  {"addr_state": "'WA'"})

# COMMAND ----------

# MAGIC %md
# MAGIC もう一度データを確認しましょう。

# COMMAND ----------

display(spark.sql("""SELECT addr_state, count(1) FROM loans_delta WHERE addr_state IN ('OR', 'WA', 'CA', 'TX', 'NY') GROUP BY addr_state"""))

# COMMAND ----------

# MAGIC %md
# MAGIC #### General Data Protection Regulation (GDPR)に基づいてテーブルからユーザーデータを削除
# MAGIC 
# MAGIC Delta Lakeテーブルから述語にマッチするデータを削除することができます。ここでは、完済したローンすべてを削除したいものとします。最初にどれだけレコードがあるのかを確認しましょう。

# COMMAND ----------

display(spark.sql("SELECT COUNT(*) FROM loans_delta WHERE funded_amnt = paid_amnt"))

# COMMAND ----------

# MAGIC %md
# MAGIC それでは削除しましょう。

# COMMAND ----------

from delta.tables import *

deltaTable = DeltaTable.forPath(spark, deltaPath)
deltaTable.delete("funded_amnt = paid_amnt")

# COMMAND ----------

# MAGIC %md
# MAGIC 完済したローンの数をチェックしてみます。

# COMMAND ----------

display(spark.sql("SELECT COUNT(*) FROM loans_delta WHERE funded_amnt = paid_amnt"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### mergeを用いた変更データのテーブルへのupsert
# MAGIC 
# MAGIC 一般的なユースケースは、あるOLAPテーブルでなされた行の変更を、OLAPワークロードの別のテーブルに複製するチェンジデータキャプチャ(CDC)です。ここでのローンデータのサンプルで使ってみるために、新たなローンや既存のローンに対する更新情報を含む別のテーブルがあるものとします。さらに、この変更テーブルは`loan_delta`テーブルと同じスキーマとなっています。SQLコマンド`MERGE`をベースとした`DeltaTable.merge()`を用いることで、これらの変更をテーブルにupsertすることができます。

# COMMAND ----------

display(spark.sql("select * from loans_delta where addr_state = 'NY' and loan_id < 30"))

# COMMAND ----------

# MAGIC %md
# MAGIC このデータにはいくつかの変更があり、ある1つのローンは完済され、別の新たなローンが追加されたものとします。

# COMMAND ----------

cols = ['loan_id', 'funded_amnt', 'paid_amnt', 'addr_state', 'closed']

items = [
  (11, 1000, 1000.0, 'NY', True),   # ローンの完済
  (12, 1000, 0.0, 'NY', False)      # 新たなローン
]

loanUpdates = spark.createDataFrame(items, cols)

# COMMAND ----------

# MAGIC %md
# MAGIC 次に、`merge`オペレーションを用いて変更データでテーブルを更新しましょう。

# COMMAND ----------

from delta.tables import *

deltaTable = DeltaTable.forPath(spark, deltaPath)

(deltaTable
  .alias("t")
  .merge(loanUpdates.alias("s"), "t.loan_id = s.loan_id") 
  .whenMatchedUpdateAll() 
  .whenNotMatchedInsertAll() 
  .execute())

# COMMAND ----------

# MAGIC %md
# MAGIC テーブルが更新されたことを確認しましょう。

# COMMAND ----------

display(spark.sql("select * from loans_delta where addr_state = 'NY' and loan_id < 30"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### insert-only mergeを用いたinsertとデータの重複排除
# MAGIC 
# MAGIC Delta LakeのmergeオペレーションはANSI標準以上の拡張構文をサポートしています。以下のような高度な機能をサポートしています。
# MAGIC 
# MAGIC - Deleteアクション: 例えば、`MERGE … WHEN MATCHED THEN DELETE`
# MAGIC - 句の条件: 例えば、`MERGE … WHEN MATCHED AND <condition> THEN ...``
# MAGIC - オプションのアクション: 全ての MATCHED 、 NOT MATCHED 句はオプションです。
# MAGIC - スター文法: 例えば、 ソースデータセットにマッチするターゲットテーブルの全てのカラムをupdate/insertするための `UPDATE *` や `INSERT *`。上で見た通り、DeltaTableにおける同等のAPIは `updateAll()` や `insertAll()` となります。
# MAGIC 
# MAGIC これによって、少ないコード量でより複雑なユースケースを表現できるようになります。例えば、過去のローンの履歴データを用いて、loan_deltaテーブルのバックフィルをしたいものとします。しかし、履歴データのいくつかはすでにテーブルにインサートされており、(すでにこれらのemailは更新されているので)それらを更新したくないものとします。(UPDATEアクションはオプションなので)INSERTアクションのみを用いた以下のmergeオペレーションを実行することで、insertを行いつつloan_idによる重複排除を行うことができます。

# COMMAND ----------

display(spark.sql("select * from loans_delta where addr_state = 'NY' and loan_id < 30"))

# COMMAND ----------

# MAGIC %md
# MAGIC このテーブルにマージしたい履歴データがいくつかあるものとします。ある過去のローンは現在のテーブルに存在しますが、履歴データは古い値を保持しているため、テーブルの現在の値を更新すべきではありません。そして、別の履歴データは現在のテーブルに存在しないので、テーブルにインサートする必要があります。

# COMMAND ----------

cols = ['loan_id', 'funded_amnt', 'paid_amnt', 'addr_state', 'closed']

items = [
  (11, 1000, 0.0, 'NY', False),
  (-100, 1000, 10.0, 'NY', False)
]

historicalUpdates = spark.createDataFrame(items, cols)

# COMMAND ----------

# MAGIC %md
# MAGIC mergeを行いましょう。

# COMMAND ----------

from delta.tables import *

deltaTable = DeltaTable.forPath(spark, deltaPath)

(deltaTable
  .alias("t")
  .merge(historicalUpdates.alias("s"), "t.loan_id = s.loan_id") 
  .whenNotMatchedInsertAll() 
  .execute())

# COMMAND ----------

# MAGIC %md
# MAGIC テーブルが更新されたことを確認しましょう。

# COMMAND ----------

display(spark.sql("select * from loans_delta where addr_state = 'NY' and loan_id < 30"))

# COMMAND ----------

# MAGIC %md
# MAGIC テーブルにおける変更は、新規ローンの追加のみであり、既存のローンは古い値に更新されないことに注意してください。

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) オペレーション履歴を用いたデータ変更の監査
# MAGIC 
# MAGIC Deltaテーブルに対する全ての変更は、テーブルのトランザクションログのコミットとして記録されます。Deltaテーブルやディレクトリに書き込みを行うと、全てのオペレーションは自動的にバージョン管理されます。テーブルの履歴を参照するために`HISTORY`コマンドを使用することができます。

# COMMAND ----------

from delta.tables import *

deltaTable = DeltaTable.forPath(spark, deltaPath)
display(deltaTable.history())

# COMMAND ----------

display(deltaTable.history(4).select("version", "timestamp", "operation", "operationParameters"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) タイムトラベルを用いたテーブルの過去のスナップショットへのクエリー
# MAGIC 
# MAGIC Delta Lakeのタイムトラベル機能を用いることで、テーブルの以前のバージョンにアクセスすることができます。この機能を活用できるユースケースには以下のようなものがあります。
# MAGIC 
# MAGIC * データ変更の監査
# MAGIC * 実験 & レポートの再現
# MAGIC * ロールバック
# MAGIC 
# MAGIC Python、Scala、SQL文法を用いてタイムスタンプ、バージョン番号を用いてクエリーを行うことができます。この例では、Python文法を用いて特定のバージョンにクエリーを行います。
# MAGIC 
# MAGIC 詳細に関しては、[Introducing Delta Time Travel for Large Scale Data Lakes](https://databricks.com/blog/2019/02/04/introducing-delta-time-travel-for-large-scale-data-lakes.html)や[ドキュメント](https://docs.delta.io/latest/delta-batch.html#deltatimetravel)をご覧ください。
# MAGIC 
# MAGIC **完済ローンを含んでいるデータの削除を行う前のテーブルをクエリーしてみましょう。**

# COMMAND ----------

previousVersion = deltaTable.history(1).select("version").first()[0] - 3

(spark.read.format("delta")
  .option("versionAsOf", previousVersion)
  .load(deltaPath)
  .createOrReplaceTempView("loans_delta_pre_delete"))

display(spark.sql("SELECT COUNT(*) FROM loans_delta_pre_delete WHERE funded_amnt = paid_amnt"))

# COMMAND ----------

# MAGIC %md
# MAGIC 削除前にはあった完済ローンと同じ数を確認することができました。

# COMMAND ----------

# MAGIC %md
# MAGIC ## クリーンアップ

# COMMAND ----------

dbutils.fs.rm(work_path, True)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
