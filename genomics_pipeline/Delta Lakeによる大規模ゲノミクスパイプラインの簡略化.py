# Databricks notebook source
# MAGIC %md
# MAGIC # Delta Lakeによる大規模ゲノミクスパイプラインの簡略化
# MAGIC 
# MAGIC #![Unified Analytics Platform for Genomics Architecture](https://pages.databricks.com/rs/094-YMS-629/images/hls_UAP4Genomics_architecture_WB.png)
# MAGIC 
# MAGIC このノートブックでは、Databricksにおける**大規模**ゲノミクスパイプラインのオペレーションと本格運用をどのように簡素化するのかを説明します。
# MAGIC 
# MAGIC ## シンプルな疑問に応えることすら困難です!
# MAGIC 
# MAGIC ゲノムデータはサイズが非常に大きいため、従来のシングルノードのバイオインフォマティクスツールでは以下のようなシンプルな疑問に答えるにも非常に時間がかかってしまいます。
# MAGIC 
# MAGIC - どれだけのサンプルをシーケンシングしたのか？
# MAGIC   - 今週は?
# MAGIC   - 今月は?
# MAGIC - それぞれいくつの変異体があったのか？
# MAGIC - コホートにおいて何のクラスの変異体を観測したのか？
# MAGIC 
# MAGIC ## 大規模ゲノミクスパイプラインのデモンストレーション
# MAGIC 
# MAGIC ここではDelta Lakeにシーケンスデータのストリーミングを流し込み、リアルタイムで分析を行う様子をデモします。
# MAGIC 
# MAGIC このノートブックはi3.2xlargeインスタンスのドライバー、8台のワーカーで構成されるDatabricks 4.3(Apache Spark 2.3.1, Scala 2.11)クラスターでテストされています。
# MAGIC 
# MAGIC **参考情報**
# MAGIC - [Simplifying Genomics Pipelines at Scale with Databricks Delta \- The Databricks Blog](https://databricks.com/blog/2019/03/07/simplifying-genomics-pipelines-at-scale-with-databricks-delta.html)
# MAGIC - [Databricks Delta Lakeによる大規模ゲノミクスパイプラインの簡略化 \- Qiita](https://qiita.com/taka_yayoi/items/2ae740ab884c26e5906e)
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作成者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>作成日</td><td>2021/06/21</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>DBR</td><td>8.3</td></tr>
# MAGIC </table>
# MAGIC 
# MAGIC <img style="margin-top:25px;" src="https://jixjiadatabricks.blob.core.windows.net/images/databricks-logo-small-new.png" width="140">

# COMMAND ----------

# ライブラリのインポート
import pyspark.sql.functions as fx
from pyspark.sql.types import StringType, IntegerType, ArrayType
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 設定
sns.set(style="white")
spark.conf.set("spark.databricks.delta.preview.enabled", "true")
spark.conf.set("spark.databricks.delta.formatCheck.enabled", "false")

# COMMAND ----------

# DBTITLE 0,functions
#
# パース用関数
#

# 変異体のコーディング
def get_coding_mutations(df):
  df = df.where(fx.col("proteinHgvs").rlike("^[p\.]")). \
          where((fx.col("mutationType") == "nonsynonymous") | 
                (fx.col("mutationType") == "synonymous") | 
                (fx.col("effect") == "stop_gained"))
  return df

# proteinHgvsのパース
def parse_proteinHgvs(hgvs):
  """
  parses proteinHgvs string into amino acid substitutions
  :param hgvs: str, proteinHgvs p.[codon1][position][codon2]
  :return: list with two amino acids (* if stop codon)
  """
  hgvs_list = list(hgvs)
  aa1 = "".join(hgvs_list[2:5])
  if (hgvs_list[-1] == "*"): # * = 終止コドン(stop codon): タンパク質合成を終了させるmRNA上のコドン(mRNA上のヌクレオチド3個から成る配列の単位)
    aa2 = hgvs_list.pop(-1)
  else:
    aa2 = "".join(hgvs_list[-3:]) # 他のコドンは3文字の短縮形
  return [aa1, aa2]

# parse_proteinHgvsをUDF(ユーザー定義関数)として定義
parse_proteinHgvs_udf = fx.udf(parse_proteinHgvs, ArrayType(StringType()))

# アミノ酸交換の取得
def get_amino_acid_substitutions(df, hgvs_col):
  """
  parse hgvs notation to get amino acid substitutions in a manageable format
  """
  df = df.withColumn("tmp", parse_proteinHgvs_udf(fx.col(hgvs_col))). \
          withColumn("reference", fx.col("tmp")[0]). \
          withColumn("alternate", fx.col("tmp")[1]). \
          drop("tmp", hgvs_col)
  return df

# アミノ酸交換のカウント
def count_amino_acid_substitution_combinations(df):
  df = df.groupBy("reference", "alternate").count(). \
          withColumnRenamed("count", "substitutions")
  return df

# COMMAND ----------

# MAGIC %md ## 初期セットアップ
# MAGIC 
# MAGIC パイプラインを構築するにあたり、まずは以下の手順で単一の`sampleId`に対応する試験のParquetファイルをDelta Lakeテーブルに書き込みます。
# MAGIC 
# MAGIC - Deltaストリームのパス`delta_stream_outpath`を指定してテーブルを作成します。
# MAGIC - 単一の`sampleId`の試験のファイルを読み込み、`delta_stream_outpath`に書き込むSparkジョブを作成します。

# COMMAND ----------

import re
from pyspark.sql.types import * 

# ログインIDからUsernameを取得
username_raw = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
# Username の英数字以外を除去し、全て小文字化。Username をファイルパスやデータベース名の一部で使用可能にするため
username = re.sub('[^A-Za-z0-9]+', '', username_raw).lower()

# データベース名
db_name = f"{username}_genomics"

# Hiveメタストアのデータベースの準備:データベースの作成
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
# Hiveメタストアのデータベースの選択
spark.sql(f"USE {db_name}")

# ファイル格納パス
work_path = f"dbfs:/tmp/{username}/dnaseq/"

print("database name: " + db_name)
print("path: " + work_path)

# COMMAND ----------

# Deltaストリームの出力パス
delta_stream_outpath = f"{work_path}annotations_etl_delta_stream/"

# ソースとなるannotations_etl_parquet
annotations_etl_parquet_path = "/databricks-datasets/genomics/annotations_etl_parquet/"

# 単一のsampleIdを指定
single_sampleId_path = annotations_etl_parquet_path + "sampleId=SRS000030_SRR709972"

# ディレクトリがある場合には削除して再作成
dbutils.fs.rm(delta_stream_outpath, True)
dbutils.fs.mkdirs(delta_stream_outpath)

# COMMAND ----------

# 単一のsampleIdの読み込み
spark.read.format("parquet").load(single_sampleId_path).write.format("delta").save(delta_stream_outpath)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delta Lakeでストリーミングを始める
# MAGIC 
# MAGIC **重要!**
# MAGIC - 以降のセルを実行するにはストリームの初期化が完了するのを待ってください。
# MAGIC - 最初は単一サンプルの情報を見ることになります。
# MAGIC - 後ほどDeltaテーブルにさらにサンプルを追加することで、テーブル、グラフがリアルタイムでアップデートされます。

# COMMAND ----------

# MAGIC %md ## シンプルな疑問に回答していきます
# MAGIC 
# MAGIC Deltaストリームに問いかける質問には以下のものが含まれます：
# MAGIC 
# MAGIC - 変異体のカウント
# MAGIC - Deltaテーブルの最初の1000行の表示
# MAGIC - テーブルのスキーマの確認

# COMMAND ----------

# 変異体のカウント
exomes = spark.readStream.format("delta").load(delta_stream_outpath)
display(exomes.groupBy("sampleId").count().withColumnRenamed("count", "variants"))

# COMMAND ----------

# Deltaテーブル`exomes`の最初の1000行を表示
display(exomes)

# COMMAND ----------

# Delltaテーブル`exomes`のスキーマを確認
exomes.printSchema()

# COMMAND ----------

# MAGIC %md ## 一塩基ヌクレオチド変異体のカウント
# MAGIC 
# MAGIC 一塩基ヌクレオチド変異体(SNV)のカウントを計算するために、以下のことを行います:
# MAGIC - referenceAlleleとalternateAlleleのフィルタリング条件に基づいて`snvs`ストリーミングデータフレームを作成します。
# MAGIC - 棒グラフでデータを参照するためにSpark SQLクエリーを実行します。
# MAGIC   - これはストリーミングDeltaテーブルであり、`exomes`Deltaテーブルに新たなデータが追加されると棒グラフは継続的に更新されることに注意してください。
# MAGIC   
# MAGIC **参考情報**
# MAGIC - [一塩基バリアント：日経バイオテクONLINE](https://bio.nikkeibp.co.jp/atcl/report/16/011900001/16/05/02/00022/)

# COMMAND ----------

snvs = exomes.where((fx.length(fx.col("referenceAllele")) == 1) & 
                    (fx.length(fx.col("alternateAllele")) == 1))
snvs.createOrReplaceTempView("snvs")

# COMMAND ----------

# MAGIC %sql
# MAGIC select referenceAllele, alternateAllele, count(1) as GroupCount 
# MAGIC   from snvs
# MAGIC  group by referenceAllele, alternateAllele
# MAGIC  order by GroupCount desc

# COMMAND ----------

# MAGIC %md ## 変異体のカウントは?
# MAGIC 
# MAGIC 変異体タイプのカウントを行うためには、Deltaテーブル`exomes`に対して`GROUP BY`を実行します。ここではドーナツチャートを使用します。

# COMMAND ----------

display(exomes.groupBy("mutationType").count())

# COMMAND ----------

# MAGIC %md
# MAGIC **注意:** ご自身のユースケースでリアルタイムでの更新が必要ない場合には、シーケンシングを行う日次、週次の定期ジョブを実行することもできます。

# COMMAND ----------

# MAGIC %md
# MAGIC ## アミノ酸置換ヒートマップの作成
# MAGIC 
# MAGIC アミノ酸置換ヒートマップを作成するために、*pandas*と*matplotlib*を使用します。
# MAGIC 
# MAGIC **重要!**
# MAGIC - 以降のセルを実行するにはストリームの初期化が完了するのを待ってください。
# MAGIC - ビルトインの`display()`以外のプロット処理はストリーミングテーブルに対して動作しません。
# MAGIC   - このため、一旦ストリームをメモリー(あるいはParquetに)に書き出して、手動で以下のセルを実行しアミノ酸置換ヒートマップを作成します。
# MAGIC   
# MAGIC **参考資料**
# MAGIC - [アミノ酸置換とは何？ Weblio辞書](https://www.weblio.jp/content/%E3%82%A2%E3%83%9F%E3%83%8E%E9%85%B8%E7%BD%AE%E6%8F%9B#:~:text=%E3%83%8E%E3%83%B3%E3%82%B7%E3%83%8E%E3%83%8B%E3%83%9E%E3%82%B9%E3%81%AA%E3%82%B3%E3%83%BC%E3%83%89%E9%85%8D%E5%88%97%E4%B8%8A,%E3%81%AE%E3%82%A2%E3%83%9F%E3%83%8E%E9%85%B8%E3%81%AB%E5%A4%89%E3%82%8F%E3%82%8B%E3%81%93%E3%81%A8%E3%80%82)

# COMMAND ----------

# アミノ酸置換のストリームの構築
coding = get_coding_mutations(exomes)
aa_substitutions = get_amino_acid_substitutions(coding.select("proteinHgvs"), "proteinHgvs")
aa_counts = count_amino_acid_substitution_combinations(aa_substitutions)
aa_counts. \
  writeStream. \
  format("memory"). \
  queryName("amino_acid_substitutions"). \
  outputMode("complete"). \
  trigger(processingTime='60 seconds'). \
  start()

# COMMAND ----------

# アミノ酸置換ヒートマップの作成
# このヒートマップはpandasとmatplotlibを使用しているのでストリーミングテーブルでは動作しません
amino_acid_substitutions = spark.read.table("amino_acid_substitutions")
max_count = amino_acid_substitutions.agg(fx.max("substitutions")).collect()[0][0]
aa_counts_pd = amino_acid_substitutions.toPandas()
aa_counts_pd = pd.pivot_table(aa_counts_pd, values='substitutions', index=['reference'], columns=['alternate'], fill_value=0)

fig, ax = plt.subplots()
with sns.axes_style("white"):
  ax = sns.heatmap(aa_counts_pd, vmax=max_count*0.4, cbar=False, annot=True, annot_kws={"size": 7}, fmt="d")
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 時間経過における変異体の数の集計
# MAGIC 
# MAGIC **重要!**
# MAGIC - 以降のセルを実行するにはストリームの初期化が完了するのを待ってください。
# MAGIC - ビルトインの`display()`以外のプロット処理はストリーミングテーブルに対して動作しません。
# MAGIC   - このため、一旦ストリームをメモリー(あるいはParquetに)に書き出して、手動で以下のセルを実行しアミノ酸置換ヒートマップを作成します。

# COMMAND ----------

# 参照: https://docs.databricks.com/spark/latest/structured-streaming/examples.html#foreachbatch-sqldw-example

# パスの設定
variant_count_test_parquet_path = f"{work_path}variant_count_test_parquet"

# Deltaフォーマットで書き出し
def write_to_delta(df, epochId):
  df.write. \
     format("delta"). \
     mode('append'). \
     save(variant_count_test_parquet_path)

# 設定
spark.conf.set("spark.sql.shuffle.partitions", "1")

# 60秒ごとにストリームを書き出すクエリーの実行
query = (
  exomes.withColumn("mutations", fx.lit("mutations")). \
         select("contigName", "start", "referenceAllele", "alternateAllele", "mutations"). \
         dropDuplicates(). \
         groupBy("mutations"). \
         count(). \
         withColumn('time', fx.lit(fx.current_timestamp())). \
         writeStream. \
         trigger(processingTime='60 seconds'). \
         foreachBatch(write_to_delta). \
         outputMode("update"). \
         start()
    )

# COMMAND ----------

# 時間経過における別個の変異体の数
variants = spark.readStream.format("delta").load(variant_count_test_parquet_path).withColumnRenamed("count", "distinct mutations")
display(variants)

# COMMAND ----------

# MAGIC %md ## ゲノミクスパイプラインにデータを追加
# MAGIC 
# MAGIC ゲノミクスパイプラインの設定が終わったので、Deltaテーブルにさらにサンプルを追加します。
# MAGIC - ストリーミングをセットアップしているので、上のグラフ(ヒートマップは除く)とテーブルはデータが投入される度に更新されます。

# COMMAND ----------

import time
files = dbutils.fs.ls(annotations_etl_parquet_path)
counter=0

# 時間間隔を持って、すべてのParquetファイルに対してループします
time.sleep(10)
for sample in files:
  counter+=1
  annotation_path = sample.path
  if ("sampleId=" in annotation_path):
    sampleId = annotation_path.split("/")[4].split("=")[1]
    variants = spark.read.format("parquet").load(str(annotation_path))
    print("running " + sampleId)
    if(sampleId != "SRS000030_SRR709972"):
      variants.write.format("delta"). \
             mode("append"). \
             save(delta_stream_outpath)
    time.sleep(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## クリーンアップ
# MAGIC 終了したら生成したデータを削除しましょう。

# COMMAND ----------

# MAGIC %python
# MAGIC 
# MAGIC # タイムアウト時間(millseconds)を変更
# MAGIC spark.conf.set("spark.sql.streaming.stopTimeout", 60000)
# MAGIC 
# MAGIC # Sreaming処理を停止
# MAGIC for s in spark.streams.active:
# MAGIC   print("Stream name:", s.name)
# MAGIC   s.stop()
# MAGIC 
# MAGIC print("Stopped.")

# COMMAND ----------

dbutils.fs.rm(delta_stream_outpath, True)
dbutils.fs.rm(variant_count_test_parquet_path, True)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
