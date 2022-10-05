# Databricks notebook source
# MAGIC %md 
# MAGIC # Databricks利用量分析ダッシュボード
# MAGIC 
# MAGIC このノートブックでは、あなた指定したオブジェクトストレージに格納されている利用量データを活用します。それぞれのチャートやレポートを日次でリフレッシュするには、以下の指示に従ってください。
# MAGIC 
# MAGIC #### このノートブックの使い方
# MAGIC 
# MAGIC 1. Cmd 2セルの`usagefilePath`にお使いのオブジェクトストレージ(バケット)名を入力します。ご参考までに構文を指定しています。
# MAGIC 1. このセル以降にあるウィジェットのフィールドを埋めます。これにはコミットのドル金額、日付範囲、それぞれのコンピュートタイプのユニットDBU価格(SKU価格)、利用量とコストをブレークダウンするために使用したいクラスタータグのキー、期間の粒度、利用量の単位(費用、DBU、累積費用、累積DBU)が含まれます。
# MAGIC 1. Cmd 2セルあるいはノートブックツールバーからを**すべてを実行**します。
# MAGIC 
# MAGIC #### 注意
# MAGIC このノートブックでは、お使いのワークスペース全てで同じユニットDBUが適用されていることを前提としています。複数のワークスペースがあり、それぞれが異なる価格プランである場合には、これらの表示結果は正確なドルの金額を示しません。

# COMMAND ----------

# 課金利用ログファイルへのパスを入力します。例: s3a://my-bucket/delivery/path/csv/dbus/
usagefilePath = "s3a://ty-db-billable-usage-log/billable-log/billable-usage/csv/"

# COMMAND ----------

# アクセスできることを確認します
dbutils.fs.ls("s3a://ty-db-billable-usage-log/")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 利用量データの準備
# MAGIC 
# MAGIC **以降のセルは編集しないでください**

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import pyspark.sql.functions as func

# 利用量のスキーマ
usageSchema = StructType([
  StructField("workspaceId", StringType(), False),
  StructField("timestamp", DateType(), False),
  StructField("clusterId", StringType(), False),
  StructField("clusterName", StringType(), False),
  StructField("clusterNodeType", StringType(), False),
  StructField("clusterOwnerUserId", StringType(), False),
  StructField("clusterCustomTags", StringType(), False),
  StructField("sku", StringType(), False),
  StructField("dbus", FloatType(), False),
  StructField("machineHours", FloatType(), False),
  StructField("clusterOwnerUserName", StringType(), False),
  StructField("tags", StringType(), False)
])

# 利用量のデータフレームを作成してキャッシュ
df = (spark.read
      .option("header", "true")
      .option("escape", "\"")
      .schema(usageSchema)
      .csv(usagefilePath)
      )

usageDF = (df.select("workspaceId",
                     "timestamp",
                     "clusterId",
                     "clusterName",
                     "clusterNodeType",
                     "clusterOwnerUserId",
                     "clusterCustomTags",
                     when(col("sku") == "STANDARD_INTERACTIVE_OPSEC", "All Purpose Compute")
                     .when(col("sku") == "STANDARD_AUTOMATED_NON_OPSEC", "Jobs Compute")
                     .when(col("sku") == "STANDARD_INTERACTIVE_NON_OPSEC", "All Purpose Compute")
                     .when(col("sku") == "LIGHT_AUTOMATED_NON_OPSEC", "Jobs Compute Light")
                     .when(col("sku") == "STANDARD_AUTOMATED_OPSEC", "Jobs Compute")
                     .when(col("sku") == "LIGHT_AUTOMATED_OPSEC", "Jobs Compute Light")
                     .when(col("sku") == "STANDARD_ALL_PURPOSE_COMPUTE", "All Purpose Compute")
                     .when(col("sku") == "STANDARD_JOBS_COMPUTE", "Jobs Compute")
                     .when(col("sku") == "STANDARD_JOBS_LIGHT_COMPUTE", "Jobs Compute Light")
                     .when(col("sku") == "PREMIUM_ALL_PURPOSE_COMPUTE", "All Purpose Compute")
                     .when(col("sku") == "PREMIUM_JOBS_COMPUTE", "Jobs Compute")
                     .when(col("sku") == "PREMIUM_JOBS_LIGHT_COMPUTE", "Jobs Compute Light")
                     .when(col("sku") == "ENTERPRISE_ALL_PURPOSE_COMPUTE", "All Purpose Compute")
                     .when(col("sku") == "ENTERPRISE_JOBS_COMPUTE", "Jobs Compute")
                     .when(col("sku") == "ENTERPRISE_JOBS_LIGHT_COMPUTE", "Jobs Compute Light")
                     .otherwise(col("sku")).alias("sku"),
                     "dbus",
                     "machineHours",
                     "clusterOwnerUserName",
                     "tags")
           .withColumn("tags", when(col("tags").isNotNull(), col("tags")).otherwise(col("clusterCustomTags")))
           .withColumn("tags", from_json("tags", MapType(StringType(), StringType())).alias("tags"))
           .drop("userId")
           .cache()
          )

# SQLコマンドを使うために一時テーブルを作成
usageDF.createOrReplaceTempView("usage")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ウィジェットの設定

# COMMAND ----------

# 動的にフィルタリングできる様にウィジェットを作成
import datetime
dbutils.widgets.removeAll()

# 日付ウィンドウのフィルター
now = datetime.datetime.now()
dbutils.widgets.text("Date - End", now.strftime("%Y-%m-%d"))
dbutils.widgets.text("Date - Beginning", now.strftime("%Y-%m-%d"))

# SKU価格。お客様のアカウントごとに固有な値を入力できる様にテキストウィジェットを作成
skus = spark.sql("select distinct(sku) from usage").rdd.map(lambda row : row[0]).collect()
for sku in skus:  # お客様固有のSKUのテキストボックスを表示
  dbutils.widgets.text("SKU Price - " + sku, ".00")

# 時系列グラフにおける時間単位
dbutils.widgets.dropdown("Time Unit", "Day", ["Day", "Month"])
timeUnit = "Time"

# コミットの金額
dbutils.widgets.text("Commit Dollars", "00.00")
commit = getArgument("Commit Dollars")

# タグのキー
tags = spark.sql("select distinct(explode(map_keys(tags))) from usage").rdd.map(lambda row : row[0]).collect()
if len(tags) > 0:
  defaultTag = tags[0]
  dbutils.widgets.dropdown("Tag Key", str(defaultTag), [str(x) for x in tags])

# 利用量のタイプ
dbutils.widgets.dropdown("Usage", "Spend", ["Spend", "DBUs", "Cumulative Spend", "Cumulative DBUs"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## データフレームの作成

# COMMAND ----------

# SKU名とレートからデータフレームを作成。これは、費用を得るために利用量データフレームとjoinされます。
skuVals = [str(sku) for sku in skus]
wigVals = [getArgument("SKU Price - " + sku) for sku in skus]

skuZip = list(zip(skuVals, wigVals)) # RDDに並列化するために、それぞれのSKUと対応するレートごとのリストオブジェクトを作成します。

skuRdd = sc.parallelize(skuZip) # RDDの作成

skuSchema = StructType([
  StructField("sku", StringType(), True),
  StructField("rate", StringType(), True)])

skuDF = spark.createDataFrame(skuRdd, skuSchema) # データフレームの作成

# COMMAND ----------

timeUnitFormat = "yyyy-MM-dd" if getArgument("Time Unit") == "Day" else "yyyy-MM"

# それぞれの期間、コミット、SKU、タグの値ごとの費用、DBU、累積費用、累積DBUを含む大規模なデータフレーム
globalDF = (usageDF
            .filter(usageDF.timestamp.between(getArgument("Date - Beginning"), getArgument("Date - End")))
            .join(skuDF, "Sku")
            .withColumn("Spend", usageDF["dbus"] * skuDF["rate"])
            .withColumn("Commit", lit(getArgument("Commit Dollars")))
            .withColumn("Tag", usageDF["tags." + getArgument("Tag Key")])
            .select(date_format("timestamp", timeUnitFormat).alias(timeUnit), "Spend", "dbus", "Sku", "Commit", "Tag")
            .groupBy(timeUnit, "Commit", "Sku", "Tag")
            .sum("Spend", "dbus")
            .withColumnRenamed("sum(dbus)", "DBUs")
            .withColumnRenamed("sum(Spend)", "Spend")
            .orderBy(timeUnit)
           )

# 単一のコミット、SKU、タグの値に対応する小規模なデータフレームを作成し、費用、DBUにタウする累積値を計算する関数
def usageBy(columnName):
  
  # 累積費用/DBUを得るためのウィンドウ関数
  cumulativeUsage = Window \
  .orderBy(timeUnit) \
  .partitionBy(columnName) \
  .rowsBetween(Window.unboundedPreceding, 0)
  
  # 指定されたカラムの値に対して当該期間にデータがない場合に行を持たないのではなく、
  # ゼロの費用、DBUの値を持つ行を追加し、適切な集約処理に加算します。
  # グラフは累積値をゼロとして解釈しないためです。
  zeros = globalDF.select(columnName).distinct().withColumn("Spend", lit(0)).withColumn("DBUs", lit(0))
  zerosByTime = globalDF.select(timeUnit).distinct().crossJoin(zeros)
  
  return (globalDF
          .select(timeUnit, columnName, "Spend", "DBUs")
          .union(zerosByTime)
          .groupBy(timeUnit, columnName)
          .sum("Spend", "DBUs")
          .withColumnRenamed("sum(DBUs)", "DBUs")
          .withColumnRenamed("sum(Spend)", "Spend")
          .withColumn("Cumulative Spend", func.sum(func.col("Spend")).over(cumulativeUsage))
          .withColumn("Cumulative DBUs", func.sum(func.col("DBUs")).over(cumulativeUsage))
          .select(timeUnit, columnName, getArgument("Usage"))
          .withColumnRenamed(getArgument("Usage"), "Usage")
          .orderBy(timeUnit)
         )

display(globalDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 利用量レポート
# MAGIC 
# MAGIC 以下のレポートを生成します。
# MAGIC </p>
# MAGIC 
# MAGIC 1. 利用量の時系列変化
# MAGIC 1. SKUごとの利用量の時系列変化
# MAGIC 1. 選択されたタグごとの利用量

# COMMAND ----------

commitMsg = ""
if getArgument("Usage") == "Cumulative Spend":
  commitMsg = "&nbsp;"*8 + "Commit = $" + getArgument("Commit Dollars")

displayHTML("<center><h2>{0} over time{1}</h2></center>".format(getArgument("Usage"), commitMsg))

# COMMAND ----------

# DBTITLE 1,利用量の時系列変化
display(usageBy("Commit"))

# COMMAND ----------

displayHTML("<center><h2>{} by SKU over time</h2></center>".format(getArgument("Usage")))

# COMMAND ----------

# DBTITLE 1,SKUごとの利用量の時系列変化
display(usageBy("Sku"))

# COMMAND ----------

displayHTML("<center><h2>{0} by {1} over time</h2></center>".format(getArgument("Usage"), getArgument("Tag Key")))

# COMMAND ----------

# DBTITLE 1,タグごとの利用量の時系列変化
display(usageBy("Tag"))

# COMMAND ----------

# MAGIC %md
# MAGIC # END
