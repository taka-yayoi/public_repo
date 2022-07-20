# Databricks notebook source
# MAGIC %md
# MAGIC # AzureにおけるCredential passthroughを用いたADLSマウントポイント上のZIPファイルへのアクセス
# MAGIC 
# MAGIC こちらのノートブックでは、資格情報パススルーを使用したADLSマウントポイントに格納されているZIPファイルの中にあるCSVファイルを解凍して保存します。
# MAGIC 
# MAGIC [Azure Active Directory 資格情報パススルーを使用して Azure Data Lake Storage にアクセスする \- Azure Databricks \| Microsoft Docs](https://docs.microsoft.com/ja-jp/azure/databricks/security/credential-passthrough/adls-passthrough)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ADLSのマウント

# COMMAND ----------

configs = {
  "fs.azure.account.auth.type": "CustomAccessToken",
  "fs.azure.account.custom.token.provider.class": spark.conf.get("spark.databricks.passthrough.adls.gen2.tokenProviderClassName")
}

# Optionally, you can add <directory-name> to the source URI of your mount point.
dbutils.fs.mount(
  source = "abfss://<コンテナ名>@<ストレージアカウント名>.dfs.core.windows.net/",
  mount_point = "/mnt/test_mnt",
  extra_configs = configs)

# COMMAND ----------

# MAGIC %md
# MAGIC アンマウント

# COMMAND ----------

dbutils.fs.unmount("/mnt/test_mnt")

# COMMAND ----------

display(dbutils.fs.ls("/mnt/test_mnt"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ADLSから直接読み込み

# COMMAND ----------

spark.read.format("csv").load("abfss://takauccontainer@takaucstorageaccount.dfs.core.windows.net/iris-dataset.csv").collect()

# COMMAND ----------

# MAGIC %md
# MAGIC 認証情報を用いたFUSEマウントポイント`/dbfs`へのアクセスは不可です。以下のようなエラーとなります。
# MAGIC 
# MAGIC ```
# MAGIC ls: cannot access '/dbfs/mnt/test_mnt/iris-dataset.csv.zip': No such file or directory
# MAGIC ```

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /dbfs/mnt/test_mnt/iris-dataset.csv.zip

# COMMAND ----------

# MAGIC %md
# MAGIC ## DBFS経由でのアクセスおよびUNZIP
# MAGIC 
# MAGIC クレディンシャルパススルーが設定されたFUSEマウントポイントがサポートされていないということは、ローカルファイルAPIでの操作ができないことを意味します。このため、Spark API経由でファイルを操作します。
# MAGIC 
# MAGIC [制限事項](https://docs.microsoft.com/ja-jp/azure/databricks/security/credential-passthrough/adls-passthrough#--limitations)
# MAGIC > [FUSE マウント](https://docs.microsoft.com/ja-jp/azure/databricks/data/databricks-file-system#fuse) (`/dbfs`) は、Databricks Runtime 7.3 LTS 以降でのみ使用できます。 資格情報のパススルーが構成されているマウント ポイントは、FUSE マウントではサポートされていません。
# MAGIC 
# MAGIC こちらでは解凍したファイルを単一のCSVファイルとして保存します。
# MAGIC 
# MAGIC **参考資料**
# MAGIC - [python \- How do I read text files within a zip file? \- Stack Overflow](https://stackoverflow.com/questions/15282651/how-do-i-read-text-files-within-a-zip-file)
# MAGIC - [Python : How to get the list of all files in a zip archive – thisPointer](https://thispointer.com/python-how-to-get-the-list-of-all-files-in-a-zip-archive/)
# MAGIC - [Apache Spark Assign The Result Of Udf To Multiple Dataframe Columns](https://www.faqcode4u.com/faq/623146/apache-spark-assign-the-result-of-udf-to-multiple-dataframe-columns)
# MAGIC - [pyspark \- Write each row of a spark dataframe as a separate file \- Stack Overflow](https://stackoverflow.com/questions/49883129/write-each-row-of-a-spark-dataframe-as-a-separate-file)

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *

schema = StructType([
     StructField("file_name", StringType(), False),
     StructField("contents", StringType(), False)
])
    
def zip_extract_file(binary_data):
    # ZIPを解凍するUDF
    import zipfile, io
    in_memory_data = io.BytesIO(binary_data)

    with zipfile.ZipFile(in_memory_data, "r") as zf:
        
        # zipファイルあたり1ファイルを想定
        target_file = zf.namelist()[0]
        
        with io.TextIOWrapper(zf.open(target_file), encoding="utf-8") as f:
            contents = f.read()
            
    return (target_file, contents)
 
decompress_func = lambda x: zip_extract_file(x)

udf_decompress = udf(decompress_func, schema)

# COMMAND ----------

# バイナリーファイルとして読み込み
df = spark.read.format("binaryFile").load("dbfs:/mnt/test_mnt/iris-dataset.csv.zip")
# UDFを適用して解凍
df = df.withColumn('decoded', udf_decompress('content')).select("decoded.file_name", "decoded.contents")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC この時点では、各行にファイル名と生データが格納されている状態です。以下のステップで単一のファイルとしてCSVを保存します。
# MAGIC 
# MAGIC **参考資料**
# MAGIC - [pyspark \- Save each row in Spark Dataframe into different file \- Stack Overflow](https://stackoverflow.com/questions/51576261/save-each-row-in-spark-dataframe-into-different-file)
# MAGIC - [python \- How to save a PySpark dataframe as a CSV with custom file name? \- Stack Overflow](https://stackoverflow.com/questions/69635571/how-to-save-a-pyspark-dataframe-as-a-csv-with-custom-file-name)
# MAGIC - [Databrikcs\(Spark\)のPysparkにて単一ファイルとしてCSVファイルを書き込む方法 \- Qiita](https://qiita.com/manabian/items/78b57741885ecda8570a)

# COMMAND ----------

# '_started'と'_committed_'で始まるファイルを書き込まないように設定
spark.conf.set("spark.sql.sources.commitProtocolClass", "org.apache.spark.sql.execution.datasources.SQLHadoopMapReduceCommitProtocol")

# '_SUCCESS'で始まるファイルを書き込まないように設定
spark.conf.set("mapreduce.fileoutputcommitter.marksuccessfuljobs","false")

# COMMAND ----------

# 最初の行からファイル名を取得
file_name_to_save = df.first().file_name

# 保存先
file_path_to_save = f'dbfs:/mnt/test_mnt/melted_csv/{file_name_to_save}'

# repartition(1)を指定して1つのファイルに保存します
# クォーテーションがあると読み込み時にパースできないので除外します
df.select("contents").repartition(1).write.option("quote", "").mode("overwrite").csv(file_path_to_save)

# COMMAND ----------

display(dbutils.fs.ls(file_path_to_save))

# COMMAND ----------

# 中身を確認するには、上の結果に基づいて適宜ファイル名を指定してください
dbutils.fs.head("dbfs:/mnt/test_mnt/melted_csv/iris-dataset.csv/part-00000-462f5c5d-eccf-4f1d-907a-10da4d4daa02-c000.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 保存したCSVの読み込み
# MAGIC 
# MAGIC 保存時に指定したファイル名はフォルダとなりますが、問題なくCSVを読み込むことができます。

# COMMAND ----------

read_df = spark.read.option("inferSchema",True).option("header", True).csv("dbfs:/mnt/test_mnt/melted_csv/iris-dataset.csv/")
display(read_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
