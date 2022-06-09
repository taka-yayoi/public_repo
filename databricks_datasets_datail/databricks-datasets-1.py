# Databricks notebook source
# MAGIC %fs
# MAGIC ls /databricks-datasets/

# COMMAND ----------

# MAGIC %md
# MAGIC ## COVID

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /databricks-datasets/COVID

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/COVID/CORD-19/2021-03-28/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/COVID/CORD-19/2021-03-28/metadata.readme"))

# COMMAND ----------

df = spark.read.option("header", True).csv("dbfs:/databricks-datasets/COVID/CORD-19/2021-03-28/metadata.csv")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## RDatasets

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/Rdatasets/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/Rdatasets/README.md"))

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/Rdatasets/data-001/csv/ggplot2/

# COMMAND ----------

df = spark.read.option("header", True).csv("dbfs:/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## adult

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/adult/README.md"))

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/adult/adult.data

# COMMAND ----------

df = spark.read.option("header", True).csv("dbfs:/databricks-datasets/adult/adult.data")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## airlines

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/airlines/part-00000

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/airlines/README.md"))

# COMMAND ----------

df = spark.read.option("header", True).csv("dbfs:/databricks-datasets/airlines/part-00000")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## amazon

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/amazon/data20K/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/amazon/README.md"))

# COMMAND ----------

df = spark.read.format("parquet").option("header", True).load("dbfs:/databricks-datasets/amazon/data20K/")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## asa

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/asa/airlines/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/asa/readme.txt"))

# COMMAND ----------

df = spark.read.option("header", True).csv("dbfs:/databricks-datasets/asa/airlines/1987.csv")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## atlas_higgs

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/atlas_higgs/

# COMMAND ----------

df = spark.read.option("header", True).csv("dbfs:/databricks-datasets/atlas_higgs/atlas_higgs.csv")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## bikeSharing

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/bikeSharing/data-001/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/bikeSharing/README.md"))

# COMMAND ----------

df = spark.read.option("header", True).csv("dbfs:/databricks-datasets/bikeSharing/data-001/day.csv")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## cctvVideos

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/cctvVideos/train_images/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/cctvVideos/readme.md"))

# COMMAND ----------

df = spark.read.format("image").load("dbfs:/databricks-datasets/cctvVideos/train_images/")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## credit-card-fraud/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/credit-card-fraud/data/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/credit-card-fraud/description.txt"))

# COMMAND ----------

df = spark.read.format("parquet").option("header", True).load("dbfs:/databricks-datasets/credit-card-fraud/data/")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## cs100

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/cs100/lab2/data-001/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/cs100/lab1/data-001/shakespeare.txt"))

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/cs100/lab2/data-001/apache.access.log.PROJECT"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## cs110x

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/cs110x/ml-1m/data-001/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/cs110x/ml-1m/data-001/README"))

# COMMAND ----------

df = spark.read.option("header", False).option("delimiter", "::").csv("dbfs:/databricks-datasets/cs110x/ml-1m/data-001/movies.dat")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## cs190/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/cs190/data-001/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/cs190/data-001/millionsong.txt"))

# COMMAND ----------

df = spark.read.option("header", False).csv("dbfs:/databricks-datasets/cs190/data-001/millionsong.txt")
display(df)

# COMMAND ----------

df = spark.read.option("header", False).option("delimiter", " ").csv("dbfs:/databricks-datasets/cs190/data-001/neuro.txt")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## data.gov

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/data.gov/irs_zip_code_data/data-001/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/data.gov/README.md"))

# COMMAND ----------

df = spark.read.option("header", True).csv("dbfs:/databricks-datasets/data.gov/irs_zip_code_data/data-001/2013_soi_zipcode_agi.csv")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## definitive-guide

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/definitive-guide/data/activity-data/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/definitive-guide/README.md"))

# COMMAND ----------

df = spark.read.format("json").load("dbfs:/databricks-datasets/definitive-guide/data/activity-data/")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## delta-sharing/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/delta-sharing/samples/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/delta-sharing/samples/README.md"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## flights/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/flights/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/flights/README.md"))

# COMMAND ----------

df = spark.read.format("csv").option("header", True).load("dbfs:/databricks-datasets/flights/departuredelays.csv")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## flower_photos/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/flower_photos/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/flower_photos/README.md"))

# COMMAND ----------

df = spark.read.format("image").load("dbfs:/databricks-datasets/flower_photos/daisy/")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## flowers/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/flowers/delta/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/flowers/README.md"))

# COMMAND ----------

df = spark.read.format("delta").load("dbfs:/databricks-datasets/flowers/delta/")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## genomics/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/genomics/1000G/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/genomics/1000G/readme.md"))

# COMMAND ----------

df = spark.read.format("parquet").load("dbfs:/databricks-datasets/genomics/1000G/dbgenomics.data/")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## hail/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/hail/data-001/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/hail/data-001/1kg_annotations.txt"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## identifying-campaign-effectiveness/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/identifying-campaign-effectiveness/subway_foot_traffic/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/identifying-campaign-effectiveness/README.md"))

# COMMAND ----------

df = spark.read.format("csv").option("header", True).load("dbfs:/databricks-datasets/identifying-campaign-effectiveness/subway_foot_traffic/foot_traffic.csv")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## iot/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/iot/

# COMMAND ----------

df = spark.read.format("json").load("dbfs:/databricks-datasets/iot/iot_devices.json")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## iot-stream/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/iot-stream/data-device/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/iot-stream/README.md"))

# COMMAND ----------

df = spark.read.format("json").load("dbfs:/databricks-datasets/iot-stream/data-device/")
display(df)

# COMMAND ----------


