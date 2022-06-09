# Databricks notebook source
# MAGIC %md
# MAGIC ## learning-spark/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/learning-spark/data-001/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/learning-spark/README.md"))

# COMMAND ----------

df = spark.read.format("csv").load("dbfs:/databricks-datasets/learning-spark/data-001/favourite_animals.csv")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## learning-spark-v2/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/learning-spark-v2/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/learning-spark-v2/README/mnm_dataset.md"))

# COMMAND ----------

df = spark.read.format("csv").option("header", True).load("dbfs:/databricks-datasets/learning-spark-v2/mnm_dataset.csv")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## lending-club-loan-stats/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/lending-club-loan-stats/

# COMMAND ----------

df = spark.read.format("csv").option("header", True).load("dbfs:/databricks-datasets/lending-club-loan-stats/LoanStats_2018Q2.csv")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## med-images/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/med-images/camelyon16/

# COMMAND ----------

# MAGIC %pip install openslide-python

# COMMAND ----------

WSI_TIF_PATH = "/databricks-datasets/med-images/camelyon16/"

import numpy as np
import openslide
import matplotlib.pyplot as plt

f, axarr = plt.subplots(1,4,sharey=True)
i=0
for pid in ["normal_034","normal_036","tumor_044", "tumor_045"]:
  path = '/dbfs/%s/%s.tif' %(WSI_TIF_PATH,pid)
  slide = openslide.OpenSlide(path)
  axarr[i].imshow(slide.get_thumbnail(np.array(slide.dimensions)//50))
  axarr[i].set_title(pid)
  i+=1
display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## media/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/media/rtb/raw_incoming_bid_stream/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/media/rtb/raw_incoming_bid_stream/README.md"))

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/media/rtb/raw_incoming_bid_stream/bidRequestSample.txt"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## mnist-digits/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/mnist-digits/data-001/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/mnist-digits/README.md"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## news20.binary/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/news20.binary/data-001/training/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/news20.binary/README.md"))

# COMMAND ----------

df = spark.read.format("parquet").load("dbfs:/databricks-datasets/news20.binary/data-001/training/")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## nyctaxi/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/nyctaxi/sample/json/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/nyctaxi/readme_nyctaxi.txt"))

# COMMAND ----------

df = spark.read.format("json").load("dbfs:/databricks-datasets/nyctaxi/sample/json/")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## nyctaxi-with-zipcodes/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/nyctaxi-with-zipcodes/subsampled/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/nyctaxi-with-zipcodes/subsampled/nyc-zips-dataset-readme.txt"))

# COMMAND ----------

df = spark.read.format("delta").load("dbfs:/databricks-datasets/nyctaxi-with-zipcodes/subsampled/")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## online_retail/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/online_retail/data-001/

# COMMAND ----------

df = spark.read.format("csv").option("header", True).load("dbfs:/databricks-datasets/online_retail/data-001/data.csv")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## overlap-join/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/overlap-join/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/overlap-join"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## power-plant/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/power-plant/data/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/power-plant/README.md"))

# COMMAND ----------

df = spark.read.format("csv").option("header", True).option("delimiter", "\t").load("dbfs:/databricks-datasets/power-plant/data/Sheet1.tsv")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## retail-org/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/retail-org/active_promotions/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/retail-org/README.md"))

# COMMAND ----------

df = spark.read.format("parquet").load("dbfs:/databricks-datasets/retail-org/active_promotions/active_promotions.parquet")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## rwe/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/rwe/ehr/csv/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/rwe/ehr/csv/README.txt"))

# COMMAND ----------

df = spark.read.format("csv").option("header", True).option("delimiter", ",").load("dbfs:/databricks-datasets/rwe/ehr/csv/allergies.csv")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## sai-summit-2019-sf/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/sai-summit-2019-sf/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/sai-summit-2019-sf/README.md"))

# COMMAND ----------

df = spark.read.format("csv").option("header", True).option("delimiter", ",").load("dbfs:/databricks-datasets/sai-summit-2019-sf/fire-calls.csv")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## sample_logs/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/sample_logs/

# COMMAND ----------

df = spark.read.format("csv").load("/databricks-datasets/sample_logs/")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## samples/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/samples/data/mllib/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/samples/data/mllib/gmm_data.txt"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## sfo_customer_survey/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/sfo_customer_survey/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/sfo_customer_survey/README.MD"))

# COMMAND ----------

df = spark.read.format("csv").option("header", True).load("dbfs:/databricks-datasets/sfo_customer_survey/2013_SFO_Customer_Survey.csv")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## sms_spam_collection/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/sms_spam_collection/data-001/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/sms_spam_collection/README.md"))

# COMMAND ----------

df = spark.read.format("csv").option("header", False).load("dbfs:/databricks-datasets/sms_spam_collection/data-001/smsData.csv")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## songs/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/songs/data-001/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/songs/README.md"))

# COMMAND ----------

df = spark.read.format("csv").option("header", False).option("delimiter", "\t").load("dbfs:/databricks-datasets/songs/data-001/part-00000")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## structured-streaming/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/structured-streaming/events/

# COMMAND ----------

df = spark.read.format("json").load("dbfs:/databricks-datasets/structured-streaming/events/file-0.json")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## timeseries/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/timeseries/Fires/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/timeseries/Fires/SFFire_readme.md"))

# COMMAND ----------

df = spark.read.format("csv").option("header", True).option("delimiter", ",").load("dbfs:/databricks-datasets/timeseries/Fires/Fire_Department_Calls_for_Service.csv")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## tpch/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/tpch/data-001/customer/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/tpch/README.md"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## warmup/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/warmup/

# COMMAND ----------

# MAGIC %md
# MAGIC ## weather/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/weather/high_temps

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/weather/README.weather_history.md"))

# COMMAND ----------

df = spark.read.format("csv").option("header", True).option("delimiter", ",").load("dbfs:/databricks-datasets/weather/high_temps")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## wiki/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/wiki/

# COMMAND ----------

df = spark.read.format("csv").load("dbfs:/databricks-datasets/wiki/")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## wikipedia-datasets/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/wikipedia-datasets/data-001/clickstream/

# COMMAND ----------

df = spark.read.format("json").load("dbfs:/databricks-datasets/wikipedia-datasets/data-001/clickstream/raw-uncompressed-json/")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## wine-quality/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/wine-quality/

# COMMAND ----------

print(dbutils.fs.head("dbfs:/databricks-datasets/wine-quality/README.md"))

# COMMAND ----------

df = spark.read.format("csv").option("header", True).option("delimiter", ";").load("dbfs:/databricks-datasets/wine-quality/winequality-red.csv")
display(df)

# COMMAND ----------


