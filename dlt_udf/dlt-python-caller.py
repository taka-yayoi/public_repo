# Databricks notebook source
import dlt
from pyspark.sql.functions import *
from pyspark.sql.types import *

# COMMAND ----------

@dlt.table
def raw_data():
  return spark.sql("SELECT id, makeItSquared(id) AS numSquared FROM RANGE(10)")

# COMMAND ----------

@dlt.table
@dlt.expect_or_drop("even_only", "passOnlyEven(numSquared) = True")
def squared_even():
  return (
    dlt.read("raw_data")
  )
