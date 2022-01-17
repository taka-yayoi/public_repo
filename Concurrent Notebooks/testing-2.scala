// Databricks notebook source
dbutils.widgets.text("Hello", "こんにちは")

// COMMAND ----------

import org.apache.spark.sql.functions._

// COMMAND ----------

val x = dbutils.widgets.get("Hello")
println(s"2つ目の作業を処理中: $x")

// COMMAND ----------

dbutils.notebook.exit("成功")
