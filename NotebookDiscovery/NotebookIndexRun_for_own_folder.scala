// Databricks notebook source
// MAGIC %md ### NotebookIndexRun
// MAGIC 
// MAGIC こちらは自分のユーザーフォルダーのみのインデックスを作成するバージョンです。フォルダーのリストを作成するので処理が並列化されます。
// MAGIC 
// MAGIC This notebook executes <a href="$./NotebookIndex">NotebookIndex</a>.
// MAGIC 
// MAGIC   - Arguments include the following:
// MAGIC     - path. The location of the notebook to run (in this case NotebookIndex).  It is assumed this notebook is in the same folder as it's run notebook.
// MAGIC     - timeoutSeconds.  Number of seconds to allow for the notebook execution (currently set at 2 hours).
// MAGIC     - folders.  Comma separated list of folders to crawl when generating the Notebook Discovery Index.
// MAGIC     - indexFilename. Where to place the generated  Notebook Discovery Index parquet file.  Typically an area mounted on Databricks.
// MAGIC     - overwrite.  Whether the Notebook Discovery Index parquet file should be overwritten if it already exists.  'False' will not overwrite. 'True' will overwrite.
// MAGIC     - parallelization. The number of concurrent requests that will be sent to Databricks through the REST APIs when creating the index (currently set at 8).
// MAGIC     

// COMMAND ----------

// MAGIC %md
// MAGIC Prepare Databricks CLI

// COMMAND ----------

// MAGIC %python
// MAGIC token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
// MAGIC host_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("browserHostName").get()
// MAGIC username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
// MAGIC 
// MAGIC dbutils.fs.put("file:///root/.databrickscfg",f"[DEFAULT]\nhost=https://{host_name}\ntoken = "+token,overwrite=True)

// COMMAND ----------

// MAGIC %python
// MAGIC import re
// MAGIC 
// MAGIC # keep alpha-numeric only
// MAGIC username_short = re.sub('[^A-Za-z0-9]+', '', username).lower()

// COMMAND ----------

// MAGIC %md
// MAGIC Get my folders

// COMMAND ----------

// MAGIC %python
// MAGIC import os
// MAGIC os.system(f"databricks workspace ls /Users/{username}/ --absolute > /tmp/folder_list")

// COMMAND ----------

// MAGIC %sh
// MAGIC cat /tmp/folder_list

// COMMAND ----------

// MAGIC %python
// MAGIC dbutils.fs.cp("file:/tmp/folder_list", f"/tmp/{username_short}/")

// COMMAND ----------

// MAGIC %python
// MAGIC sparkDF = spark.read.format("csv").option("header", "false").option("inferSchema", "true").load(f"/tmp/{username_short}/folder_list")
// MAGIC 
// MAGIC display(sparkDF)
// MAGIC 
// MAGIC sparkDF.registerTempTable("df")

// COMMAND ----------

// MAGIC %md
// MAGIC [python \- How to turn a pandas dataframe row into a comma separated string \- Stack Overflow](https://stackoverflow.com/questions/37877708/how-to-turn-a-pandas-dataframe-row-into-a-comma-separated-string)

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC df = spark.sql("SELECT * FROM df ORDER BY _c0")
// MAGIC pdf = df.toPandas()
// MAGIC x = pdf.to_string(header=False,
// MAGIC                   index=False,
// MAGIC                   index_names=False).split('\n')
// MAGIC 
// MAGIC folders_list = [','.join(ele.split(",")) for ele in x]
// MAGIC #print(folders_list)
// MAGIC stripped_folders_list = [s.strip() for s in folders_list]
// MAGIC folders_str = str(stripped_folders_list)
// MAGIC folders_str = folders_str.replace("'", "").replace("[", "").replace("]", "")
// MAGIC print(folders_str)

// COMMAND ----------

// MAGIC %python
// MAGIC rc = dbutils.notebook.run("NotebookIndex",\
// MAGIC                               72000,\
// MAGIC                               {"folders" : f"{folders_str}",
// MAGIC                                             "indexFilename" : f"/tmp/{username_short}/index",\
// MAGIC                                             "overwrite" : "True",\
// MAGIC                                             "parallelization" : "8"})
// MAGIC 
// MAGIC print(rc)

// COMMAND ----------

/*
val rc = dbutils.notebook.run(path="NotebookIndex",
                              timeoutSeconds=72000,
                              arguments=Map("folders" -> """
                                                            /Users/takaaki.yayoi@databricks.com/
                                                            """,
                                            "indexFilename" -> "/tmp/takaakiyayoidatabrickscom/index",
                                            "overwrite" -> "True",
                                            "parallelization" -> "32"))

println(rc)
*/

// COMMAND ----------


