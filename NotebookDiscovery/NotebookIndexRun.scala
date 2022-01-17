// Databricks notebook source
// MAGIC %md ### NotebookIndexRun
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

val rc = dbutils.notebook.run(path="NotebookIndex",
                              timeoutSeconds=72000,
                              arguments=Map("folders" -> """
                                                            /Users/takaaki.yayoi@databricks.com/
                                                            """,
                                            "indexFilename" -> "/tmp/takaakiyayoidatabrickscom/index",
                                            "overwrite" -> "True",
                                            "parallelization" -> "32"))

println(rc)

// COMMAND ----------


