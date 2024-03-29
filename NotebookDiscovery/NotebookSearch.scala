// Databricks notebook source
// MAGIC %md ### NotebookSearch
// MAGIC Search/Filter the Notebook Discovery Index parquet file generated by NotebookIndex.  Since the index is a parquet file, any of the fields can be readily queried using standard operations such as filter and contains.
// MAGIC <br/>The Notebook Discovery Index parquet file has the following structure.
// MAGIC ```
// MAGIC  |-- nbLang: string (nullable = true)
// MAGIC  |-- nbName: string (nullable = true)
// MAGIC  |-- nbFolder: string (nullable = true)
// MAGIC  |-- nbOrigId: long (nullable = true)
// MAGIC  |-- nbUrl: string (nullable = true)
// MAGIC  |-- cPos: double (nullable = true)
// MAGIC  |-- cOrigId: long (nullable = true)
// MAGIC  |-- cDateTime: long (nullable = true)
// MAGIC  |-- cText: string (nullable = true)
// MAGIC  |-- cLang: string (nullable = true)
// MAGIC  |-- cUrl: string (nullable = true)
// MAGIC ```
// MAGIC with the following descriptions for each field:
// MAGIC - nbLang: notebook language associated with the notebook when it was created (scala, python, sql, or r)
// MAGIC - nbName: notebook name associated with the notebook when it was created
// MAGIC - nbFolder: notebook folders (if there are any parent folders for the notebook)
// MAGIC - nbOrigId: notebook original id (used for constructing links to the notebook)
// MAGIC - nbUrl: notebook url so it can quickly be followed
// MAGIC - cPos: command relative position in the notebook
// MAGIC - cOrigId: command original id  (used for constructing links to the command)
// MAGIC - cDateTime: command submitted datetime (last time the command was submitted)
// MAGIC - cText: command text (we don't include the results)
// MAGIC - cLang: command language.  Includes not only the base scala, python, sql, and R included for nbLang but also markdown (%md), run (%run), shell (%sh), and dbfs (%fs)
// MAGIC - cUrl: command url so it can quickly be followed

// COMMAND ----------

// MAGIC %md ##### imports

// COMMAND ----------

import org.apache.spark.storage.StorageLevel
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._

// COMMAND ----------

// MAGIC %md ##### read the Notebook Discovery Index file
// MAGIC Read in the Notebook Discovery Index parquet file (generated by NotebookIndex) that contains all of the commands for all of the notebooks.

// COMMAND ----------

// NotebookIndexRunで作成したインデックスparquetファイルのパス
val indexFilename = "/tmp/takaakiyayoidatabrickscom/index/"

val notebookCommands = spark.read.parquet(indexFilename)
                            .persist(StorageLevel.DISK_ONLY)

println(notebookCommands.count)

// COMMAND ----------

// MAGIC %md ##### helper functions for displaying search results
// MAGIC 
// MAGIC - displaySearchResults. Display a table of HTML results with links to the notebooks and commands.
// MAGIC   - num.  Number of results to display from the search results (default is false).
// MAGIC   - distinctNotebooks.  False to display results at the command level,  true to roll up results to the notebook level (default is false).

// COMMAND ----------

def searchResultRow(lang: String, folder: String, nbName: String, nbUrl: String, cUrl: String, distinctNotebooks: Boolean): String = { 
  
    // URL書き換え用
    var url_replace = false
    var nbUrl_r: String = ""
    var cUrl_r: String = ""

    if(url_replace) {
      nbUrl_r = nbUrl.replace("oregon.cloud.databricks.com", "e2-demo-west.cloud.databricks.com")
      cUrl_r = cUrl.replace("oregon.cloud.databricks.com", "e2-demo-west.cloud.databricks.com")
    } else {
      nbUrl_r = nbUrl
      cUrl_r = cUrl   
    }
  
    var result = "<tr>" +
                    "<td>" + lang+ "</td>" +
                    "<td>" + folder+ "</td>" +
                    "<td>" + "<a href='" + nbUrl_r + "'>" + nbName + "</a>" + "</td>" +
                    "<td>" + "<a href='" + cUrl_r + "'>" + "cmd" + "</a>" + "</td>" +
                 "</tr>"
    if (distinctNotebooks) {
      result = "<tr>" +
                    "<td>" + lang+ "</td>" +
                    "<td>" + folder+ "</td>" +
                    "<td>" + "<a href='" + nbUrl_r + "'>" + nbName + "</a>" + "</td>" +
                 "</tr>"
    }
    return result
}
  
val searchResultRowUDF = udf[String,String,String,String,String,String,Boolean](searchResultRow)

def displaySearchResults(searchResults: Dataset[Row], num: Integer = 50, distinctNotebooks:Boolean = false) = {
  var tableHeader = "<table><style>table, th, td {border: 1px solid black; border-collapse: collapse;} th, td {padding: 15px;} th {text-align: left;} table tr:nth-child(even) {background-color: lightgray;}</style><tr><th>language</th><th>folder</th><th>notebook</th><th>command</th></tr>"
  if (distinctNotebooks) {
     tableHeader = "<table><style>table, th, td {border: 1px solid black; border-collapse: collapse;} th, td {padding: 15px;} th {text-align: left;} table tr:nth-child(even) {background-color: lightgray;}</style><tr><th>language</th><th>folder</th><th>notebook</th></tr>"
  }
  val tableFooter = "</table>"
  var rows = searchResults.limit(num)
                          .withColumn("html",searchResultRowUDF($"nbLang",$"nbFolder",$"nbName",$"nbUrl",$"cUrl",lit(distinctNotebooks)))
                          .select($"html")
                          .collect
                          .map(rec => rec.getAs[String](0))
                          .mkString("")
  if (distinctNotebooks) {
    rows = searchResults.select($"nbLang",$"nbFolder",$"nbName",$"nbUrl")
                        .distinct
                        .limit(num)
                        .withColumn("html",searchResultRowUDF($"nbLang",$"nbFolder",$"nbName",$"nbUrl",lit(""),lit(distinctNotebooks)))
                        .select($"html")
                        .collect
                        .map(rec => rec.getAs[String](0))
                        .mkString("")
  }
  val results = tableHeader + rows + tableFooter
  displayHTML(results)
}
  

// COMMAND ----------

// MAGIC %md ##### search for %run commands 

// COMMAND ----------

displaySearchResults(notebookCommands.filter($"cLang" === "run"),num=10)

// COMMAND ----------

// MAGIC %md ##### search for %sh commands

// COMMAND ----------

displaySearchResults(notebookCommands.filter($"cLang" === "shell"),num=10)

// COMMAND ----------

// MAGIC %md ##### search for notebooks in folder '/Users/d.mcbeath@elsevier.com/' with a language of 'scala' and command text contains 'collect_list'

// COMMAND ----------

displaySearchResults(notebookCommands.filter($"cLang" === "scala")
                                     .filter($"cText".contains("collect_list"))
                                     .filter($"nbFolder".startsWith("/Users/takaaki.yayoi@databricks.com/")))

// COMMAND ----------

// MAGIC %md ##### search for scala notebooks that were submitted since July 30, 2021
// MAGIC - roll up results to the notebook level

// COMMAND ----------


import java.text.SimpleDateFormat
val myDate = "2021-07-30"
val dateFormat = new SimpleDateFormat("yyyy-MM-dd")
val date = dateFormat.parse(myDate)
val epoch = date.getTime()

displaySearchResults(notebookCommands.filter($"cLang" === "scala")
                                     .filter($"cDatetime" > epoch),
                     distinctNotebooks=true)


// COMMAND ----------

displaySearchResults(notebookCommands.filter($"cLang" === "python")
                                     .filter($"cText".contains("cross validation"))
                                     .filter($"nbFolder".startsWith("/Users/takaaki.yayoi@databricks.com/")))

// COMMAND ----------

var similarity_index = "/tmp/takaakiyayoidatabrickscom/index-similarity"

val similartiyCommands = spark.read.parquet(similarity_index)

//displaySearchResults(similartiyCommands, num=10)
display(similartiyCommands)

// COMMAND ----------


