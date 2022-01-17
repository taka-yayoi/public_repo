// Databricks notebook source
// MAGIC %md ### NotebookIndex
// MAGIC 
// MAGIC Create the Notebook Discovery Index parquet file for all of the notebook information needed to power NotebookSearch and NotebookSimilarity.  We first leverage the Databricks [workspace/list](https://docs.databricks.com/dev-tools/api/latest/workspace.html#list) API to crawl the directories and keep a list of all the notebooks we encounter.   We then use the Databricks [workspace/export](https://docs.databricks.com/dev-tools/api/latest/workspace.html#export) API to export the DBC (archive) for each of the notebooks we previously identified.  The number of concurrent requests to Databricks via the REST APIs can be controlled through the parallelization parameter.  Lastly, we extract the json record for the notebook from the archive, generate a CommandRecord structure record and write the Notebook Discovery Index parquet file.
// MAGIC 
// MAGIC Since every command in a notebook will have a record in the index file, it is easy to support search with command granularity.  Search results can then be rolled up to the notebook level if desired.  
// MAGIC The resulting Notebook Discovery Index parquet file will have the following structure.  
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

import scala.collection.mutable.ListBuffer

import java.net.URI
import java.util.Base64
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.ByteArrayInputStream

import org.apache.http.impl.client.HttpClients
import org.apache.http.impl.client.CloseableHttpClient
import org.apache.http.client.utils.URIBuilder
import org.apache.http.client.methods.HttpGet
import org.apache.http.client.methods.CloseableHttpResponse
import org.apache.http.HttpEntity

import org.apache.commons.io.IOUtils

import org.json.JSONArray
import org.json.JSONObject

// COMMAND ----------

// MAGIC %md ##### parameters

// COMMAND ----------

val folders = dbutils.widgets.get("folders")
val indexFilename = dbutils.widgets.get("indexFilename")
val overwrite = dbutils.widgets.get("overwrite")
val parallelization = dbutils.widgets.get("parallelization").toInt

// COMMAND ----------

// MAGIC %md ##### validate parameters

// COMMAND ----------

def fileExists(file: String): Boolean = {
  import java.io.FileNotFoundException
  var rc = true
  try {
    dbutils.fs.ls(file)
  } catch {
    case foo: FileNotFoundException => {rc = false}
  }
  rc
}

// verify overwrite parameter
if (overwrite.toLowerCase != "false" && overwrite.toLowerCase != "true") {
  dbutils.notebook.exit("FAILED.  Overwrite parameter '%s' must be either 'True' or 'False'.".format(overwrite))
}

// very index file does not already exist if overwrite is false
if (overwrite.toLowerCase == "false") {
  if (fileExists(indexFilename)) {
    dbutils.notebook.exit("FAILED.  Index file '%s' already exists.".format(indexFilename))
  }
}

// COMMAND ----------

// MAGIC %md ##### variables

// COMMAND ----------

val Array(scheme,domain) = dbutils.notebook.getContext().apiUrl.getOrElse("").split("://")
val bearer = dbutils.notebook.getContext().apiToken.getOrElse("")

// COMMAND ----------

// MAGIC %md ##### RetryUtils
// MAGIC Allow for a number of retries when sending requests to the Databricks REST APIs.  This is necessary because we can sometimes exceed a threshold which would ultimately fail the request.  In those situations, we would want to retry the request.  This code was modeled after [RetryUtils.scala](https://github.com/delta-io/delta-sharing/blob/main/spark/src/main/scala/io/delta/sharing/spark/util/RetryUtils.scala) which is available under [Apache 2.0 License](http://www.apache.org/licenses/LICENSE-2.0).

// COMMAND ----------

package org.elsevierlabs.NotebookDiscovery.util

object RetryUtils {

  import java.io.{InterruptedIOException, IOException}
  import scala.util.control.NonFatal

  def runWithExponentialBackoff[T](numRetries: Int)(func: => T): T = {
    var times = 0
    var sleepMs = 100
    while (true) {
      times += 1
      try {
        return func
      } catch {
        case NonFatal(e) if shouldRetry(e) && times <= numRetries =>
          println(s"Sleeping $sleepMs ms to retry because of error: ${e.getMessage}", e)
          Thread.sleep(sleepMs)
          sleepMs *= 2
      }
    }
    throw new IllegalStateException("Should not happen")
  }

  def shouldRetry(t: Throwable): Boolean = {
    t match {
      case e: UnexpectedHttpStatus =>
        if (e.statusCode == 429) { // Too Many Requests
          true
        } else if (e.statusCode >= 500 && e.statusCode < 600) { // Internal Error
          true
        } else {
          println(s"Not retrying because of statusCode ${e.statusCode} and error: ${e.getMessage}")
          false
        }
      case _: InterruptedException => false
      case _: InterruptedIOException => false
      case _: IOException => true
      case _ => false
    }
  }
}

class UnexpectedHttpStatus(message: String, val statusCode: Int) extends IllegalStateException(message)

// COMMAND ----------

// MAGIC %md ##### getDBRestResponse
// MAGIC Helper function to process a Databricks REST API request.

// COMMAND ----------

def getDBRestResponse(httpGet: HttpGet): JSONObject =  {
  
  import org.elsevierlabs.NotebookDiscovery.util._
  
  // retry is currently hard-coded to 3 attempts
  RetryUtils.runWithExponentialBackoff(3) {
    // create the http client
    val client: CloseableHttpClient = HttpClients.createDefault()
  
    // response object
    var json = new JSONObject("{}")
  
    // execute the request
    val response: CloseableHttpResponse = client.execute(httpGet)
    val statusCode: Int = response.getStatusLine().getStatusCode()
  
    // process the response
    val entity: HttpEntity = response.getEntity()
    val content= IOUtils.toString(entity.getContent(),"UTF-8")
    if (statusCode == 200) {
      json = new JSONObject(content)
    } else if (statusCode == 404) {
      // Don't retry and return an empty json response
    } else if (statusCode == 400 && content.contains("{\"error\":\"DatabricksServiceException: BAD_REQUEST")) {
      println("Notebook contents too large and will be ignored for Path: '%s' Query '%s'".format(httpGet.getURI().getPath(),httpGet.getURI().getQuery()))
      // Don't retry and return an empty json response
    } else {
      println("Path: '%s' Query '%s'".format(httpGet.getURI().getPath(),httpGet.getURI().getQuery()))
      println(json.toString(4))
      client.close()
      throw new UnexpectedHttpStatus(s"HTTP request failed with status: $statusCode $content", statusCode)
    }
  
    client.close()
    return json
  }
}

// COMMAND ----------

// MAGIC %md ##### walkPath
// MAGIC Recursive function to walk the specified workspace path collecting notebooks in the path or in one of it's children directories. 
// MAGIC <br/>Libraries and other repos are currently ignored.

// COMMAND ----------

def walkPath(path: String, notebooks: ListBuffer[String]): ListBuffer[String] = {
  
  // build the url
  val uri: URI = new URIBuilder()
			         .setScheme(scheme)
			         .setHost(domain)
			         .setPath("/api/2.0/workspace/list")
			         .setParameter("path", path)
			         .build()
  
  // create the http get request
  val httpGet: HttpGet = new HttpGet(uri)
  
  // set the authorization header
  httpGet.setHeader("Authorization", "Bearer " + bearer)
  
  // get the response
  val json: JSONObject = getDBRestResponse(httpGet)
  
  // process the json response
  if (json.has("objects")) {
    val objects: JSONArray = json.getJSONArray("objects")
    for (i <- 0 to objects.length - 1) {
	  val obj: JSONObject = objects.getJSONObject(i)
      val notebookPath: String = obj.getString("path")
	  if (obj.getString("object_type").compareTo("DIRECTORY") == 0) {
		walkPath(notebookPath,notebooks)
	  } else if (obj.getString("object_type").compareTo("NOTEBOOK") == 0) {
		notebooks += ((notebookPath))
	  } else {
		// ignore as it is a Library or Repo
      }
    }
  }
  
  // return the notebooks
  return notebooks
  
}

// COMMAND ----------

// MAGIC %md ##### exportNotebook
// MAGIC Request the DBC archive format for the specified notebook.  There should only be a single file in the returned zipped archive.  Once unzipped, a json string representing the notebook contents will be returned.

// COMMAND ----------

def exportNotebook(notebookPath: String): String = {
  
  //println("processing: " + notebookPath)
  
  // init for an empty notebook
  var notebook: String = "{}"
  
  // build the url
  val uri: URI = new URIBuilder()
			         .setScheme(scheme)
			         .setHost(domain)
			         .setPath("/api/2.0/workspace/export")
			         .setParameter("path", notebookPath)
                     .setParameter("format", "DBC")
			         .build()
  
  // create the http get request
  val httpGet: HttpGet = new HttpGet(uri)
  
  // set the authorization header
  httpGet.setHeader("Authorization", "Bearer " + bearer)
  
  // get the response
  val json: JSONObject = getDBRestResponse(httpGet)
  
  // process the json response
  if (json.has("content")) {
	// response is base64 encoded
	val encodedContent: String = json.getString("content")
	val decodedBytes: Array[Byte]  = Base64.getDecoder().decode(encodedContent)
					
	// DBC archive is zipped
	val zis: ZipInputStream = new ZipInputStream(new ByteArrayInputStream(decodedBytes))
	var zipEntry: ZipEntry = zis.getNextEntry()
      
    // there should be only 1 entry as we are getting only one notebook
	if (zipEntry != null) {
	  val is: InputStreamReader = new InputStreamReader(zis)
      val sb: StringBuilder =new StringBuilder()
	  val br: BufferedReader = new BufferedReader(is)
	  var read: String = br.readLine()
	  while(read != null) {
		sb.append(read)
		read = br.readLine()
	  }
	  notebook = sb.toString()
	}
	zis.closeEntry()
	zis.close()
  } else {
    println("*** no content returned for notebook while processing notebook: " + notebookPath)
  }
  
  // return the notebook
  return notebook
  //return "{'test':1}"
}

// COMMAND ----------

// MAGIC %md ##### createCommandRecords
// MAGIC 
// MAGIC Create a CommandRecord for every command in the notebook. 

// COMMAND ----------

case class CommandRecord(nbLang: String,      // notebook language
                         nbName: String,      // notebook name
                         nbFolder: String,    // notebook folders
                         nbOrigId: Long,      // notebook original id
                         nbUrl: String,       // notebook url
                         cPos: Double,        // command relative position in the notebook
                         cOrigId: Long,       // command original id
                         cDateTime: Long,     // command submitted datetime
                         cText: String,       // command text
                         cLang: String,       // command language
                         cUrl: String)        // command url

def createCommandRecords(notebook: String, notebookContent: String): ListBuffer[CommandRecord] = {
  
  // create a buffer for the command records
  val commandRecords: ListBuffer[CommandRecord] = ListBuffer[CommandRecord]()
  
  // parse the notebook json  
  val json: JSONObject = new JSONObject(notebookContent)
  
  // make sure we don't have an empty json file (in other words export had problems retrieving the notebook)
  if (json.has("language")) {
    
    // get the notebook language (value specified when the notebook was created)
    var nbLang: String = json.getString("language")

    // get the notebook name (value specified when the notebook was created or renamed)
    val nbName: String = json.getString("name")
  
    // get the parent folders for the notebook if there are any
    var nbFolder: String = notebook
    nbFolder = nbFolder.substring(0,nbFolder.lastIndexOf(nbName))
  
    // get the notebook original id and build a url for the notebook so it can be easily retrieved
    val nbOrigId: Long = json.getLong("origId")
    val nbUrl = scheme + "://" + domain + "/#notebook/" + nbOrigId
  
    // process the commands (cells)
    if (json.has("commands")) {
      val commands: JSONArray = json.getJSONArray("commands")
      for (i <- 0 to commands.length - 1) {
        val command: JSONObject = commands.getJSONObject(i)
      
        // get the command original id so we can later build a url to the command
        val cOrigId: Long = command.getLong("origId")
      
        // get the submitted datetime for the command
        val cDateTime: Long = command.getLong("submitTime")
      
        // get the command text 
        val cText: String = command.getString("command")
      
        // get the command relative position in the notebook
        val cPos: Double = command.getDouble("position")
      
        // get the command language (it can be different than the notebook language).  Specifically look at the magic commands.
        var cLang: String = ""
        cText.trim().split("\\s+")(0) match {
          case "%md"  => cLang = "markdown"
          case "%scala"  => cLang = "scala"
          case "%python"  => cLang = "python"
          case "%sql"  => cLang = "sql"
          case "%r"  => cLang = "r"
          case "%run"  => cLang = "run"
          case "%sh"  => cLang = "shell"
          case "%fs"  => cLang = "dbfs"
          // default to the notebook language
          case _  => cLang = nbLang
        }
  
        // build a url for the command
        val cUrl: String = scheme + "://" + domain + "/#notebook/" + nbOrigId + "/command/" + cOrigId
      
        // create a record for the command and add to the buffer for the notebook
        commandRecords += (CommandRecord(nbLang, nbName, nbFolder, nbOrigId, nbUrl, cPos, cOrigId, cDateTime, cText, cLang, cUrl))
      }
    }  
  }
  
  // return the command records for the notebook
  return commandRecords
  
}

// COMMAND ----------

// MAGIC %md ##### Mainline
// MAGIC - Process the folders, create the CommandRecords, and write the Notebook Discovery Index parquet file

// COMMAND ----------

val foldersArr: Array[String] = folders.split(",")
val commands = sc.parallelize(foldersArr,parallelization)
               .mapPartitions(iter => {
                 var commandRecords: ListBuffer[CommandRecord] = ListBuffer[CommandRecord]()
                 iter.foreach(rec => {
                   val folder = rec.trim()
                   var notebooks: ListBuffer[String] = ListBuffer[String]()
                   notebooks = walkPath(folder, notebooks)
                   for (notebook <- notebooks) {
                     val notebookContent: String = exportNotebook(notebook)
                     commandRecords = commandRecords ++ createCommandRecords(notebook, notebookContent)
                   }
                 })
                 commandRecords.iterator
               })
               .toDS

if (overwrite.toLowerCase == "true") {
  commands.write.mode("overwrite").parquet(indexFilename)
} else {
  commands.write.parquet(indexFilename)
}

dbutils.notebook.exit("OK")
