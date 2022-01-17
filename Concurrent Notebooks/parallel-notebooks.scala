// Databricks notebook source
// MAGIC %md
// MAGIC 並列にノートブックを実行する関数を定義します。このノートブックはライブラリであるかのように関数を定義することができます。

// COMMAND ----------

import scala.concurrent.{Future, Await}
import scala.concurrent.duration._
import scala.util.control.NonFatal

// COMMAND ----------

case class NotebookData(path: String, timeout: Int, parameters: Map[String, String] = Map.empty[String, String])

def parallelNotebooks(notebooks: Seq[NotebookData]): Future[Seq[String]] = {
  import scala.concurrent.{Future, blocking, Await}
  import java.util.concurrent.Executors
  import scala.concurrent.ExecutionContext
  import com.databricks.WorkflowException

  val numNotebooksInParallel = 4 
  // あまりに多くのノートブックを並列実行しようとすると、一度にジョブをサブミットした際にドライバーがクラッシュする恐れがあります
  // このコードは並列実行するノートブックの数を制限します
  implicit val ec = ExecutionContext.fromExecutor(Executors.newFixedThreadPool(numNotebooksInParallel))
  val ctx = dbutils.notebook.getContext()
  
  Future.sequence(
    notebooks.map { notebook => 
      Future {
        dbutils.notebook.setContext(ctx)
        if (notebook.parameters.nonEmpty)
          dbutils.notebook.run(notebook.path, notebook.timeout, notebook.parameters)
        else
          dbutils.notebook.run(notebook.path, notebook.timeout)
      }
      .recover {
        case NonFatal(e) => s"ERROR: ${e.getMessage}"
      }
    }
  )
}

def parallelNotebook(notebook: NotebookData): Future[String] = {
  import scala.concurrent.{Future, blocking, Await}
  import java.util.concurrent.Executors
  import scala.concurrent.ExecutionContext.Implicits.global
  import com.databricks.WorkflowException

  val ctx = dbutils.notebook.getContext()
  // ここで用いる最もシンプルなインタフェースは、並列で多くのノートブックをサブミットすることに対する防御機構は持っていません
  Future {
    dbutils.notebook.setContext(ctx)
    
    if (notebook.parameters.nonEmpty)
      dbutils.notebook.run(notebook.path, notebook.timeout, notebook.parameters)
    else
      dbutils.notebook.run(notebook.path, notebook.timeout)
    
  }
  .recover {
    case NonFatal(e) => s"ERROR: ${e.getMessage}"
  }
}

