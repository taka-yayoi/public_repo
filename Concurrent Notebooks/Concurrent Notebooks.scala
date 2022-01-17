// Databricks notebook source
// MAGIC %md 
// MAGIC # ノートブックの同時実行
// MAGIC 
// MAGIC このノートブックでは、ノートブックを同時実行する様々な方法を説明します。子のノートブックの出力を表示するには、``Notebook job #nnn``のリンクをクリックします。
// MAGIC 
// MAGIC #### 参考資料
// MAGIC 
// MAGIC - [FutureとPromise · Scala研修テキスト](https://scala-text.github.io/scala_text/future-and-promise.html)
// MAGIC - [Scala の Future \- Qiita](https://qiita.com/4245Ryomt/items/63bcf0bf0bab3a99f6b5)

// COMMAND ----------

// MAGIC %md  
// MAGIC 注意: それぞれのノートブックの出力は互いに影響を及ぼしません。ファイルやテーブルに書き込む際には、出力パスをウィジェットを通じてパラメーター化する必要があります。これは同じクラスターで2人のユーザーが同じノートブックを実行するのと同じことです。彼らは状態を共有しません。

// COMMAND ----------

// MAGIC %md
// MAGIC ## ヘルパー関数の定義
// MAGIC 
// MAGIC ``parallel-notebooks``ノートブックは、ライブラリのように関数を定義します。

// COMMAND ----------

// MAGIC %run "./parallel-notebooks"

// COMMAND ----------

// MAGIC %md 
// MAGIC 残りのコマンドでは、いろいろな方法で``NotebookData``関数を呼び出します。

// COMMAND ----------

// MAGIC %md 
// MAGIC ## バージョン1: シーケンスを指定
// MAGIC 
// MAGIC 最初のバージョンでは、ノートブックのシーケンス、タイムアウト、パラメーターをの組み合わせを作成し、``parallel notebooks``ノートブックで定義されている``parallelNotebooks``関数に引き渡します。``parallelNotebooks``関数は、実行スレッド数を制限することでドライバーが過負荷になることを防ぎます。

// COMMAND ----------

import scala.concurrent.Await
import scala.concurrent.duration._
import scala.language.postfixOps

val notebooks = Seq(
  NotebookData("testing", 15),
  NotebookData("testing-2", 15, Map("Hello" -> "イエス")),
  NotebookData("testing-2", 15, Map("Hello" -> "あるいは")),
  NotebookData("testing-2", 15, Map("Hello" -> "たくさんのノートブックを")),
  NotebookData("testing-2", 1, Map("Hello" -> "並列で")) // 意図的にタイムアウトによる失敗を引き起こします
)

val res = parallelNotebooks(notebooks)

Await.result(res, 30 seconds) // これはブロック用の呼び出しです
res.value

// COMMAND ----------

// MAGIC %md
// MAGIC ## バージョン2: Futureのシーケンスを指定
// MAGIC 
// MAGIC 次のバージョンでは、個々の並列ノートブックを定義します。上と同じパラメーターを受け取りますが、それぞれで処理を行うため、処理されるべき結果のシーケンスを指定する必要があります。受け取る際には、Futureである必要があります。

// COMMAND ----------

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import scala.language.postfixOps
/*
注意: これは若干マニュアルの作業ですが、最初のバージョンと同じ結果を得ることができます。
*/
val n1 = parallelNotebook(NotebookData("testing", 15))
val n2 = parallelNotebook(NotebookData("testing-2", 15, Map("Hello" -> "イエス")))
val res = Future.sequence(List(n1, n2)) 

Await.result(res, 30 minutes) // これはブロック用の呼び出しです
res.value

// COMMAND ----------

// MAGIC %md
// MAGIC ## バージョン3: Futureを個別に指定
// MAGIC 
// MAGIC 最後のバージョンは最もマニュアルな方法です。戻されるべきFutureを個々に定義します。

// COMMAND ----------

val ctx = dbutils.notebook.getContext()
/*
注意: このバージョンでは、上のバージョンに組み込まれた構造を説明しています。
*/

val myHappyNotebookTown1 = Future {
  dbutils.notebook.setContext(ctx)
  println("Starting First --")
  val x = dbutils.notebook.run("testing", 15)
  println("Finished First -- ")
  x
}

val myHappyNotebookTown2 = Future {
  dbutils.notebook.setContext(ctx)
  println("Starting Second --")
  val x = dbutils.notebook.run("testing-2", 15, Map("Hello" -> "イエス"))
  println("Finished Second -- ")
  x
}

val res = Future.sequence(List(myHappyNotebookTown1, myHappyNotebookTown2))

Await.result(res, 30 minutes) // これはブロック用の呼び出しです
res.value

// COMMAND ----------

// MAGIC %md
// MAGIC それぞれのサンプルは、何が自動で行われるのかを除いて全く同じものとなります。これは全て同じ結果を得るために使用されます。
