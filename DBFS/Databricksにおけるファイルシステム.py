# Databricks notebook source
# MAGIC %md
# MAGIC # はじめに
# MAGIC 
# MAGIC Databricksでファイルを取り扱う際には、Databricks File System(DBFS)を理解する必要があります。
# MAGIC 
# MAGIC 本ノートブックでは、DBFSの概要をご説明するとともに、具体的な使用例をデモします。
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2021/02/03</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>クラスター</td><td>7.4ML</td></tr>
# MAGIC </table>
# MAGIC <img style="margin-top:25px;" src="https://sajpstorage.blob.core.windows.net/workshop20210205/databricks-logo-small-new.png" width="140">

# COMMAND ----------

# MAGIC %md 
# MAGIC # Databricks File System(DBFS)
# MAGIC 
# MAGIC Databricks File System(DBFS)はDatabricksのワークスペースにマウントされる分散ファイルシステムです。Databricksクラスターから利用することができます。DBFSはクラウドのオブジェクトストレージを抽象化するものであり、以下のメリットをもたらします：<br><br>
# MAGIC 
# MAGIC - オブジェクトストレージ(S3/Azure Blob Storageなど)追加の認証情報なしにオブジェクトストレージにアクセスすることができる
# MAGIC - ストレージURLではなく、ディレクトリ、ファイルの文法に従ってファイルにアクセスできる
# MAGIC - ファイルはオブジェクトストレージで永続化されるので、クラスタが削除されてもデータが残る
# MAGIC 
# MAGIC 詳細はこちらを参照ください。
# MAGIC 
# MAGIC [Databricks File System (DBFS) — Databricks Documentation](https://docs.databricks.com/data/databricks-file-system.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## DBFS root
# MAGIC 
# MAGIC DBFSにおけるデフォルトの場所は「DBFS root」と呼びます。DBFS rootにはいくつかのデータが格納されています。<br><br>
# MAGIC 
# MAGIC - /FileStore: インポートされたデータファイル、生成されたグラフプロット、アップロードされたライブラリが格納されます。[詳細はこちら](https://docs.databricks.com/data/filestore.html)
# MAGIC - /databricks-datasets: サンプルのデータセットが格納されます。[詳細はこちら](https://docs.databricks.com/data/databricks-datasets.html)
# MAGIC - /databricks-results: クエリ結果の全データをダウンロードする際に生成されるファイルが格納されます。
# MAGIC - /databricks/init: クラスタノードのinit scriptが格納されます。
# MAGIC - /user/hive/warehouse: Databricksで管理するHiveテーブルのメタデータ及びテーブルデータが格納されます。
# MAGIC 
# MAGIC **注意**
# MAGIC マウントポイント`/mnt`に書き込まれるデータはDBFS rootの外となります。DBFS rootに書き込み権限があったとしても、マウントしたオブジェクトストレージにデータを書き込むことをお勧めします。

# COMMAND ----------

# MAGIC %md 
# MAGIC ## UIからDBFSにアクセス
# MAGIC 
# MAGIC 1. 画面左のDataアイコン![](https://sajpstorage.blob.core.windows.net/workshop20210205/data-icon.png)をクリックします。
# MAGIC 2. 画面上部の「DBFS」ボタンをクリックすることでDBFSの階層構造を参照できます。 <br>
# MAGIC **注意** 管理者の方によって「DBFS browser」が有効になっていることを確認ください。
# MAGIC ![](https://sajpstorage.blob.core.windows.net/workshop20210205/browse.png)
# MAGIC 
# MAGIC この他にも、CLI、DBFS API(REST)、Databricksファイルシステムユーティリティ、Spark API、ローカルファイルAPIからもDBFSにアクセスできます。

# COMMAND ----------

# MAGIC %md ## DBFSとローカルドライバーノードのパス
# MAGIC 
# MAGIC Databricksでファイルを操作する際には、DBFSにアクセスしているのか、ローカルのクラスタードライバーノードのファイルシステムにアクセスしているのかを意識する必要があります。
# MAGIC 
# MAGIC ノートブックからファイルシステムにアクセスする際には、`%fs`、`%sh`といったマジックコマンド、Databrikcsファイルシステムユーティリティ`dbutils.fs`などを使用します。
# MAGIC 
# MAGIC APIやコマンドによって、パスを指定した際、DBFSを参照するのか、ローカルファイルシステムを参照するのかのデフォルトの挙動が異なりますので注意ください。
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>コマンド</th><th>デフォルト</th><th>DBFSへのアクセス</th><th>ローカルファイルシステムへのアクセス</th></tr>
# MAGIC   <tr><td>%fs</td><td>DBFS root</td><td></td><td>パスの先頭に`file:/`を追加</td></tr>
# MAGIC   <tr><td>%sh</td><td>ローカルドライバノード</td><td>パスの先頭に`/dbfs`を追加</td><td></td></tr>
# MAGIC   <tr><td>dbutils.fs</td><td>DBFS root</td><td></td><td>パスの先頭に`file:/`を追加</td></tr>
# MAGIC   <tr><td>pythonのos.コマンド</td><td>ローカルドライバノード</td><td>パスの先頭に`/dbfs`を追加</td><td></td></tr>
# MAGIC   
# MAGIC </table>

# COMMAND ----------

# DBTITLE 1,DBFS rootを参照
# MAGIC %fs ls /tmp

# COMMAND ----------

# MAGIC %sh ls /dbfs/tmp

# COMMAND ----------

# DBTITLE 1,ドライバのローカルファイルシステムを参照
# MAGIC %fs ls file:/tmp

# COMMAND ----------

# MAGIC %sh ls /tmp

# COMMAND ----------

# MAGIC %md 
# MAGIC # デモ
# MAGIC 
# MAGIC FileStoreはファイルを保存したファイルをブラウザから直接参照できる特別なフォルダです。以下のような使い方が可能です。<br><br>
# MAGIC 
# MAGIC 1. HTMLやJavaScriptを保存してブラウザから直接アクセスする。アクセスする際には`DisplayHTML`を使う。
# MAGIC 2. 出力結果を保存してローカルのデスクトップにファイルを保存する。

# COMMAND ----------

# MAGIC %md 
# MAGIC ## `displayHTML()`で使うJavaScriptライブラリをダウンロードし、一旦ドライバのローカルディスクに保存します。

# COMMAND ----------

# MAGIC %scala
# MAGIC import sys.process._

# COMMAND ----------

# MAGIC %scala
# MAGIC "sudo apt-get -y install wget" !!

# COMMAND ----------

# MAGIC %scala
# MAGIC "wget -P /tmp http://d3js.org/d3.v3.min.js" !!

# COMMAND ----------

# MAGIC %md 
# MAGIC ## `file:/tmp`のファイル一覧を表示し、ローカルディスクにファイルが保存されたことを確認します。

# COMMAND ----------

# MAGIC %scala
# MAGIC display(dbutils.fs.ls("file:/tmp/d3.v3.min.js"))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## ローカルの`/tmp`に保存されているファイルをブラウザから直接参照できるように、`/FileStore/customjs`にコピーします

# COMMAND ----------

# MAGIC %scala
# MAGIC dbutils.fs.mkdirs("/FileStore/customjs")
# MAGIC dbutils.fs.cp("file:/tmp/d3.v3.min.js", "/FileStore/customjs/d3.v3.min.js")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## `/FileStore/customjs`のファイル一覧を表示し、ファイルがコピーされたことを確認します

# COMMAND ----------

# MAGIC %scala
# MAGIC display(dbutils.fs.ls("/FileStore/customjs"))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 保存したJavaScriptライブラリをブラウザから参照する際には、`/files`から参照することになるので、パスは`/files/customjs/d3.v3.min.js`となります

# COMMAND ----------

# MAGIC %scala
# MAGIC displayHTML(s"""
# MAGIC <!DOCTYPE html>
# MAGIC <meta charset="utf-8">
# MAGIC <body>
# MAGIC <script src="/files/customjs/d3.v3.min.js"></script>
# MAGIC <script>
# MAGIC var width = 200
# MAGIC var height = 200
# MAGIC var vertices = d3.range(100).map(function(d) {
# MAGIC   return [Math.random() * width, Math.random() * height];
# MAGIC });
# MAGIC 
# MAGIC var lineFunction = d3.svg.line()
# MAGIC                          .x(function(d) { return d[0]; })
# MAGIC                          .y(function(d) { return d[1]; })
# MAGIC                          .interpolate("linear");
# MAGIC 
# MAGIC //The SVG Container
# MAGIC var svgContainer = d3.select("body").append("svg")
# MAGIC                                     .attr("width", 200)
# MAGIC                                     .attr("height", 200);
# MAGIC 
# MAGIC //The line SVG Path we draw
# MAGIC var lineGraph = svgContainer.append("path")
# MAGIC                             .attr("d", lineFunction(vertices))
# MAGIC                             .attr("stroke", "blue")
# MAGIC                             .attr("stroke-width", 2)
# MAGIC                             .attr("fill", "none");
# MAGIC </script>
# MAGIC   """)
