// Databricks notebook source
// MAGIC %md
// MAGIC # TPCDSデータの生成
// MAGIC 
// MAGIC このノートブックは`park-sql-perf`ライブラリ(https://github.com/databricks/spark-sql-perf )を用いてTPCDSデータを生成します。
// MAGIC 
// MAGIC このライブラリのパッケージング済みのバージョンは https://github.com/BlueGranite/tpc-ds-dataset-generator/tree/master/lib から取得できます。
// MAGIC 
// MAGIC **注意**
// MAGIC - 上のパッケージング済みのバージョン(`spark-sql-perf_2.12-0.5.1-SNAPSHOT.jar`)をクラスターライブラリとしてインストールしてください。

// COMMAND ----------

// MAGIC %md
// MAGIC ## パラメータ設定

// COMMAND ----------

// MAGIC %python
// MAGIC # 重要: 実行するクラスターのワーカーインスタンスの数で更新してください!!!
// MAGIC num_workers=6

// COMMAND ----------

// TCPDSデータをS3バケット上に作成する場合にはマウント
//dbutils.fs.mount("s3a://tpc-benchmarks/", "/mnt/performance-datasets", "sse-s3")

// COMMAND ----------

// TPC-DSデータがダウンロードされるベースディレクトリ
val base_dir = "/tmp/takaaki.yayoi@databricks.com/blog"

// COMMAND ----------

// 重要: パラメーターを設定します!!!
// TPCDSのスケールファクター
val scaleFactor = "1"

// データフォーマット
val format = "parquet"
// falseの場合、decimalではなくfloat型が使用されます
val useDecimal = true
// falseの場合、dateではなく文字列型が使用されます
val useDate = true
// trueの場合、パーティションキーのnull行は除外されます
val filterNull = true
// trueの場合、生成時にパーティションは単一のファイルにまとめられます
val shuffle = true

// データを生成するs3/dbfsのパス
val rootDir = s"${base_dir}/tpcds/sf$scaleFactor-$format/useDecimal=$useDecimal,useDate=$useDate,filterNull=$filterNull"
// 生成するデータベースの名前
// val databaseName = s"tpcds_sf${scaleFactor}" +
//   s"""_${if (useDecimal) "with" else "no"}decimal""" +
//   s"""_${if (useDate) "with" else "no"}date""" +
//   s"""_${if (filterNull) "no" else "with"}nulls"""
val databaseName = "taka_jumpstart_db" // 明示的に指定

// COMMAND ----------

// 指定されたパラメータでテーブルスキーマを作成します
import com.databricks.spark.sql.perf.tpcds.TPCDSTables

val tables = new TPCDSTables(sqlContext, dsdgenDir = "/tmp/tpcds-kit/tools", scaleFactor = scaleFactor, useDoubleForDecimal = !useDecimal, useStringForDate = !useDate)

// COMMAND ----------

// MAGIC %md
// MAGIC ## dsgenのインストール
// MAGIC 
// MAGIC 以下の2つのセルは全てのエグゼキューターノードにdsgenをインストールするためのハックです。**2つ目の結果が空であることを確認してください!!**

// COMMAND ----------

// MAGIC %python
// MAGIC import os
// MAGIC import subprocess
// MAGIC import time
// MAGIC import socket
// MAGIC # 最初にクラスターに変更したバージョンのdsdgenをインストールします
// MAGIC def install(x):
// MAGIC   p = '/tmp/install.sh'
// MAGIC   if (os.path.exists('/tmp/tpcds-kit/tools/dsdgen')): 
// MAGIC     time.sleep(1)
// MAGIC     return "", ""
// MAGIC   with open(p, 'w') as f:    
// MAGIC     f.write("""#!/bin/bash
// MAGIC     sudo apt-get update
// MAGIC     sudo apt-get -y --force-yes install gcc make flex bison byacc git
// MAGIC 
// MAGIC     cd /tmp/
// MAGIC     git clone https://github.com/databricks/tpcds-kit.git
// MAGIC     cd tpcds-kit/tools/
// MAGIC     make -f Makefile.suite
// MAGIC     /tmp/tpcds-kit/tools/dsdgen -h
// MAGIC     """)
// MAGIC   os.chmod(p, 555)
// MAGIC   p = subprocess.Popen([p], stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
// MAGIC   out, err = p.communicate()
// MAGIC   return socket.gethostname(), out, err
// MAGIC 
// MAGIC # 注意: クラスターに対してこれでアップデートします
// MAGIC # ノードあたり最低1つのジョブがあることを確認してください
// MAGIC sc.range(0, num_workers, 1, num_workers).map(install).collect()

// COMMAND ----------

// MAGIC %python
// MAGIC # 全てのノードにdsdgenがインストールされていることをチェックします。結果が空ではない場合、上のステップを再実行してください
// MAGIC import os
// MAGIC import socket
// MAGIC import time
// MAGIC def fileThere(x):
// MAGIC   time.sleep(0.1)
// MAGIC   return socket.gethostname(), os.path.exists('/tmp/tpcds-kit/tools/dsdgen'), 
// MAGIC   
// MAGIC sc.range(0, num_workers, 1, num_workers).map(fileThere).filter(lambda x: not x[1]).collect()

// COMMAND ----------

// MAGIC %md
// MAGIC ## データの生成
// MAGIC 
// MAGIC 6ノードのCPUクラスターで10分程度かかります。

// COMMAND ----------

// データ生成のチューニング:

import org.apache.spark.deploy.SparkHadoopUtil
// parquetライターによって使用されるメモリを制限
SparkHadoopUtil.get.conf.set("parquet.memory.pool.ratio", "0.1")
// snappyによる圧縮:
sqlContext.setConf("spark.sql.parquet.compression.codec", "snappy")
// TPCDSには約2000の日付があります
spark.conf.set("spark.sql.shuffle.partitions", "2000")
// あまりに大きいファイルは書き込みません
sqlContext.setConf("spark.sql.files.maxRecordsPerFile", "20000000")

val dsdgen_partitioned=10000 // SF10000+での推奨値
val dsdgen_nonpartitioned=10 // 生成時に多くの並列度を必要としない小さなテーブル

// COMMAND ----------

// val tableNames = Array("") // Array("") = generate all.
//val tableNames = Array("call_center", "catalog_page", "catalog_returns", "catalog_sales", "customer", "customer_address", "customer_demographics", "date_dim", "household_demographics", "income_band", "inventory", "item", "promotion", "reason", "ship_mode", "store", "store_returns", "store_sales", "time_dim", "warehouse", "web_page", "web_returns", "web_sales", "web_site") // all tables

// 全ての小さなディメンジョンデーブルの作成
val nonPartitionedTables = Array("call_center", "catalog_page", "customer", "customer_address", "customer_demographics", "date_dim", "household_demographics", "income_band", "item", "promotion", "reason", "ship_mode", "store",  "time_dim", "warehouse", "web_page", "web_site")
nonPartitionedTables.foreach { t => {
  tables.genData(
      location = rootDir,
      format = format,
      overwrite = true,
      partitionTables = true,
      clusterByPartitionColumns = shuffle,
      filterOutNullPartitionValues = filterNull,
      tableFilter = t,
      numPartitions = dsdgen_nonpartitioned)
}}
println("Done generating non partitioned tables.")

// 最大/潜在的に難易度が高いテーブルが最後に生成されるようにする
val partitionedTables = Array("inventory", "web_returns", "catalog_returns", "store_returns", "web_sales", "catalog_sales", "store_sales") 
partitionedTables.foreach { t => {
  tables.genData(
      location = rootDir,
      format = format,
      overwrite = true,
      partitionTables = true,
      clusterByPartitionColumns = shuffle,
      filterOutNullPartitionValues = filterNull,
      tableFilter = t,
      numPartitions = dsdgen_partitioned)
}}
println("Done generating partitioned tables.")

// COMMAND ----------

// MAGIC %md
// MAGIC ## データベースの作成
// MAGIC 
// MAGIC 6ノードのCPUクラスターで20分程度かかります。

// COMMAND ----------

sql(s"drop database if exists $databaseName cascade")
sql(s"create database $databaseName")

// COMMAND ----------

sql(s"use $databaseName")

// COMMAND ----------

tables.createExternalTables(rootDir, format, databaseName, overwrite = true, discoverPartitions = true)

// COMMAND ----------

// MAGIC %md
// MAGIC ### データベースの確認

// COMMAND ----------

// MAGIC %sql
// MAGIC show tables

// COMMAND ----------

// MAGIC %sql
// MAGIC select ss.*, dd.*, (dd.d_year*100 + dd.d_moy) as month_id
// MAGIC from store_sales ss
// MAGIC inner join date_dim dd on ss.ss_sold_date_sk=dd.d_date_sk 
// MAGIC inner join item i on ss.ss_item_sk=i.i_item_sk

// COMMAND ----------

// Deltaテーブルが書き込まれるパス
// これはデモ用ノートブックのソースデータとして使用するdeltaパスとなります
val store_sales_delta_path = "/tmp/takaaki.yayoi@databricks.com/sales_store_tpcds"

// COMMAND ----------

val df = spark.sql("""
select ss.ss_quantity, ss.ss_sales_price, ss.ss_net_paid, ss.ss_store_sk, ss.ss_customer_sk, ss.ss_ticket_number, dd.d_date, dd.d_weekend, dd.d_holiday,  i.i_category, (dd.d_year*100 + dd.d_moy) as month_id
from store_sales ss
inner join date_dim dd on ss.ss_sold_date_sk=dd.d_date_sk 
inner join item i on ss.ss_item_sk=i.i_item_sk
""")
df.write.format("delta").mode("overwrite").partitionBy("month_id").save(store_sales_delta_path)

// COMMAND ----------

// MAGIC %md
// MAGIC ### OPTIMIZE(Z-ordering)

// COMMAND ----------

// MAGIC %sql
// MAGIC -- Cmd20の値でパスを変更してください
// MAGIC optimize delta.`/tmp/takaaki.yayoi@databricks.com/sales_store_tpcds`
// MAGIC zorder by d_date, i_category

// COMMAND ----------

// MAGIC %md
// MAGIC # END
