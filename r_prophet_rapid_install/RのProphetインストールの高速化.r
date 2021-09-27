# Databricks notebook source
# MAGIC %md
# MAGIC # RのProphetインストールの高速化
# MAGIC 
# MAGIC Rのprophetをインストールする際、依存しているrstanのコンパイルが走るため20分程度時間を要するため、打開策として以下の処理を行う。
# MAGIC <br><br>
# MAGIC 1. `$R_LIBS`に配置されているインストール済みのライブラリをzipし、DBFSに退避
# MAGIC 1. init scriptでDBFSからunzipを行いライブラリを配備
# MAGIC 
# MAGIC **参考情報**
# MAGIC - [R\-User\-Guide/package\_management\.md at master · marygracemoesta/R\-User\-Guide](https://github.com/marygracemoesta/R-User-Guide/blob/master/Developing_on_Databricks/package_management.md#older-package-versions)
# MAGIC - [CRAN \- Package prophet](https://cran.r-project.org/web/packages/prophet/index.html)
# MAGIC - [【 unzip 】コマンド――ZIPファイルからファイルを取り出す：Linux基本コマンドTips（35） \- ＠IT](https://www.atmarkit.co.jp/ait/articles/1607/26/news014.html#sample1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## DBFS上に退避先を作成

# COMMAND ----------

# MAGIC %python
# MAGIC dbutils.fs.mkdirs("/databricks/r_archives")

# COMMAND ----------

# MAGIC %md
# MAGIC ## インストール済みのライブラリをzipにしてDBFSに退避
# MAGIC 
# MAGIC 以下の作業は、手動でprophetのインストールが完了した状態で行うこと。依存関係があるため、$R_LIBS以下を全てバックアップする。

# COMMAND ----------

# MAGIC %sh
# MAGIC ls $R_LIBS

# COMMAND ----------

# MAGIC %sh
# MAGIC zip -r /databricks/driver/r_libraries_20210521.zip $R_LIBS
# MAGIC cp /databricks/driver/r_libraries_20210521.zip /dbfs/databricks/r_archives

# COMMAND ----------

# MAGIC %python
# MAGIC display(dbutils.fs.ls("/databricks/r_archives"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## v8のインストールおよびRライブラリを展開するinit scriptの作成
# MAGIC 
# MAGIC 下のスクリプトを実行後、`/databricks/scripts/r-env-install.sh`をクラスターのinit scriptとして設定して下さい。

# COMMAND ----------

# MAGIC %python
# MAGIC dbutils.fs.put("/databricks/scripts/r-env-install.sh","""
# MAGIC sudo apt-get update
# MAGIC sudo apt-get install -y libv8-dev
# MAGIC unzip /dbfs/databricks/r_archives/r_libraries_20210521.zip -d /
# MAGIC """, True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 動作確認
# MAGIC 
# MAGIC init scriptを設定したクラスターが起動した後に動作確認を行う。

# COMMAND ----------

library(prophet)

# COMMAND ----------

packageVersion("prophet")

# COMMAND ----------

# MAGIC %md
# MAGIC ## init script logの確認(ご参考)

# COMMAND ----------

# MAGIC %fs ls /cluster-logs/0311-045451-beads211/init_scripts/

# COMMAND ----------

# MAGIC %python
# MAGIC display(dbutils.fs.ls("/cluster-logs/0311-045451-beads211/init_scripts/0311-045451-beads211_10_0_12_15/"))

# COMMAND ----------

# MAGIC %sh
# MAGIC cat /dbfs/cluster-logs/0311-045451-beads211/init_scripts/0311-045451-beads211_10_0_12_15/20210520_090222_00_v8-install-test.sh.stderr.log

# COMMAND ----------

# MAGIC %sh
# MAGIC cat /dbfs/cluster-logs/0311-045451-beads211/init_scripts/0311-045451-beads211_10_0_12_15/20210520_090222_00_v8-install-test.sh.stdout.log

# COMMAND ----------

# MAGIC %md
# MAGIC # END
