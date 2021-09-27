# Databricks notebook source
# MAGIC %md # init scriptによる日本語フォントのインストール
# MAGIC <br>
# MAGIC - 以下のスクリプトを実行すると、DBFSの `/databricks/scripts/japanese-font-install.sh` にinit scriptが作成されます。
# MAGIC - Clustersにて、日本語フォントを使用するクラスターにinit scriptを指定してください。
# MAGIC ![](https://sajpstorage.blob.core.windows.net/workshop202103-r-mlflow/Screen Shot 2021-02-21 at 11.12.15.png)
# MAGIC 
# MAGIC ** 参考情報 **
# MAGIC - [Cluster node initialization scripts — Databricks Documentation](https://docs.databricks.com/clusters/init-scripts.html)
# MAGIC - [Configure clusters — Databricks Documentation](https://docs.databricks.com/clusters/configure.html#cluster-log-delivery) init scriptの動作をロギングすることができます

# COMMAND ----------

# init script格納ディレクトリの作成
dbutils.fs.mkdirs("dbfs:/databricks/scripts/")

# COMMAND ----------

# init scriptの作成
dbutils.fs.put("/databricks/scripts/japanese-font-install.sh","""
#!/bin/bash
apt-get install fonts-takao-mincho fonts-takao-gothic fonts-takao-pgothic -y""", True)

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/databricks/scripts/japanese-font-install.sh"))
