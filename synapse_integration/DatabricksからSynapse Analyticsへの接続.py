# Databricks notebook source
# MAGIC %md
# MAGIC # Synapse Analyticsにアクセスしてデータの読み書きを行う
# MAGIC 
# MAGIC **参考資料**
# MAGIC - [DatabricksとAzure Synapse Analyticsの連携 \- Qiita](https://qiita.com/taka_yayoi/items/7d830a6a273dadd94c2a)
# MAGIC - [チュートリアル:Azure Synapse Analytics の使用を開始する \- Azure Synapse Analytics \| Microsoft Docs](https://docs.microsoft.com/ja-jp/azure/synapse-analytics/get-started)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Synapse Workspaceの作成
# MAGIC 
# MAGIC ここではデモの目的でダミーのワークスペースを作成しています。既存のワークスペースが存在する場合には、そちらを参照してください。
# MAGIC 
# MAGIC ここでは以下の設定を行なっています。
# MAGIC 
# MAGIC |項目|設定値|
# MAGIC |:--|:--|
# MAGIC |workspace名|taka-workspace|
# MAGIC |ストレージアカウント名|takaaccount|
# MAGIC |ストレージコンテナー名|users|
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/demo20210509-synapse/create_workspace.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 専用のSQLプールを作成
# MAGIC 
# MAGIC ここでは「mysqlpool」と言うSQLプールを作成しています。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/demo20210509-synapse/create_sql_pool.png)

# COMMAND ----------

# MAGIC %md ## Synapseワークスペースの設定の確認

# COMMAND ----------

# MAGIC %md
# MAGIC ### ファイアウォール設定の確認
# MAGIC 
# MAGIC DatabricksからSynapseにアクセスできるようにするためには、Synapseのファイアウォールの設定で「Azure サービスおよびリソースに、このワークスペースへのアクセスを許可する」がオンになっていることを確認します。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/demo20210509-synapse/firewall.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ### SQL 管理ユーザー名の確認
# MAGIC 
# MAGIC ワークスペースの概要に表示される「SQL 管理ユーザー名」をメモしておいてください。また、パスワードもご確認ください。これらはSynapse接続時に指定する必要があります。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/demo20210509-synapse/sqladmin.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ### SQLプールのマスターパスワードが作成されていることを確認
# MAGIC 
# MAGIC DatabricksからSQLプールにアクセスして操作を行う場合には、当該SQLプールでマスターパスワードが作成されている必要があります。作成されていない場合には、当該SQLプールで以下のSQLを実行してください。
# MAGIC <br>
# MAGIC - [Create a Database Master Key \- SQL Server \| Microsoft Docs](https://docs.microsoft.com/en-us/sql/relational-databases/security/encryption/create-a-database-master-key?view=sql-server-ver15)
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/demo20210509-synapse/master_password.png)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## ストレージアカウントのアクセスキーを設定
# MAGIC 
# MAGIC ストレージアカウントのアクセスキーは、ホーム > ストレージアカウントで「キーの表示」をクリックし、表示されるKey1を指定します。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/demo20210509-synapse/storage_account_access_key.png)

# COMMAND ----------

# MAGIC %md
# MAGIC 以下ではアクセスキーやパスワードを平文で記載していますが、本運用の際にはシークレットの活用をご検討ください。
# MAGIC <br><br>
# MAGIC **参考資料**
# MAGIC - [Databricksにおけるシークレットの管理 \- Qiita](https://qiita.com/taka_yayoi/items/338ef0c5394fe4eb87c0)

# COMMAND ----------

storage_account_key = "<<ストレージアカウントのアクセスキー>>"

spark.conf.set("fs.azure.account.key.takaaccount.blob.core.windows.net", storage_account_key)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## JDBCユーザー名とパスワードで接続しデータを読み込む
# MAGIC <br>
# MAGIC - `hostname` Workspace SQL endpointを指定
# MAGIC - `database` SQL pool名を指定
# MAGIC - `dbuser` SQL 管理ユーザー名を`ユーザー名@ワークスペース名`の形式で指定
# MAGIC - `dbpassword` = SQL 管理ユーザーのパスワードを指定
# MAGIC - `storage_account` Workspace作成時に指定したストレージアカウント名を指定
# MAGIC - `container_name` = Workspace作成時に指定したストレージコンテナー名を指定
# MAGIC - `table_name` = 読み取るテーブル名を指定

# COMMAND ----------

hostname = "taka-workspace.sql.azuresynapse.net"
database = "mysqlpool"
dbuser = "sqladminuser@taka-workspace"
dbpassword = "<<SQL管理ユーザーのパスワード>>"
storage_account = "takaaccount"
container_name = "users"
table_name = "NYCTaxiTripSmall"

# Get some data from an Azure Synapse table.
df = spark.read \
  .format("com.databricks.spark.sqldw") \
  .option("url", "jdbc:sqlserver://{0}:1433;database={1};user={2};password={3};trustServerCertificate=false;hostNameInCertificate=*.sql.azuresynapse.net;loginTimeout=30;".format(hostname, database, dbuser, dbpassword)) \
  .option("tempDir", "wasbs://{0}@{1}.blob.core.windows.net/tempdir".format(container_name, storage_account)) \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("dbTable", table_name) \
  .load()

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Synapse上にテーブルを作成する
# MAGIC 
# MAGIC ここでは、Databricksデータセットに格納されているダイアモンドのデータセットをSynapse上に作成します。

# COMMAND ----------

dataFrame = "/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv"
diamonds = spark.read.format("csv").option("header","true")\
  .option("inferSchema", "true").load(dataFrame)

# COMMAND ----------

display(diamonds)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC [書き込みのセマンティクス](https://qiita.com/taka_yayoi/items/7d830a6a273dadd94c2a#%E6%9B%B8%E3%81%8D%E8%BE%BC%E3%81%BF%E3%81%AE%E3%82%BB%E3%83%9E%E3%83%B3%E3%83%86%E3%82%A3%E3%82%AF%E3%82%B9)
# MAGIC 
# MAGIC > Azure SynapseコネクターはCOPY文をサポートしています。COPY文は、外部テーブルを作成することなしにデータロードが可能で、データロードに必要な権限が少なくてすみ、Azure Synapseに対して高速なデータ投入を可能とする便利な方法を提供します。

# COMMAND ----------

# COPY文の使用を強制
spark.conf.set("spark.databricks.sqldw.writeSemantics", "copy")

# 書き込み先のテーブル名
write_table_name = "diamonds"

diamonds.write \
  .format("com.databricks.spark.sqldw") \
  .option("url", "jdbc:sqlserver://{0}:1433;database={1};user={2};password={3};trustServerCertificate=false;hostNameInCertificate=*.sql.azuresynapse.net;loginTimeout=30;".format(hostname, database, dbuser, dbpassword)) \
  .option("tempDir", "wasbs://{0}@{1}.blob.core.windows.net/tempdir".format(container_name, storage_account)) \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("dbTable", write_table_name) \
  .save()

# COMMAND ----------

# MAGIC %md
# MAGIC Synapse Analytics側でデータを確認します。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/demo20210509-synapse/diamonds.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
