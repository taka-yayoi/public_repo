# Databricks notebook source
# MAGIC %md
# MAGIC # Jobs APIサンプル
# MAGIC 
# MAGIC [Jobs API 2\.1](https://docs.databricks.com/dev-tools/api/latest/jobs.html)を用いることで、REST API経由でジョブを操作することができます。

# COMMAND ----------

# MAGIC %md
# MAGIC ## トークンの取得
# MAGIC 
# MAGIC APIにアクセスするために、パーソナルアクセストークンが必要となります。パーソナルアクセストークンは、サイドメニューの**Settings > User Settings**を開き、**Access Tokens**で**Generate New Token**をクリックします。トークンが表示されるのでコピーしておきます。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/demo20220129-pytorch-mlflow/PAT.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## シークレットによるトークン管理
# MAGIC 
# MAGIC ノートブック上でトークンを使う際には、シークレットに格納することをお勧めします。
# MAGIC 
# MAGIC [Databricksにおけるシークレットの管理 \- Qiita](https://qiita.com/taka_yayoi/items/338ef0c5394fe4eb87c0)

# COMMAND ----------

import os

# トークンは機密性の高い情報なので、ノートブックに直接記載しません。事前にCLIでシークレットとして登録しておいたトークンを呼び出します
token = dbutils.secrets.get("demo-token-takaaki.yayoi", "token")
os.environ["DATABRICKS_TOKEN"] = token

# COMMAND ----------

# MAGIC %md
# MAGIC ## REST APIの呼び出し

# COMMAND ----------

import requests
import json

def call_api(job_id, flower):
  # 使用しているワークスペースのホスト名を指定します
  instance_id = '<Databricksホスト名>'

  api_version = '/api/2.0'
  api_command = '/jobs/run-now'
  url = f"https://{instance_id}{api_version}{api_command}"
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}'}

  params = {
    "job_id": job_id,
    "notebook_params": {
      "flower": flower
    }
  }
  #print(params)

  response = requests.post(
    url = url,
    headers = headers,
    json = params
  )

  print(json.dumps(json.loads(response.text), indent = 2))

# COMMAND ----------

call_api(157442, "sunflowers")

# COMMAND ----------

# MAGIC %md
# MAGIC ジョブの一覧を取得することもできます。

# COMMAND ----------

import requests
import json

instance_id = 'e2-demo-west.cloud.databricks.com'

api_version = '/api/2.0'
api_command = '/jobs/list'
url = f"https://{instance_id}{api_version}{api_command}"
headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}'}

response = requests.get(
  url = url,
  headers = headers
)

print(json.dumps(json.loads(response.text), indent = 2))

# COMMAND ----------

# MAGIC %md
# MAGIC # END
