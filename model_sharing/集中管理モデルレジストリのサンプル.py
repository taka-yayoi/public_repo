# Databricks notebook source
# MAGIC %md # 集中管理モデルレジストリのサンプル
# MAGIC 
# MAGIC このノートブックでは、現在使用しているワークスペースでMLflowトラッキングサーバーにモデルを記録し、異なるワークスペースのモデルレジストリにモデルを登録をする方法を説明します。
# MAGIC 
# MAGIC [Databricksワークスペース間における機械学習モデルの共有 \- Qiita](https://qiita.com/taka_yayoi/items/ddddbf9637fe03607b4f)
# MAGIC 
# MAGIC ## セットアップ
# MAGIC 
# MAGIC 1. モデルレジストリワークスペースでアクセストークンを作成します。
# MAGIC 1. このワークスペースでは、シークレットを作成し、アクセストークンとリモートのワークスペース情報を格納します。最も簡単な方法はDatabricks CLIを使うことですが、シークレットREST APIを使用することもできます。
# MAGIC 
# MAGIC   a. シークレットスコープを作成します: `databricks secrets create-scope --scope <scope>`.  
# MAGIC   b. ターゲットワークスペースに対するユニークな名称をつけます。これを `<prefix>` と呼びます。次に3つのシークレットを作成します:
# MAGIC     * `databricks secrets put --scope <scope> --key <prefix>-host`. モデルレジストリワークスペースのホスト名を入力します。
# MAGIC     * `databricks secrets put --scope <scope> --key <prefix>-token`. モデルレジストリワークスペースのアクセストークンを入力します。
# MAGIC     * `databricks secrets put --scope <scope> --key <prefix>-workspace-id`. モデルレジストリのワークスペースIDを入力します。ワークスペースIDはワークスペースIDのURLから取得することができます。
# MAGIC 
# MAGIC **注意**
# MAGIC - このノートブックを実行する前に、ノートブックの上部にあるノートブックパラメーターのフィールドにモデルレジストリワークスペースに対応するシークレットスコープとキープレフィックスを入力してください。
# MAGIC - MLランタイムが必要です。

# COMMAND ----------

dbutils.widgets.text('1_registry_secret_scope', '')
dbutils.widgets.text('2_registry_secret_key_prefix', '')
scope = str(dbutils.widgets.get('1_registry_secret_scope'))
key = str(dbutils.widgets.get('2_registry_secret_key_prefix'))

registry_uri = 'databricks://' + scope + ':' + key if scope and key else None

print(registry_uri)

# COMMAND ----------

# MAGIC %md ## 新規モデルの登録
# MAGIC 
# MAGIC モデルを登録する方法は3つあります。モデルを登録することで新規モデルバージョンを作成します。
# MAGIC   1. `MlflowClient.create_model_version`
# MAGIC   2. `<flavor>.log_model`
# MAGIC   3. `mlflow.register_model`

# COMMAND ----------

import uuid
prefix = uuid.uuid4().hex[0:4]  # クラッシュを避けるために、モデル名に対するユニークなプレフィクスを指定します。
model1_name = f'{prefix}_model1'
model2_name = f'{prefix}_model2'
model3_name = f'{prefix}_model3'

print(model1_name, model2_name, model3_name)

# COMMAND ----------

# サンプルのモデル実装
import mlflow
import mlflow.pyfunc

class SampleModel(mlflow.pyfunc.PythonModel):
  def predict(self, ctx, input_df):
      return 7

artifact_path = 'sample_model'

# COMMAND ----------

# MAGIC %md ### 方法1 `MlflowClient.create_model_version`を用いたモデルバージョンの作成

# COMMAND ----------

# MLflow Trackingへのモデルの記録
from mlflow.tracking.artifact_utils import get_artifact_uri

with mlflow.start_run() as new_run:
  mlflow.pyfunc.log_model(  
      python_model=SampleModel(),
      artifact_path=artifact_path,
  )
  run1_id = new_run.info.run_id
  source = get_artifact_uri(run_id=run1_id, artifact_path=artifact_path)

# COMMAND ----------

# ローカルのトラッキングサーバー、リモートレジストリサーバーをポイントするためにMlflowClientのインスタンスを作成
from mlflow.tracking.client import MlflowClient
client = MlflowClient(tracking_uri=None, registry_uri=registry_uri)

model = client.create_registered_model(model1_name)
client.create_model_version(name=model1_name, source=source, run_id=run1_id)

# COMMAND ----------

# MAGIC %md この時点で、リモートのモデルレジストリワークスペースで新規モデルバージョンを確認できます。

# COMMAND ----------

# MAGIC %md ### 方法2 `mlflow.register_model`を用いたモデルバージョンの作成
# MAGIC 
# MAGIC この方法では、モデルが存在しない場合には登録済みモデルも作成します。

# COMMAND ----------

mlflow.set_registry_uri(registry_uri)
mlflow.register_model(model_uri=f'runs:/{run1_id}/{artifact_path}', name=model2_name)

# COMMAND ----------

# MAGIC %md ### 方法3 `<flavor>.log_model`を用いたモデルの作成
# MAGIC 
# MAGIC この方法では、モデルが存在しない場合には登録済みモデルも作成します。

# COMMAND ----------

mlflow.set_registry_uri(registry_uri)

with mlflow.start_run() as new_run:
  mlflow.pyfunc.log_model(
    python_model=SampleModel(),
    artifact_path=artifact_path,
    registered_model_name=model3_name, # このパラメーターを指定することでモデル&バージョンを作成します
  )

# COMMAND ----------

# MAGIC %md ## リモートモデルレジストリにおけるモデルの管理

# COMMAND ----------

client = MlflowClient(tracking_uri=None, registry_uri=registry_uri)

# COMMAND ----------

model_names = [m.name for m in client.list_registered_models() if m.name.startswith(prefix)]
print(model_names)

# COMMAND ----------

# model2の説明文を更新
client.update_registered_model(model2_name, description='For ranking')
client.get_registered_model(model2_name)

# COMMAND ----------

# model3のステージングを変更
client.transition_model_version_stage(model3_name, 1, 'Staging')
client.get_model_version(model3_name, 1)

# COMMAND ----------

# model1をモデルレジストリから削除
model1_version = client.get_model_version(model1_name, 1)  # あとでファイルをクリーンアップするのに必要になります
client.delete_registered_model(model1_name)
assert model1_name not in [m.name for m in client.list_registered_models()]

# COMMAND ----------

# MAGIC %md ## リモートレジストリからのモデルのダウンロード
# MAGIC 
# MAGIC モデルをダウンロードするには4つの方法があります。
# MAGIC   1. レジストリのURIを明示的に指定した`mlflow.<flavor>.load_model`
# MAGIC   2. `registry_uri`セットを用いた`mlflow.<flavor>.load_model`
# MAGIC   3. `ModelsArtifactRepository`の使用
# MAGIC   4. REST API (`DbfsRestArtifactRepository`)を用いたモデルファイルの格納場所の取得およびダウンロード

# COMMAND ----------

# MAGIC %md ### 方法1: レジストリのURIを明示的に指定した`mlflow.<flavor>.load_model`

# COMMAND ----------

model = mlflow.pyfunc.load_model(f'models://{scope}:{key}@databricks/{model3_name}/Staging')
model.predict(1)

# COMMAND ----------

# MAGIC %md ### 方法2: `registry_uri`セットを用いた`mlflow.<flavor>.load_model`

# COMMAND ----------

mlflow.set_registry_uri(registry_uri)
model = mlflow.pyfunc.load_model(f'models:/{model3_name}/Staging')
model.predict(1)

# COMMAND ----------

# MAGIC %md ### 方法3: `ModelsArtifactRepository`の使用
# MAGIC 
# MAGIC モデルファイルをモデルフレームワークにロードせずにダウンロードするために、`ArtifactRepository`を使用することができます。`ModelsArtifactRepository`はモデルレジストリのオペレーションに最も適したサブクラスです。リモートレジストリを指定するためには、`mlflow.set_registry_uri`を用いて`registry_uri`を設定するか、以下のように`ModelsArtifactRepository`に情報を直接指定することができます。

# COMMAND ----------

import os
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
local_path = ModelsArtifactRepository(f'models://{scope}:{key}@databricks/{model3_name}/Staging').download_artifacts('')
os.listdir(local_path)

# COMMAND ----------

# MAGIC %md ### 方法4: REST API (`DbfsRestArtifactRepository`)を用いたモデルファイルの格納場所の取得およびダウンロード
# MAGIC 
# MAGIC Pythonでは、この方法と等価である`ModelsArtifactRepository.download_artifacts`(方法3)をお勧めします。しかし、別の文脈でREST APIを用いてダウンロードをどのように行うのかを理解する意味ではこのサンプルは有用です。

# COMMAND ----------

from six.moves import urllib
from mlflow.store.artifact.dbfs_artifact_repo import DbfsRestArtifactRepository

version = client.get_latest_versions(model3_name, ['Staging'])[0].version
uri = client.get_model_version_download_uri(model3_name, version)
path = urllib.parse.urlparse(uri).path
local_path = DbfsRestArtifactRepository(f'dbfs://{scope}:{key}@databricks{path}').download_artifacts('')
os.listdir(local_path)

# COMMAND ----------

# MAGIC %md ## クリーンアップ
# MAGIC 
# MAGIC リモートレジストリのモデルとモデルアーティファクトの中間コピーを削除します。

# COMMAND ----------

def delete_version_tmp_files(version):
  import posixpath
  location = posixpath.dirname(version.source)
  if registry_uri == 'databricks':
    dbutils.fs.rm(location, recurse=True)
  else:
    from mlflow.utils.databricks_utils import get_databricks_host_creds
    from mlflow.utils.rest_utils import http_request
    import json
    response = http_request(
      host_creds=get_databricks_host_creds(registry_uri), 
      endpoint='/api/2.0/dbfs/delete',
      method='POST',
      json=json.loads('{"path": "%s", "recursive": "true"}' % (location))
    )

def archive_and_delete(name):
  try:
    client.transition_model_version_stage(name, 1, 'Archived')
  finally:
    client.delete_registered_model(name)

# COMMAND ----------

delete_version_tmp_files(model1_version)
delete_version_tmp_files(client.get_model_version(model2_name, 1))
delete_version_tmp_files(client.get_model_version(model3_name, 1))

archive_and_delete(model2_name)
archive_and_delete(model3_name)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
