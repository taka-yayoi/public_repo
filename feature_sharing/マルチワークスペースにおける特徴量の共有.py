# Databricks notebook source
# MAGIC %md # 集中管理特徴量ストアのサンプル
# MAGIC 
# MAGIC このノートブックでは、リモートの特徴量ストアワークスペース(workspace B)に特徴量テーブルを作成します。そして、リモートの特徴量テーブルを用いてモデルをトレーニングし、異なるリモートワークスペース(workspace C)のモデルレジストリにモデルを登録します。
# MAGIC 
# MAGIC [Databricksワークスペース間で特徴量テーブルを共有する \- Qiita](https://qiita.com/taka_yayoi/items/33ac44f965adfe726f1f)
# MAGIC 
# MAGIC ## ノートブックのセットアップ
# MAGIC 
# MAGIC 1. 特徴量テーブルが作成されるワークスペース(workspace B)でアクセストークンを作成します。
# MAGIC 1. ローカルワークスペース(workspace A)でリモートワークスペースの情報とアクセストークンを格納するシークレットを作成します。最も簡単な方法はDatabricks CLIを使うことですが、Secrets REST APIを使用することもできます。
# MAGIC 
# MAGIC   a. シークレットスコープを作成します: `databricks secrets create-scope --scope <scope>`.  
# MAGIC   b. ターゲットワークスペースに対するユニークな名称をつけます。これを `<prefix>` と呼びます。次に3つのシークレットを作成します:
# MAGIC     * `databricks secrets put --scope <scope> --key <prefix>-host`. 特徴量ストアのワークスペース(workspace B)のホスト名を入力します。
# MAGIC     * `databricks secrets put --scope <scope> --key <prefix>-token`. 特徴量ストアのワークスペースのアクセストークンを入力します。
# MAGIC     * `databricks secrets put --scope <scope> --key <prefix>-workspace-id`. 特徴量ストアのワークスペースIDを入力します。ワークスペースIDはワークスペースIDのURLから取得することができます。
# MAGIC 
# MAGIC **注意**
# MAGIC - このノートブックを実行する前に、ノートブックの上部にあるノートブックパラメーターのフィールドに、特徴量ストアのワークスペース(workspace B)、モデルレジストリワークスペース(workspace C)に対応するシークレットスコープとキープレフィックスを入力してください。なお、workspace Bとworkspace Cを同じワークスペースにすることも可能です。
# MAGIC - MLランタイム(10.2以降を推奨)が必要です。

# COMMAND ----------

dbutils.widgets.text('feature_store_secret_scope', '')
dbutils.widgets.text('feature_store_secret_key_prefix', '')

dbutils.widgets.text('model_registry_secret_scope', '')
dbutils.widgets.text('model_registry_secret_key_prefix', '')

fs_scope = str(dbutils.widgets.get('feature_store_secret_scope'))
fs_key = str(dbutils.widgets.get('feature_store_secret_key_prefix'))

mr_scope = str(dbutils.widgets.get('model_registry_secret_scope'))
mr_key = str(dbutils.widgets.get('model_registry_secret_key_prefix'))

feature_store_uri = f'databricks://{fs_scope}:{fs_key}' if fs_scope and fs_key else None
model_registry_uri = f'databricks://{mr_scope}:{mr_key}' if mr_scope and mr_key else None

print("feature_store_uri", feature_store_uri)
print("model_registry_uri", model_registry_uri)

# COMMAND ----------

# MAGIC %md ## 特徴量テーブルのセットアップ
# MAGIC 
# MAGIC このステップでは、特徴量テーブルのためのデータベースを作成し、リモートの特徴量テーブルを作成するために使用するデータフレーム`features_df`を作成します。

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS feature_store_multi_workspace;

# COMMAND ----------

from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    FloatType,
)

feature_table_name = "feature_store_multi_workspace.feature_table"

feature_table_schema = StructType(
    [
        StructField("user_id", IntegerType(), False),
        StructField("user_feature", FloatType(), True),
    ]
)

features_df = spark.createDataFrame(
    [
        (123, 100.2),
        (456, 12.4),
    ],
    feature_table_schema,
)

# COMMAND ----------

# MAGIC %md ## リモート特徴量テーブルの作成
# MAGIC 
# MAGIC リモート特徴量テーブルを作成するためのAPIコールは、お使いのクラスターのDatabricks MLランタイムのバージョンに依存します。
# MAGIC 
# MAGIC - Databricks Runtime 10.2 ML以降では `FeatureStoreClient.create_table`　を使います。
# MAGIC - Databricks Runtime 10.1 ML以前では `FeatureStoreClient.create_feature_table` を使います。
# MAGIC 
# MAGIC このステップでは、リモートのワークスペース(Workspace B)に特徴量テーブルを作成します。

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient(feature_store_uri=feature_store_uri, model_registry_uri=model_registry_uri)

# COMMAND ----------

# Databricks Runtime 10.2 ML以降ではこちらのコマンドを使用します
fs.create_table(
    feature_table_name,
    primary_keys="user_id",
    df=features_df,
    description="Sample feature table",
)

# COMMAND ----------

# Databricks Runtime 10.1 ML以前では、以下をコメントを解除して実行します。

#fs.create_feature_table(
#    feature_table_name,
#    "user_id",
#    features_df=features_df,
#    description="Sample feature table",
#)

# COMMAND ----------

# MAGIC %md この時点で特徴量ストアのワークスペースで新規特徴量テーブルを確認することができます。

# COMMAND ----------

# MAGIC %md ## モデルをトレーニングするためにリモートの特徴量テーブルを読み込む

# COMMAND ----------

import mlflow

class SampleModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return model_input.sum(axis=1, skipna=False)

# COMMAND ----------

record_table_schema = StructType(
    [
        StructField("id", IntegerType(), False),
        StructField("income", IntegerType(), False),
    ]
)

record_table = spark.createDataFrame(
    [
        (123, 10000),
        (456, 20000),
        (789, 30000),
    ],
    record_table_schema,
)

# COMMAND ----------

from databricks.feature_store import FeatureLookup

feature_lookups = [
    FeatureLookup(
        table_name=feature_table_name,
        feature_name="user_feature",
        lookup_key="id",
    ),
]

# COMMAND ----------

training_set = fs.create_training_set(
    record_table,
    feature_lookups=feature_lookups,
    exclude_columns=["id"],
    label="income",
)

# トレーニングデータセットのロード。load_df()は、モデルトレーニングを行うためにsklearnに引き渡すことのできるデータフレームを返却します。
training_df = training_set.load_df()

# COMMAND ----------

# MAGIC %md ## リモートモデルレジストリのワークスペース(workspace C)にモデルを登録

# COMMAND ----------

with mlflow.start_run() as new_run:
  fs.log_model(
      SampleModel(),
      artifact_path="model",
      flavor=mlflow.pyfunc,
      training_set=training_set,
      registered_model_name="multi_workspace_fs_model",
  )

# COMMAND ----------

# MAGIC %md この時点で、リモートのモデルレジストリワークスペースで新規モデルバージョンを確認できます。

# COMMAND ----------

# MAGIC %md ## バッチ推論を行うためにリモートモデルレジストリのモデルを使用

# COMMAND ----------

# モデルURIの取得
model_uri = f"models:/multi_workspace_fs_model/1"

# モデルから予測を取得するためにscore_batchの呼び出し
with_predictions = fs.score_batch(model_uri, record_table.drop("income"))

# COMMAND ----------

display(with_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
