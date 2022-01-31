# Databricks notebook source
# MAGIC %md # MLflowクイックスタートpart 2: Amazon SageMakerを用いたモデルのサービング
# MAGIC 
# MAGIC [このガイドの最初のパート](https://qiita.com/taka_yayoi/items/093f68cb0983cf683647)、**MLflowクイックスタート: トレーニングとロギング**では、モデルのトレーニング、MLflowトラッキングサーバーへのメトリクス、パラメーター、モデルのロギングにフォーカスしました。
# MAGIC 
# MAGIC ##### 注意: このノートブックでは *Run All* を使用しないでください。SageMakerにモデルをデプロイし、更新するのに数分を要し、モデルがアクティブになるまでモデルを検索することができません。
# MAGIC 
# MAGIC ガイドの本パートは、以下のセクションから構成されています。
# MAGIC 
# MAGIC #### セットアップ
# MAGIC * MLflowトラッキングUIを用いてデプロイするモデルを選択します。
# MAGIC 
# MAGIC #### モデルのデプロイ
# MAGIC * MLflow APIを用いて選択したモデルをSageMakerにデプロイします。
# MAGIC * デプロイしたモデルのステータスを確認します。
# MAGIC   * デプロイしたモデルがアクティブであり、クエリーできるかを確認します。
# MAGIC 
# MAGIC #### デプロイしたモデルのクエリー
# MAGIC * デプロイしたモデルが評価できる入力ベクトルをロードします。
# MAGIC * 入力を用いてデプロイしたモデルをクエリーします。
# MAGIC 
# MAGIC #### デプロイメントの管理
# MAGIC * MLflow APIを用いてデプロイしたモデルを更新します。
# MAGIC * 更新したモデルをクエリーします。
# MAGIC 
# MAGIC #### デプロイメントのクリーンアップ
# MAGIC * MLflow APIを用いてモデルデプロイメントを削除します。
# MAGIC 
# MAGIC クイックスタートチュートリアルのパート1と同様に、このノートブックはscikit-learnの`diabetes`データセットでトレーニングしたElasticNetモデルを使用します。

# COMMAND ----------

# MAGIC %md ## 前提条件
# MAGIC 
# MAGIC [クイックスタートガイドのパート1](https://qiita.com/taka_yayoi/items/093f68cb0983cf683647)のMLflowクイックスタートノートブックのElasticNetモデルを前提とします。

# COMMAND ----------

# MAGIC %md ### セットアップ

# COMMAND ----------

# MAGIC %md
# MAGIC 1. 以下のクラスターを使用していることを確認してください: 
# MAGIC   * **Python Version:** Python 3
# MAGIC   * アタッチされているIAMロールがSageMakerデプロイメントをサポートしていること。SageMakerデプロイメント用IAMロールの設定に関しては、[機械学習モデルをSageMakerにデプロイするのためのAWS認証設定のセットアップ](https://qiita.com/taka_yayoi/items/e26e4546cdaf13069aad)をご覧ください。
# MAGIC   
# MAGIC 1. Databricksランタイムを実行しているのであれば、必要なライブラリをインストールするためにCmd 5のコメントを解除して実行してください。Databricks機械学習ランタイムを実行しているのであれば、必要なライブラリはインストール済みなので、このステップをスキップしてください。
# MAGIC 1. このノートブックをクラスターにアタッチします。

# COMMAND ----------

#dbutils.library.installPyPI("mlflow", version="1.0.0", extras="extras")
#dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC [クイックスタートガイドのパート1](https://qiita.com/taka_yayoi/items/093f68cb0983cf683647)のElasticNetトレーニングのランに紐づけられているランIDを選択します。MLflow UIのラン詳細ページのエクスペリメントランからランIDとモデルパスを取得することができます。
# MAGIC 
# MAGIC ![image](https://docs.databricks.com/_static/images/mlflow/mlflow-deployment-example-run-info.png)

# COMMAND ----------

# MAGIC %md ### リージョン、ランID、モデルURIの設定
# MAGIC 
# MAGIC **注意**: 新規リージョンごとに新規SageMakerエンドポイントを作成する必要があります。

# COMMAND ----------

region = "ap-northeast-1"
run_id1 = "b844cb3ddfef4c85b77d845c3dd892f6"
model_uri = "runs:/" + run_id1 + "/model"

# COMMAND ----------

# MAGIC %md ### モデルのデプロイ
# MAGIC 
# MAGIC このセクションでは、**セットアップ**で選択したモデルをSageMakerにデプロイします。

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC AmazonのElastic Container Registry (ECR)のDockerイメージを指定します。SageMakerはモデルをサービングする際にこのイメージを使用します。
# MAGIC ローカルマシンでMLflow CLI:`mlflow sagemaker build-and-push-container`を実行し、`mlflow-pyfunc`イメージをビルドし、イメージをECRのリポジトリにアップロードします。ECRリポジトリに`mlflow-pyfunc`イメージが作成されます。アップロード完了後、DockerイメージのURLを取得することができます。
# MAGIC 
# MAGIC 上記CLIを実行する前に以下の設定を行なっていることを確認してください。
# MAGIC - ローカルマシンでAWS CLIのインストール、設定を行います。
# MAGIC - PyPIで`mlflow`、　`boto3`をインストールしておきます。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/demo20220129-pytorch-mlflow/ECR.png)

# COMMAND ----------

# MAGIC %md
# MAGIC MLflowの`deploy`関数の引数として引き渡す`mlflow-pyfunc`イメージに対するECR URLを定義します。

# COMMAND ----------

# 以下のECR Dockerイメージに対するURLを<ECR-URL>で置き換えてください。
# ECR URLは以下のフォーマットである必要があります: {account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:{tag}
image_ecr_url = "<AWS account id>.dkr.ecr.ap-northeast-1.amazonaws.com/mlflow-pyfunc:1.23.1"

# COMMAND ----------

# MAGIC %md 
# MAGIC SageMakerにトレーニングしたモデルをデプロイするためにMLflowのSageMaker APIを使用します。`mlflow.sagemaker.deploy()`関数は、SageMakerエンドポイントと、エンドポイントに必要となる全てのSageMaker中間オブジェクトを作成します。

# COMMAND ----------

import mlflow.sagemaker as mfs
app_name = "diabetes-class-v1"
mfs.deploy(app_name=app_name, model_uri=model_uri, image_url=image_ecr_url, region_name=region, mode="create")

# COMMAND ----------

# MAGIC %md #### これで一つの関数を使用するだけで、みなさんのモデルがSageMakerにデプロイされました。

# COMMAND ----------

# MAGIC %md 
# MAGIC 以下のセルを実行することで新規SageMakerエンドポイントのステータスを確認することができます。
# MAGIC 
# MAGIC **注意**: アプリケーションのステータスは**Creating**であるべきです。ステータスが**InService**になるまで待ってください。それまではクエリーのリクエストは失敗します。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/demo20220129-pytorch-mlflow/sagemaker_endpoint_inservice.png)

# COMMAND ----------

import boto3

def check_status(app_name):
  sage_client = boto3.client('sagemaker', region_name=region)
  endpoint_description = sage_client.describe_endpoint(EndpointName=app_name)
  endpoint_status = endpoint_description["EndpointStatus"]
  return endpoint_status

print("Application status is: {}".format(check_status(app_name)))

# COMMAND ----------

# MAGIC %md ### デプロイモデルのクエリー

# COMMAND ----------

# MAGIC %md #### `diabetes`データセットからサンプル入力をロードします

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn import datasets

# diabetesデータセットをロードします
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# デプロイしたElasticNetモデルに対するサンプル入力として動作するpandasデータフレームを作成します
Y = np.array([y]).transpose()
d = np.concatenate((X, Y), axis=1)
cols = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'progression']
data = pd.DataFrame(d, columns=cols)
query_df = data.drop(["progression"], axis=1).iloc[[0]]

# サンプルの入力データフレームを`split`オリエンテーションを用いてJSONでシリアライズされたpandasデータフレームに変換します
input_json = query_df.to_json(orient="split")

# COMMAND ----------

print("Using input dataframe JSON: {}".format(input_json))

# COMMAND ----------

# MAGIC %md #### HTTPリクエストを送信してサンプルインプットを評価します
# MAGIC 
# MAGIC `boto3`で提供される`sagemaker-runtime` APIを用いてSageMakerエンドポイントREST APIにクエリーを実行します。

# COMMAND ----------

import json

def query_endpoint(app_name, input_json):
  client = boto3.session.Session().client("sagemaker-runtime", region)
  
  response = client.invoke_endpoint(
      EndpointName=app_name,
      Body=input_json,
      ContentType='application/json; format=pandas-split',
  )
  preds = response['Body'].read().decode("ascii")
  preds = json.loads(preds)
  print("Received response: {}".format(preds))
  return preds

print("Sending batch prediction request with input dataframe json: {}".format(input_json))

# デプロイされたモデルにポストすることで入力を評価します
prediction1 = query_endpoint(app_name=app_name, input_json=input_json)

# COMMAND ----------

# MAGIC %md ### デプロイメントの管理
# MAGIC 
# MAGIC 異なるランの出力を用いて、デプロイされたモデルを更新することができます。異なるElasticNetトレーニングランに紐づけられたランIDを指定します。

# COMMAND ----------

run_id2 = "b61d30cd47444c05b65ac081d387367b"
model_uri = "runs:/" + run_id2 + "/model"

# COMMAND ----------

# MAGIC %md 
# MAGIC `replace`モードで`mlflow.sagemaker.deploy()`をコールします。これにより、新たなランIDに対応するモデルを用いて`diabetes-class`アプリケーションエンドポイントを更新します。

# COMMAND ----------

mfs.deploy(app_name=app_name, model_uri=model_uri, image_url=image_ecr_url, region_name=region, mode="replace")

# COMMAND ----------

# MAGIC %md **注意**: エンドポイントのステータスは **Updating** となります。エンドポイントのステータスが **InService** になって初めて、クエリーは更新されたモデルに対してリクエストを行うようになります。

# COMMAND ----------

print("Application status is: {}".format(check_status(app_name)))

# COMMAND ----------

# MAGIC %md 
# MAGIC 更新されたモデルをクエリーします。異なる予測結果を受け取るはずです。

# COMMAND ----------

prediction2 = query_endpoint(app_name=app_name, input_json=input_json)

# COMMAND ----------

# MAGIC %md 予測結果を比較します。

# COMMAND ----------

print("Run ID: {} Prediction: {}".format(run_id1, prediction1)) 
print("Run ID: {} Prediction: {}".format(run_id2, prediction2))

# COMMAND ----------

# MAGIC %md ### デプロイメントのクリーンアップ
# MAGIC 
# MAGIC モデルデプロイメントが不要となった場合には、削除するために`mlflow.sagemaker.delete()`関数を実行します。

# COMMAND ----------

# 指定されたアプリケーションと関連づけられたSageMakerモデルと設定を削除するには、archive=Falseオプションを指定します。
mfs.delete(app_name=app_name, region_name=region, archive=False)

# COMMAND ----------

# MAGIC %md 
# MAGIC アプリケーションに紐づけられたSageMakerエンドポイントが削除されていることを確認します。

# COMMAND ----------

def get_active_endpoints(app_name):
  sage_client = boto3.client('sagemaker', region_name=region)
  app_endpoints = sage_client.list_endpoints(NameContains=app_name)["Endpoints"]
  return list(filter(lambda en : en == app_name, [str(endpoint["EndpointName"]) for endpoint in app_endpoints]))
  
print("The following endpoints exist for the `{an}` application: {eps}".format(an=app_name, eps=get_active_endpoints(app_name)))

# COMMAND ----------

# MAGIC %md
# MAGIC # END
