# Databricks notebook source
# MAGIC %md
# MAGIC # カスタムモデルのトレーニング、SageMakerへのデプロイメント
# MAGIC 
# MAGIC 機械学習モデルの推論処理の前処理として「`モデルを呼び出す際、IDを指定し推論に用いるデータをS3から取得する`」という要件がある場合、`PyFunc`として前処理を含むモデルを実装・トレーニングし、モデルをSageMakerのエンドポイントにデプロイするアプローチを取る必要があります。
# MAGIC 
# MAGIC 本ノートブックでは、ワインの品質を予測するモデルを構築しますが、前処理を追加したカスタムモデルをSageMakerのエンドポイントにデプロイし、推論に用いるデータのIDを指定して推論を実行します。
# MAGIC 
# MAGIC **注意**
# MAGIC - 要件が`モデルを呼び出す際に前処理を追加したい`のみであり、S3へのアクセスが不要であれば、`PyFunc`による実装のみで十分です。トレーニングしたモデルをMLflowのモデルサービングで運用することが可能です。
# MAGIC - 上記要件に`S3へのアクセス`が追加される場合、SageMakerのエンドポイントにモデルのデプロイを行う必要があります。
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2022/02/04</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.1</td></tr>
# MAGIC   <tr><td>クラスター</td><td>10.2ML</td></tr>
# MAGIC </table>
# MAGIC 
# MAGIC <img style="margin-top:25px;" src="https://sajpstorage.blob.core.windows.net/workshop20210205/databricks-logo-small-new.png" width="140">
# MAGIC 
# MAGIC **要件**
# MAGIC - 以下の処理を行うクラスターには、SageMakerおよびS3にアクセスできるインスタンスプロファイルをアタッチします。
# MAGIC - クラスターライブラリとしてPyPIから以下をインストールします。
# MAGIC   - s3fs
# MAGIC   - sagemaker
# MAGIC   - botocore
# MAGIC 
# MAGIC **参考資料**
# MAGIC - [Databricksにおけるカスタムモデルのトレーニング、SageMakerエンドポイントへのデプロイメント \- Qiita](https://qiita.com/taka_yayoi/items/3b2c86b2334cda557d4c)
# MAGIC - [機械学習モデルをSageMakerにデプロイするのためのAWS認証設定のセットアップ \- Qiita](https://qiita.com/taka_yayoi/items/e26e4546cdaf13069aad)
# MAGIC - [SageMakerへのscikit\-learnモデルのデプロイメント \- Qiita](https://qiita.com/taka_yayoi/items/79f024874075ea9e1cd3)
# MAGIC - [Databricksにおけるインスタンスプロファイルを用いたS3バケットへのセキュアなアクセス \- Qiita](https://qiita.com/taka_yayoi/items/446c7971be354f88c679)
# MAGIC - [Preprocess input data before making predictions using Amazon SageMaker inference pipelines and Scikit\-learn \| AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/preprocess-input-data-before-making-predictions-using-amazon-sagemaker-inference-pipelines-and-scikit-learn/)

# COMMAND ----------

# MAGIC %md
# MAGIC ## AWSの設定
# MAGIC 
# MAGIC 以下の手順に従ってSageMakerにアクセスするためのロールを作成します。
# MAGIC 
# MAGIC [機械学習モデルをSageMakerにデプロイするのためのAWS認証設定のセットアップ \- Qiita](https://qiita.com/taka_yayoi/items/e26e4546cdaf13069aad)
# MAGIC 
# MAGIC モデルがアクセスするデータが格納されているS3にアクセスするためのインラインポリシーを追加します。
# MAGIC 
# MAGIC ```
# MAGIC {
# MAGIC     "Version": "2012-10-17",
# MAGIC     "Statement": [
# MAGIC         {
# MAGIC             "Effect": "Allow",
# MAGIC             "Action": [
# MAGIC                 "s3:ListBucket"
# MAGIC             ],
# MAGIC             "Resource": [
# MAGIC                 "arn:aws:s3:::<S3バケット名>"
# MAGIC             ]
# MAGIC         },
# MAGIC         {
# MAGIC             "Effect": "Allow",
# MAGIC             "Action": [
# MAGIC                 "s3:PutObject",
# MAGIC                 "s3:GetObject",
# MAGIC                 "s3:DeleteObject",
# MAGIC                 "s3:PutObjectAcl"
# MAGIC             ],
# MAGIC             "Resource": [
# MAGIC                 "arn:aws:s3:::<S3バケット名>/*"
# MAGIC             ]
# MAGIC         }
# MAGIC     ]
# MAGIC }
# MAGIC ```
# MAGIC 
# MAGIC 以下の手順に従って、上記SageMaker用ロールのインスタンスプロファイルARNを用いてDatabricksでインスタンスプロファイルを作成します。
# MAGIC 
# MAGIC [Databricksにおけるインスタンスプロファイルを用いたS3バケットへのセキュアなアクセス \- Qiita](https://qiita.com/taka_yayoi/items/446c7971be354f88c679#%E3%82%B9%E3%83%86%E3%83%83%E3%83%975-databricks%E3%81%AB%E3%82%A4%E3%83%B3%E3%82%B9%E3%82%BF%E3%83%B3%E3%82%B9%E3%83%97%E3%83%AD%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB%E3%82%92%E8%BF%BD%E5%8A%A0%E3%81%99%E3%82%8B)
# MAGIC 
# MAGIC Databricksのデプロイに使用したロールに以下のポリシーを追加します。
# MAGIC 
# MAGIC ```
# MAGIC {
# MAGIC     "Version": "2012-10-17",
# MAGIC     "Statement": [
# MAGIC         {
# MAGIC             "Action": [
# MAGIC                 "iam:PassRole"
# MAGIC             ],
# MAGIC             "Resource": [
# MAGIC                 "arn:aws:iam::<AWSアカウントID>:role/<上で作成したSageMaker用ロール>"
# MAGIC             ],
# MAGIC             "Effect": "Allow"
# MAGIC         }
# MAGIC     ]
# MAGIC }
# MAGIC ```

# COMMAND ----------

#%pip freeze

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルのトレーニング & ロギング

# COMMAND ----------

# MAGIC %md ### トレーニング、テストデータセットの準備

# COMMAND ----------

import pandas as pd

white_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-white.csv", sep=";")
red_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-red.csv", sep=";")

red_wine['is_red'] = 1
white_wine['is_red'] = 0

data = pd.concat([red_wine, white_wine], axis=0)

# カラム名から空白を削除
data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

# qualityを2値に変換します
high_quality = (data.quality >= 7).astype(int)
data.quality = high_quality
display(data)

# COMMAND ----------

# トレーニング、テストデータセットに分割します
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, random_state=123)
X_train = train.drop(["quality"], axis=1)
X_test = test.drop(["quality"], axis=1)
y_train = train.quality
y_test = test.quality

# COMMAND ----------

# MAGIC %md
# MAGIC ### カスタムモデルの定義
# MAGIC 
# MAGIC PyFuncのカスタムモデルとして定義することで、推論を行う前に任意の処理を組み込むことことができます。
# MAGIC 
# MAGIC **注意** 前処理の中でS3へのアクセスが必要な場合、MLflowのモデルサービング用サーバーではS3にアクセスするためのインスタンスプロファイルを使用できないため、SageMakerエンドポイントを使用する必要があります。

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle
import time

# カスタムモデルに必要なライブラリ
import boto3
import fsspec
import s3fs

# PyFuncのカスタムモデルとして定義
class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  """
  sklearnのモデルをトレーニング、使用するクラス
  """
  
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    """これは抽象化関数です。sklearnモデルを取り出すためにメソッドをカスタマイズします。
        Args:
            context ([type]): モデルのアーティファクトが格納されるMLflowコンテキスト
            model_input ([type]): データを検索するキー
        Returns:
            [type]: ロードされたモデルアーティファクト
    """
    
    import csv
    import pandas as pd
    import boto3
    import s3fs
    import os

    # S3バケット名
    bucket_name = '<s3バケット名>'
    # ファイルパス
    savepath = 'wine_quality_w_id.csv'
    #bucket_path = f's3://{bucket_name}/{savepath}'
    #print(bucket_path)
    
    # S3からのデータ取得
    # 本処理を実行するクラスターにS3にアクセス可能なインスタンスプロファイルをアタッチすること    
    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket=bucket_name, Key=savepath)
    file = response["Body"]
    df = pd.read_csv(file, header=0, delimiter=",", low_memory=False)
    #print(df)
    
    # SageMakerエンドポイント経由の場合、引数はDataFrameとして渡されるのでDataFrameから値を取り出す     
    if isinstance(model_input, pd.DataFrame):
      row_id = int(model_input['data'].iloc[0])
    else:
      row_id = int(model_input)
    
    print("model_input:", model_input)
    
    target_data = df.drop("quality", axis=1)    
    target_data = target_data[target_data['lot_id'] == row_id].drop("lot_id", axis=1)
    print("target_data:", target_data)
    
    return self.model.predict_proba(target_data)[:,1]    

# COMMAND ----------

# MAGIC %md
# MAGIC ### モデルのトレーニング

# COMMAND ----------

# mlflow.start_runは、このモデルのパフォーマンスを追跡するための新規MLflowランを生成します。
# コンテキスト内で、使用されたパラメーターを追跡するためにmlflow.log_param、精度のようなメトリクスを追跡するために
# mlflow.log_metricを呼び出します。
with mlflow.start_run(run_name='untuned_random_forest') as run:
  n_estimators = 10
  model = RandomForestClassifier(n_estimators=n_estimators, random_state=np.random.RandomState(123))
  model.fit(X_train, y_train)

  # predict_probaは[prob_negative, prob_positive]を返却するので、出力を[:, 1]でスライスします。
  predictions_test = model.predict_proba(X_test)[:,1]
  auc_score = roc_auc_score(y_test, predictions_test)
  mlflow.log_param('n_estimators', n_estimators)
  # メトリックとしてROC曲線のAUCを使用します。
  mlflow.log_metric('auc', auc_score)
  wrappedModel = SklearnModelWrapper(model)

  # モデルの入出力スキーマを定義するシグネチャをモデルとともに記録します。
  # モデルがデプロイされた際に、入力を検証するためにシグネチャが用いられます。
  # TODO:シグネチャの定義
  #signature = infer_signature(np.array([0]), wrappedModel.predict(None, np.array([0])))
  #signature = infer_signature(0, wrappedModel.predict(None, 0))
  #print(signature)
  
  # MLflowにはモデルをサービングする際に用いられるconda環境を作成するユーティリティが含まれています。
  # 必要な依存関係がconda.yamlに保存され、モデルとともに記録されます。
  conda_env =  _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), 
                             "scikit-learn=={}".format(sklearn.__version__),
                             "boto3==1.20.1",
                             "fsspec=={}".format(fsspec.__version__),
                             "s3fs=={}".format(s3fs.__version__),
                             "botocore==1.23.24"
                            ],
        additional_conda_channels=None,
    )
  #mlflow.pyfunc.log_model("random_forest_model", python_model=wrappedModel, conda_env=conda_env, signature=signature)
  mlflow.pyfunc.log_model("random_forest_model", python_model=wrappedModel, conda_env=conda_env)

# COMMAND ----------

# MAGIC %md
# MAGIC ### モデルの動作確認

# COMMAND ----------

# ラン(トレーニング)のIDを取得
previous_run_id = run.info.run_id

# COMMAND ----------

import mlflow

# トラッキングされたモデルのロード
loaded_model = mlflow.pyfunc.load_model(f"runs:/{previous_run_id}/random_forest_model")

# COMMAND ----------

#dbutils.fs.ls("s3://ty-db-data-bucket/")

# COMMAND ----------

import numpy as np

# IDを指定して予測を実行
out_predict = loaded_model.predict("8")
print(out_predict)

# COMMAND ----------

#out_predict = loaded_model.predict(["0", "1", "2", "3", "4", "5", "6", "7"])
#print(out_predict)

# COMMAND ----------

# MAGIC %md
# MAGIC ## カスタムモデルのサービング
# MAGIC 
# MAGIC MLflowで記録されたモデルをSageMakerのエンドポイントにデプロイします。

# COMMAND ----------

# MAGIC %md
# MAGIC ### モデルのデプロイ
# MAGIC 
# MAGIC SageMakerエンドポイントにモデルをデプロイする際には、AmazonのElastic Container Registry (ECR)のDockerイメージを指定します。SageMakerはモデルをサービングする際にこのイメージを使用します。イメージの作成にはMLflow CLIを使用します。
# MAGIC 
# MAGIC MLflow CLIコマンドを実行する前に以下の設定を行なっていることを確認してください。
# MAGIC - ローカルマシンでAWS CLIのインストール、設定を行います。
# MAGIC   - [AWS\-CLIの初期設定のメモ \- Qiita](https://qiita.com/reflet/items/e4225435fe692663b705)
# MAGIC - ローカルマシンでPyPIを使って`mlflow`、`boto3`をインストールしておきます。
# MAGIC - ローカルマシンでDockerが動いていることを確認します。
# MAGIC 
# MAGIC ローカルマシンでMLflow CLI:`mlflow sagemaker build-and-push-container`を実行し、`mlflow-pyfunc`イメージをビルドし、イメージをECRのリポジトリにアップロードします。ECRリポジトリに`mlflow-pyfunc`イメージが作成されます。アップロード完了後、DockerイメージのURLを取得することができます。これは初回のみの操作です。以降、エンドポイントを作成する際にこのイメージのURLを指定します。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/demo20220129-pytorch-mlflow/ECR.png)

# COMMAND ----------

region = "ap-northeast-1"
run_id1 = previous_run_id
model_uri = "runs:/" + run_id1 + "/random_forest_model"

# COMMAND ----------

# 以下のECR Dockerイメージに対するURLを<ECR-URL>で置き換えてください。
# ECR URLは以下のフォーマットである必要があります: {account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:{tag}
image_ecr_url = "<AWSアカウントID>.dkr.ecr.ap-northeast-1.amazonaws.com/mlflow-pyfunc:1.23.1"

# COMMAND ----------

import mlflow.sagemaker as mfs
app_name = "wine-quality" # SageMakerエンドポイント名

# COMMAND ----------

# MAGIC %md
# MAGIC 以下の引数`mode`は以下の通り指定します。
# MAGIC 
# MAGIC - エンドポイントの新規作成:`create`
# MAGIC - エンドポイントの更新:`replace`

# COMMAND ----------

# 4-5分かかります
mfs.deploy(app_name=app_name, model_uri=model_uri, image_url=image_ecr_url, region_name=region, mode="replace") # mode create/replace

# COMMAND ----------

# MAGIC %md
# MAGIC 以下のセルを実行することで新規SageMakerエンドポイントのステータスを確認することができます。
# MAGIC 
# MAGIC **注意**: アプリケーションのステータスはこの時点では**Creating**です。ステータスが**InService**になるまで待ってください。それまではクエリーのリクエストは失敗します。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/demo20220129-pytorch-mlflow/sagemaker_endpoint_inservice.png)

# COMMAND ----------

# エンドポイントのステータスを確認するヘルパー関数
def check_status(app_name):
  sage_client = boto3.client('sagemaker', region_name=region)
  endpoint_description = sage_client.describe_endpoint(EndpointName=app_name)
  endpoint_status = endpoint_description["EndpointStatus"]
  return endpoint_status

print("Application status is: {}".format(check_status(app_name)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### エンドポイントからモデルの呼び出し
# MAGIC 
# MAGIC HTTPリクエストを送信してサンプルインプットを評価します。
# MAGIC 
# MAGIC `boto3`で提供される`sagemaker-runtime` APIを用いてSageMakerエンドポイントREST APIにクエリーを実行します。

# COMMAND ----------

import json
import numpy as np

# パラメータ指定
data = [{"data": "8"}] # 推論に用いるデータのIDを指定します
input_json = json.dumps(data)
print(input_json)

def query_endpoint(app_name, input_json):
  client = boto3.session.Session().client("sagemaker-runtime", region)
  
  response = client.invoke_endpoint(
      EndpointName=app_name,
      Body=input_json,
      ContentType='application/json',
  )
  
  preds = response['Body'].read().decode("ascii")
  preds = json.loads(preds)
  print("Received response: {}".format(preds))
  return preds

print("Sending batch prediction request with input dataframe json: {}".format(input_json))

# デプロイされたモデルにポストすることで入力を評価します
prediction1 = query_endpoint(app_name=app_name, input_json=input_json)

# COMMAND ----------

# MAGIC %md
# MAGIC ### エンドポイントの削除

# COMMAND ----------

# 指定されたアプリケーションと関連づけられたSageMakerモデルと設定を削除するには、archive=Falseオプションを指定します。
mfs.delete(app_name=app_name, region_name=region, archive=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
