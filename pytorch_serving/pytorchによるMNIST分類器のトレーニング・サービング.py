# Databricks notebook source
# MAGIC %md
# MAGIC # MLflowによるPyTorch MNIST分類器のトラッキング・サービング
# MAGIC 
# MAGIC 本ノートブックでは、Pytorch LightningによるMNIST分類器をMLflowでトラッキングし、サービングするところまでをデモします。REST APIへの入力が画像になるので、MLflowのtensorサポートを活用します。
# MAGIC 
# MAGIC [MLflowでTensorの入力をサポートしました \- Qiita](https://qiita.com/taka_yayoi/items/3e439dc5df7257fd41db)
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2022/01/29</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>クラスター</td><td>10.2ML</td></tr>
# MAGIC </table>
# MAGIC <img style="margin-top:25px;" src="https://sajpstorage.blob.core.windows.net/workshop20210205/databricks-logo-small-new.png" width="140">
# MAGIC 
# MAGIC **注意**
# MAGIC 
# MAGIC - モデルサービングで`Cannot register 2 metrics with the same name`エラーが生じる場合、tensorflowのバージョンが2.6であることに起因する可能性があります。最新のランタイム10.2MLであればバージョン2.7になるので本エラーを回避できます。
# MAGIC   - [\[Solved\] Cannot register 2 metrics with the same name: /tensorflow/api/keras/optimizers \- Exception Error](https://exerror.com/cannot-register-2-metrics-with-the-same-name-tensorflow-api-keras-optimizers/)
# MAGIC 
# MAGIC - MLflowのオートロギングを用いてモデルをトラッキングします。オートロギングが対応しているバージョンのpytorch-lightningをインストールします。
# MAGIC   - [mlflow\.pytorch — MLflow 1\.23\.1 documentation](https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html#module-mlflow.pytorch)
# MAGIC 
# MAGIC   > Autologging is known to be compatible with the following package versions: **1.0.5 <= pytorch-lightning <= 1.5.9**. Autologging may not succeed when used with package versions outside of this range.

# COMMAND ----------

# MAGIC %md
# MAGIC ## pytorch lightningのインストール

# COMMAND ----------

# MAGIC %pip freeze | grep tensorflow

# COMMAND ----------

# MAGIC %pip install pytorch_lightning==1.5.9

# COMMAND ----------

import os

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

import numpy as np

try:
    from torchmetrics.functional import accuracy
except ImportError:
    from pytorch_lightning.metrics.functional import accuracy

# MLflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

# COMMAND ----------


# auto-loggingで記録された情報を表示
def print_auto_logged_info(r):

    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルの定義
# MAGIC 
# MAGIC こちらはトレーニングループのみを含む最もシンプルなサンプルです(バリデーション、テストなし)。
# MAGIC 
# MAGIC **注意** `LightningModule`はPyTorchの`nn.Module`です。単にいくつかの役立つ機能を持っているだけです。

# COMMAND ----------

from pytorch_lightning import LightningModule, Trainer

class MNISTModel(pl.LightningModule):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        #acc = accuracy(loss, y) # エラーになるのでコメントアウト

        # PyTorchのロガーを使って精度情報を記録
        self.log("train_loss", loss, on_epoch=True)
        #self.log("acc", acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

# COMMAND ----------

# MAGIC %md
# MAGIC ## トレーニングおよびMLflowによるトラッキング
# MAGIC 
# MAGIC 精度指標のメトリクスは、上で`on_epoch=True`を指定しているので、エポックごとに記録されます。

# COMMAND ----------

# MLflowのエンティティを全てオートロギング
mlflow.pytorch.autolog()

# モデルを初期化
mnist_model = MNISTModel()

# MNISTデータセットのDataLoaderを初期化
train_ds = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=32)

# トレーナーを初期化
trainer = Trainer(
    gpus=0, # CPU
    max_epochs=20,
    progress_bar_refresh_rate=20,
)

# モデルをトレーニング ⚡
with mlflow.start_run() as run: # run IDを取得するためにブロックを宣言
  trainer.fit(mnist_model, train_loader)

# COMMAND ----------

# MAGIC %md
# MAGIC 画面右上の**Experiment**ボタンで表示される一覧でモデルを確認することができます。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/demo20220129-pytorch-mlflow/experiments.png)
# MAGIC 
# MAGIC 日付の右にある![](https://docs.databricks.com/_images/external-link.png)アイコンをクリックすることで、さらに詳細を確認することができます。こちらでは、エポックごとのメトリクスの変化を確認することも可能です。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/demo20220129-pytorch-mlflow/metrics_graph.png)
# MAGIC 
# MAGIC メトリクスをクリックするとグラフが表示されます。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/demo20220129-pytorch-mlflow/metrics.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ### トラッキング情報の確認

# COMMAND ----------

# 自動で記録されたパラメーター、メトリクスを表示
print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

# COMMAND ----------

# MAGIC %md
# MAGIC ### TendorBoardの活用
# MAGIC 
# MAGIC ノートブック上で直接TensorBoardを活用することができます。
# MAGIC 
# MAGIC [データサイエンティスト向けの10個のシンプルなDatabricksノートブック tips & tricks \- Qiita](https://qiita.com/taka_yayoi/items/ba0294e20a19cdb1fe10#4-pytorchtensorflow%E3%81%AB%E3%81%8A%E3%81%91%E3%82%8Btensorboard%E3%83%9E%E3%82%B8%E3%83%83%E3%82%AF%E3%82%B3%E3%83%9E%E3%83%B3%E3%83%89)

# COMMAND ----------

# MAGIC %load_ext tensorboard
# MAGIC %tensorboard --logdir /databricks/driver/lightning_logs/

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルによる分類
# MAGIC 
# MAGIC これはデモなので、トレーニングデータセットの一部を用いて分類を行ないます。
# MAGIC 
# MAGIC [PyTorch 1\.0 \- How to predict single images \- mnist example? \- PyTorch Forums](https://discuss.pytorch.org/t/pytorch-1-0-how-to-predict-single-images-mnist-example/32394/2)
# MAGIC 
# MAGIC 下のセルでは画像を確認するためにmatplotlibのimshowを用いて、ノートブック上に画像を表示しています。 
# MAGIC 
# MAGIC `TODO: RGBの調整`

# COMMAND ----------

from matplotlib.pyplot import imshow

single_loaded_img = train_loader.dataset.data[0]
imshow(single_loaded_img)

single_loaded_img_conv = single_loaded_img[None, None]
single_loaded_img_conv = single_loaded_img_conv.type('torch.FloatTensor') # DoubleTensorの代替

out_predict = mnist_model(single_loaded_img_conv)
print(out_predict)

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルをMLflowモデルレジストリに登録

# COMMAND ----------

# モデルをMLflowモデルレジストリに登録しましょう
model_name = 'mnist_example'
registered_model_name = "pytorch_tensor"

# COMMAND ----------

# MAGIC %md
# MAGIC ### シグネチャの追加
# MAGIC 
# MAGIC シグネチャを登録することで、モデルの入力に対するスキーマ強制が可能となります。
# MAGIC 
# MAGIC - [MLflowでTensorの入力をサポートしました \- Qiita](https://qiita.com/taka_yayoi/items/3e439dc5df7257fd41db)
# MAGIC - [Python: MLflow Models を使ってみる \- CUBE SUGAR CONTAINER](https://blog.amedama.jp/entry/mlflow-models#Signature-%E3%82%92%E8%BF%BD%E5%8A%A0%E3%81%99%E3%82%8B)
# MAGIC - [テンソルの基礎  \|  TensorFlow Core](https://www.tensorflow.org/guide/tensor?hl=ja)

# COMMAND ----------

input_img_np = single_loaded_img_conv.to('cpu').detach().numpy().copy()
out_predict_np = out_predict.to('cpu').detach().numpy().copy()

# COMMAND ----------

# MLflowモデルレジストリに格納するためにtensor入力を用いてモデルのシグネチャを作成します
signature = infer_signature(input_img_np, out_predict_np)

# どのように見えるかを確認します
print(signature)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 入力サンプルの準備
# MAGIC 
# MAGIC - MLflowモデルレジストリに格納する入力サンプルを作成します
# MAGIC - 入力サンプルをモデルレジストリに登録しておくと、モデルサービングの画面で入力サンプルを用いて簡単に動作確認を行うことができます
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/demo20220129-pytorch-mlflow/post_example.png)

# COMMAND ----------

# np.expand_dims() は、第2引数の axis で指定した場所の直前に dim=1 を挿入します
input_example = np.expand_dims(input_img_np[0], axis=0)

# COMMAND ----------

input_example.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ### モデルレジストリへの登録
# MAGIC 
# MAGIC 上で準備したシグネチャ、入力サンプルを指定してモデルレジストリに登録します。

# COMMAND ----------

mlflow.pytorch.log_model(mnist_model, model_name, signature=signature, input_example=input_example, registered_model_name=registered_model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルレジストリからモデルをロードして分類
# MAGIC 
# MAGIC モデルレジストリにモデルを登録すると、モデルバージョン固有のURIでモデルをロードすることができるようになります。

# COMMAND ----------

# モデルをロードしてサンプルの予測を実行しましょう
model_version = "1"
loaded_model = mlflow.pytorch.load_model(f"models:/{registered_model_name}/{model_version}")
#loaded_model = mlflow.pytorch.load_model(f"runs:/edcdc804d8ad4d809732cbebe537fad2/mnist_example") # using run id

# COMMAND ----------

data = train_loader.dataset.data[0]
print("data.type:", type(data))
print("data.shape:", data.shape)

imshow(data)

data_conv = data[None, None]
data_conv = data_conv.type('torch.FloatTensor') # DoubleTensorの代替

out_predict = loaded_model(data_conv)

print("分類結果:", out_predict)

# COMMAND ----------

# MAGIC %md
# MAGIC ## REST APIを通じたモデルの呼び出し
# MAGIC 
# MAGIC 上記モデルレジストリに移動し、モデルサービングを有効化します。
# MAGIC ![](https://sajpstorage.blob.core.windows.net/demo20220129-pytorch-mlflow/enable_model_serving.png)
# MAGIC 
# MAGIC これまでのステップでtensorを受け取れるtensorflowモデルをモデルレジストリに登録しているので、REST API経由で画像分類が行えます。REST APIを使用する際には、パーソナルアクセストークンを発行し、REST API呼び出しの中にBearerトークンとして埋め込む必要があります。パーソナルアクセストークンは、サイドメニューの**Settings > User Settings**を開き、**Access Tokens**で**Generate New Token**をクリックします。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/demo20220129-pytorch-mlflow/PAT.png)
# MAGIC 
# MAGIC モデルサービングを有効化してもモデルが**Pending**から**Ready**にならない場合、モデルのデプロイに失敗している可能性があります。サービングの画面下部の**Logs**でエラーが起きていないか確認してください。
# MAGIC ![](https://sajpstorage.blob.core.windows.net/demo20220129-pytorch-mlflow/serving_log.png)
# MAGIC 
# MAGIC [Databricksにおけるモデルサービング \- Qiita](https://qiita.com/taka_yayoi/items/b5a5f83beb4c532cf921#rest-api%E3%83%AA%E3%82%AF%E3%82%A8%E3%82%B9%E3%83%88%E3%81%AB%E3%82%88%E3%82%8B%E3%82%B9%E3%82%B3%E3%82%A2%E3%83%AA%E3%83%B3%E3%82%B0)
# MAGIC 
# MAGIC **参考資料**
# MAGIC - [RESTful API  \|  TFX  \|  TensorFlow](https://www.tensorflow.org/tfx/serving/api_rest#request_format_2)
# MAGIC <br>**Request format**
# MAGIC ```
# MAGIC {
# MAGIC   // (Optional) Serving signature to use.
# MAGIC   // If unspecifed default serving signature is used.
# MAGIC   "signature_name": <string>,
# MAGIC 
# MAGIC   // Input Tensors in row ("instances") or columnar ("inputs") format.
# MAGIC   // A request can have either of them but NOT both.
# MAGIC   "instances": <value>|<(nested)list>|<list-of-objects>
# MAGIC   "inputs": <value>|<(nested)list>|<object>
# MAGIC }
# MAGIC ```
# MAGIC 
# MAGIC - [NumPy配列ndarrayとPython標準のリストを相互に変換 \| note\.nkmk\.me](https://note.nkmk.me/python-numpy-list/)
# MAGIC   - リスト型listをNumPy配列ndarrayに変換: `numpy.array()`
# MAGIC   - NumPy配列ndarrayをリスト型listに変換: `tolist()`

# COMMAND ----------

# MAGIC %md
# MAGIC [Databricksにおけるシークレットの管理 \- Qiita](https://qiita.com/taka_yayoi/items/338ef0c5394fe4eb87c0)

# COMMAND ----------

import os

# トークンは機密性の高い情報なので、ノートブックに直接記載しません。事前にCLIでシークレットとして登録しておいたトークンを呼び出します
token = dbutils.secrets.get("demo-token-takaaki.yayoi", "token")
os.environ["DATABRICKS_TOKEN"] = token

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd

# tensorをエンドポイントに引き渡す際のフォーマットに変換
def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  # モデルのREST APIエンドポイント(モデルサービングの画面で確認できます)
  url = f'https://e2-demo-west.cloud.databricks.com/model/{registered_model_name}/{model_version}/invocations'
  #print(url)
  
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}'}
  
  # datasetがデータフレームの場合はJSONに変換、そうでない場合はtensorを渡す際のJSONにフォーマットに変換
  data_json = dataset.to_dict(orient='split') if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  #print(data_json)
  
  # API呼び出し
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# COMMAND ----------

data = train_loader.dataset.data[10]
print("data.type:", type(data))
print("data.shape:", data.shape)
imshow(data)

data_conv = data[None, None]
data_conv = data_conv.type('torch.FloatTensor') # DoubleTensorの代替
print(data_conv.shape)

# モデルサービングは、比較的小さいデータバッチにおいて低レーテンシーで予測するように設計されています。
served_predictions = score_model(data_conv)
print("分類結果:", served_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ファイルを指定して分類を実行
# MAGIC 
# MAGIC 以下のセルのロジックは、ローカルマシンで画像を指定してモデルを呼び出して分類することを想定しています。

# COMMAND ----------

from PIL import Image
img = Image.open("/dbfs/FileStore/shared_uploads/takaaki.yayoi@databricks.com/five.jpg")
#img = Image.open("/dbfs/FileStore/shared_uploads/takaaki.yayoi@databricks.com/zero.jpg")
tf_image = np.array(img)
#print(tf_image)
#print(tf_image.tolist())

# 0-1に正規化
#vmin = tf_image.min()
#vmax = tf_image.max()
#tf_image_norm = (tf_image - vmin).astype(float) / (vmax - vmin).astype(float)
#print(tf_image_norm)

# Signatureに合わせます
tf_image = np.expand_dims(tf_image, axis=0)
input_example = np.expand_dims(tf_image, axis=0)

served_predictions = score_model(input_example)
print("分類結果:", served_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
