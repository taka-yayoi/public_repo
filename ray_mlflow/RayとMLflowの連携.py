# Databricks notebook source
# MAGIC %md
# MAGIC # Ray TuneとMLflowトラッキング: Pytorch Lightningトレーニングの最適化
# MAGIC 
# MAGIC このノートブックでは、Ray TuneとMLflowトラッキングの新たなインテグレーションを活用することで、どのようにPytorchのLightningモデルをトラッキングし、最適化するのかを、ステップバイステップで説明します。Ray Tuneは並列でのハイパーパラメーター探索に、MLflowは実験のロギングに活用します。さあ、始めましょう！
# MAGIC 
# MAGIC **参考資料**
# MAGIC - [RayとMLflow: 分散機械学習アプリケーションの本格運用 \- Qiita](https://qiita.com/taka_yayoi/items/078a5a0a74b18acdb03b)
# MAGIC - [How the Integrations Between Ray & MLflow Aids Distributed ML Production \- The Databricks Blog](https://databricks.com/blog/2021/02/03/ray-mlflow-taking-distributed-machine-learning-applications-to-production.html)
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Amog Kamsetty / Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2021/05/23</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>クラスター</td><td>DBR8.0ML</td></tr>
# MAGIC </table>
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/themes/databricks/assets/images/header_logo_2x.png" alt="logo" width="150"/> 

# COMMAND ----------

# MAGIC %md
# MAGIC ## ライブラリのインストール

# COMMAND ----------

# MAGIC %md
# MAGIC まず初めに必要なライブラリ、MLflow(MLランタイムの場合はインストール済みです)、RayとPytorch Lightningをインストールします。
# MAGIC 
# MAGIC Rayのインストールにおいては、インテグレーションを活用するために最新のwheelをインストールする必要があります。しかし、Ray 1.2リリース以降は安定版をインストールするだけで大丈夫です。
# MAGIC 
# MAGIC 以下のリンク先から使用しているOS、Pythonのバージョンに合致するものを選択する必要があります。
# MAGIC [Installing Ray — Ray v2\.0\.0\.dev0](https://docs.ray.io/en/master/installation.html)

# COMMAND ----------

# MAGIC %pip install pytorch-lightning==1.1.8
# MAGIC %pip install pytorch-lightning-bolts
# MAGIC %pip install torchvision
# MAGIC #%pip install mlflow
# MAGIC 
# MAGIC # Python v3.8、Ubuntu OS向けのRayの最新Wheelsをインストールします
# MAGIC # Ray 1.2リリース後は以下の行のインストールは不要となります
# MAGIC %pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-2.0.0.dev0-cp38-cp38-manylinux2014_x86_64.whl
# MAGIC 
# MAGIC # Rayおよび、その他のRay Tuneの依存ライブラリをインストールします 
# MAGIC %pip install ray[tune]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pytorch Lightningモデルの定義

# COMMAND ----------

# MAGIC %md
# MAGIC 次に使用するPytorch Lightningモデルを定義します。この場合、Pytorch Lightningチュートリアルで使用されているMNISTデータセットに対するシンプルなモデルを使用します。

# COMMAND ----------

import torch
from torch.nn import functional as F
import pytorch_lightning as pl

class LightningMNISTClassifier(pl.LightningModule):
    def __init__(self, config, data_dir=None):
        super(LightningMNISTClassifier, self).__init__()

        self.data_dir = data_dir or os.getcwd()
        self.lr = config["lr"]
        layer_1, layer_2 = config["layer_1"], config["layer_2"]

        # mnist画像は(1, 28, 28)の形状です (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, layer_1)
        self.layer_2 = torch.nn.Linear(layer_1, layer_2)
        self.layer_3 = torch.nn.Linear(layer_2, 10)
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = torch.relu(x)
        x = self.layer_2(x)
        x = torch.relu(x)
        x = self.layer_3(x)
        x = torch.log_softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        acc = self.accuracy(logits, y)
        self.log("ptl/train_loss", loss)
        self.log("ptl/train_accuracy", acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        acc = self.accuracy(logits, y)
        return {"val_loss": loss, "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)

# COMMAND ----------

# MAGIC %md
# MAGIC ## トレーニング関数の作成、モデルのロギング

# COMMAND ----------

# MAGIC %md
# MAGIC モデルをトレーニングするための関数を定義します。
# MAGIC 
# MAGIC ここで大切なのは、トレーニング関数に `mlflow_mixin` デコレータを追加することです。デコレータを追加することで、トレーニング関数で、あらゆるmlflow.trackingメソッドを呼び出すことができ、適切なMLflowランが自動的に記録されます。
# MAGIC 
# MAGIC この例では、トレーニングの開始前にシンプルに `mlflow.pytorch.autolog()` を呼び出します。これにより、明示的に記録を指示することなしに、Pytorch Lightningモデルの全てのメトリクス、パラメーター、モデルアーティファクトを自動的にロギングします。
# MAGIC 
# MAGIC そして、Ray TuneとPytorch Lightningの連携によって、 `TuneReportCallback` がモデルのトレーニングが行われた際に中間結果をTuneにレポートします。
# MAGIC 
# MAGIC 最後に必要なことは、適切なDatabricksの認証情報を設定し、リモートのプロセスからDatabricksサーバーにログインできるようにするということです。

# COMMAND ----------

from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
from ray.tune.integration.mlflow import mlflow_mixin
from ray.tune.integration.pytorch_lightning import TuneReportCallback

@mlflow_mixin
def train_mnist_tune(config):
    host = config["mlflow"]["tracking_uri"]
    token = config["mlflow"]["token"]
    if "databricks" in host:
      import os
      os.environ["DATABRICKS_HOST"] = host
      os.environ["DATABRICKS_TOKEN"] = token
    model = LightningMNISTClassifier(config, config["data_dir"])
    dm = MNISTDataModule(
        data_dir=data_dir, num_workers=0, batch_size=config["batch_size"])
    metrics = {"val_loss": "ptl/val_loss", "val_acc": "ptl/val_accuracy"}
    mlflow.pytorch.autolog()
    trainer = pl.Trainer(
        max_epochs=config["num_epochs"],
        gpus=config["num_gpus"],
        progress_bar_refresh_rate=0,
        callbacks=[TuneReportCallback(metrics, on="validation_end")])
    trainer.fit(model, dm)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ハイパーパラメーター探索の実行

# COMMAND ----------

# MAGIC %md
# MAGIC Ray Tuneを使うために最後に必要なことは、我々のモデルに対する並列ハイパーパラメーター探索処理を起動することです。
# MAGIC 
# MAGIC ここでは、使用するハイパーパラメーターの探索空間をシンプルに定義します。そして、ハイパーパラメーター設定に基づいて、Ray Tuneが多くのトレーニングを並列実行することで、マシンやクラスターの全てのリソースを最大限に活用します。トレーニング関数を定義する際に `mlflow_mixin` デコレータを使用したので、それぞれのトレーニングの実行(ラン)はMLflowに記録されます。
# MAGIC 
# MAGIC ここでは、MLflowのトラッキングURI、あるいは、実験ログや結果を格納する場所を指定する必要があります。これは、ローカルのディレクトリ、サーバーのURI、HTTPサーバーやDatabricksのワークスペースでも構いません。
# MAGIC 
# MAGIC そして最後に、 `mixin` はリモートのRayプロセスからMLflowにロギングするので、適切なDatabricksの認証情報がプロセスに引き渡される必要があります。Databricks上のMLflowを使っていない場合には、このステップは不要です。

# COMMAND ----------

# MAGIC %md
# MAGIC ### データのダウンロード

# COMMAND ----------

data_dir = "./mnist"
# データのダウンロード
MNISTDataModule(data_dir=data_dir).prepare_data()

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLflowの設定

# COMMAND ----------

import mlflow

# 既存のエクスペリメント名、存在しない場合には新たにDatabricksワークスペースに作成したエクスペリメント名を指定します
experiment_name = "/Users/takaaki.yayoi@databricks.com/20210521_Ray_MLflow/Ray integration"

mlflow.set_experiment(experiment_name)

# COMMAND ----------

# Databricks認証情報の取得
import os
from mlflow.tracking._tracking_service import utils
# 注意: リモートトラッキングサーバーにロギングしていない場合、例えば、ローカルファイルシステムにロギングしている場合には、 get_host_creds は undefined となります
host_creds = utils._get_store().get_host_creds()
token = host_creds.token
host = host_creds.host

# COMMAND ----------

# MAGIC %md
# MAGIC ### ハイパーパラメーターの定義

# COMMAND ----------

from ray import tune

# 1 GPUを使用してトレーニングを行う場合には1を指定してください
num_gpus_per_run = 0

# ハイパーパラメーター探索空間の指定
# Tuneはこの探索空間を分解して、トレーニング関数に設定を引き渡します
config = {
        "layer_1": tune.choice([32, 64, 128]),
        "layer_2": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
        # MLflowの設定を引き渡します。Databricksで実行していない場合にはトークンは不要です
        "mlflow": {
            "experiment_name": experiment_name,
            "tracking_uri": host,
            "token": token
        },
        "data_dir": "./mnist",
        "num_epochs": 2,
        "num_gpus": num_gpus_per_run
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ### tune.runの呼び出し

# COMMAND ----------

# MAGIC %md
# MAGIC 全ての準備が整ったので、 `tune.run` を呼び出します。出力結果から、Tuneが数多くの異なるトレーニングランを並列で実行していることが分かると思います。これによって、容易に並列ハイパーパラメーターチューニングを実行できます。また、 `resources_per_trial` フィールドを指定することでGPUを活用することもできます。

# COMMAND ----------

# DatabricksでRayを動作させるHackについては以下を参照ください
# https://forums.databricks.com/questions/45772/is-there-step-by-step-guide-how-to-setup-ray-clust.html
import sys

sys.stdout.fileno = lambda: False

# COMMAND ----------

import ray

# COMMAND ----------

analysis = tune.run(
        train_mnist_tune,
        resources_per_trial={
            "cpu": 1,
            "gpu": num_gpus_per_run
        },
        metric="val_loss",
        mode="min",
        config=config,
        # ハイパーパラメーター探索空間からいくつの異なるサンプルを試すか?
        num_samples=2,
        name="tune_mnist")

print("探索から得られたベストなハイパーパラメーター: ", analysis.best_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLflowを使ったランの可視化、クエリー

# COMMAND ----------

# MAGIC %md
# MAGIC 実験の処理が終了した後は、MLflowを使って、UIでランを可視化したり、プログラムで問い合わせを行うことができます。

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()
runs = client.search_runs(
            [mlflow.get_experiment_by_name(experiment_name).experiment_id])

# COMMAND ----------

# ベストなランを検索し、ベストランのメトリクスを最終結果として記録します
best_val_loss = float("inf")
for r in runs:
    if "best_run" not in r.data.tags and r.data.metrics["ptl/val_loss"] < best_val_loss:
        best_run = r
        best_train_loss = r.data.metrics["ptl/train_loss"]
        best_val_acc = r.data.metrics["ptl/val_accuracy"]
        best_train_acc = r.data.metrics["ptl/train_accuracy"]

# COMMAND ----------

mlflow.end_run()
with mlflow.start_run():
  mlflow.set_tag("best_run", best_run.info.run_id)
  mlflow.log_metrics(
      {
          "train_loss": best_train_loss,
          "val_loss": best_val_loss,
          "train_accuracy": best_train_acc,
          "val_accuracy": best_val_acc
      }
  )

# COMMAND ----------

# MAGIC %md
# MAGIC # END
