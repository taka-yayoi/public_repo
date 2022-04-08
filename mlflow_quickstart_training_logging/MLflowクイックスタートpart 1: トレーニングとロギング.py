# Databricks notebook source
# MAGIC %md # MLflowクイックスタート: トレーニングとロギング  
# MAGIC 
# MAGIC このチュートリアルは、MLflow [ElasticNet Diabetes example](https://github.com/mlflow/mlflow/tree/master/examples/sklearn_elasticnet_diabetes)をベースとしています。このノートブックでは、モデルのパラメーター、メトリクス、モデル自身、グラフのようなアーティファクトをDatabricksがホストするトラッキングサーバーに記録するモデルトレーニングプロセスのトラッキングに、MLflowをどのように使用するのかを説明しています。これには、記録された結果をMLflowトラッキングUIで参照する方法も含まれています。
# MAGIC 
# MAGIC このノートブックでは、scikit-learnの`diabetes`データセットを使用しており、BMI、血圧などの指標に基づいて進行状態を示すメトリクス(1年後の病状の進展を示す定量的指標)を予測します。scikit-learnのElasticNet線形回帰モデルを使用して、`alpha`と`l1_ratio`をチューニングします。ElasticNetの詳細に関しては、以下を参照ください。
# MAGIC   * [Elastic net regularization](https://en.wikipedia.org/wiki/Elastic_net_regularization)
# MAGIC   * [Regularization and Variable Selection via the Elastic Net](https://web.stanford.edu/~hastie/TALKS/enet_talk.pdf)
# MAGIC   
# MAGIC ## 要件
# MAGIC * このノートブックではDatabricks Runtime 6.4以降、あるいはDatabricks Runtime 6.4 ML以降が必要となります。Databricks Runtime 5.5 LTSあるいはDatabricks Runtime 5.5 LTS MLが稼働しているPython 3を使用することができます。
# MAGIC * Databricksランタイムが稼働しているクラスターを使用している場合には、MLflowをインストールする必要があります。"Install a library on a cluster" ([AWS](https://docs.databricks.com/libraries/cluster-libraries.html#install-a-library-on-a-cluster)|[Azure](https://docs.microsoft.com/azure/databricks/libraries/cluster-libraries#--install-a-library-on-a-cluster)|[GCP](https://docs.gcp.databricks.com/libraries/cluster-libraries.html#install-a-library-on-a-cluster))を参照ください。**Library Source**でPyPIを選択し、**Package**フィールドに`mlflow`を入力ください。
# MAGIC * DatabricksランタイムMLが稼働しているクラスターを使用しているのであれば、MLflowはすべてインストールされています。
# MAGIC 
# MAGIC ### 注意
# MAGIC このノートブックは、DatabricksがホストしているMLflowトラッキングサーバーを使用していることを前提としています。DatabricksのMLflowトラッキングサーバーをプレビューしたいのであれば、アクセスをリクエストするためにはDatabricksの営業窓口にお問い合わせください。自身でトラッキングサーバーをセットアップするには、MLflowの[Tracking Servers](https://www.mlflow.org/docs/latest/tracking.html#mlflow-tracking-servers)の手順に従ってセットアップを行い、[mlflow.set_tracking_uri](https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri)を実行してトラッキングサーバーへの接続を設定してください。

# COMMAND ----------

# MAGIC %md ## ライブラリのインポート、データセットのロード

# COMMAND ----------

# ライブラリのインポート
import os
import warnings
import sys

import pandas as pd
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import lasso_path, enet_path
from sklearn import datasets

# mlflowのインポート
import mlflow
import mlflow.sklearn

# 糖尿病データセットのロード
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# pandas DataFrameの作成 
Y = np.array([y]).transpose()
d = np.concatenate((X, Y), axis=1)
cols = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'progression']
data = pd.DataFrame(d, columns=cols)

# COMMAND ----------

# MAGIC %md ## ElasticNet descent pathをプロットするための関数の作成
# MAGIC 
# MAGIC `plot_enet_descent_path()`関数:
# MAGIC 
# MAGIC * 指定した*l1_ratio*に対するElasticNetモデルの[ElasticNet Descent Path](http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_coordinate_descent_path.html)のプロットを作成し保存します。
# MAGIC * `display()`を用いてノートブックに表示することができる画像を返却します。
# MAGIC * クラスタードライバーノードに`ElasticNet-paths.png`を保存します。

# COMMAND ----------

def plot_enet_descent_path(X, y, l1_ratio):
    # 計算パス
    eps = 5e-3  # 小さいほどパスは長くなります

    # グローバルの画像変数の参照
    global image
    
    print("Computing regularization path using ElasticNet.")
    alphas_enet, coefs_enet, _ = enet_path(X, y, eps=eps, l1_ratio=l1_ratio, fit_intercept=False)

    # 結果の表示
    fig = plt.figure(1)
    ax = plt.gca()

    colors = cycle(['b', 'r', 'g', 'c', 'k'])
    neg_log_alphas_enet = -np.log10(alphas_enet)
    for coef_e, c in zip(coefs_enet, colors):
        l1 = plt.plot(neg_log_alphas_enet, coef_e, linestyle='--', c=c)

    plt.xlabel('-Log(alpha)')
    plt.ylabel('coefficients')
    title = 'ElasticNet Path by alpha for l1_ratio = ' + str(l1_ratio)
    plt.title(title)
    plt.axis('tight')

    # 画像の表示
    image = fig
    
    # 画像の保存
    fig.savefig("ElasticNet-paths.png")

    # プロットのクローズ
    plt.close(fig)

    # 画像の返却
    return image    

# COMMAND ----------

# MAGIC %md ## 糖尿病モデルのトレーニング
# MAGIC 
# MAGIC `train_diabetes()`関数は、入力パラメーター*in_alpha*と*in_l1_ratio*に基づいてElasticNet線形回帰モデルをトレーニングします。
# MAGIC 
# MAGIC この関数は以下を記録するためにMLflowトラッキングを使用します:
# MAGIC * parameters
# MAGIC * metrics
# MAGIC * model
# MAGIC * 上で定義した`plot_enet_descent_path()`関数が作成した画像
# MAGIC 
# MAGIC **Tip:** 新規MLflowのランを作成するためには`with mlflow.start_run:`を使用することをお勧めします。コードが成功したか、エラーで失敗したかに関係なく`with`コンテキストはMLflowのランをクローズするので、`mlflow.end_run`を呼び出す必要はありません。

# COMMAND ----------

# train_diabetes
#   ElasticNetを用いて糖尿病の進行具合を予測するためにskleanのDiabetesデータセットを使用します
#       予測する"progression"カラムはベースラインのあと1年での糖尿病の進行度合いを示す定量的指標となります
#       http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
def train_diabetes(data, in_alpha, in_l1_ratio):
  
  # メトリクスの評価
  def eval_metrics(actual, pred):
      rmse = np.sqrt(mean_squared_error(actual, pred))
      mae = mean_absolute_error(actual, pred)
      r2 = r2_score(actual, pred)
      return rmse, mae, r2

  warnings.filterwarnings("ignore")
  np.random.seed(40)

  # (0.75, 0.25)でトレーニングデータセット、テストデータセットに分割
  train, test = train_test_split(data)

  # 予測する"progression"カラムはベースラインのあと1年での糖尿病の進行度合いを示す定量的指標となります
  train_x = train.drop(["progression"], axis=1)
  test_x = test.drop(["progression"], axis=1)
  train_y = train[["progression"]]
  test_y = test[["progression"]]

  if float(in_alpha) is None:
    alpha = 0.05
  else:
    alpha = float(in_alpha)
    
  if float(in_l1_ratio) is None:
    l1_ratio = 0.05
  else:
    l1_ratio = float(in_l1_ratio)
  
  # MLflowランのスタート; the "with"キーワードを指定することで、このセルがクラッシュしてもランのクローズを保証できます
  with mlflow.start_run():
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    # ElasticNetモデルのメトリクスの出力
    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # mlflow UI向けにmlflow属性を記録
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(lr, "model")
    modelpath = "/dbfs/mlflow/test_diabetes/model-%f-%f" % (alpha, l1_ratio)
    mlflow.sklearn.save_model(lr, modelpath)
    
    # plot_enet_descent_pathの呼び出し
    image = plot_enet_descent_path(X, y, l1_ratio)
    
    # アーティファクトの記録(出力ファイル)
    mlflow.log_artifact("ElasticNet-paths.png")

# COMMAND ----------

# MAGIC %md ## 異なるパラメーターによる実験
# MAGIC 
# MAGIC 異なるパラメーターを指定して`train_diabetes`を呼び出します。これら全てのランをMLflowエクスペリメントで可視化することができます。

# COMMAND ----------

# MAGIC %fs rm -r dbfs:/mlflow/test_diabetes

# COMMAND ----------

# alphaとl1_ratioを0.01, 0.01に
train_diabetes(data, 0.01, 0.01)

# COMMAND ----------

display(image)

# COMMAND ----------

# alphaとl1_ratioを0.01, 0.75に
train_diabetes(data, 0.01, 0.75)

# COMMAND ----------

display(image)

# COMMAND ----------

# alphaとl1_ratioを0.01, .5に
train_diabetes(data, 0.01, .5)

# COMMAND ----------

display(image)

# COMMAND ----------

# alphaとl1_ratioを0.01, 1に
train_diabetes(data, 0.01, 1)

# COMMAND ----------

display(image)

# COMMAND ----------

# MAGIC %md ## MLflow UIでエクスペリメント、ラン、ノートブックのバージョンを参照
# MAGIC 
# MAGIC 結果を参照するには、このページの右上の**Experiment**をクリックします。エクスペリメントサイドバーが表示されます。このサイドバーにはこのノートブック上のランのパラメーター、メトリクスを表示します。循環している矢印アイコンをクリックすることで、最新のランが表示されるように画面をリフレッシュすることができます。
# MAGIC 
# MAGIC パラメーター、メトリクスを持つランの一覧を含むノートブックのエクスペリメントを参照するには、**Experiment Runs**の右にある矢印アイコンをクリックします。新規タブにエクスペリメントページが表示されます。テーブルの**Source**カラムには、個々のランに紐づけられているノートブックのバージョンへのリンクが含まれています。
# MAGIC 
# MAGIC 特定のランの詳細を見るには、当該ランの**Start Time**カラムのリンクをクリックします。あるいは、エクスペリメントサイドバーで、ランの日時の右側にあるアイコンをクリックします。
# MAGIC 
# MAGIC 詳細に関しては、"ノートブックエクスペリメントの参照"を参照ください。 ([AWS](https://qiita.com/taka_yayoi/items/ba0c7f46ff7c3dbf87bb#%E3%83%8E%E3%83%BC%E3%83%88%E3%83%96%E3%83%83%E3%82%AF%E3%82%A8%E3%82%AF%E3%82%B9%E3%83%9A%E3%83%AA%E3%83%A1%E3%83%B3%E3%83%88%E3%81%AE%E5%8F%82%E7%85%A7)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/tracking#view-notebook-experiment)|[GCP](https://docs.gcp.databricks.com/applications/mlflow/tracking.html#view-notebook-experiment)).
