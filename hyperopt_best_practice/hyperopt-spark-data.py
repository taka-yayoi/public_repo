# Databricks notebook source
# MAGIC %md ## Hyperopt: 異なるサイズのデータセットに対するベストプラクティス
# MAGIC 
# MAGIC このノートブックでは、異なるサイズのデータセットを対象に、Hyperoptクラス`SparkTrials`を使用する際のガイドラインを説明します。
# MAGIC * 小規模 (~10MB以下)
# MAGIC * 中規模 (~100MB)
# MAGIC * 大規模 (~1GB以上)
# MAGIC 
# MAGIC このノートブックではランダムに生成したデータセットを使用します。ここでのゴールはLASSOモデルにおける正則化パラメーター`alpha`をチューニングすることです。
# MAGIC 
# MAGIC 要件:
# MAGIC * Databricks Runtime 6.4 ML, あるいは Databricks Runtime 7.0 ML 以降
# MAGIC * 2台のワーカーノード

# COMMAND ----------

import numpy as np
import os, shutil, tempfile
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK
from sklearn import linear_model, datasets, model_selection

# COMMAND ----------

# ユーティリティメソッド

def gen_data(bytes):
  """
  指定されたバイト数に基づき、ランダム回帰問題に対するトレーニング/テストデータを生成
  (X_train, X_test, y_train, y_test)を返却
  """
  n_features = 100
  n_samples = int((1.0 * bytes / (n_features + 1)) / 8)
  X, y = datasets.make_regression(n_samples=n_samples, n_features=n_features, random_state=0)
  return model_selection.train_test_split(X, y, test_size=0.2, random_state=1)

def train_and_eval(data, alpha):
  """
  入力値のalphaとトレーニングデータを用いてLASSOモデルをトレーニングし、テストデータを用いて評価
  """
  X_train, X_test, y_train, y_test = data  
  model = linear_model.Lasso(alpha=alpha)
  model.fit(X_train, y_train)
  loss = model.score(X_test, y_test)
  return {"loss": loss, "status": STATUS_OK}

def tune_alpha(objective):
  """
  alphaを入力としlossを返却するobjectiveをチューニングするためにHyperoptのSparkTrialsを使用
  ベストなalphaを返却
  """
  best = fmin(
    fn=objective,
    space=hp.uniform("alpha", 0.0, 10.0),
    algo=tpe.suggest,
    max_evals=4,
    trials=SparkTrials(parallelism=2))
  return best["alpha"]

# COMMAND ----------

# MAGIC %md ### 小規模データセット (~10MB以下)
# MAGIC 
# MAGIC データセットが小さい場合には、ドライバーにデータをロードし、目的関数から直接呼び出します。`SparkTrials`は自動でデータと目的関数をワーカーにブロードキャストします。オーバーヘッドは無視できるものです。

# COMMAND ----------

# 10MBのデータセットを生成
data_small = gen_data(10 * 1024 * 1024) 

# COMMAND ----------

def objective_small(alpha):
  # 小規模データは直接呼び出します。
  return train_and_eval(data_small, alpha)

tune_alpha(objective_small)

# COMMAND ----------

# MAGIC %md ### 中規模データセット (~100MB)
# MAGIC 
# MAGIC 中規模データセットを目的変数から直接呼び出すことが非効率になる場合があります。目的関数のコードを変更すると、再度データをブロードキャストする必要が出てきます。Sparkを使用する際に明示的にデータをブロードキャストし、ワーカーのブロードキャスト変数から値を取り出すことをお勧めします。

# COMMAND ----------

# 100MBのデータを生成
data_medium = gen_data(100 * 1024 * 1024)
# 中規模データにおいては最初にブロードキャストします
bc_data_medium = sc.broadcast(data_medium)

# COMMAND ----------

def objective_medium(alpha):
  # ブロードキャストされたデータをワーカーにロードします
  data = bc_data_medium.value
  return train_and_eval(data, alpha)

tune_alpha(objective_medium)

# COMMAND ----------

# MAGIC %md ### 大規模データセット (~1GB以上)
# MAGIC 
# MAGIC 大規模データセットのブロードキャストには膨大なクラスターリソースが必要となります。データをDBFSに格納し、DBFSローカルファイルインタフェースを用いてワーカーにデータをロードすることを検討してください。

# COMMAND ----------

# ユーティリティメソッド

def save_to_dbfs(data):
  """
  入力データ(numpy配列のタプル)をDBFS上の一時ファイルとして保存しパスを返却
  """
  # 最初にローカルファイルとしてデータを保存
  data_filename = "data.npz"
  local_data_dir = tempfile.mkdtemp()
  local_data_path = os.path.join(local_data_dir, data_filename)
  np.savez(local_data_path, *data)
  
  # クラスターノードにシェアできるようにデータをDBFSに移動
  dbfs_tmp_dir = "/dbfs/ml/tmp/hyperopt"
  os.makedirs(dbfs_tmp_dir, exist_ok=True)
  dbfs_data_dir = tempfile.mkdtemp(dir=dbfs_tmp_dir)  
  dbfs_data_path = os.path.join(dbfs_data_dir, data_filename)  
  shutil.move(local_data_path, dbfs_data_path)
  return dbfs_data_path

def load(path):
  """
  保存データ(numpy配列のタプル)のロード
  """
  return list(np.load(path).values())

# COMMAND ----------

# 　1GBのデータセットを生成
data_large = gen_data(1000 * 1024 * 1024) 
# 大規模データセットにおいては最初にDBFSに保存します
data_large_path = save_to_dbfs(data_large)

# COMMAND ----------

def objective_large(alpha):
  # DBFSからワーカーにデータをロードします
  data = load(data_large_path)
  return train_and_eval(data, alpha)

tune_alpha(objective_large)

# COMMAND ----------

# 大規模データセットを削除
shutil.rmtree(data_large_path, ignore_errors=True)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
