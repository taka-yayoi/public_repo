# Databricks notebook source
# MAGIC %md 
# MAGIC # Hyperoptの分散処理と自動化MLflowトラッキング
# MAGIC 
# MAGIC [Hyperopt](https://github.com/hyperopt/hyperopt)はハイパーパラメーターチューニングのためのPythonライブラリです。Databricks機械学習ランタイムには、最適化、エンハンスされたバージョンのHyperopt、自動化されたMLflowトラッキング、分散チューニングのための`SparkTrials`クラスが含まれています。
# MAGIC 
# MAGIC このノートブックでは、シングルマシンのPython MLアルゴリズムのハイパーパラメーターチューニングをどのようにスケールアップするのか、MLflowを用いて結果をトラッキングするのかを説明します。パート1では、シングルマシンのHyperoptのワークフローを作成します。パート2では、Sparkクラスターでワークフローの計算処理を分散するための`SparkTrials`の使い方を学びます。

# COMMAND ----------

# MAGIC %md ## 必要なパッケージのインポート、データセットのロード

# COMMAND ----------

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials

import mlflow

# COMMAND ----------

# scikit-learnからirisデータセットのロード
iris = iris = load_iris()
X = iris.data
y = iris.target

# COMMAND ----------

# MAGIC %md ## Part 1. シングルマシンのHyperoptワークフロー
# MAGIC 
# MAGIC Hyperoptワークフローのステップを以下に示します:  
# MAGIC 1. 最小化する関数を定義します  
# MAGIC 2. ハイパーパラメーターの探索空間を定義します
# MAGIC 3. 探索アルゴリズムを選択します 
# MAGIC 4. Hyperoptの`fmin()`を用いてチューニングアルゴリズムを実行します
# MAGIC 
# MAGIC 詳細に関しては[Hyperopt documentation](https://github.com/hyperopt/hyperopt/wiki/FMin)をご覧ください。

# COMMAND ----------

# MAGIC %md ### 最小化する関数の定義
# MAGIC 
# MAGIC このサンプルでは、サポートベクトルマシンの分類器を使用します。正則化パラメーター`C`の最適な値を見つけ出すことが目的となります。
# MAGIC 
# MAGIC Hyperoptワークフローにおけるコードの大部分は目的関数となります。このサンプルでは、[scikit-learnのサポートベクトルマシン分類器](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)を使用します。

# COMMAND ----------

def objective(C):
    # サポートベクトルマシンの分類モデルの作成
    clf = SVC(C)
    
    # モデルのパフォーマンスを比較するために交差検証の精度を使用します
    accuracy = cross_val_score(clf, X, y).mean()
    
    # Hyperoptは目的関数を最小化しようとします。精度が高いほど良いモデルを意味するので、負の精度を返却しなくてはなりません
    return {'loss': -accuracy, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md ### ハイパーパラメーターに対する探索空間の定義
# MAGIC 
# MAGIC 探索空間とパラメーターの表現の定義の詳細については[Hyperopt docs](https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions)をご覧ください。

# COMMAND ----------

search_space = hp.lognormal('C', 0, 1.0)

# COMMAND ----------

# MAGIC %md ### 探索アルゴリズムの選択
# MAGIC 
# MAGIC 主な2つの選択肢は以下のものとなります:
# MAGIC * `hyperopt.tpe.suggest`: Tree of Parzen Estimators, 過去の結果に基づいて探索するハイパーパラメーターセッティングをイテレーティブ、適合的に選択するベイジアンアプローチ
# MAGIC * `hyperopt.rand.suggest`: ランダムサーチ、探索空間をサンプリングする非適合的アプローチ

# COMMAND ----------

algo=tpe.suggest

# COMMAND ----------

# MAGIC %md Hyperoptの`fmin()`を用いてチューニングアルゴリズムを実行します。
# MAGIC 
# MAGIC テストすべきハイパーパラメーター空間における最大ポイント数として`max_evals`を設定します。すなわち、フィッティング、評価するモデルの最大数となります。

# COMMAND ----------

argmin = fmin(
  fn=objective,
  space=search_space,
  algo=algo,
  max_evals=16)

# COMMAND ----------

# Cに対するベストな値を表示します
print("Best value found: ", argmin)

# COMMAND ----------

# MAGIC %md ## Part 2. Apache SparkとMLflowを用いた分散チューニング
# MAGIC 
# MAGIC チューニングを分散するには、`fmin()`に引数`SparkTrials`という`Trials`を追加します。
# MAGIC 
# MAGIC `SparkTrials`は2つのオプションの引数を受け取ります:  
# MAGIC * `parallelism`: 同時にフィッティング、評価するモデルの数。デフォルトでは利用可能なSparkのタスクのスロット数となります。
# MAGIC * `timeout`: Maximum time (in seconds) that `fmin()`が実行される最長の時間(秒)です。デフォルトでは制限がありません。
# MAGIC 
# MAGIC このサンプルでは、Cmd 7で定義される非常にシンプルな目的関数を使用します。この場合、関数はクイックに実行され、Sparkジョブ起動のオーバーヘッドが計算時間の大部分を占めるので、分散されるケースでの計算処理がより多くの時間を要するようになります。典型的なリアルワールドの問題においては、目的関数はより複雑なものとなり、計算処理を分散させるために`SparkTrials`を用いることで、シングルマシンのチューニングよりも高速に処理を行うことができます。
# MAGIC 
# MAGIC 自動化MLflowトラッキングはデフォルトで有効化されています。使用するためには、例に示しているように`fmin()`の呼び出しの前に`mlflow.start_run()`を呼び出します。

# COMMAND ----------

from hyperopt import SparkTrials

# SparkTrialsクラスのAPIドキュメントを表示するには以下の行のコメントを解除してください。
#help(SparkTrials)

# COMMAND ----------

spark_trials = SparkTrials()

with mlflow.start_run():
  argmin = fmin(
    fn=objective,
    space=search_space,
    algo=algo,
    max_evals=16,
    trials=spark_trials)

# COMMAND ----------

# Cに対するベストな値を表示します
print("Best value found: ", argmin)

# COMMAND ----------

# MAGIC %md 
# MAGIC ノートブックに関連づけられているMLflowエクスペリメントを参照するためには、右上のノートブックコンテキストバーの**Experiment**をクリックします。ここでは、すべてのランを参照することができます。MLflow UIでランを参照するには、**Experiment Runs**の一番右にあるアイコンをクリックします。
# MAGIC 
# MAGIC `C`のチューニングによる効果を検証するには:
# MAGIC 
# MAGIC 1. 結果のランを選択して**Compare**をクリックします
# MAGIC 1. Scatter PlotでX軸に**C**、Y軸に**loss**を選択します
