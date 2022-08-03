# Databricks notebook source
# MAGIC %md # Databricks機械学習クイックスタート: モデルトレーニング
# MAGIC 
# MAGIC このノートブックでは、Databricksにおける機械学習モデルトレーニングの概要を説明します。モデルをトレーニングするには、Databricks機械学習ランタイムにプレインストールされているscikit-learnのようなライブラリを利用することできます。加えて、トレーニングしたモデルをトラッキングするためにMLflowを利用したり、ハイパーパラメーターチューニングをスケールさせるために、HyperoptとSparkTrialsを活用することができます。
# MAGIC 
# MAGIC このチュートリアルでは以下を行います:
# MAGIC - Part 1: MLflowトラッキングを用いたシンプルな分類モデルのトレーニング
# MAGIC - Part 2: より良いモデルを得るためのHyperoptを用いたハイパーパラメーターチューニング
# MAGIC 
# MAGIC モデルのライフサイクル管理やモデルの推論などDatabricksにおける機械学習の本格運用に関しては、[エンドツーエンドのサンプル](https://qiita.com/taka_yayoi/items/f48ccd35e0452611d81b)をご覧ください。 
# MAGIC 
# MAGIC ### 要件
# MAGIC 
# MAGIC - Databricks機械学習ランタイム7.5以降が稼働しているクラスター

# COMMAND ----------

# MAGIC %md 
# MAGIC ### ライブラリ
# MAGIC 
# MAGIC 必要なライブラリをインポートします。これらのライブラリは[Databricks機械学習ランタイム](https://qiita.com/taka_yayoi/items/824b507019d3ade7eedc)にプレインストールされており、互換性とパフォーマンスに関してチューニングされています。

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.ensemble

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope

# COMMAND ----------

# MAGIC %md 
# MAGIC ### データのロード
# MAGIC 
# MAGIC このチュートリアルでは、異なるワインのサンプルからなるデータセットを使用します。この[データセット](https://archive.ics.uci.edu/ml/datasets/Wine)は、UCI Machine Learning Repositoryからのもので、[DBFS](https://qiita.com/taka_yayoi/items/897264c486e179d72247)に格納されています。ここでのゴールは、品質に基づいて赤ワイン・白ワインを分類するというものです。
# MAGIC 
# MAGIC 他のデータソースからデータをアップロード、ロードする方法の詳細に関しては、データの取り扱いに関するドキュメント([AWS](https://docs.databricks.com/data/index.html)|[Azure](https://docs.microsoft.com/azure/databricks/data/index))をご覧ください。

# COMMAND ----------

# MAGIC %md
# MAGIC 以下のコマンドはCommunity Editionの制限に対応するためのものです。詳細は[こちら](https://databricks.com/jp/international-blogs/get-started-with-databricks-community-edition-jp)の**よくある質問**を参照ください。

# COMMAND ----------

dbutils.fs.cp('dbfs:/databricks-datasets/wine-quality/winequality-white.csv', 'file:/tmp/winequality-white.csv')
dbutils.fs.cp('dbfs:/databricks-datasets/wine-quality/winequality-red.csv', 'file:/tmp/winequality-red.csv')

# COMMAND ----------

# データのロード、前処理
white_wine = pd.read_csv("/tmp/winequality-white.csv", sep=';')
red_wine = pd.read_csv("/tmp/winequality-red.csv", sep=';')
white_wine['is_red'] = 0.0
red_wine['is_red'] = 1.0
data_df = pd.concat([white_wine, red_wine], axis=0)

# ワイン品質に基づく分類ラベルの定義
data_labels = data_df['quality'] >= 7
data_df = data_df.drop(['quality'], axis=1)

# 80/20でトレーニング、テストデータセットに分割
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
  data_df,
  data_labels,
  test_size=0.2,
  random_state=1
)

# COMMAND ----------

# MAGIC %md ## Part 1. 分類モデルのトレーニング

# COMMAND ----------

# MAGIC %md ### MLflowトラッキング
# MAGIC [MLflowトラッキング](https://www.mlflow.org/docs/latest/tracking.html)を用いることで、お使いの機械学習トレーニングのコード、パラメーター、モデルを整理することができます。 
# MAGIC 
# MAGIC [*autologging*](https://www.mlflow.org/docs/latest/tracking.html#automatic-logging)を用いることで、自動MLflowトラッキングを有効化することができます。

# COMMAND ----------

# このノートブックにおけるMLflowオートロギングを有効化
mlflow.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC 次に、MLflowランのコンテキストで分類器をトレーニングし、トレーニングしたモデル、関連するメトリクス、パラメーターを自動で記録します。
# MAGIC 
# MAGIC モデルのAUCスコアやテストデータセットのような追加メトリクスを記録することもできます。

# COMMAND ----------

with mlflow.start_run(run_name='gradient_boost') as run:
  model = sklearn.ensemble.GradientBoostingClassifier(random_state=0)
  
  # モデル、パラメーター、トレーニングメトリクスが自動で記録されます
  model.fit(X_train, y_train)

  predicted_probs = model.predict_proba(X_test)
  roc_auc = sklearn.metrics.roc_auc_score(y_test, predicted_probs[:,1])
  
  # テストデータにおけるAUCスコアは自動で記録されないので、手動で記録します
  mlflow.log_metric("test_auc", roc_auc)
  print("Test AUC of: {}".format(roc_auc))

# COMMAND ----------

# MAGIC %md
# MAGIC 画面右上の**エクスペリメント**に**1**というバッジが表示されているかと思います。こちらをクリックすることで記録されたモデルを確認することができます。
# MAGIC 
# MAGIC このモデルのパフォーマンスに満足できないのであれば、異なるハイパーパラメータを指定して別のモデルをトレーニングします。

# COMMAND ----------

# 後で参照できるようにrun_nameを指定して新たなランを実行します
with mlflow.start_run(run_name='gradient_boost') as run:
  model_2 = sklearn.ensemble.GradientBoostingClassifier(
    random_state=0, 
    
    # n_estimatorsに対して新たなパラメーターを指定してみます
    n_estimators=200,
  )
  model_2.fit(X_train, y_train)

  predicted_probs = model_2.predict_proba(X_test)
  roc_auc = sklearn.metrics.roc_auc_score(y_test, predicted_probs[:,1])
  mlflow.log_metric("test_auc", roc_auc)
  print("Test AUC of: {}".format(roc_auc))

# COMMAND ----------

# MAGIC %md ### MLflowランの参照
# MAGIC 
# MAGIC 記録されたトレーニングランを参照するには、エクスペリメントサイドバーを表示するために、ノートブックの右上の**エクスペリメント**アイコンをクリックします。必要であれば、最新の状態を表示するためにリフレッシュアイコンをクリックします。
# MAGIC 
# MAGIC <img width="350" src="https://docs.databricks.com/_static/images/mlflow/quickstart/experiment-sidebar-icons.png"/>
# MAGIC 
# MAGIC より詳細なMLflowエクスペリメントページ([AWS](https://docs.databricks.com/applications/mlflow/tracking.html#notebook-experiments)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/tracking#notebook-experiments))を表示するために、エクスペリメントページアイコン(右上矢印のアイコン)をクリックすることができます。このページでは、ランを比較したり、特定のランの詳細を参照できます。
# MAGIC 
# MAGIC <img width="800" src="https://docs.databricks.com/_static/images/mlflow/quickstart/compare-runs.png"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ### モデルのロード
# MAGIC 
# MAGIC MLflow APIを用いて特定のランの結果にアクセスすることもできます。以下のセルのコードでは、特定のMLflowランにおいてトレーニングされたモデルをロードして予測を行う方法を示しています。[MLflowランページ](https://qiita.com/taka_yayoi/items/ba0c7f46ff7c3dbf87bb#%E3%83%8E%E3%83%BC%E3%83%88%E3%83%96%E3%83%83%E3%82%AF%E3%82%A8%E3%82%AF%E3%82%B9%E3%83%9A%E3%83%AA%E3%83%A1%E3%83%B3%E3%83%88%E3%81%AE%E5%8F%82%E7%85%A7)で特定のモデルをロードするためのコードスニペットを参照することもできます。

# COMMAND ----------

# モデルが記録された後は、別のノートブックやジョブからロードすることができます
# mlflow.pyfunc.load_modelを用いることで一般的なAPIを用いたモデルの予測が可能になります
model_loaded = mlflow.pyfunc.load_model(
  'runs:/{run_id}/model'.format(
    run_id=run.info.run_id
  )
)

predictions_loaded = model_loaded.predict(X_test)
predictions_original = model_2.predict(X_test)

# ロードしたモデルはオリジナルと一致しなくてはなりません
assert(np.array_equal(predictions_loaded, predictions_original))

# COMMAND ----------

# MAGIC %md ## Part 2. ハイパーパラメーターチューニング
# MAGIC 
# MAGIC ここまでで、あなたはシンプルなモデルをトレーニングし、あなたの作業を整理するためにMLflowサービスを使用しました。このセクションでは、Hyperoptを用いてどのように洗練されたチューニングを行うのかをカバーします。

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hyperopt、SparkTrialsによる並列トレーニング
# MAGIC 
# MAGIC [Hyperopt](http://hyperopt.github.io/hyperopt/)はハイパーパラメーターチューニングのためのPythonライブラリです。DatabricksにおけるHyperoptの詳細に関しては[ドキュメント](https://qiita.com/taka_yayoi/items/238ecf8b038151b84bc1)を参照ください。
# MAGIC 
# MAGIC 並列でハイパーパラメーターの探索、複数のモデルのトレーニングを実行するために、HyperoptとSparkTrialsを活用することができます。これにより、モデルのパフォーマンス最適化に必要な時間を短縮することができます。MLflowとらっきんぐはHyperoptとインテグレーションされているので、モデルとパラメーターを自動で記録します。
# MAGIC 
# MAGIC **注意**
# MAGIC Community Editionのクラスターで以下を実行すると約7分かかります。速度を上げるにはクラスター構成のノード数を増やし、並列度を上げるといった対策が可能です。

# COMMAND ----------

# 探索する検索空間を定義
search_space = {
  'n_estimators': scope.int(hp.quniform('n_estimators', 20, 1000, 1)),
  'learning_rate': hp.loguniform('learning_rate', -3, 0),
  'max_depth': scope.int(hp.quniform('max_depth', 2, 5, 1)),
}

def train_model(params):
  # ワーカーごとにautologgingを有効化
  mlflow.autolog()
  with mlflow.start_run(nested=True):
    model_hp = sklearn.ensemble.GradientBoostingClassifier(
      random_state=0,
      **params
    )
    model_hp.fit(X_train, y_train)
    predicted_probs = model_hp.predict_proba(X_test)
    # テストデータセットのAUCに基づくチューニング
    # 実運用の環境においては、別に検証用データセットを用いることになるかもしれません
    roc_auc = sklearn.metrics.roc_auc_score(y_test, predicted_probs[:,1])
    mlflow.log_metric('test_auc', roc_auc)
    
    # 損失関数を-1*auc_scoreに設定し、fminがauc_scoreを最大化するようにします
    return {'status': STATUS_OK, 'loss': -1*roc_auc}

# SparkTrialsは、Sparkのワーカーを用いてチューニングを分散します
# 並列度を上げれば処理速度を向上しますが、それぞれのハイパーパラメーターのトライアルは他のトライアルから得られる情報が減ります
# 小規模クラスターやDatabricksコミュニティエディションではparallelism=2を設定するようにしてください
spark_trials = SparkTrials(
  parallelism=8
)

with mlflow.start_run(run_name='gb_hyperopt') as run:
  # 最大のAUCを達成するパラメーターを見つけるためにhyperoptを使用します
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=32,
    trials=spark_trials)

# COMMAND ----------

# MAGIC %md ### ベストモデルを取得するためにランを検索
# MAGIC 
# MAGIC すべてのランはMLflowによってトラッキングされているので、MLflowのsearch runs APIを用いて、最も高いAUCを出したベストのモデルのメトリクスとパラメーターを取得することができます。
# MAGIC 
# MAGIC チューニングされたモデルは、Part 1でトレーニングしたシンプルなモデルよりも高いパフォーマンスを示すべきです。

# COMMAND ----------

# テストデータセットにおけるAUCでランをソートします。同じ順位の場合には最新のランを採用します
best_run = mlflow.search_runs(
  order_by=['metrics.test_auc DESC', 'start_time DESC'],
  max_results=10,
).iloc[0]
print('Best Run')
print('AUC: {}'.format(best_run["metrics.test_auc"]))
print('Num Estimators: {}'.format(best_run["params.n_estimators"]))
print('Max Depth: {}'.format(best_run["params.max_depth"]))
print('Learning Rate: {}'.format(best_run["params.learning_rate"]))

best_model_pyfunc = mlflow.pyfunc.load_model(
  'runs:/{run_id}/model'.format(
    run_id=best_run.run_id
  )
)
best_model_predictions = best_model_pyfunc.predict(X_test[:5])
print("Test Predictions: {}".format(best_model_predictions))

# COMMAND ----------

# MAGIC %md ### UIで複数のランを比較
# MAGIC 
# MAGIC Part 1と同じように、**エクスペリメント**サイドバーの上にある外部リンクアイコンからアクセスできるMLflowエクスペリメント詳細ページでランを参照、比較することができます。
# MAGIC 
# MAGIC エクスペリメント詳細ページで、親のランを展開するために"+"アイコンをクリックし、親以外の全てのランを選択、**比較**をクリックします。それぞれのパラメーターのメトリックに対するインパクトを表示するparallel coordinates plotを用いて、異なるランを可視化することができます。
# MAGIC <img width="800" src="https://docs.databricks.com/_static/images/mlflow/quickstart/parallel-plot.png"/>
