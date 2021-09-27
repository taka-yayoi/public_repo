# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow、AutoMLによるモデルアンサンブル
# MAGIC 
# MAGIC ### 前提条件
# MAGIC * Databricks Runtime 8.3 ML以降(9.1 MLで動作確認)
# MAGIC 
# MAGIC ##### 仮説
# MAGIC * アンサンブルはよりロバストで優れたパフォーマンスを達成するものですが、構築、維持が困難なものです。しかし、AutoMLがユーザーからその重荷を取り除きます。
# MAGIC * AutoMLはベストモデルを示すだけでなく、トレーニングされた全てのモデルのサマリーを提供します。
# MAGIC * これによって、ユーザーは単一のモデルよりアンサンブルモデルが優れているかどうかを容易に確認できる様になります。
# MAGIC * 全てのモデルのラン(トレーニング)が記録されているので、特にこのオプションを検証することが容易となります。
# MAGIC 
# MAGIC ##### データセット
# MAGIC * このサンプルでは、Kaggleで提供されている[telco dataset](https://www.kaggle.com/blastchar/telco-customer-churn)を用いて、潜在的な顧客離脱を予測します。

# COMMAND ----------

# MAGIC %md ## データ準備
# MAGIC 
# MAGIC 事前に上記リンク先からCSVファイルをダウンロードし、上のメニューの**File > Upload Data**でDBFSにCSVファイルをアップロードしてください。

# COMMAND ----------

import re
from pyspark.sql.types import * 

# Username を取得
username_raw = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
# Username の英数字以外を除去し、全て小文字化。Username をファイルパスやデータベース名の一部で使用可能にするため。
username = re.sub('[^A-Za-z0-9]+', '', username_raw).lower()

print("username:", username)

# COMMAND ----------

# 生データのロード
input_df = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/takaaki.yayoi@databricks.com/WA_Fn_UseC__Telco_Customer_Churn.csv")

# トレーニングデータセット・テストデータセットの分割
train_df, test_df = input_df.randomSplit([0.90, 0.1], seed=42)

#　new_dfは推論テストで使用します
new_df = test_df 
display(train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### エンコーディング(文字列のchurnを数値に)

# COMMAND ----------

# DBTITLE 0,Encoding (churnString to churn numeric)
from pyspark.sql.functions import when, col
test_df = test_df.withColumn("churn", when(test_df.Churn == 'Yes' ,1).otherwise(0))
train_df = train_df.withColumn("churn", when(train_df.Churn == 'Yes' ,1).otherwise(0))

# COMMAND ----------

# MAGIC %md
# MAGIC ### ノートブックの後半でテストするために予測値を削除

# COMMAND ----------

# DBTITLE 0,Drop the predicted value for testing later in the notebook
import sklearn.metrics
import numpy as np
test_pdf = test_df.toPandas()
y_test = test_pdf["churn"]
X_test = test_pdf.drop("churn", axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## トレーニング
# MAGIC 
# MAGIC ターゲットカラムとトレーニング時間に関する停止条件を指定して、AutoMLによるモデルのトレーニングを実施します。

# COMMAND ----------

# DBTITLE 0,Use AutoML to train models by specifying the target column and expectations around training time
import databricks.automl

# データ出力先
data_dir = f"dbfs:/tmp/{username}/ensemble_automl/"
dbutils.fs.rm(data_dir, True)

# COMMAND ----------

automl_models = databricks.automl.classify(train_df, 
                                   target_col = "churn",
                                   data_dir= data_dir,
                                   timeout_minutes=60, 
                                   max_trials=1000) 

# COMMAND ----------

automl_models

# COMMAND ----------

# MAGIC %md
# MAGIC ### Experiment IDの取得

# COMMAND ----------

# DBTITLE 0,Get the Experiment Id
import mlflow
from mlflow.tracking.client import MlflowClient

client = MlflowClient()
experiment_id = automl_models.experiment.experiment_id
experiment_id

# COMMAND ----------

# MAGIC %md
# MAGIC ### ベストモデルの特定

# COMMAND ----------

# DBTITLE 0,Determine the best model 
print(automl_models.best_trial.model_description)
best_model_uri = automl_models.best_trial.model_path
metrics = automl_models.best_trial.metrics
print('accuracy=', metrics['val_accuracy_score'], ' f1 score=', metrics['val_f1_score'], ' precision=', metrics['val_precision_score'],  \
                ' recall=',metrics['val_recall_score'],  ' roc_auc_score=',metrics['val_roc_auc_score'])
predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri=best_model_uri, result_type="integer")
test_df = test_df.withColumn("bestModel", predict_udf())
display(test_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ベストモデルに対するコンフュージョンマトリクスの取得

# COMMAND ----------

# DBTITLE 0,Fetch Confusion Matrix for Best Model
model = mlflow.sklearn.load_model(best_model_uri)
sklearn.metrics.plot_confusion_matrix(model, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### エクスペリメントのランにおける全ての異なるアーキテクチャからトップモデルを特定

# COMMAND ----------

# MAGIC %md
# MAGIC #### F1スコアに基づき異なるアーキテクチャでのベストモデルに対するコンフュージョンマトリクスを生成

# COMMAND ----------

# DBTITLE 0,Generate Confusion Matrix for Best Models in the Different Architectures by F1 Score
model_types = ['DecisionTree', 'LogisticRegression', 'RandomForest', 'LGBM', 'XGB']
for model_type in model_types:
  filter_str = "params.classifier LIKE '" + model_type + "%'"
  print(filter_str)
  
  models = (client.search_runs(experiment_ids=experiment_id, filter_string=filter_str, order_by=["metrics.val_f1_score DESC"]))
  
  
  if models:
    model = models[0]
    metrics = model.data.metrics
    print('accuracy=', metrics['val_accuracy_score'], ' f1 score=', metrics['val_f1_score'], ' precision=', metrics['val_precision_score'],  \
                ' recall=',metrics['val_recall_score'],  ' roc_auc_score=',metrics['val_roc_auc_score'])
    best_runId = model.info.run_uuid
    model_uri = f"runs:/{best_runId}/model"

    predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type="integer")
    test_df = test_df.withColumn(model_type, predict_udf())
    
    model = mlflow.sklearn.load_model(model_uri)  
    disp = sklearn.metrics.plot_confusion_matrix(model, X_test, y_test)
    disp.ax_.set_title(model_type)

display(test_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### アンサンブルのオプション

# COMMAND ----------

# MAGIC %md 
# MAGIC #### いくつのモデル？
# MAGIC 
# MAGIC ##### 投票戦略
# MAGIC 
# MAGIC * 多数決？
# MAGIC * 75%以上？
# MAGIC * 組み合わせ？

# COMMAND ----------

# MAGIC %md
# MAGIC ### バイアスと個々のモデルのエラーを削減するための個々のモデルによる投票

# COMMAND ----------

# DBTITLE 0,Voting by individual models to reduce bias and errors of individual models
@udf('integer')
def ensembleAll(decision_tree, logistic_regression, random_forest, light_gbm, xgboost):
  votes = decision_tree +logistic_regression + random_forest + light_gbm + xgboost
  if votes >= 4:
    return 1
  else:
    return 0

@udf('integer')
def ensembleTop4(decision_tree, random_forest, light_gbm, xgboost):
  votes = decision_tree + random_forest + light_gbm + xgboost
  if votes >= 3:
    return 1
  else:
    return 0
  
@udf('integer')
def ensembleTop3(random_forest, light_gbm, xgboost):
  votes = random_forest + light_gbm + xgboost
  if votes >= 2:
    return 1
  else:
    return 0
  
@udf('integer')
def ensembleTop2(light_gbm, xgboost):
  votes = light_gbm + xgboost
  if votes >= 1:
    return 1
  else:
    return 0

# COMMAND ----------

# MAGIC %md 
# MAGIC ### AutoMLのベストモデルと様々なアンサンブルの比較

# COMMAND ----------

# MAGIC %md
# MAGIC #### アンサンブルのスコア

# COMMAND ----------

# DBTITLE 0,Ensemble Score
model_type='Ensemble'
test_df = test_df.withColumn('FullEnsemble', ensembleAll('DecisionTree','LogisticRegression','RandomForest','LGBM', 'XGB'))
test_df = test_df.withColumn('ensembleTop4', ensembleTop4('DecisionTree','RandomForest','LGBM', 'XGB'))
test_df = test_df.withColumn('ensembleTop3', ensembleTop3('RandomForest','LGBM', 'XGB'))
test_df = test_df.withColumn('ensembleTop2', ensembleTop2('LGBM', 'XGB'))
display(test_df)   

# COMMAND ----------

# MAGIC %md
# MAGIC #### アンサンブルのメトリクス

# COMMAND ----------

# DBTITLE 0,Ensemble Metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


def plotCM(y_pred, model_type):
  labels = [0,1]
  cm = confusion_matrix(y_test, y_pred, labels)
  print(cm)
  fig = plt.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(cm)
  plt.title(model_type)
  fig.colorbar(cax)
  ax.set_xticklabels([''] + labels)
  ax.set_yticklabels([''] + labels)
  plt.xlabel('Predicted Label')
  plt.ylabel('True Label')
  plt.show()
  print("F1 Score: ", f1_score(y_test, y_pred, average="macro"))
  print("Precision Score: ", precision_score(y_test, y_pred, average="macro"))
  print("Recall Score: ", recall_score(y_test, y_pred, average="macro")) 

# COMMAND ----------

# MAGIC %md
# MAGIC #### ベストモデルと様々なアンサンブルのコンフュージョンマトリクスの生成

# COMMAND ----------

# DBTITLE 0,Generate Confusion Matrices for the best model and the various ensembles
pdf = test_df.toPandas()

y_pred = pdf["bestModel"]
plotCM(y_pred, 'bestModel"')

y_pred = pdf["FullEnsemble"]
plotCM(y_pred, 'FullEnsemble"')

y_pred = pdf["ensembleTop4"]
plotCM(y_pred, 'ensembleTop4')

y_pred = pdf["ensembleTop3"]
plotCM(y_pred, 'ensembleTop3')

y_pred = pdf["ensembleTop2"]
plotCM(y_pred, 'ensembleTop2')

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLFlow: アンサンブル管理のオプション
# MAGIC 
# MAGIC レジストリへのプロモーション

# COMMAND ----------

# MAGIC %md #### アンサンブルの最終的なモデルのリスト

# COMMAND ----------

# DBTITLE 0,Final List of Models in the Ensemble
model_types = ['DecisionTree', 'RandomForest', 'LGBM', 'XGB']

# COMMAND ----------

# MAGIC %md
# MAGIC ### オプション#1: それぞれのモデルを別々に記録

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1 - アンサンブルのモデルを別々にレジストリに記録し、ステージング/プロダクションにプロモーション

# COMMAND ----------

# DBTITLE 0,1 - Log each model of ensemble separately in registry, promote to staging/production
for model_type in model_types:
  filter_str = "params.classifier LIKE '" + model_type + "%'"
  model_name = model_type
  model = (client.search_runs(experiment_ids=experiment_id, filter_string=filter_str, order_by=["metrics.val_f1_score DESC"]))[0]
  best_runId = model.info.run_uuid
  model_uri = f"runs:/{best_runId}/model"
  model_details = mlflow.register_model(model_uri, model_name)

# モデルはステージング、プロダクションにプロモーションされる場合があります
model_stage = 'Production'
for model_name in model_types:
    client.transition_model_version_stage(model_name, 1, stage=model_stage)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2 - 登録されている全てのモデルをロードし、予測のためにアンサンブルを使用

# COMMAND ----------

# DBTITLE 0,2 - Load all registered models and use ensemble to predict
def ensemble_inference(df):
  from mlflow.tracking import MlflowClient
  client = mlflow.tracking.MlflowClient()
  
  # 4モデル全てをロード: プロダクションの最新モデル
  for model_name in model_types:
    model_version = client.get_latest_versions(model_name, stages=[model_stage])[0].version
    model_uri = "models:/{model_name}/{model_stage}".format(model_name=model_name, model_stage=model_stage)
    predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type="integer")
  
    # スコアリング
    df = df.withColumn(model_name, predict_udf())
  
  # udf
  ensemble_predict_df = df.withColumn('prediction', ensembleTop4('DecisionTree','RandomForest','LGBM', 'XGB'))
  display(ensemble_predict_df)
  return ensemble_predict_df

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3 - 新規データのスコアリング

# COMMAND ----------

# DBTITLE 0,3 - Score new data
ensemble_predict_df = ensemble_inference(new_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### オプション#2: 単一のアンサンブルモデルを記録
# MAGIC 
# MAGIC ##### アンサンブルを単一モデルと取り扱える様にpyfuncを使用
# MAGIC * モデルを登録
# MAGIC * アンサンブルを参照した推論
# MAGIC * https://databricks.com/notebooks/dff/01_dff_model.html

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1 - 個々のモデルをロード

# COMMAND ----------

# DBTITLE 0,1 - Load individual models
filter_str = "params.classifier LIKE 'DecisionTree%'"
model = (client.search_runs(experiment_ids=experiment_id, filter_string=filter_str, order_by=["metrics.val_f1_score DESC"]))[0]
best_runId = model.info.run_uuid
DecisionTree_model_uri = f"runs:/{best_runId}/model"    
DecisionTree_model = mlflow.sklearn.load_model(DecisionTree_model_uri) 


filter_str = "params.classifier LIKE 'RandomForest%'"
model = (client.search_runs(experiment_ids=experiment_id, filter_string=filter_str, order_by=["metrics.val_f1_score DESC"]))[0]
best_runId = model.info.run_uuid
RandomForest_model_uri = f"runs:/{best_runId}/model"    
RandomForest_model = mlflow.sklearn.load_model(RandomForest_model_uri)  


filter_str = "params.classifier LIKE 'LGBM%'"
model = (client.search_runs(experiment_ids=experiment_id, filter_string=filter_str, order_by=["metrics.val_f1_score DESC"]))[0]
best_runId = model.info.run_uuid
LGBM_model_uri = f"runs:/{best_runId}/model"    
LGBM_model = mlflow.sklearn.load_model(LGBM_model_uri)

filter_str = "params.classifier LIKE 'XGB%'"
model = (client.search_runs(experiment_ids=experiment_id, filter_string=filter_str, order_by=["metrics.val_f1_score DESC"]))[0]
best_runId = model.info.run_uuid
XGB_model_uri = f"runs:/{best_runId}/model"    
XGB_model = mlflow.sklearn.load_model(XGB_model_uri) 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2 - 個々のモデルを渡すアンサンブルpyfuncモデルを作成し、トラッキングサーバーに記録

# COMMAND ----------

# DBTITLE 0,2 - create an ensemble pyfunc model by passing each individual model to it and log in tracking server
import functools

class Ensemble(mlflow.pyfunc.PythonModel):
  def __init__(self, DecisionTree, RandomForest, LGBM, XGB):
    self.DecisionTree = DecisionTree
    self.RandomForest = RandomForest
    self.LGBM = LGBM
    self.XGB = XGB
  
  def load_context(self, context):
    import numpy as np
    import pandas as pd

    return

  # 指定されたモデルの数に基づいてモデルを決定するヘルパー関数
  def decide(self, votes, num_scores):
    # マルチクラスにおいては0-Nの結果を返却する必要があるためアウトプット、返却ロジックを変更する必要があります
    if votes >= int(num_scores/2) + 1:
      return 1
    else:
      return 0

  # スコアは他の分類器から得られる一連の予測結果のリストとなります
  def ensembleTopN(self, *scores):    
    # マルチクラスに対する投票を作成する際にはこの行を変更する必要があります
    votes = functools.reduce(lambda x, y: x+y, scores)
    num_scores = len(scores)
    decide_with_num_scores = functools.partial(self.decide, num_scores=num_scores)
    decide_vec = np.vectorize(decide_with_num_scores)
    # これは2値分類なので0か1を返却します    
    return decide_vec(votes)

    
  # 入力はpandasデータフレームかシリーズとなります       
  def predict(self, context, model_input):
    import pandas as pd
      
    dt = self.DecisionTree.predict(model_input)
    rf = self.RandomForest.predict(model_input)
    lgbm = self.LGBM.predict(model_input)
    xgb = self.XGB.predict(model_input)
    ensemble = self.ensembleTopN(
      dt,rf,lgbm,xgb
    )
    return pd.DataFrame({
      "DecisionTreePredictions": dt,
      "RandomForestPredictions": rf,
      "LGBMPredictions": lgbm,
      "XGBPredictions": xgb,
      "EnsemblePredeictions": ensemble
    })

with mlflow.start_run(experiment_id=experiment_id) as ensemble_run:
  mlflow.log_param("DecisionTree", DecisionTree_model_uri)
  mlflow.log_param("RandomForest", RandomForest_model_uri)
  mlflow.log_param("LGBM", LGBM_model_uri)
  mlflow.log_param("XGB", XGB_model_uri)
  
  mlflow.pyfunc.log_model("Ensemble", python_model=Ensemble(DecisionTree_model, RandomForest_model, LGBM_model, XGB_model))
  
print(ensemble_run.info.run_uuid)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3 - トラッキングサーバーからモデルの詳細を取得しテスト推論を実施

# COMMAND ----------

# DBTITLE 0,3 - Perform test inference using model details from tracking server
import mlflow
# 前回のランからアンサンブルモデルのランURIを生成
single_ensemble_model = f'runs:/{ensemble_run.info.run_uuid}/Ensemble'
# PyFuncモデルとしてモデルをロード
loaded_model = mlflow.pyfunc.load_model(single_ensemble_model)
  

# pandasデータフレームに対する予測を実施
import pandas as pd
import numpy as np
loaded_model.predict(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4 - アンサンブルモデルをレジストリにプロモーション

# COMMAND ----------

# DBTITLE 0,4 - Promote ensemble model to registry
model_name = 'single_ensemble_model'
model_desc = "Combined Ensemble Model which pickles the 'DecisionTree', 'RandomForest', 'LGBM', 'XGB' models."
client = MlflowClient()
client.create_registered_model(model_name)

# 登録されたモデル名に基づいて新規バージョンのモデルを作成
model_uri = "runs:/{}/Ensemble".format(ensemble_run.info.run_uuid)
print(model_uri)
# 4381625721814023
# 0a2d4b07fea642e1b5b216c7519c5d69
artifact_path = f"dbfs:/databricks/mlflow-tracking/{experiment_id}/{ensemble_run.info.run_uuid}/artifacts/Ensemble"
mv = client.create_model_version(model_name, artifact_path, ensemble_run.info.run_id, description=model_desc)
print("Name: {}".format(mv.name))
print("Version: {}".format(mv.version))
print("Description: {}".format(mv.description))
print("Status: {}".format(mv.status))
print("Stage: {}".format(mv.current_stage))


# COMMAND ----------

# MAGIC %md
# MAGIC #### 5 - モデルをプロダクションに移行

# COMMAND ----------

# DBTITLE 0,5 - Transition Model to Production
client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Production"
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6 - レジストリからアンサンブルをロードし、新規データに対する推論を実施

# COMMAND ----------

# DBTITLE 0,6 - Load ensemble from registry to do inference on new data
import mlflow
import functools
single_ensemble_model = 'models:/{}/Production'.format(model_name)
# PyFuncモデルとしてモデルをロード
loaded_model = mlflow.pyfunc.load_model(single_ensemble_model)
  

# pandasデータフレームに対する予測の実施
import pandas as pd
import numpy as np
loaded_model.predict(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
