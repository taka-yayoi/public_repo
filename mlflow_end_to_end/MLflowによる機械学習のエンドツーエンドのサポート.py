# Databricks notebook source
# MAGIC %md
# MAGIC # 構造化データに対する機械学習モデルのトレーニング：エンドツーエンドのサンプル
# MAGIC 
# MAGIC このデモでは以下のステップをカバーします。
# MAGIC - データのインポート
# MAGIC - Seabornとmatplotlibによるデータの可視化
# MAGIC - データセットに対して機械学習モデルをトレーニングする際、ハイパーパラメーター探索を並列実行
# MAGIC - MLflowによるハイパーパラメーター探索結果の確認
# MAGIC - ベストモデルをMLflowに登録
# MAGIC - Spark UDFを用いて登録済みモデルを別のデータセットに適用
# MAGIC - 低レーテンシーリクエストに対応するためのモデルサービング
# MAGIC 
# MAGIC この例では、ワインの物理化学的特性に基づいて、ポルトガルの"Vinho Verde"ワインの品質を予測するモデルを構築します。
# MAGIC 
# MAGIC この例では、UCI機械学習リポジトリのデータ[*
# MAGIC Modeling wine preferences by data mining from physicochemical properties*](https://www.sciencedirect.com/science/article/pii/S0167923609001377?via%3Dihub) [Cortez et al., 2009]を活用します。
# MAGIC 
# MAGIC ## 要件
# MAGIC このノートブックではDatabricks MLランタイムが必要です。Databrikcs 7.3 LTS ML以前を利用している場合には、CloudPickleライブラリをアップデートする必要があります。この場合、Cmd 3の`%pip install`コマンドのコメントを解除して実行してください。
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2021/08/10</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>クラスター</td><td>8.3ML</td></tr>
# MAGIC </table>
# MAGIC 
# MAGIC <img style="margin-top:25px;" src="https://sajpstorage.blob.core.windows.net/workshop20210205/databricks-logo-small-new.png" width="140">

# COMMAND ----------

# MAGIC %md ## OSS MLflowとの違い
# MAGIC 
# MAGIC ![](https://databricks.com/wp-content/uploads/2021/06/Table-1.png)

# COMMAND ----------

# Databrikcs 7.3 LTS ML以前を利用している場合にのみ以下のコマンドの実行が必要です。
#%pip install --upgrade cloudpickle

# COMMAND ----------

# MAGIC %md ## データのインポート
# MAGIC 
# MAGIC このセクションでは、サンプルデータからpandasデータフレームにデータを読み込みます。

# COMMAND ----------

import pandas as pd

white_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-white.csv", sep=";")
red_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-red.csv", sep=";")

# COMMAND ----------

# MAGIC %md
# MAGIC ワインが赤ワインか白ワインかを示す"is_red"カラムを追加して、二つのデータフレームを一つのデータセットにマージします。

# COMMAND ----------

red_wine['is_red'] = 1
white_wine['is_red'] = 0

data = pd.concat([red_wine, white_wine], axis=0)

# カラム名から空白を削除
data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

# COMMAND ----------

data.head()

# COMMAND ----------

# MAGIC %md ## データの可視化
# MAGIC 
# MAGIC モデルをトレーニングする前に、Seaborn、matplotlibを用いてデータを可視化します。

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 目的変数のqualityのヒストグラムをプロットします。

# COMMAND ----------

import seaborn as sns
sns.distplot(data.quality, kde=False)

# COMMAND ----------

# MAGIC %md 
# MAGIC qualityは3から9に正規分布しているように見えます。
# MAGIC 
# MAGIC quality >= 7のワインを高品質と定義します。

# COMMAND ----------

high_quality = (data.quality >= 7).astype(int)
data.quality = high_quality

# COMMAND ----------

# MAGIC %md 
# MAGIC 特徴量と2値ラベルの間の相関を見るにはボックスプロットが有用です。

# COMMAND ----------

import matplotlib.pyplot as plt

dims = (3, 4)

f, axes = plt.subplots(dims[0], dims[1], figsize=(25, 15))
axis_i, axis_j = 0, 0
for col in data.columns:
  if col == 'is_red' or col == 'quality':
    continue # カテゴリ変数にボックスプロットは使用できません
  sns.boxplot(x=high_quality, y=data[col], ax=axes[axis_i, axis_j])
  axis_j += 1
  if axis_j == dims[1]:
    axis_i += 1
    axis_j = 0

# COMMAND ----------

# MAGIC %md 上のボックスプロットから、いくつかの変数がqualityに対する単変量予測子として優れていることがわかります。
# MAGIC <br><br>
# MAGIC - alcoholのボックスプロットにおいては、高品質ワインのアルコール含有量の中央値は、低品質のワインの75%パーセンタイルよりも大きな値となっています。
# MAGIC - densityのボックスプロットにおいては、低品質ワインの密度は高品質ワインよりも高い値を示しています。密度は品質と負の相関があります。

# COMMAND ----------

# MAGIC %md ## データの前処理
# MAGIC 
# MAGIC モデルのトレーニングの前に、欠損値のチェックを行い、データをトレーニングデータとバリデーションデータに分割します。

# COMMAND ----------

data.isna().any()

# COMMAND ----------

# MAGIC %md 欠損値はありませんでした。

# COMMAND ----------

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, random_state=123)
X_train = train.drop(["quality"], axis=1)
X_test = test.drop(["quality"], axis=1)
y_train = train.quality
y_test = test.quality

# COMMAND ----------

# MAGIC %md ## ベースラインモデルの構築
# MAGIC 
# MAGIC 出力が2値であり、複数の変数間での相互関係がある可能性があることから、このタスクにはランダムフォレスト分類器が適しているように見えます。
# MAGIC 
# MAGIC 以下のコードでは、scikit-learnを用いてシンプルな分類器を構築します。モデルの精度を追跡するためにMLflowを用い、後ほど利用するためにモデルを保存します。

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

# sklearnのRandomForestClassifierのpredictメソッドは、2値の分類結果(0、1)を返却します。
# 以下のコードでは、それぞれのクラスに属する確率を返却するpredict_probaを用いる、ラッパー関数SklearnModelWrapperを構築します。

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]

  
# mlflow.start_runは、このモデルのパフォーマンスを追跡するための新規MLflowランを生成します。
# コンテキスト内で、使用されたパラメーターを追跡するためにmlflow.log_param、精度のようなメトリクスを追跡するために
# mlflow.log_metricを呼び出します。
with mlflow.start_run(run_name='untuned_random_forest'):
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
  signature = infer_signature(X_train, wrappedModel.predict(None, X_train))
  
  # MLflowにはモデルをサービングする際に用いられるconda環境を作成するユーティリティが含まれています。
  # 必要な依存関係がconda.yamlに保存され、モデルとともに記録されます。
  conda_env =  _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(sklearn.__version__)],
        additional_conda_channels=None,
    )
  mlflow.pyfunc.log_model("random_forest_model", python_model=wrappedModel, conda_env=conda_env, signature=signature)

# COMMAND ----------

# MAGIC %md
# MAGIC サ二ティチェックとして、モデルによって出力される特徴量の重要度を確認します。

# COMMAND ----------

feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns.tolist(), columns=['importance'])
feature_importances.sort_values('importance', ascending=False)

# COMMAND ----------

# MAGIC %md 
# MAGIC 先ほどボックスプロットで見たように、品質を予測するのにアルコールと密度が重要であることがわかります。

# COMMAND ----------

# MAGIC %md 
# MAGIC MLflowにROC曲線のAUCを記録しました。右上の**Experiment**をクリックして、エクスペリメントランのサイドバーを表示します。
# MAGIC 
# MAGIC このモデルはAUC0.89を達成しました。
# MAGIC 
# MAGIC ランダムな分類器のAUCは0.5となり、それよりAUCが高いほど優れていると言えます。詳細は、[Receiver Operating Characteristic Curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve)を参照ください。

# COMMAND ----------

# MAGIC %md #### MLflowモデルレジストリにモデルを登録
# MAGIC 
# MAGIC モデルレジストリにモデルを登録することで、Databricksのどこからでもモデルを容易に参照できるようになります。
# MAGIC 
# MAGIC 以下のセクションでは、どのようにプログラム上から操作をするのかを説明しますが、UIを用いてモデルを登録することもできます。"Create or register a model using the UI" ([AWS](https://docs.databricks.com/applications/machine-learning/manage-model-lifecycle/index.html#create-or-register-a-model-using-the-ui)|[Azure](https://docs.microsoft.com/azure/databricks/applications/machine-learning/manage-model-lifecycle/index#create-or-register-a-model-using-the-ui))を参照ください。

# COMMAND ----------

run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "untuned_random_forest"').iloc[0].run_id

# COMMAND ----------

# モデルレジストリにモデルを登録します
model_name = "wine_quality_taka"
model_version = mlflow.register_model(f"runs:/{run_id}/random_forest_model", model_name)

# モデル登録に数秒を要するので、待ち時間を挿入します。
time.sleep(15)

# COMMAND ----------

# MAGIC %md 
# MAGIC Modelsページでモデルを確認できるはずです。Modelsページを表示するには、左のサイドバーでModelsアイコンをクリックします。
# MAGIC 
# MAGIC 次に、このモデルをproductionに移行し、モデルレジストリからモデルをこのノートブックにロードします。

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Production",
)

# COMMAND ----------

# MAGIC %md 
# MAGIC Modelsページでは、モデルバージョンが`Production`ステージにあると表示されます。
# MAGIC 
# MAGIC これで、`models:/wine_quality/production`のパスでモデルを参照することができます。

# COMMAND ----------

model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")

# サニティチェック: この結果はMLflowで記録されたAUCと一致すべきです
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')

# COMMAND ----------

# MAGIC %md ## 新たなモデルを用いたエクスペリメント
# MAGIC 
# MAGIC ハイパーパラメーターチューニングを行わなくても、このランダムフォレストモデルはうまく動きました。
# MAGIC 
# MAGIC 以下のコードでは、より精度の高いモデルをトレーニングするためにxgboostライブラリを使用します。HyperoptとSparkTrialsを用いて、複数のモデルを並列にトレーニングするために、ハイパーパラメーター探索を並列で処理します。上のコードと同様に、パラメーター設定、パフォーマンスをMLflowでトラッキングします。

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope
from math import exp
import mlflow.xgboost
import numpy as np
import xgboost as xgb

search_space = {
  'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
  'learning_rate': hp.loguniform('learning_rate', -3, 0),
  'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
  'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
  'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
  'objective': 'binary:logistic',
  'seed': 123, # トレーニングの再現性を確保するためにシードを設定します。
}

def train_model(params):
  # MLflowのオートロギングによって、ハイパーパラメーターとトレーニングしたモデルは自動的にMLflowに記録されます。
  mlflow.xgboost.autolog()
  with mlflow.start_run(nested=True):
    train = xgb.DMatrix(data=X_train, label=y_train)
    test = xgb.DMatrix(data=X_test, label=y_test)
    # xgbが評価メトリクスを追跡できるようにテストセットを渡します。XGBoostは、評価メトリクスに改善が見られなくなった際にトレーニングを中止します。
    booster = xgb.train(params=params, dtrain=train, num_boost_round=1000,\
                        evals=[(test, "test")], early_stopping_rounds=50)
    predictions_test = booster.predict(test)
    auc_score = roc_auc_score(y_test, predictions_test)
    mlflow.log_metric('auc', auc_score)

    signature = infer_signature(X_train, booster.predict(train))
    mlflow.xgboost.log_model(booster, "model", signature=signature)
    
    # fminがauc_scoreを最大化するようにlossに-1*auc_scoreを設定します。
    return {'status': STATUS_OK, 'loss': -1*auc_score, 'booster': booster.attributes()}

# 並列度が高いほどスピードを改善できますが、ハイパーパラメータの探索において最適とは言えません。
# max_evalsの平方根が並列度の妥当な値と言えます。
spark_trials = SparkTrials(parallelism=10)

# "xgboost_models"という親のランの子ランとして、それぞれのハイパーパラメーターの設定が記録されるようにMLflowランのコンテキスト内でfminを実行します。
with mlflow.start_run(run_name='xgboost_models'):
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=96,
    trials=spark_trials, 
    rstate=np.random.RandomState(123)
  )

# COMMAND ----------

# MAGIC %md #### MLflowを用いて結果を確認
# MAGIC 
# MAGIC Experiment Runsサイドバーを開いて、ランを参照します。メニューを表示するために、下向き矢印の隣にあるDateをクリックし`auc`を選択し、aucメトリックの順でランを並び替えます。一番高いaucは0.92となっています。ベースラインモデルを上回りました！
# MAGIC 
# MAGIC MLflowはそれぞれのランのパフォーマンスメトリクスとパラメーターをトラッキングします。Experiment Runsサイドバーの一番上にある右上向きの矢印アイコン<img src="https://docs.databricks.com/_static/images/icons/external-link.png"/>をクリックすることで、MLflowランの一覧に移動することができます。

# COMMAND ----------

# MAGIC %md 
# MAGIC 次に、どのようにハイパーパラメータの選択がAUCと相関しているのかを見てみましょう。"+"アイコンをクリックして、親のランを展開し、親以外の全てのランを選択し、"Compare"をクリックします。Parallel Coordinates Plotを選択します。
# MAGIC 
# MAGIC メトリックに対するパラメーターのインパクトを理解するために、Parallel Coordinates Plotは有用です。プロットの右上にあるピンクのスライダーをドラッグすることで、AUCの値のサブセット、対応するパラメーターの値をハイライトすることができます。以下のプロットでは、最も高いAUCの値をハイライトしています。
# MAGIC 
# MAGIC <img src="https://docs.databricks.com/_static/images/mlflow/end-to-end-example/parallel-coordinates-plot.png"/>
# MAGIC 
# MAGIC 最もパフォーマンスの良かったランの全てが、`reg_lambda`と`learning_rate`において低い値を示していることに注意してください。
# MAGIC 
# MAGIC これらのパラメーターに対してより低い値を探索するために、さらなるハイパーパラメーターチューニングを実行することもできますが、ここではシンプルにするために、そのステップをデモに含めていません。

# COMMAND ----------

# MAGIC %md 
# MAGIC それぞれのハイパーパラメーターの設定において生成されたモデルを記録するためにMLflowを用いました。以下のコードでは、最も高いパフォーマンスを示したランを検索し、モデルレジストリにモデルを登録します。

# COMMAND ----------

best_run = mlflow.search_runs(order_by=['metrics.auc DESC']).iloc[0]
print(f'AUC of Best Run: {best_run["metrics.auc"]}')

# COMMAND ----------

# MAGIC %md #### MLflowモデルレジストリのProductionステージにある`wine_quality_taka`モデルを更新
# MAGIC 
# MAGIC はじめに、`wine_quality_taka`という名前でベースラインモデルをモデルレジストリに保存しました。さらに精度の高いモデルができましたので、`wine_quality_taka`を更新します。

# COMMAND ----------

new_model_version = mlflow.register_model(f"runs:/{best_run.run_id}/model", model_name)

# モデル登録に数秒を要するので、待ち時間を挿入します。
time.sleep(15)

# COMMAND ----------

# MAGIC %md 
# MAGIC 左のサイドバーで**Models**をクリックし、`wine_quality_taka`に二つのバージョンが存在することを確認します。
# MAGIC 
# MAGIC 以下のコードで新バージョンをproductionに移行します。

# COMMAND ----------

# 古いモデルバージョンをアーカイブします。
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Archived"
)

# 新しいモデルバージョンをProductionに昇格します。
client.transition_model_version_stage(
  name=model_name,
  version=new_model_version.version,
  stage="Production"
)

# COMMAND ----------

# MAGIC %md load_modelを呼び出すクライアントは新たなモデルを受け取ります。

# COMMAND ----------

# このコードは上の"ベースラインモデルの構築"と同じものです。新たなモデルを利用するためにクライアント側での変更は不要です！
model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')

# COMMAND ----------

# MAGIC %md ## バッチ推論
# MAGIC 
# MAGIC 新たなデータのコーパスに対してモデルを評価したいというシナリオは数多く存在します。例えば、新たなデータバッチを手に入れたり、同じデータコーパスに対して二つのモデルを比較することなどが考えられます。
# MAGIC 
# MAGIC 以下のコードでは、並列に処理を行うためにSparkを用い、Deltaテーブルに格納されたデータに対してモデルの評価を行います。

# COMMAND ----------

# 新たなデータコーパスをシミュレートするために、既存のX_trainデータをDeltaテーブルに保存します。
# 実際の環境では、本当に新たなデータバッチとなります。
spark_df = spark.createDataFrame(X_train)
# Deltaテーブルの保存先(適宜変更してください)
table_path = "dbfs:/tmp/takaakiyayoidatabrickscom/delta/wine_data"
# すでにコンテンツが存在する場合には削除します
dbutils.fs.rm(table_path, True)
spark_df.write.format("delta").save(table_path)

# COMMAND ----------

# MAGIC %md モデルをSparkのUDF(ユーザー定義関数)としてロードし、Deltaテーブルに適用できるようにします。

# COMMAND ----------

import mlflow.pyfunc

apply_model_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{model_name}/production")

# COMMAND ----------

# 新規データをDeltaから読み込みます
new_data = spark.read.format("delta").load(table_path)

# COMMAND ----------

display(new_data)

# COMMAND ----------

from pyspark.sql.functions import struct

# 新規データにモデルを適用します
udf_inputs = struct(*(X_train.columns.tolist()))

new_data = new_data.withColumn(
  "prediction",
  apply_model_udf(udf_inputs)
)

# COMMAND ----------

# それぞれの行には予測結果が紐づけられています。
# xgboostの関数はデフォルトでは確率を出力せず、予測結果が[0, 1]に限定されないことに注意してください。
display(new_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルサービング
# MAGIC 
# MAGIC 低レーテンシーでの予測を行うようにモデルを運用するためには、MLflowのモデルサービング([AWS](https://docs.databricks.com/applications/mlflow/model-serving.html)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/model-serving))を利用して、モデルをエンドポイントにデプロイします。
# MAGIC 
# MAGIC 以下のコードでは、どのようにREST APIを用いてデプロイしたモデルから予測結果を得るのかを説明します。

# COMMAND ----------

# MAGIC %md
# MAGIC モデルのエンドポイントにリクエストするためには、Databricksのトークンが必要です。(右上のプロファイルアイコンの下の)User Settingページでトークンを生成することができます。
# MAGIC 
# MAGIC トークンなど機密性の高い情報はノートブックに記述すべきではありません。シークレットに保存するようにしてください。
# MAGIC 
# MAGIC [Databricksにおけるシークレットの管理 \- Qiita](https://qiita.com/taka_yayoi/items/338ef0c5394fe4eb87c0)

# COMMAND ----------

import os

# 事前にCLIでシークレットにトークンを登録しておきます
token = dbutils.secrets.get("demo-token-takaaki.yayoi", "token")

os.environ["DATABRICKS_TOKEN"] = token

# COMMAND ----------

# MAGIC %md
# MAGIC 左のサイドバーで**Models**をクリックし、登録されているワインモデルに移動します。servingタブをクリックし、**Enable Serving**をクリックします。
# MAGIC 
# MAGIC 次に、**Call The Model**で、リクエストを送信するためのPythonコードスニペットを表示するために**Python**ボタンをクリックします。コードをこのノートブックにコピーします。次のセルと同じようなコードになるはずです。
# MAGIC 
# MAGIC Databricksの外からリクエストするために、このトークンを利用することもできます。

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = 'https://e2-demo-west.cloud.databricks.com/model/wine_quality_taka/2/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}'}
  data_json = dataset.to_dict(orient='split') if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC エンドポイントから得られるモデルの予測結果と、ローカルでモデルを評価した結果は一致すべきです。

# COMMAND ----------

# モデルサービングは、比較的小さいデータバッチにおいて低レーテンシーで予測するように設計されています。
num_predictions = 5
served_predictions = score_model(X_test[:num_predictions])
model_evaluations = model.predict(X_test[:num_predictions])
# トレーニングしたモデルとデプロイされたモデルの結果を比較します。
pd.DataFrame({
  "Model Prediction": model_evaluations,
  "Served Model Prediction": served_predictions,
})

# COMMAND ----------

# MAGIC %md ## モデルのアクセス権管理
# MAGIC 
# MAGIC MLflowモデルレジストリに登録されたMLflowモデルに対しては、6つのレベルのアクセス権を設定できます:**No Permissions**(アクセス権なし)、**Read**(読み取り)、**Edit**(編集)、**Manage Staging Versions**(ステージングバージョンの管理)、**Manage Production Versions**(プロダクションバージョンの管理)、**Manage**(管理)です。
# MAGIC 
# MAGIC 以下のテーブルに権限ごとにできることを示します。
# MAGIC 
# MAGIC | できること| No Permissions | Read | Edit | Manage Staging Versions | Manage Production Versions | Manage | 
# MAGIC |:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
# MAGIC |モデルの作成   |x   |x   |x   |x   |x   |x   |
# MAGIC |モデル詳細、バージョン、ステージ変更リクエスト、アクティビティ、アーティファクトダウンロードURIの表示   |   | x  |x   |x   |x   |x   |
# MAGIC |モデルバージョンステージ変更依頼   |   | x  | x  | x  | x  | x  | 
# MAGIC |新バージョンのモデルの追加   |   |   | x  | x  | x  | x  | 
# MAGIC |モデルの記述更新   |   |   |  x | x  |  x |  x | 
# MAGIC |モデルのステージング状態の変更   |   |   |   |  x(None/Archived/Stagingのみ) | x  |  x |   
# MAGIC |モデルのステージング状態変更リクエストの承認・却下   |   |   |   |  x(None/Archived/Stagingのみ) |   x| x  |   
# MAGIC |モデルのステージング状態変更リクエストのキャンセル(リクエストのキャンセルを参照)   |   |   |   |   |   |  x |   
# MAGIC |アクセス権の変更   |   |   |   |   |   | x  |   
# MAGIC |モデル名の変更   |   |   |   |   |   | x  |   
# MAGIC |モデル及びバージョンの削除   |   |   |   |   |   | x  |   
# MAGIC 
# MAGIC ##### リクエストのキャンセル
# MAGIC > - ステージング状態変更リクエストをした本人は、リクエストをキャンセルできます。

# COMMAND ----------

# MAGIC %md
# MAGIC # END
