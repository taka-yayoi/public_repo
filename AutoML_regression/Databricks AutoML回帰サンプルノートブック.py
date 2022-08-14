# Databricks notebook source
# MAGIC %md # AutoML回帰サンプルノートブック
# MAGIC 
# MAGIC ## 要件
# MAGIC Databricks Runtime for Machine Learning 8.3以降が必要です。

# COMMAND ----------

# MAGIC %md ## カルフォルニア住宅データセット
# MAGIC 
# MAGIC このデータセットは1990年のアメリカの国勢調査から作成されたものです。一行が国勢調査のブロックグループとなります。目的変数はカルフォルニアの地域における住宅価格の中央値です。

# COMMAND ----------

import sklearn
input_pdf = sklearn.datasets.fetch_california_housing(as_frame=True)
display(input_pdf.frame)

# COMMAND ----------

# MAGIC %md ## トレーニングデータセットとテストデータセットの分割

# COMMAND ----------

from sklearn.model_selection import train_test_split

train_pdf, test_pdf = train_test_split(input_pdf.frame, test_size=0.01, random_state=42)
display(train_pdf)

# COMMAND ----------

# MAGIC %md # トレーニング
# MAGIC 
# MAGIC 以下のコマンドでAutoMLを実行します。モデルが予測すべき目的変数を、引数`target_col`で指定する必要があります。実行が完了すると、トレーニングでベストなモデルを生成したノートブックにアクセスして、コードを確認することができます。このノートブックには特徴量の重要度のグラフも含まれています。

# COMMAND ----------

from databricks import automl
summary = automl.regress(train_pdf, target_col="MedHouseVal", timeout_minutes=30)

# COMMAND ----------

# MAGIC %md 以下のコマンドでAutoMLの出力に関する情報を表示します。

# COMMAND ----------

help(summary)

# COMMAND ----------

# MAGIC %md # 次のステップ
# MAGIC 
# MAGIC - 上でリンクされているノートブックやエクスペリメントを確認します。
# MAGIC - ベストモデルのノートブックのメトリクスが好適に見える場合には、次の推論セクションに移動します。
# MAGIC - ベストモデルとして生成されたモデルを改善したいと考える場合には、以下を実施します。
# MAGIC   - ベストトライアルのノートブックをコピーしてオープンします。
# MAGIC   - モデルを改善するために必要な修正を加えます。例えば、異なるハイパーパラメーターを指定します。
# MAGIC   - モデルに満足できたら、トレーニングされたモデルが記録されているアーティファクトのURIを記録します。このURIをCmd12の`model_uri`に指定します。

# COMMAND ----------

# MAGIC %md
# MAGIC # 推論
# MAGIC 
# MAGIC 新たなデータを用いて予測を行う際に、AutoMLでトレーニングしたモデルを活用することが可能です。以下の例では、pandasデータフレームのデータに対してどのように予測を行うのか、Sparkデータフレームに対して予測を行うために、モデルをどのようにSparkのUDF(ユーザー定義関数)として登録するのかをデモします。

# COMMAND ----------

# MAGIC %md ## pandasデータフレーム

# COMMAND ----------

model_uri = summary.best_trial.model_path
# model_uri = "<model-uri-from-generated-notebook>"

# COMMAND ----------

import mlflow

# テストデータセットの準備
y_test = test_pdf["MedHouseVal"]
X_test = test_pdf.drop("MedHouseVal", axis=1)

# ベストモデルによる推論の実行
model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(X_test)
test_pdf["MedHouseVal_predicted"] = predictions
display(test_pdf)

# COMMAND ----------

# MAGIC %md ## Sparkデータフレーム

# COMMAND ----------

# テストデータセットの準備
test_df = spark.createDataFrame(test_pdf)

#features = mlflow.pyfunc.load_model(model_uri).metadata.get_input_schema().column_names()
predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)
display(test_df.withColumn("MedHouseVal_predicted", predict_udf()))

# COMMAND ----------

# MAGIC %md ## テスト
# MAGIC 
# MAGIC 実運用環境において、最終的なベストモデルがどれだけの性能を発揮するのかを見積もるために、ホールドアウトしておいたテストセットで予測を行います。

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# データセットの準備
y_pred = test_pdf["MedHouseVal_predicted"]
test = pd.DataFrame({"Predicted":y_pred,"Actual":y_test})
test = test.reset_index()
test = test.drop(["index"], axis=1)

# グラフのプロット
fig= plt.figure(figsize=(16,8))
plt.plot(test[:50])
plt.legend(["Actual", "Predicted"])
sns.jointplot(x="Actual", y="Predicted", data=test, kind="reg");
