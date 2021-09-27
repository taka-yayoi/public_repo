# Databricks notebook source
# MAGIC %md # AutoML分類サンプルノートブック
# MAGIC 
# MAGIC ## 要件
# MAGIC Databricks Runtime for Machine Learning 8.3以降が必要です。

# COMMAND ----------

# MAGIC %md ## 国勢調査 年収データセット
# MAGIC 
# MAGIC このデータセットには1994年の国勢調査データベースから取得された国勢調査データが含まれています。それぞれの行が個人を表しています。ここでのゴールは、当該人物が年収5万ドルより多いか否かを判断することです。分類結果は**income**列に文字列`<=50K`あるいは`>50k`で表現されます。

# COMMAND ----------

from pyspark.sql.types import DoubleType, StringType, StructType, StructField

schema = StructType([
  StructField("age", DoubleType(), False),
  StructField("workclass", StringType(), False),
  StructField("fnlwgt", DoubleType(), False),
  StructField("education", StringType(), False),
  StructField("education_num", DoubleType(), False),
  StructField("marital_status", StringType(), False),
  StructField("occupation", StringType(), False),
  StructField("relationship", StringType(), False),
  StructField("race", StringType(), False),
  StructField("sex", StringType(), False),
  StructField("capital_gain", DoubleType(), False),
  StructField("capital_loss", DoubleType(), False),
  StructField("hours_per_week", DoubleType(), False),
  StructField("native_country", StringType(), False),
  StructField("income", StringType(), False)
])
input_df = spark.read.format("csv").schema(schema).load("/databricks-datasets/adult/adult.data")

# COMMAND ----------

# MAGIC %md ## トレーニングデータセットとテストデータセットの分割

# COMMAND ----------

train_df, test_df = input_df.randomSplit([0.99, 0.01], seed=42)
display(train_df)

# COMMAND ----------

# MAGIC %md # トレーニング
# MAGIC 
# MAGIC 以下のコマンドでAutoMLを実行します。モデルが予測すべき目的変数を、引数`target_col`で指定する必要があります。実行が完了すると、トレーニングでベストなモデルを生成したノートブックにアクセスして、コードを確認することができます。このノートブックには特徴量の重要度のグラフも含まれています。

# COMMAND ----------

from databricks import automl
summary = automl.classify(train_df, target_col="income", timeout_minutes=30)

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

model_uri = summary.best_trial.model_path
# model_uri = "<model-uri-from-generated-notebook>"

# COMMAND ----------

# MAGIC %md ## pandasデータフレーム

# COMMAND ----------

import mlflow

# テストデータセットの準備
test_pdf = test_df.toPandas()
y_test = test_pdf["income"]
X_test = test_pdf.drop("income", axis=1)

# ベストモデルによる推論の実行
model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(X_test)
test_pdf["income_predicted"] = predictions
display(test_pdf)

# COMMAND ----------

# MAGIC %md ## Sparkデータフレーム

# COMMAND ----------

predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type="string")
display(test_df.withColumn("income_predicted", predict_udf()))

# COMMAND ----------

# MAGIC %md ## テスト
# MAGIC 
# MAGIC 実運用環境において、最終的なベストモデルがどれだけの性能を発揮するのかを見積もるために、ホールドアウトしておいたテストセットで予測を行います。以下の図では正しい予測と誤った予測をブレークダウンしています。

# COMMAND ----------

import sklearn.metrics

model = mlflow.sklearn.load_model(model_uri)
sklearn.metrics.plot_confusion_matrix(model, X_test, y_test)
