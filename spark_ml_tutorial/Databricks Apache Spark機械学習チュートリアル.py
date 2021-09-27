# Databricks notebook source
# MAGIC %md
# MAGIC # MLlib事始め - 二値分類の例

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC このチュートリアルは、Apache SparkのMLlibに慣れることを目的としています。ここでは、デモグラフィック情報に基づいて、年収が5万ドル以上か否かを分類する二値分類問題に取り組みます。データセットは、[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Adult)にあるものを利用します。既にこちらのデータはDatabricksランタイムに格納されています。このノートブックでは、データ処理、機械学習パイプライン、機械学習アルゴリズムなどのMLlibの機能をデモンストレーションします。
# MAGIC 
# MAGIC このノートブックでは以下のステップを踏みます：
# MAGIC 
# MAGIC 1. データセットの読み込み
# MAGIC 1. 特徴量の前処理
# MAGIC 1. モデルの定義
# MAGIC 1. パイプラインの構築
# MAGIC 1. モデルの評価
# MAGIC 1. ハイパーパラメーターのチューニング
# MAGIC 1. 予測の実行、モデル性能の評価
# MAGIC 
# MAGIC **要件**
# MAGIC - Databricks MLランタイム7.0以上が必要です。

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1. データセットの読み込み

# COMMAND ----------

# MAGIC %md
# MAGIC データの最初の行を見るためにDatabricksのユーティリティを使います

# COMMAND ----------

# MAGIC %fs head --maxBytes=1024 databricks-datasets/adult/adult.data

# COMMAND ----------

# MAGIC %md 
# MAGIC データセットにはカラム名が含まれていないため、カラム名とデータタイプを指定するスキーマを作成します

# COMMAND ----------

schema = """`age` DOUBLE,
`workclass` STRING,
`fnlwgt` DOUBLE,
`education` STRING,
`education_num` DOUBLE,
`marital_status` STRING,
`occupation` STRING,
`relationship` STRING,
`race` STRING,
`sex` STRING,
`capital_gain` DOUBLE,
`capital_loss` DOUBLE,
`hours_per_week` DOUBLE,
`native_country` STRING,
`income` STRING"""

dataset = spark.read.csv("/databricks-datasets/adult/adult.data", schema=schema)

# COMMAND ----------

# MAGIC %md 
# MAGIC データセットをランダムにトレーニングデータとテストデータに分割します。再現性確保のために乱数のシードを設定しています。
# MAGIC 
# MAGIC あらゆる前処理を実行する前にデータを分割すべきです。これにより、モデルを評価する際、テストデータが未知のデータに近い状態を維持することができます。

# COMMAND ----------

trainDF, testDF = dataset.randomSplit([0.8, 0.2], seed=42)
print("トレーニングデータ:", trainDF.cache().count()) # 何回かトレーニングするのでデータをキャッシュします
print("テストデータ:", testDF.count())

# COMMAND ----------

# MAGIC %md データを確認しましょう

# COMMAND ----------

display(trainDF)

# COMMAND ----------

# MAGIC %md `hours_per_week`の数の分布はどうなっているでしょうか？

# COMMAND ----------

display(trainDF.select("hours_per_week").summary())

# COMMAND ----------

# MAGIC %md `education`はどうなっているでしょうか？

# COMMAND ----------

display(trainDF
        .groupBy("education")
        .count()
        .sort("count", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## バックグラウンド: Transformers、estimators、pipelines
# MAGIC 
# MAGIC 本ノートブックで説明するMLlibの機械学習における重要な3つのコンセプトは、**Transformers**、**Estimators**、そして、**Pipelines**です。
# MAGIC <br>
# MAGIC - **Transformer**: データフレームをインプットとして新たなデータフレームを返却します。Transformersは、データから学習は行わず、モデル学習のためのデータを準備するか、学習したMLlibモデルで予測を行うために、単にルールベースの変換処理を適用します。`.transform()`メソッドでtransformerを呼び出すことができます。
# MAGIC 
# MAGIC - **Estimator**: `.fit()`メソッドを用いてデータフレームからパラメーターを学習(fit)し、モデルを返却します。モデルはtransformerです。
# MAGIC 
# MAGIC - **Pipeline**: 複数のステップを容易に実行できるように単一のワークフローにまとめます。機械学習モデル作成には、多くのケースで異なるステップが含まれ、それらを繰り返す必要があります。パイプラインを用いることでこのプロセスを自動化することができます。
# MAGIC 
# MAGIC 詳細はこちらを参照ください:
# MAGIC [ML Pipelines(英語)](https://spark.apache.org/docs/latest/ml-pipeline.html#ml-pipelines)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2. 特徴量の前処理
# MAGIC 
# MAGIC このノートブックのゴールは、データセットに含まれる特徴量(教育レベル、既婚・未婚、職業など)から`income`のレベルを予測するというものです。この最初のステップは、MLlibで利用できるように特徴量を操作、前処理することです。

# COMMAND ----------

# MAGIC %md ### カテゴリー変数を数値に変換する
# MAGIC 
# MAGIC 線形回帰、ロジスティック回帰などの学習アルゴリズムでは、特徴量が数値である必要があります。上記の成人データセットでは、教育、職業、既婚・未婚のデータがカテゴリー変数となっています。
# MAGIC 
# MAGIC 以下のコードでは、カテゴリー変数を0か1のみを取る数値変数に変換するために、どのように`StringIndexer`と`OneHotEncoder`を使用するのかを説明します。
# MAGIC 
# MAGIC - `StringIndexer`は、文字列のカラムをラベルのインデックスに変換します。例えば、"red"、"blue"、"green"をそれぞれ0、1、2に変換します。 
# MAGIC - `OneHotEncoder`は、カテゴリー変数のインデックスを二進数のベクトルにマッピングします。当該レコードのカテゴリー変数のインデックスに該当するベクトルの要素に"1"が割り当てられます。
# MAGIC 
# MAGIC SparkにおけるOne-hotエンコーディングは2段階のプロセスとなります。最初にStringIndexerを使い、OneHotEncoderを呼び出します。以下のコードブロックでは、StringIndexerとOneHotEncoderを定義しますが、データにはまだ適用しません。
# MAGIC 
# MAGIC 詳細はこちらを参照ください:   
# MAGIC - [StringIndexer(英語)](http://spark.apache.org/docs/latest/ml-features.html#stringindexer)   
# MAGIC - [OneHotEncoder(英語)](https://spark.apache.org/docs/latest/ml-features.html#onehotencoder)

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder

categoricalCols = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex"]

# 以下の２行はestimatorとなります。後ほどデータセットを変換する際に適用することになる関数を返却します。
stringIndexer = StringIndexer(inputCols=categoricalCols, outputCols=[x + "Index" for x in categoricalCols]) 
encoder = OneHotEncoder(inputCols=stringIndexer.getOutputCols(), outputCols=[x + "OHE" for x in categoricalCols]) 

# ラベルとなるカラム("income")も("<=50K"、">50K")の二つの値をとる文字列のカラムとなります。
# こちらもStringIndexerを使って数値に変換します。
labelToIndex = StringIndexer(inputCol="income", outputCol="label")

# COMMAND ----------

# MAGIC %md 
# MAGIC このノートブックでは、特徴量エンジニアリングとモデル構築のステップ全てを一つのパイプラインにまとめます。ただ、その前に上のコードブロックで構築した`stringIndexer`estimatorを適用することでestimatorやtransformerがどのように動作するのかを詳しく見てみましょう。
# MAGIC 
# MAGIC データセットを変換する`StringIndexerModel`を返却するように`.fit()`メソッドを呼び出します。
# MAGIC 
# MAGIC そして、`StringIndexerModel`の`.transform()`メソッドを呼び出すことで、カラムが追加された新たなデータフレームが返却されます。必要であれば、表示結果を右にスクロールして追加されたカラムを参照してください。
# MAGIC 
# MAGIC 詳細はこちらを参照ください: [StringIndexerModel(英語)](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/StringIndexerModel.html)

# COMMAND ----------

stringIndexerModel = stringIndexer.fit(trainDF)
display(stringIndexerModel.transform(trainDF))

# COMMAND ----------

# MAGIC %md ### 全ての特徴量カラムを一つの特徴量ベクトルにまとめます
# MAGIC 
# MAGIC 多くのMLlibアルゴリズムでは、入力として単一の特徴量カラムが必要となります。それぞれの行の特徴量カラムは、予測に用いる特徴量に対応するベクトルを保持します。
# MAGIC 
# MAGIC MLlibは、一連のカラムから単一のベクトルカラムを作成する`VectorAssembler`transformerを提供します。
# MAGIC 
# MAGIC 下のコードブロックではどのようにVectorAssemblerを使用するのかを説明します。
# MAGIC 
# MAGIC 詳細はこちらを参照ください: [VectorAssembler(英語)](https://spark.apache.org/docs/latest/ml-features.html#vectorassembler)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# ここには、データセットの数値カラムとone-hotエンコードされた２値のベクトル両方が含まれます。
numericCols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
assemblerInputs = [c + "OHE" for c in categoricalCols] + numericCols
vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

# COMMAND ----------

# MAGIC %md ## Step 3. モデルの定義
# MAGIC 
# MAGIC 本ノートブックでは[ロジスティック回帰(英語)](https://spark.apache.org/docs/latest/ml-classification-regression.html#logistic-regression)モデルを使います。

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol="features", labelCol="label", regParam=1.0)

# COMMAND ----------

# MAGIC %md ## Step 4. パイプラインの構築
# MAGIC 
# MAGIC `Pipeline`は、transformers、estimatorsが含まれる順番付きのリストです。データセットに適用する変換処理の再現性を確保し、自動化するために、パイプラインを定義することができます。
# MAGIC 
# MAGIC `StringIndexer`で見たのと同様に、`Pipeline`もestimatorです。`pipeline.fit()`メソッドが、transformerである`PipelineModel`を返却します。
# MAGIC 
# MAGIC 詳細はこちらを参照ください:
# MAGIC [Pipelines(英語)](https://spark.apache.org/docs/latest/ml-pipeline.html#ml-pipelines)  

# COMMAND ----------

from pyspark.ml import Pipeline

# これまでに作成したステージを組み合わせてパイプラインを定義します
pipeline = Pipeline(stages=[stringIndexer, encoder, labelToIndex, vecAssembler, lr])

# パイプラインモデルを定義します
pipelineModel = pipeline.fit(trainDF)

# テストデータセットにパイプラインモデルを適用します
predDF = pipelineModel.transform(testDF)

# COMMAND ----------

# MAGIC %md モデルによる予測結果を表示します。`features`カラムは、one-hotエンコーディングを実行した後、多くのケースで要素のほとんどが0となる[sparse vector(英語)](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.linalg.SparseVector.html#pyspark.ml.linalg.SparseVector)となります。

# COMMAND ----------

display(predDF.select("features", "label", "prediction", "probability"))

# COMMAND ----------

# MAGIC %md ## Step 5. モデルの評価
# MAGIC 
# MAGIC `display`コマンドにはROCカーブを表示するオプションが組み込まれています。

# COMMAND ----------

display(pipelineModel.stages[-1], predDF.drop("prediction", "rawPrediction", "probability"), "ROC")

# COMMAND ----------

# MAGIC %md 
# MAGIC モデル評価において、ROCカーブのAUC(Area Under the Curve)を計算するために`BinaryClassificationEvaluator`を用い、精度を評価するために`MulticlassClassificationEvaluator`を用います。
# MAGIC 
# MAGIC 詳細はこちらを参照ください:
# MAGIC - [BinaryClassificationEvaluator(英語)](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.BinaryClassificationEvaluator.html#binaryclassificationevaluator)  
# MAGIC - [MulticlassClassificationEvaluator(英語)](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.MulticlassClassificationEvaluator.html#multiclassclassificationevaluator)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

bcEvaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
print(f"Area under ROC curve: {bcEvaluator.evaluate(predDF)}")

mcEvaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print(f"Accuracy: {mcEvaluator.evaluate(predDF)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6. ハイパーパラメーターのチューニング
# MAGIC 
# MAGIC MLlibはハイパーパラメーターチューニングと交差検証(cross validation)の手段を提供します。
# MAGIC - ハイパーパラメータチューニングにおいては、`ParamGridBuilder`を用いることで、モデルのハイパーパラメーターの探索空間を定義できます。
# MAGIC - 交差検証においては、`CrossValidator`を用いることで、estimator(入力データセットに適用するパイプライン)、evaluator、ハイパーパラメーターの探索空間、交差検証のフォールド数を定義できます。
# MAGIC 
# MAGIC 詳細はこちらを参照ください:
# MAGIC - [交差検証（クロスバリデーション）とは？合わせてグリッドサーチに関しても学ぼう！ \| AI Academy Media](https://aiacademy.jp/media/?p=263)
# MAGIC - [Model selection using cross-validation(英語)](https://spark.apache.org/docs/latest/ml-tuning.html)  
# MAGIC - [ParamGridBuilder(英語)](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.ParamGridBuilder.html#paramgridbuilder)  
# MAGIC - [CrossValidator(英語)](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidator.html#crossvalidator)   

# COMMAND ----------

# MAGIC %md
# MAGIC モデルをチューニングするために、`ParamGridBuilder`と`CrossValidator`を使用します。本例においては、`CrossValidator`での検証において、3種類の`regParam`、3種類の`elasticNetParam`から生成される、3 x 3 = 9のハイパーパラメーターの組み合わせを使用します。

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             .build())

# COMMAND ----------

# MAGIC %md 
# MAGIC MLlibの`CrossValidator`を呼び出した際、Databricksは[MLflow](https://mlflow.org/)を用いて、自動的に全てのランを追跡します。MLflowのUI([AWS](https://docs.databricks.com/applications/mlflow/index.html)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/))を用いて、構築したモデルを比較することができます。下のセルの実行後、画面右上にある**Experiment**ボタンを押してみてください。
# MAGIC 
# MAGIC 本例では、作成したパイプラインをestimatorとします。

# COMMAND ----------

# 3フォールドのCrossValidatorを作成
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=bcEvaluator, numFolds=3, parallelism = 4)

# 交差検証の実施。交差検証からベストなモデルを得るために処理に数分かかる場合があります。
cvModel = cv.fit(trainDF)

# COMMAND ----------

# MAGIC %md ## Step 7. 予測の実行、モデル性能の評価
# MAGIC 
# MAGIC テストデータセットに対する予測を行うために、交差検証によって特定されたベストモデルを用い、AUCによるモデルの性能を評価します。

# COMMAND ----------

# テストデータセットに対する予測を行うために、交差検証によって特定されたベストモデルを使用
cvPredDF = cvModel.transform(testDF)

# AUCと精度を用いてモデルの性能を評価 
print(f"Area under ROC curve: {bcEvaluator.evaluate(cvPredDF)}")
print(f"Accuracy: {mcEvaluator.evaluate(cvPredDF)}")

# COMMAND ----------

# MAGIC %md 予測結果のデータセットを見てみます。`prediction`カラムの値が0の場合、`<=50K`、1の場合`>50K`と予測したことを意味します。

# COMMAND ----------

display(cvPredDF)

# COMMAND ----------

# MAGIC %md
# MAGIC また、SQLを用いることで、予測結果を年齢別、職業別に集計することができます。SQLを実行するために、予測結果のデータセットから一時ビューを作成します。

# COMMAND ----------

cvPredDF.createOrReplaceTempView("finalPredictions")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT occupation, prediction, count(*) AS count
# MAGIC FROM finalPredictions
# MAGIC GROUP BY occupation, prediction
# MAGIC ORDER BY occupation

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT age, prediction, count(*) AS count
# MAGIC FROM finalPredictions
# MAGIC GROUP BY age, prediction
# MAGIC ORDER BY age
