# Databricks notebook source
# MAGIC %md
# MAGIC # MLlibを使い始める - 2値分類のサンプル

# COMMAND ----------

# MAGIC %md
# MAGIC このチュートリアルはApache Spark MLlibを使い始められるように設計されています。ここでは2値分類問題を取り扱います。デモグラフィックデータに基づいて、ある個人の収入が$50,000より高いかどうかを予測できるでしょうか？[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Adult)からのデータセット使用しており、このデータはDatabricksランタイム上で提供されています。このノートブックでは、データの前処理、機械学習パイプライン、いくつかの機械学習アルゴリズムなどMLlibで利用できる幾つかの機能をデモンストレーションします。
# MAGIC 
# MAGIC このノートブックには以下のステップが含まれています:
# MAGIC 
# MAGIC 1. データセットのロード
# MAGIC 1. 特徴量の処理
# MAGIC 1. モデルの定義
# MAGIC 1. パイプラインの構築
# MAGIC 1. モデルの評価
# MAGIC 1. ハイパーパラメーターチューニング
# MAGIC 1. 予測及びモデルパフォーマンスの評価
# MAGIC 
# MAGIC ## 要件
# MAGIC 
# MAGIC Databricksランタイム7.0以降、あるいはDatabricks機械学習ランタイム7.0以降。Databricksランタイム6.x、Databricks機械学習ランタイム6.xを使用している場合には、([AWS](https://docs.databricks.com/getting-started/spark/machine-learning.html)|[Azure](https://docs.microsoft.com/azure/databricks/getting-started/spark/machine-learning/))のノートブックをご覧ください。

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1. データセットのロード

# COMMAND ----------

# MAGIC %md
# MAGIC データの最初の数行を表示するためにDatabricksユーティリティを使用します。

# COMMAND ----------

# MAGIC %fs head --maxBytes=1024 databricks-datasets/adult/adult.data

# COMMAND ----------

# MAGIC %md 
# MAGIC データセットにはカラム名が含まれていないので、カラム名とデータ型を割り当てるためにスキーマを作成します。

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
# MAGIC トレーニングセットとテストセットにランダムに分割し、再現性確保のためのシードを設定します。
# MAGIC 
# MAGIC 前処理を行う前にデータを分割するのがベストです。これによって、モデルを評価する際にテストデータセットが新規のデータをより近い形でシミュレーションすることが可能となります。

# COMMAND ----------

trainDF, testDF = dataset.randomSplit([0.8, 0.2], seed=42)
print(trainDF.cache().count()) # トレーニングデータに複数回アクセスするのでキャッシュします
print(testDF.count())

# COMMAND ----------

# MAGIC %md データを確認してみましょう。

# COMMAND ----------

display(trainDF)

# COMMAND ----------

# MAGIC %md `hours_per_week`の数の分布はどのようになっているでしょうか？

# COMMAND ----------

display(trainDF.select("hours_per_week").summary())

# COMMAND ----------

# MAGIC %md `education`のステータスはどうなっているでしょうか？

# COMMAND ----------

display(trainDF
        .groupBy("education")
        .count()
        .sort("count", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## バックグラウンド: トランスフォーマー、エスティメーター、パイプライン
# MAGIC 
# MAGIC このノートブックで説明されるMLlibの機械学習においては3つの重要なコンセプトがあります。**トランスフォーマー**、**エスティメーター**、**パイプライン**です。
# MAGIC 
# MAGIC - **トランスフォーマー(Transformer)**: データフレームを入力とし新たなデータフレームを返却します。トランスフォーマーはデータからパラメーターを学習せず、モデルトレーニングのためのデータを準備する、あるいはトレーニングしたMLlibモデルを用いた予測を生成するために、シンプルにルールベースの変換処理を適用します。トランスフォーマーは`.transform()`メソッドで呼び出すことができます。
# MAGIC 
# MAGIC - **エスティメーター(Estimator)**: `.fit()`メソッド経由でデータフレームからパラメーターを学習(あるいは"フィット")し、トランスフォーマーであるモデルを返却します。
# MAGIC 
# MAGIC - **パイプライン(Pipeline)**: 複数のステップを容易に実行できるように単一のワークフローにまとめます。機械学習んモデルの生成においては、通常異なる数多くのステップのセットアップが含まれ、それらに対してイテレーションを行います。パイプラインは、このプロセスの自動化に役立ちます。
# MAGIC 
# MAGIC 詳細はこちら:
# MAGIC - [ML Pipelines](https://spark.apache.org/docs/latest/ml-pipeline.html#ml-pipelines)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2. 特徴量の前処理
# MAGIC 
# MAGIC このノートブックのゴールはデータセットに含まれる特徴量(教育レベル、既婚・未婚、職業など)から`income`のレベルを予測するモデルを構築することです。MLlibで必要となるフォーマットになるように、特徴量を操作、前処理することが最初のステップとなります。

# COMMAND ----------

# MAGIC %md ### カテゴリー変数を数値に変換
# MAGIC 
# MAGIC 線形回帰、ロジスティック回帰のようないくつかの機械学習アルゴリズムでは数値の特徴量が求められます。Adultデータセットには学歴、職業、既婚・未婚のようなカテゴリー変数が含まれています。
# MAGIC 
# MAGIC 以下のコードブロックでは、カテゴリー変数を0あるいは1のみの値を取る一連の数値変数に変換するために、`StringIndexer`と`OneHotEncoder`の使い方を説明しています。
# MAGIC 
# MAGIC - `StringIndexer`は、文字列のカラムをラベルインデックスのカラムに変換します。例えば、 "red"、"blue"、"green"という値を0、1、2に変換します。
# MAGIC - `OneHotEncoder`は、カテゴリーのインデックスを2値のベクトルにマッピングします。それぞれの行で最大1つの"1"が存在し、それがその行のカテゴリー変数のインデックスに対応します。
# MAGIC 
# MAGIC Sparkにおけるワンホットエンコーディングは2ステップのプロセスになります。最初にStringIndexerを使用して、次にOneHotEncoderを使用します。以下のコードブロックではStringIndexerとOneHotEncoderを定義していますが、まだデータには適用していません。
# MAGIC 
# MAGIC 詳細はこちら:   
# MAGIC - [StringIndexer](http://spark.apache.org/docs/latest/ml-features.html#stringindexer)   
# MAGIC - [OneHotEncoder](https://spark.apache.org/docs/latest/ml-features.html#onehotencoder)

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder

categoricalCols = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex"]

# 以下の2行はエスティメーターです。後ほどデータセットを変換するために適用する関数を返却します
stringIndexer = StringIndexer(inputCols=categoricalCols, outputCols=[x + "Index" for x in categoricalCols]) 
encoder = OneHotEncoder(inputCols=stringIndexer.getOutputCols(), outputCols=[x + "OHE" for x in categoricalCols]) 

# ラベルのカラム("income")も文字列です。2つの値を取り得ます。"<=50K" と ">50K" です
# StringIndexerを用いて数値に変換します
labelToIndex = StringIndexer(inputCol="income", outputCol="label")

# COMMAND ----------

# MAGIC %md 
# MAGIC このノートブックでは、特徴量エンジニアリングとモデリングのステップすべてをまとめるパイプラインを構築します。しかし、上のコードブロックで作成した`stringIndexer`を適用することで、エスティメーターとトランスフォーマーがどの容易動作するのかを詳しくみてみましょう。
# MAGIC 
# MAGIC データセットを変換するのに使用する`StringIndexerModel`を返却してもらうために`.fit()`メソッドを呼び出します。
# MAGIC 
# MAGIC `StringIndexerModel`の`.transform()`メソッドは、新たなカラムが追加された新規のデータフレームを返却します。必要であれば、右側にスクロールして新規のカラムを見てみてください。
# MAGIC 
# MAGIC 詳細はこちら: 
# MAGIC - [StringIndexerModel](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/StringIndexerModel.html)

# COMMAND ----------

stringIndexerModel = stringIndexer.fit(trainDF)
display(stringIndexerModel.transform(trainDF))

# COMMAND ----------

# MAGIC %md ### すべての特徴量カラムを1つの特徴量ベクトルに結合
# MAGIC 
# MAGIC 多くのMLlibアルゴリズムは入力として1つの特徴量カラムを要求します。このカラムのそれぞれの行には、予測で用いる一連の特徴量に対応するデータポイントのベクトルが含まれます。
# MAGIC 
# MAGIC MLlibはカラムのリストから1つのベクトルを作成するために、`VectorAssembler`トランスフォーマーを提供します。
# MAGIC 
# MAGIC 以下のコードブロックではVectorAssemblerの使い方を説明しています。
# MAGIC 
# MAGIC 詳細はこちら: 
# MAGIC - [VectorAssembler](https://spark.apache.org/docs/latest/ml-features.html#vectorassembler)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# これにはデータセットの数値カラムと、ワンホットエンコーディングされた2値ベクトルカラムの両方が含まれています
numericCols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
assemblerInputs = [c + "OHE" for c in categoricalCols] + numericCols
vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

# COMMAND ----------

# MAGIC %md ## Step 3. モデルの定義
# MAGIC 
# MAGIC このノートブックでは、[ロジスティック回帰](https://spark.apache.org/docs/latest/ml-classification-regression.html#logistic-regression)モデルを使用します。

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol="features", labelCol="label", regParam=1.0)

# COMMAND ----------

# MAGIC %md ## Step 4. パイプラインの構築
# MAGIC 
# MAGIC `Pipeline`はトランスフォーマーとエスティメーターの順序ありリストです。データセットに適用される変換処理を自動化し、再現性を確保するためにパイプラインを定義することができます。このステップではパイプラインを定義し、テストデータセットに適用します。
# MAGIC 
# MAGIC `StringIndexer`で見たのと同じように、`Pipeline`はエスティメーターです。`pipeline.fit()`メソッドはトランスフォーマーである`PipelineModel`を返却します。
# MAGIC 
# MAGIC 詳細はこちら: 
# MAGIC - [Pipelines](https://spark.apache.org/docs/latest/ml-pipeline.html#ml-pipelines)  

# COMMAND ----------

from pyspark.ml import Pipeline

# 以前のステップで作成したステージに基づいてパイプラインを定義
pipeline = Pipeline(stages=[stringIndexer, encoder, labelToIndex, vecAssembler, lr])

# パイプラインモデルの定義
pipelineModel = pipeline.fit(trainDF)

# テストデータセットにパイプラインモデルを適用
predDF = pipelineModel.transform(testDF)

# COMMAND ----------

# MAGIC %md 
# MAGIC モデルの予測結果を表示します。`features`カラムはワンホットエンコーディングされる場合にはよく起こり得る、多くのゼロから構成される[sparse vector](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.linalg.SparseVector.html#pyspark.ml.linalg.SparseVector)となります。

# COMMAND ----------

display(predDF.select("features", "label", "prediction", "probability"))

# COMMAND ----------

# MAGIC %md ## Step 5. モデルの評価
# MAGIC 
# MAGIC `display`コマンドにはビルトインのROC曲線オプションがあります。

# COMMAND ----------

display(pipelineModel.stages[-1], predDF.drop("prediction", "rawPrediction", "probability"), "ROC")

# COMMAND ----------

# MAGIC %md 
# MAGIC モデルを評価するためには、ROC曲線のAUC(曲線の下部の面積)を評価するために`BinaryClassificationEvaluator`を用い、精度を評価するために`MulticlassClassificationEvaluator`を使用します。
# MAGIC 
# MAGIC 詳細はこちら:  
# MAGIC - [BinaryClassificationEvaluator](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.BinaryClassificationEvaluator.html#binaryclassificationevaluator)  
# MAGIC - [MulticlassClassificationEvaluator](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.MulticlassClassificationEvaluator.html#multiclassclassificationevaluator)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

bcEvaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
print(f"Area under ROC curve: {bcEvaluator.evaluate(predDF)}")

mcEvaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print(f"Accuracy: {mcEvaluator.evaluate(predDF)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6. ハイパーパラメーターチューニング
# MAGIC 
# MAGIC MLlibはハイパーパラメーターチューニングと交差検証を行うための手段を提供しています。
# MAGIC - ハイパーパラメーターチューニングに対しては、`ParamGridBuilder`を用いることで、一連のモデルハイパーパラメーターに対するグリッドサーチを定義することができます。
# MAGIC - 交差検証については、`CrossValidator`を用いることで、エスティメーター(入力データセットに適用するパイプライン)、エバリュエーター、ハイパーパラメーターのグリッド空間、交差検証に用いるフォールド数を指定することができます。
# MAGIC   
# MAGIC 詳細はこちら:   
# MAGIC - [Model selection using cross-validation](https://spark.apache.org/docs/latest/ml-tuning.html)  
# MAGIC - [ParamGridBuilder](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.ParamGridBuilder.html#paramgridbuilder)  
# MAGIC - [CrossValidator](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidator.html#crossvalidator)   

# COMMAND ----------

# MAGIC %md
# MAGIC モデルをチューニングするために `ParamGridBuilder` と `CrossValidator` を使用します。このサンプルでは、3つの値の`regParam`と3つの値の`elasticNetParam`を用いるので、 `CrossValidator`で検証する 3 x 3 = 9 のハイパーパラメーターの組み合わせが存在します。

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             .build())

# COMMAND ----------

# MAGIC %md 
# MAGIC MLlibで`CrossValidator`を呼び出す際、Databrikcsは[MLflow](https://mlflow.org/)を用いて自動ですべてのラン(トレーニングの実行)をトラッキングします。それぞれのモデルを日隠すためにMLflowのUI([AWS](https://docs.databricks.com/applications/mlflow/index.html)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/)|[GCP](https://docs.gcp.databricks.com/applications/mlflow/index.html))を活用することもできます。
# MAGIC 
# MAGIC このサンプルでは、エスティメーターとして作成したパイプラインを使用します。

# COMMAND ----------

# 3-foldのCrossValidatorの作成
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=bcEvaluator, numFolds=3, parallelism = 4)

# 交差検証の実行。このステップは数分要し、交差検証の結果得られたベストなモデルを返却します
cvModel = cv.fit(trainDF)

# COMMAND ----------

# MAGIC %md ## Step 7. 予測の実行及びモデルパフォーマンスの評価
# MAGIC 
# MAGIC テストデータセットに対する予測を行うために、交差検証で特定されたベストなモデルを使用し、ROC曲線のAUCを用いてモデルのパフォーマンスを評価します。

# COMMAND ----------

# テストデータセットに対する予測を行うために交差検証で特定されたベストなモデルを使用
cvPredDF = cvModel.transform(testDF)

# ROC曲線のAUCと精度に基づいてモデルのパフォーマンスを評価
print(f"Area under ROC curve: {bcEvaluator.evaluate(cvPredDF)}")
print(f"Accuracy: {mcEvaluator.evaluate(cvPredDF)}")

# COMMAND ----------

# MAGIC %md
# MAGIC SQLコマンドを用いて、年齢、職業ごとの予測結果を表示することができます。予測結果のデータセットに対する一時ビューを作成します。

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

# COMMAND ----------

# MAGIC %md
# MAGIC # END
