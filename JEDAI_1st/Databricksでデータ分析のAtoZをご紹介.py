# Databricks notebook source
# MAGIC %md
# MAGIC # Databricksでデータ分析のAtoZをご紹介
# MAGIC 
# MAGIC データ分析におけるAtoZとは何でしょうか？
# MAGIC ![](https://sajpstorage.blob.core.windows.net/workshop20210428-jedai/DA-A-Z.jpg)
# MAGIC 
# MAGIC - データ分析はビジネス上の課題を解決するための手段の一つです。
# MAGIC - データ分析というと予測モデル構築が脚光を浴びがちですが、データをビジネス価値につなげる長い道のりのほんの一部です。
# MAGIC - 本日は、データをビジネス価値創出につなげる道のりを、実例含めてご紹介いたします。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 自己紹介
# MAGIC ![](https://sajpstorage.blob.core.windows.net/workshop20210428-jedai/self_introduction_v1.png)
# MAGIC 
# MAGIC Qiitaに色々投稿してます。
# MAGIC - [Databricksクイックスタートガイド \- Qiita](https://qiita.com/taka_yayoi/items/125231c126a602693610)

# COMMAND ----------

# MAGIC %md
# MAGIC ## データ分析の(終わり無き)長い道のり
# MAGIC 個人的経験も踏まえたものです。最後までたどり着けないプロジェクトもたくさんありました...
# MAGIC <br><br>
# MAGIC 1. ビジネス課題の特定
# MAGIC 1. データ分析における仮説の立案
# MAGIC 1. データ分析アプローチの検討
# MAGIC 1. データソースの調査、分析データの入手
# MAGIC 1. 分析データの読み込み
# MAGIC 1. 探索的データ分析(EDA:Exploratory Data Analysis)
# MAGIC 1. 分析データの前処理
# MAGIC 1. 分析アルゴリズムの検討
# MAGIC 1. 分析パイプラインのレビュー
# MAGIC 1. モデルの構築
# MAGIC 1. モデルの評価
# MAGIC 1. モデルのチューニング
# MAGIC 1. モデルのデプロイ
# MAGIC 1. 精度・性能のモニタリング

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ1〜4 ヒアリング + ホワイトボードを前にしたディスカッション
# MAGIC いきなりデータを触り始めるプロジェクトはまず存在しません。データ分析には必ずビジネスにつながる目的があるべきです。
# MAGIC - ステップ1 ビジネス課題の特定<br>
# MAGIC > あるマーケティング担当者の悩み 「マーケティングを効率的に進めるために、年収の高いお客様を簡単に特定できないだろうか？」<br>
# MAGIC **ビジネス課題： 富裕層を特定することによるマーケティングの効率化**<br>
# MAGIC 
# MAGIC - ステップ2 データ分析における仮説の立案
# MAGIC > あるデータサイエンティストの思い 「デモグラフィック情報から収入を予測できれば、ユーザー情報登録時に年収を予測できるのではないか？」<br>
# MAGIC **データ分析における仮説: デモグラフィック情報から年収を予測できる**<br>
# MAGIC 
# MAGIC - ステップ3 データ分析アプローチの検討
# MAGIC > マーケティング担当者とデータサイエンティストの議論 「具体的な年収を予測するのではなく、年収が一定額以上か未満かを識別するだけで十分ではないか」<br>
# MAGIC **データ分析アプローチ: 年収が5万ドル以上か否かを分類する二値分類問題に取り組む**<br>
# MAGIC 
# MAGIC - ステップ4 データソースの調査、分析データの入手
# MAGIC > データサイエンティストとDWH担当者の会話 「過去に蓄積したデモグラフィック情報と年収情報は利用できそうだ」<br>
# MAGIC **分析データ: 過去に蓄積したデモグラフィック情報、年収情報**
# MAGIC 
# MAGIC ちなみに意外と大変なのは、ステップ4において、データの由来、スキーマ、更新頻度などの確認です。いろんな人に聞いて回らないとわからないケースも。

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ5 データの読み込み
# MAGIC 
# MAGIC 上で述べたとおり、デモグラフィック情報に基づいて、年収が5万ドル以上か否かを分類する二値分類問題に取り組みます。データセットは、[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Adult)にあるものを利用します。このデータは既にDatabricksランタイムに格納されています。このノートブックでは、データ処理、機械学習パイプライン、機械学習アルゴリズムなどのMLlibの機能をデモンストレーションします。
# MAGIC 
# MAGIC その前に、Databricksワークスペースの画面構成を簡単にご説明します。
# MAGIC 
# MAGIC 参考情報：
# MAGIC - [Databricksワークスペースのコンセプト \- Qiita](https://qiita.com/taka_yayoi/items/78bf647c40a906d90db0)
# MAGIC - [Databricksノートブックを使う \- Qiita](https://qiita.com/taka_yayoi/items/dfb53f63aed2fbd344fc)
# MAGIC - [Databricksにおけるデータのインポート、読み込み、変更 \- Qiita](https://qiita.com/taka_yayoi/items/4fa98b343a91b8eaa480)

# COMMAND ----------

# MAGIC %md
# MAGIC 大抵の場合、分析データを入手した後にすることはデータの中身の確認でしょう。Databricksでは柔軟にノートブック上での作業を行えるように、多くの`マジックコマンド`がサポートされています。下のセルにある`%fs`もその一つです。Databricksのファイルシステムに格納されているファイルの一部を表示します。
# MAGIC 
# MAGIC 参考情報： 
# MAGIC - [Databricksの言語マジックコマンド](https://qiita.com/taka_yayoi/items/dfb53f63aed2fbd344fc#%E6%B7%B7%E6%88%90%E8%A8%80%E8%AA%9E)
# MAGIC - [Databricksにおけるファイルシステム \- Qiita](https://qiita.com/taka_yayoi/items/e16c7272a7feb5ec9a92)

# COMMAND ----------

# MAGIC %fs head --maxBytes=1024 databricks-datasets/adult/adult.data

# COMMAND ----------

# MAGIC %md 
# MAGIC データセットにはカラム名が含まれていないため、カラム名とデータタイプを指定するスキーマを作成します。作成したスキーマを指定してCSVファイルを読み込みます。

# COMMAND ----------

# スキーマの定義
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

# ファイルを読み込みます
dataset = spark.read.csv("/databricks-datasets/adult/adult.data", schema=schema)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ6 探索的データ分析(EDA:Exploratory Data Analysis)
# MAGIC 
# MAGIC 個人的にはデータを理解するEDAは非常に重要だと思っています。EDAを通じて取り扱うデータの素性を理解することで、以降の分析での手戻りを減らすことができます。
# MAGIC 
# MAGIC 参考情報：
# MAGIC - [Databricksにおけるデータの可視化 \- Qiita](https://qiita.com/taka_yayoi/items/36a307e79e9433121c38)

# COMMAND ----------

# MAGIC %md 
# MAGIC モデル構築、評価に向けて、データセットをランダムにトレーニングデータとテストデータに分割します。また、再現性確保のために乱数のシードを設定しています。
# MAGIC 
# MAGIC あらゆる前処理を実行する前の**生の状態**でデータを分割すべきです。これにより、モデルを評価する際、テストデータが未知のデータに近い状態を維持することができます。

# COMMAND ----------

trainDF, testDF = dataset.randomSplit([0.8, 0.2], seed=42)
print("トレーニングデータ:", trainDF.cache().count()) # 何回かトレーニングするのでデータをキャッシュします
print("テストデータ:", testDF.count())

# COMMAND ----------

# MAGIC %md データを確認しましょう。EDA! EDA!

# COMMAND ----------

# データを確認するには、とにかくdisplay!
display(trainDF)

# COMMAND ----------

# MAGIC %md `hours_per_week`(週当たりの勤務時間)の分布はどうなっているでしょうか？

# COMMAND ----------

display(trainDF.select("hours_per_week").summary())

# COMMAND ----------

# MAGIC %md `education`(最終学歴)はどうなっているでしょうか？
# MAGIC 
# MAGIC 表を見て、全体の傾向を把握するのには限界があります。百のテーブルは一のグラフにしかずです。そんな時にはlet's可視化！![](https://docs.databricks.com/_images/chart-button.png)をクリック！

# COMMAND ----------

# 最終学歴でグルーピングして件数をカウント、カウントの昇順でソートして表示
display(trainDF
        .groupBy("education")
        .count()
        .sort("count", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## インターミッション: Transformers、estimators、pipelines
# MAGIC 
# MAGIC 本ノートブックで説明するMLlibの機械学習における重要な3つのコンセプトは、**Transformers**、**Estimators**、そして、**Pipelines**です。
# MAGIC <br>
# MAGIC - **Transformer**: データフレームをインプットとして新たなデータフレームを返却します。Transformersは、データから学習は行わず、モデル学習のためのデータを準備するか、学習したMLlibモデルで予測を行うために、単にルールベースの変換処理を適用します。`.transform()`メソッドでtransformerを呼び出すことができます。
# MAGIC 
# MAGIC - **Estimator**: `.fit()`メソッドを用いてデータフレームからパラメーターを学習(fit)し、モデルを返却します。モデルはtransformerです。
# MAGIC 
# MAGIC - **Pipeline**: 複数のステップを容易に実行できるように単一のワークフローにまとめます。機械学習モデル作成には、多くのケースで異なるステップが含まれ、それらを繰り返す必要があります。パイプラインを用いることでこのプロセスを自動化することができます。
# MAGIC 
# MAGIC 参考情報:
# MAGIC - [ML Pipelines(英語)](https://spark.apache.org/docs/latest/ml-pipeline.html#ml-pipelines)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ7 分析データの前処理
# MAGIC 
# MAGIC ここでのゴールは、データセットに含まれる特徴量(教育レベル、既婚・未婚、職業など)から、年収`income`のレベルを予測するというものです。まず、MLlibで利用できるように特徴量を操作、前処理を行います。いわゆる、特徴量エンジニアリングです。

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
# MAGIC そして、`StringIndexerModel`の`.transform()`メソッドを呼び出すことで、特徴量を変換結果を格納するカラム`...Index`が追加された新たなデータフレームが返却されます。必要であれば、表示結果を右にスクロールして追加されたカラムを参照してください。
# MAGIC 
# MAGIC 参考情報:
# MAGIC - [StringIndexerModel(英語)](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/StringIndexerModel.html)

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
# MAGIC 参考情報: 
# MAGIC - [VectorAssembler(英語)](https://spark.apache.org/docs/latest/ml-features.html#vectorassembler)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# ここには、データセットの数値カラムとone-hotエンコードされた２値のベクトル両方が含まれます。
numericCols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
assemblerInputs = [c + "OHE" for c in categoricalCols] + numericCols
vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ8 分析アルゴリズムの検討
# MAGIC 分析アルゴリズムは、データとビジネス課題に応じて使い分けることになると思います。EDAの過程でも「この辺りのアルゴリズムだろうな」などと当たりをつけながらデータを見ていきます。
# MAGIC <br><br>
# MAGIC - お客様を分類したい -> クラスタリング(k-meansや階層型クラスタリング)
# MAGIC - 値やラベルを予測したい -> 回帰、決定木、ランダムフォレスト、SVM、NN、etc.
# MAGIC - 画像や動画を判別したい -> ディープラーニングなど
# MAGIC - 組み合わせを予測したい -> アソシエーション分析
# MAGIC - テキストを分類したい -> BERTなど

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ9 分析パイプラインのレビュー
# MAGIC 
# MAGIC どのタイミングでレビューを行うのかは、チームやプロジェクトによるかと思います。とは言え、リモートワークが浸透した昨今、ここまで開発したロジックを簡単にシェアできる機能があると便利だと思いませんか？Databricksにはあります！以下の機能を利用いただくことで、チーム間で連携しながら開発を進めることができます。
# MAGIC <br><br>
# MAGIC - コメント機能
# MAGIC - 同時編集機能
# MAGIC - アクセス権管理機能
# MAGIC - バージョン管理機能
# MAGIC - git連携機能
# MAGIC - アーカイブ機能
# MAGIC - (地味ですが)セルへのリンク機能

# COMMAND ----------

# MAGIC %md ## ステップ10 モデルの構築
# MAGIC 
# MAGIC 本ノートブックでは[ロジスティック回帰(英語)](https://spark.apache.org/docs/latest/ml-classification-regression.html#logistic-regression)モデルを使います。
# MAGIC 
# MAGIC 参考情報：
# MAGIC - [Scikit\-learn でロジスティック回帰（クラス分類編） \- Qiita](https://qiita.com/0NE_shoT_/items/b702ab482466df6e5569)

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol="features", labelCol="label", regParam=1.0)

# COMMAND ----------

# MAGIC %md ### パイプラインの構築
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

# MAGIC %md ## ステップ11 モデルの評価
# MAGIC 
# MAGIC `display`コマンドにはROCカーブを表示するオプションが組み込まれています。
# MAGIC 
# MAGIC 参考情報：
# MAGIC - [ROC曲線](https://oku.edu.mie-u.ac.jp/~okumura/stat/ROC.html)

# COMMAND ----------

display(pipelineModel.stages[-1], predDF.drop("prediction", "rawPrediction", "probability"), "ROC")

# COMMAND ----------

# MAGIC %md 
# MAGIC モデル評価において、ROCカーブのAUC(Area Under the Curve)を計算するために`BinaryClassificationEvaluator`を用い、精度を評価するために`MulticlassClassificationEvaluator`を用います。
# MAGIC 
# MAGIC 参考情報：
# MAGIC - [曲線下の面積（AUC）](https://oku.edu.mie-u.ac.jp/~okumura/stat/ROC.html)
# MAGIC > ROC曲線下の面積（Area under the curve， AUC）は分類器（分類のアルゴリズム）の性能の良さを表します。0から1までの値をとり，完全な分類が可能なときの面積は1で，ランダムな分類の場合は0.5になります。
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
# MAGIC ## ステップ12 モデルのチューニング
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
# MAGIC 
# MAGIC <a href="https://www.mlflow.org/docs/latest/index.html"><img width=100 src="https://www.mlflow.org/docs/latest/_static/MLflow-logo-final-black.png" title="MLflow Documentation — MLflow 1.15.0 documentation"></a>
# MAGIC 
# MAGIC 参考情報：
# MAGIC - [PythonによるDatabricks MLflowクイックスタートガイド \- Qiita](https://qiita.com/taka_yayoi/items/dd81ac0da656bf883a34)

# COMMAND ----------

# 3フォールドのCrossValidatorを作成
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=bcEvaluator, numFolds=3, parallelism = 4)

import mlflow
import mlflow.spark
 
with mlflow.start_run():
  # 交差検証の実施。交差検証からベストなモデルを得るために処理に数分かかる場合があります。
  cvModel = cv.fit(trainDF)
  
  # ベストモデルをテストデータで評価しロギングします。
  test_metric = bcEvaluator.evaluate(cvModel.transform(testDF))
  mlflow.log_metric('test_' + bcEvaluator.getMetricName(), test_metric) 
  
  # ベストモデルをロギングします。
  mlflow.spark.log_model(spark_model=cvModel.bestModel, artifact_path='best-model') 

# COMMAND ----------

# MAGIC %md ## ステップ13 モデルのデプロイ
# MAGIC 
# MAGIC ベストモデルをデプロイ(配備)して、予測を実行します。デプロイには、いくつかの方法があります。
# MAGIC <br><br>
# MAGIC - 特定の場所にモデルをデプロイし、バッチやストリーミングでデータを流し込んで予測を行う
# MAGIC - モデルサーバーにデプロイし、REST APIなどで予測を行う(モデルサービング)
# MAGIC 
# MAGIC MLflowを活用することで、モデルサービング及びモデルの本格デプロイに向けた承認フローなどを容易に構築することができます。
# MAGIC 
# MAGIC ここでは、テストデータセットに対する予測を行うために、交差検証によって特定されたベストモデルを用います。
# MAGIC 
# MAGIC 参考情報：
# MAGIC - [MLflow guide \| Databricks on AWS(英語)](https://docs.databricks.com/applications/mlflow/index.html)
# MAGIC - [Track machine learning training runs \| Databricks on AWS(英語)](https://docs.databricks.com/applications/mlflow/tracking.html)

# COMMAND ----------

# テストデータセットに対する予測を行うために、交差検証によって特定されたベストモデルを使用
cvPredDF = cvModel.transform(testDF)

# COMMAND ----------

# MAGIC %md ### 予測結果に対する分析

# COMMAND ----------

# MAGIC %md 予測結果のデータセットを見てみます。`prediction`カラムの値が0の場合、`<=50K`、1の場合`>50K`と予測したことを意味します。

# COMMAND ----------

display(cvPredDF)

# COMMAND ----------

# MAGIC %md
# MAGIC また、SQLを用いることで、予測結果を年齢別、職業別に集計することができます。SQLを実行するために、予測結果のデータセットから一時ビューを作成します。
# MAGIC 
# MAGIC 参考情報：
# MAGIC - [Databricksにおけるデータベースおよびテーブル \- Qiita](https://qiita.com/taka_yayoi/items/e7f6982dfbee7fc84894)

# COMMAND ----------

cvPredDF.createOrReplaceTempView("finalPredictions")

# COMMAND ----------

# MAGIC %md 職業ごとの予測結果
# MAGIC 
# MAGIC - **0** 年収が`<=50K`
# MAGIC - **1** 年収が`>50K`

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT occupation, prediction, count(*) AS count
# MAGIC FROM finalPredictions
# MAGIC GROUP BY occupation, prediction
# MAGIC ORDER BY occupation

# COMMAND ----------

# MAGIC %md 年齢ごとの予測結果
# MAGIC 
# MAGIC - **0** 年収が`<=50K`
# MAGIC - **1** 年収が`>50K`

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT age, prediction, count(*) AS count
# MAGIC FROM finalPredictions
# MAGIC GROUP BY age, prediction
# MAGIC ORDER BY age

# COMMAND ----------

# MAGIC %md ## ステップ14 精度・性能のモニタリング
# MAGIC 
# MAGIC 実運用においては、モデルが定常的に目標とする精度を達成しているのかをモニタリングする必要があります。モデルの経年劣化「ドリフト」を検知した場合には、再学習を行うなどの対応が必要となります。
# MAGIC ![](https://databricks.com/wp-content/uploads/2019/09/model_drift.jpg)
# MAGIC 
# MAGIC なお、ドリフトには以下の種類があります。
# MAGIC 
# MAGIC **概念ドリフト(concept drift)**
# MAGIC 
# MAGIC > 目標変数の統計的属性が変化した時、予測しようとする本当の概念もまた変化します。例えば、不正トランザクションにおいては、新たな手口が生まれてくると、不正の定義自体を見直さなくてはなりません。このような変化は概念ドリフトを引き起こします。
# MAGIC 
# MAGIC **データドリフト(data drift)**
# MAGIC 
# MAGIC > 入力データから選択された特徴量を用いてモデルをトレーニングします。入力データの統計的特性に変化が生じた際、モデルの品質に影響を及ぼします。例えば、季節性によるデータの変化、個人的嗜好の変化、トレンドなどは入力データのドリフトを引き起こします。
# MAGIC 
# MAGIC **上流データの変化(upstream data changes)**
# MAGIC 
# MAGIC > モデル品質に影響を与えうるデータパイプライン上流でのオペレーションの変更が生じる場合があります。例えば、特徴量のエンコーディングにおいて華氏から摂氏に変更があったり、特徴量の生成が停止されることでnullや欠損値になるなどです。
# MAGIC 
# MAGIC ここでは、ベストモデルをAUCで評価しますが、実際には評価は一度限りではなく、継続的なオペレーションになります。これが「終わり無き」所以です。
# MAGIC 
# MAGIC 参考情報：
# MAGIC - [機械学習の本格運用：デプロイメントからドリフト検知まで \- Qiita](https://qiita.com/taka_yayoi/items/879506231b9ec19dc6a5)

# COMMAND ----------

# AUCと精度を用いてモデルの性能を評価 
print(f"Area under ROC curve: {bcEvaluator.evaluate(cvPredDF)}")
print(f"Accuracy: {mcEvaluator.evaluate(cvPredDF)}")

# COMMAND ----------

# MAGIC %md 本日は、データ分析の終わり無き旅路の一部をお見せしました。
# MAGIC 
# MAGIC ご覧いただけたように、(ホワイトボード、ヒアリングを除く)全ての作業をDatabricksのワークスペースで完結することができます。
# MAGIC 
# MAGIC 是非、Databricksをご利用ください！

# COMMAND ----------

# MAGIC %md # END
