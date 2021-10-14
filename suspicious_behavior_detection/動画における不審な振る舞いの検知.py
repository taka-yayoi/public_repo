# Databricks notebook source
# DBTITLE 0,Video Identification of Suspicious Behavior: Preparation
# MAGIC %md 
# MAGIC 
# MAGIC # 動画における不審な振る舞いの検知
# MAGIC 
# MAGIC このノートブックでは以下の処理を行います。
# MAGIC 
# MAGIC - `動画における不審な振る舞いの検知：準備`で処理したデータを使用
# MAGIC - トレーニングデータのロード
# MAGIC - トレーニングデータを用いたモデルのトレーニング
# MAGIC - モデルを用いてテスト用データに対する予測を実施
# MAGIC - 動画には不審な行動があるのか？
# MAGIC 
# MAGIC このノートブックで用いるソースデータは、[EC Funded CAVIAR project/IST 2001 37540](http://homepages.inf.ed.ac.uk/rbf/CAVIAR/)で確認できます。
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2018/09/mnt_raela_video_splash.png" width=900/>
# MAGIC 
# MAGIC ### 前提条件
# MAGIC * 画像と特徴量のデータセットを準備するためには`動画における不審な振る舞いの検知：準備`を実行してください。
# MAGIC 
# MAGIC ### クラスター設定
# MAGIC - 推奨クラスター設定
# MAGIC  * Databricksランタイムバージョン: `Databricks Runtime for ML` (例: 8.4 ML, 9.1 ML, etc.)
# MAGIC  * Driver: 64GB RAMのインスタンス (例: `Azure: Standard_D16s_v3, AWS: r4.4xlarge`)
# MAGIC  * Workers: 2台の64GB RAMのインスタンス (例: `Azure: Standard_D16s_v3, AWS: r4.4xlarge`)
# MAGIC  * Python: `Python 3`
# MAGIC 
# MAGIC 
# MAGIC ### 手動でのインストールが必要な項目
# MAGIC 
# MAGIC 手動でのインストールに関しては、**Upload a Python PyPI package or Python Egg** [Databricks](https://docs.databricks.com/user-guide/libraries.html#upload-a-python-pypi-package-or-python-egg) | [Azure Databricks](https://docs.azuredatabricks.net/user-guide/libraries.html#upload-a-python-pypi-package-or-python-egg)を参照ください。
# MAGIC  
# MAGIC * Pythonライブラリ:
# MAGIC  * `opencv-python`: 3.4.2 
# MAGIC 
# MAGIC ### Databricks機械学習ラインタイムにインストール済みのライブラリ
# MAGIC 
# MAGIC ここでは*Databricks Runtime for ML*を使用するので、以下のライブラリのインストールは**不要**です。
# MAGIC * Pythonライブラリ:
# MAGIC  * `h5py`: 2.7.1
# MAGIC  * `tensorflow`: 1.7.1
# MAGIC  * `keras`: 2.1.5 (TensorFlowバックエンドを使用)
# MAGIC  * *`import tensorflow as tf; print(tf.__version__)`で確認できます*
# MAGIC 
# MAGIC * JARs:
# MAGIC  * `spark-deep-learning-1.0.0-spark2.3-s_2.11.jar`
# MAGIC  * `tensorframes-0.3.0-s_2.11.jar`
# MAGIC  * *クラスターのSpark UI > Environmentで確認できます*
# MAGIC  
# MAGIC  
# MAGIC #### 参考資料
# MAGIC 
# MAGIC - [Databricks機械学習ランタイムを用いた動画における不審な振る舞いの検知 \- Qiita](https://qiita.com/taka_yayoi/items/0dff172a79040ec5cfb6)
# MAGIC - [画像アプリケーションに対するDatabricksリファレンスソリューション \- Qiita](https://qiita.com/taka_yayoi/items/41be81691df3f7c8e6bf)
# MAGIC - [Databricksにおける画像の取り扱い \- Qiita](https://qiita.com/taka_yayoi/items/8d4b1b61699d68a34e58)
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2021/10/14</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>クラスター</td><td>5.5ML(現在のランタイムで廃止されているDeepImageFeaturizerを使用しているため)</td></tr>
# MAGIC </table>
# MAGIC <img style="margin-top:25px;" src="https://jixjiadatabricks.blob.core.windows.net/images/databricks-logo-small-new.png" width="140">

# COMMAND ----------

# DBTITLE 1,動画設定と表示用ヘルパー関数のインクルード
# MAGIC %run ./video_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## トレーニングデータのロード
# MAGIC <br>
# MAGIC * 前のステップで生成したトレーニングデータセットを含むParquetファイルを読み込みます
# MAGIC * 手でラベル付けしたデータを読み込みます 

# COMMAND ----------

# joinする前にプレフィクスを追加
prefix = "dbfs:" + targetImgPath

# 手でラベル付したデータを読み込み 
from pyspark.sql.functions import expr
labels = spark.read.csv(labeledDataPath, header=True, inferSchema=True)
labels_df = labels.withColumn("filePath", expr("concat('" + prefix + "', ImageName)")).drop('ImageName')

# Parquetフォーマットの特徴量データを読み込み
featureDF = spark.read.parquet(imgFeaturesPath)

# ラベルと特徴量をjoinしてトレーニングデータセットを作成
train = featureDF.join(labels_df, featureDF.origin == labels_df.filePath).select("features", "label", featureDF.origin)

# トレーニングで使用する画像の数を確認
train.count()

# COMMAND ----------

display(labels_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ロジスティック回帰モデルのトレーニング

# COMMAND ----------

# DBTITLE 0,ロジスティック回帰モデルのトレーニング
from pyspark.ml.classification import LogisticRegression

# LogisticRegressionモデルのフィッティング
lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
lrModel = lr.fit(train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## テストデータに対する予測

# COMMAND ----------

# DBTITLE 0,テストデータに対する予測
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel

# テストデータのロード
featuresTestDF = spark.read.parquet(imgFeaturesTestPath)

# テストデータに対する予測の実施
result = lrModel.transform(featuresTestDF)
result.createOrReplaceTempView("result")

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

# StructTypeの最初と2番目の要素を抽出
firstelement=udf(lambda v:float(v[0]),FloatType())
secondelement=udf(lambda v:float(v[1]),FloatType())

# 2番目の要素が我々が必要としている確率です
predictions = result.withColumn("prob2", secondelement('probability'))
predictions.createOrReplaceTempView("predictions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 動画に不審な振る舞いはあったのか？

# COMMAND ----------

# MAGIC %sql
# MAGIC select origin, probability, prob2, prediction from predictions where prediction = 1  order by prob2 desc

# COMMAND ----------

# MAGIC %md
# MAGIC ## トップ3のフレームを確認
# MAGIC 
# MAGIC `prob2`カラムに基づいて最も不審と判断されたトップ3のフレームを確認します。

# COMMAND ----------

displayImg(f"{work_path}/videos/cctvFrames/test/Fight_OneManDownframe0017.jpg")

# COMMAND ----------

displayImg(f"{work_path}/videos/cctvFrames/test/Fight_OneManDownframe0016.jpg")

# COMMAND ----------

displayImg(f"{work_path}/videos/cctvFrames/test/Fight_OneManDownframe0019.jpg")

# COMMAND ----------

# MAGIC %md ## ソース動画の確認
# MAGIC 
# MAGIC 不審な画像のソースとなっている動画を確認してみます。
# MAGIC 
# MAGIC ![](https://s3.us-east-2.amazonaws.com/databricks-dennylee/media/Fight_OneManDown.gif)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
