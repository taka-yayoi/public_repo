# Databricks notebook source
# DBTITLE 0,Video Identification of Suspicious Behavior: Preparation
# MAGIC %md 
# MAGIC 
# MAGIC # 動画における不審な振る舞いの検知：準備
# MAGIC 
# MAGIC このノートブックでは動画データに以下の処理を行い、データの**準備**を行います。
# MAGIC 
# MAGIC - 個々の画像を抽出し、画像をDBFS/クラウドストレージに保存することで画像を処理します。
# MAGIC - 保存された画像にSparkのDeep Learning Pipelinesの`DeepImageFeaturizer`を適用して、画像の特徴量を抽出し、ParquetフォーマットでDBFS/クラウドストレージに保存します。
# MAGIC   - この操作をトレーニングデータセット、テストデータセットの両方に適用します。
# MAGIC 
# MAGIC このノートブックで用いるソースデータは、[EC Funded CAVIAR project/IST 2001 37540](http://homepages.inf.ed.ac.uk/rbf/CAVIAR/)で確認できます。
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2018/09/mnt_raela_video_splash.png" width=900/>
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

# DBTITLE 1,トレーニング用動画のディレクトリを表示
display(dbutils.fs.ls(srcVideoPath))

# COMMAND ----------

# DBTITLE 1,動画のサンプル
# MAGIC %md
# MAGIC こちらが不審な振る舞いを特定するためにトレーニングを行うサンプル動画です。
# MAGIC * このデータのソースは[EC Funded CAVIAR project/IST 2001 37540](http://homepages.inf.ed.ac.uk/rbf/CAVIAR/)です。
# MAGIC 
# MAGIC ![](https://databricks.com/wp-content/uploads/2018/09/Browse2.gif)

# COMMAND ----------

# DBTITLE 1,動画の処理 - ビデオフレームの抽出
# MAGIC %md
# MAGIC OpenCV (`cv2`)を用いて、MPG動画からJPG画像を抽出します。

# COMMAND ----------

# CV2を用いた画像の抽出および保存
def extractImagesSave(src, tgt):
  import cv2
  import uuid
  import re

  ## 1秒ごとにビデオフレームを抽出し、JPGとして保存
  def extractImages(pathIn):
      count = 0
      srcVideos = "/dbfs" + src + "(.*).mpg"
      p = re.compile(srcVideos)
      vidName = str(p.search(pathIn).group(1))
      vidcap = cv2.VideoCapture(pathIn)
      success,image = vidcap.read()
      success = True
      while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
        success,image = vidcap.read()
        print ('Read a new frame: ', success)
        
        if success:
          cv2.imwrite("/dbfs" + tgt + vidName + "frame%04d.jpg" % count, image)     # フレームをJPEGファイルとして保存
          count = count + 1
          print ('Wrote a new frame')
        
  ## 全ての動画からフレームを抽出しDBFSに保存
  def createFUSEpaths(dbfsFilePath):
    return "/dbfs/" + dbfsFilePath[0][6:]
  
  # fileList RDDを構築
  fileList = dbutils.fs.ls(src)
  FUSEfileList = map(createFUSEpaths, fileList)
  FUSEfileList_rdd = sc.parallelize(FUSEfileList)
  
  # ディレクトリを確実に作成
  dbutils.fs.mkdirs(tgt)
  
  # 画像の抽出および保存
  FUSEfileList_rdd.map(extractImages).count()

  
# 空のファイルの削除
def removeEmptyFiles(pathDir):
  import sys
  import os

  rootDir = '/dbfs' + pathDir
  for root, dirs, files in os.walk(rootDir):
    for f in files:
      fileName = os.path.join(root, f)
      if os.path.getsize(fileName) == 0:
        print ("empty fileName: %s \n" % fileName)
        os.remove(fileName)


# COMMAND ----------

# MAGIC %md # トレーニングデータセット

# COMMAND ----------

# DBTITLE 1,トレーニング画像の抽出
# 画像の抽出
extractImagesSave(srcVideoPath, targetImgPath)

# 空のファイルの削除
removeEmptyFiles(targetImgPath)

# 動画から抽出した画像ファイルの一覧を表示
display(dbutils.fs.ls(targetImgPath))

# COMMAND ----------

# DBTITLE 1,トレーニング画像の確認
from pyspark.ml.image import ImageSchema

trainImages = ImageSchema.readImages(targetImgPath)
display(trainImages)

# COMMAND ----------

# DBTITLE 1,DeepImageFeaturizerを用いた特徴量抽出
# MAGIC %md
# MAGIC InceptionV3モデルを通じて画像の特徴量を生成するために、[Spark Deep Learning Pipelines](https://github.com/databricks/spark-deep-learning)の`DeepImageFeaturizer`を使用します。

# COMMAND ----------

# DBTITLE 0,Save Features Function
# DeepImageFeaturizerを用いて画像特徴量を抽出して保存 
def saveImageFeatures(images, filePath):
  from sparkdl import DeepImageFeaturizer

  # DeepImageFeaturizerとInceptionV3を用いてfeaturizerを構築
  featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")

  # 画像をメタデータ(origin, height, width, nChannels, mode, data)、特徴量(udt)に変換
  features = featurizer.transform(images)

  # 特徴量情報をParquetファイルフォーマットとして出力
  # この処理には数分かかります
  dbutils.fs.mkdirs(filePath)

  # 保存した特徴量から画像ファイル名(imgFileName)のみを抽出
  features.select("image.origin", "features").coalesce(2).write.mode("overwrite").parquet(filePath)

# COMMAND ----------

# DBTITLE 1,トレーニング用の画像特徴量を保存
saveImageFeatures(trainImages, imgFeaturesPath)

# COMMAND ----------

# Parquetファイルを確認
display(dbutils.fs.ls(imgFeaturesPath))

# COMMAND ----------

# MAGIC %md # テストデータセット

# COMMAND ----------

# DBTITLE 1,テスト用動画のディレクトリ
display(dbutils.fs.ls(srcTestVideoPath))

# COMMAND ----------

# DBTITLE 1,テスト用画像の抽出
# 画像の抽出
extractImagesSave(srcTestVideoPath, targetImgTestPath)

# 空のファイルの削除
removeEmptyFiles(targetImgTestPath)

# 動画から抽出した画像ファイルの一覧を表示
display(dbutils.fs.ls(targetImgTestPath))

# COMMAND ----------

# DBTITLE 1,テスト用画像の確認
from pyspark.ml.image import ImageSchema

testImages = ImageSchema.readImages(targetImgTestPath)
display(testImages)

# COMMAND ----------

# DBTITLE 1,テスト用画像の特徴量を保存
saveImageFeatures(testImages, imgFeaturesTestPath)

# COMMAND ----------

# Parquetファイルの確認
display(dbutils.fs.ls(imgFeaturesTestPath))

# COMMAND ----------

# MAGIC %md
# MAGIC # つづく
