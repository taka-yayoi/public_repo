# Databricks notebook source
# MAGIC %md
# MAGIC # コーヒー物体検知: 単一画像からの検知
# MAGIC 
# MAGIC <img src="https://pages.databricks.com/rs/094-YMS-629/images/SBUX-DB.png"/>
# MAGIC 
# MAGIC このノートブックでは、画像におけるシンプルな物体検知の例(コーヒー、カップ、椅子など)をお見せします。Databricks MLランタイムに加えて、[Image AI](https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl)ライブラリを活用します。
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2022/09/08</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>クラスター</td><td>7.3ML</td></tr>
# MAGIC </table>
# MAGIC <img style="margin-top:25px;" src="https://jixjiadatabricks.blob.core.windows.net/images/databricks-logo-small-new.png" width="140">
# MAGIC 
# MAGIC **参考資料**
# MAGIC - [Official English Documentation for ImageAI\! — ImageAI 2\.1\.6 documentation](https://imageai.readthedocs.io/en/latest/index.html)

# COMMAND ----------

# MAGIC %md ## セットアップ
# MAGIC 
# MAGIC **[ImageAI](https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/VIDEO.md):**の一部である**事前学習済み**モデルを追加するためにインストールを行います。
# MAGIC <br><br>
# MAGIC - **ImageAI**: ImageAI本体
# MAGIC - **Yolo V3**: You only look once (YOLO)は最先端のリアルタイム物体検知システムです。ここでは[YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)に記載されているV3を使用します。これは画像向けのモデルです。
# MAGIC - **Resnet**: 特徴量抽出のためにResidual Network model(h5 formatフォーマット)を使用します。このモデルは動画向けです。

# COMMAND ----------

# MAGIC %md ### ImageAIのインストール

# COMMAND ----------

# MAGIC %pip install tensorflow==2.4.1
# MAGIC %pip install keras==2.5.0rc0
# MAGIC 
# MAGIC %pip install imageai

# COMMAND ----------

# MAGIC %md
# MAGIC ### ファイルパスの設定
# MAGIC 
# MAGIC このノートブックではユーザー名に基づいてファイル格納パスを設定します。以下のパスを使用します。
# MAGIC <br><br>
# MAGIC - `/tmp/20210902_workshop/<ユーザー名>/`: 検知対象データ、検出結果などを保存
# MAGIC - `/FileStore/tmp/20210902_workshop/<ユーザー名>/`: 検出結果参照用

# COMMAND ----------

import re
from pyspark.sql.types import * 

# ログインIDからUsernameを取得
username_raw = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
# Username の英数字以外を除去し、全て小文字化。Username をファイルパスやデータベース名の一部で使用可能にするため。
username = re.sub('[^A-Za-z0-9]+', '', username_raw).lower()

# DBFS上のファイル格納パス
work_path = f"/tmp/20210902_workshop/{username}/"

# パスをシェル環境変数に設定して、シェルコマンドから参照できるようにします
import os
os.environ['work_path'] = work_path

print("path: " + work_path)

# COMMAND ----------

# MAGIC %md ### 作業用ディレクトリの初期化

# COMMAND ----------

dbutils.fs.rm(work_path, recurse=True) # 一度削除
dbutils.fs.mkdirs(work_path) # ディレクトリ作成

# COMMAND ----------

# MAGIC %md ### 検知対象ファイルのダウンロード

# COMMAND ----------

# MAGIC %sh
# MAGIC wget -P /dbfs${work_path} https://sajpstorage.blob.core.windows.net/workshop20210615-object-detection/five_drinks.jpg

# COMMAND ----------

# MAGIC %md ### 学習済みモデルのダウンロード
# MAGIC 
# MAGIC Yoloをダウンロードします。

# COMMAND ----------

# MAGIC %sh
# MAGIC ## Install the ImageAI pre-trained models: YoloV3, ResNet
# MAGIC wget -P /dbfs$work_path https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5
# MAGIC #wget -P /dbfs$work_path https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5

# COMMAND ----------

# ダウンロードされていることを確認します
display(dbutils.fs.ls(work_path))

# COMMAND ----------

## ------------------------------
## ユーティリティ関数
## ------------------------------
# displayVid(): クラウドストレージ上の動画を表示
def displayVid(filepath):
  return displayHTML("""
  <video width="480" height="320" controls>
  <source src="/files/%s" type="video/mp4">
  </video>
  """ % filepath)

# displayDbfsVid(): DBFSの動画を表示
def displayDbfsVid(filepath):
  return displayHTML("""
  <video width="480" height="320" controls>
  <source src="/dbfs/%s" type="video/mp4">
  </video>
  """ % filepath)

# displayImg(): DBFS/クラウドストレージの動画を表示
def displayImg(filepath):
  dbutils.fs.cp(filepath, "FileStore/%s" % filepath)
  return displayHTML("""
  <img src="/files/%s" width="800">
  """ % filepath)

# COMMAND ----------

# MAGIC %md ## 単一画像からコーヒーの検知
# MAGIC 
# MAGIC YoloV3と以下のスターバックスのアイスコーヒーの画像を用いて物体検知を行います。
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/workshop20210615-object-detection/five_drinks.jpg" width="600"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ### TensoFlowの設定
# MAGIC 
# MAGIC #### 参考情報
# MAGIC - [Using allow\_growth memory option in Tensorflow and Keras \| by Kobkrit Viriyayudhakorn \| Kobkrit](https://kobkrit.com/using-allow-growth-memory-option-in-tensorflow-and-keras-dc8c8081bc96)

# COMMAND ----------

# TensorFlowの設定
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.visible_device_list = "0"
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5

tf.compat.v1.Session(config=config)
tf.compat.v1.set_random_seed(42) # 再現性確保のために乱数のシードを設定

# COMMAND ----------

# ImageAIからObjectDetectionをインポート
from imageai.Detection import ObjectDetection
import os
import h5py

# 実行パス、分析画像、出力先のパスを指定
execution_path = f"/dbfs{work_path}"
execution_path_source = f"/dbfs{work_path}five_drinks.jpg"
execution_path_results = f"/dbfs{work_path}result.jpg"

print(execution_path)
print(execution_path_source)
print(execution_path_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 物体検知

# COMMAND ----------

# YoloV3モデルとObjectDetectionを使用
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel()

detections = detector.detectObjectsFromImage(
  input_image = execution_path_source, # 検知対象画像パス
  output_image_path = execution_path_results, # 検知結果画像パス
  minimum_percentage_probability = 30 # 確率の閾値
)

# COMMAND ----------

detections

# COMMAND ----------

# MAGIC %md #### 結果の確認
# MAGIC 
# MAGIC - dining tableの確率は30%程度ですが、他のcupに関しては60-90%程度の確率となっています。
# MAGIC - percentage_probabilityの後に表示されているのは検知箇所の座標です。

# COMMAND ----------

# 画像をSparkデータフレームとして読み込み
extractedImages = spark.read.format("image").load(f"{work_path}/result.jpg")
display(extractedImages)

# COMMAND ----------

# 画像を直接表示
displayImg(f"{work_path}/result.jpg")

# COMMAND ----------

# MAGIC %md ## クリーンアップ

# COMMAND ----------

dbutils.fs.rm(work_path, recurse=True)
dbutils.fs.rm(f"/FileStore{work_path}", recurse=True)

# COMMAND ----------

# MAGIC %md ## 今後の進め方
# MAGIC 
# MAGIC ここでは事前学習済みのモデルを用いてImageAIを使用しました。この後の進め方としては以下のことが考えられます。
# MAGIC <br><br>
# MAGIC - 直接`keras-yolov3`を用いて、よりスターバックスに特化したモデル(コーヒー、焼き菓子、カップなど)をトレーニングする。この際にはMLflow、TensorBoardをご活用いただけます。
# MAGIC - モデルのスピードを改善するためにスケールアップする
# MAGIC - より大量の画像、あるいは動画を分析するために *分散処理* を行う
# MAGIC 
# MAGIC 
# MAGIC #### 参考資料
# MAGIC - [YOLOv3 Keras版実装に関して関連記事のまとめ \- Qiita](https://qiita.com/tfukumori/items/9e62c4a7acd7ea6ddd3b)
# MAGIC - [keras\-yolo3 で独自データセットに対して学習させる方法 \| 理系ディア](https://rikeidia.site/archives/236)
# MAGIC - [rplab/Bacterial\-Identification](https://github.com/rplab/Bacterial-Identification)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
