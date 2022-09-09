# Databricks notebook source
# MAGIC %md openslideのPythonバインディングのインストール

# COMMAND ----------

# MAGIC %pip install openslide-python

# COMMAND ----------

dbutils.widgets.removeAll()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 機械学習による病理画像分析の自動化
# MAGIC 
# MAGIC ### Whole Slide Image Segmentation
# MAGIC 
# MAGIC こちらは以下の記事で引用されているサンプルノートブックとなります。
# MAGIC 
# MAGIC > [Databricksにおける機械学習による病理画像分析の自動化 \- Qiita](https://qiita.com/taka_yayoi/items/3929677d4e0c9dffaef4)
# MAGIC 
# MAGIC このノートブックでは、癌細胞の転移の確率をWhole Slide Image(WSI)上にマッピングするモデルをトレーニングするワークフローにおいて、ステップバイステップでSparkのディープラーニングの機能を説明します。
# MAGIC 
# MAGIC ここでは、[Camelyon16 Grand Challenge](http://gigadb.org/dataset/100439)から得られるWSI、転移している領域を手書きした注釈を使用します。
# MAGIC 
# MAGIC まず初めに、注釈データに基づいて腫瘍・正常パッチ(画像の断片)を生成するためにApache Sparkの並列分散処理機能を活用します。
# MAGIC 
# MAGIC **参考論文**
# MAGIC - [病理診断におけるデジタル化と AI の現状(PDF)](https://www.jstage.jst.go.jp/article/haigan/60/2/60_81/_pdf/-char/ja)
# MAGIC 
# MAGIC > ただし WSI による病理画像は極めてサイズが大きく， そのまま CNN で解析することはできない. 任意の大きさ(128×128px 程度)の切り取り画像(パッチ)に切り分け， パッチごとに解析を行うことが一般的である
# MAGIC 
# MAGIC 後段のモデルでは生成したパッチの特徴量を使用します。この特徴量を計算するために事前学習済みディープラーニングモデルを使います。このような考え方は、事前学習モデルから新しいモデルに知識(特徴量のエンコーディングなど)を移し替えることから、*転移学習*と呼ばれます。
# MAGIC 
# MAGIC このノートブックでは、以下を実施します：
# MAGIC * OpenSlideを用いた注釈に基づく腫瘍・正常パッチの生成
# MAGIC * 推論のためのデータ準備
# MAGIC * 分散処理による特徴量エンジニアリング
# MAGIC   * Apache Sparkのバイナリーファイルデータソースによるデータロード
# MAGIC   * 特徴量抽出のためのモデルのロードおよび準備
# MAGIC   * 事前学習済みモデルによる特徴量の計算
# MAGIC * スライドのセグメントに対して腫瘍・正常の予測を行う二値分類器のトレーニング
# MAGIC * (MLflowでロギングされた)トレーニング済みモデルを用いて新規スライドにおける腫瘍の確率に基づくヒートマップの投影
# MAGIC 
# MAGIC 要件：
# MAGIC * Databricksランタイム5.5ML以上
# MAGIC * 2〜8台のワーカーから構成されるr4シリーズ(AWS)、Dシリーズ(Azure)のインスタンス
# MAGIC * クラスターへのopenslide-pythonのインストール(cmd1、cmd7を参照)
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2021/05/18</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.3</td></tr>
# MAGIC   <tr><td>クラスター</td><td>DBR7.3ML</td></tr>
# MAGIC </table>
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/themes/databricks/assets/images/header_logo_2x.png" alt="logo" width="150"/> 

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ1: データ準備
# MAGIC _癌の転移箇所の輪郭から腫瘍・正常パッチを生成します。_
# MAGIC 
# MAGIC 転移箇所の輪郭情報（領域を構成するポリゴン）は病理学者によって目視で作成されたものであり、XML形式で提供されています。ここでは、[Baidu Research NCRF project](https://github.com/baidu-research/NCRF)において作成された注釈情報の編集版のコピーを使用します。この注釈情報を用いて、腫瘍を含むパッチ(画像のサブセット)、正常な細胞を含むパッチを生成します。次に、これらラベル付けされたパッチを用いて、どの領域に主要の転移があるのかを識別する_画像セグメンテーションモデル_をトレーニングします。
# MAGIC 
# MAGIC これらの画像を操作する際、[OpenSlideライブラリ](https://openslide.org/)を使用します。ライブラリをインストールするためには、マシンが起動する際に実行される[init script](https://docs.databricks.com/user-guide/clusters/init-scripts.html)を使用します。

# COMMAND ----------

# MAGIC %md
# MAGIC このワークフローを実行する際に、最初に`openslide-tools`をクラスターにインストールする`Init Script`を作成する必要があります。このスクリプトを[使用するクラスターにアタッチ](https://docs.databricks.com/user-guide/clusters/init-scripts.html#configure-a-cluster-scoped-init-script)する必要があります。initスクリプトを作成した後、あなたが使用するクラスターの`Init Script`設定ペインに`/openslide/openslide-tools.sh`のパスを追加してください。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/demo20210518-digital-pathology/init_script.png)

# COMMAND ----------

# openslide-toolsをインストールするinitスクリプトを /openslide/openslide-tools.sh として作成します。
# initスクリプト作成・設定後はこちらのセルの実行は不要です。
dbutils.fs.mkdirs('/openslide/')
dbutils.fs.rm('/openslide/openslide-tools.sh', True)
dbutils.fs.put('/openslide/openslide-tools.sh',
               """
               #!/bin/bash
               apt-get install -y openslide-tools
               """)

# COMMAND ----------

# MAGIC %md
# MAGIC 続ける前にクラスターを再起動しましょう。この作業はこのワークフローを**初めて**実行する際にのみ必要であることに注意してください。

# COMMAND ----------

# DBTITLE 1,ノートブックの初期化及び注釈データの取得
import os
import subprocess
from pyspark.sql import functions as F
import re

# Username を取得
username_raw = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
# Username の英数字以外を除去し、全て小文字に変換
username = re.sub('[^A-Za-z0-9]+', '', username_raw).lower()
# ファイル格納パス
work_path = f"/tmp/{username}/digital-pathology/"

# パラメーターを設定するウィジェットを追加します
dbutils.widgets.text('PATCH_SIZE',"299")
dbutils.widgets.text('MAX_N_PATCHES','100000')
#dbutils.widgets.text('MAX_N_PATCHES','10000')

PATCH_SIZE=int(dbutils.widgets.get('PATCH_SIZE'))
MAX_N_PATCHES=int(dbutils.widgets.get('MAX_N_PATCHES'))
BASE_PATH=work_path

print(f"MAX_N_PATCHES:{MAX_N_PATCHES} PATCH_SIZE:{PATCH_SIZE} BASE_PATH:{BASE_PATH}")

# クリーンアップ
dbutils.fs.rm(BASE_PATH, recurse=True)

# 注釈データのパス
WSI_TIF_PATH = "/databricks-datasets/med-images/camelyon16/"
LEVEL = 0

ANNOTATIONS_PATH = BASE_PATH+"/annotations/"
PATCH_PATH = BASE_PATH+"/patches/"
  
for path in [BASE_PATH, PATCH_PATH]:
  if not os.path.exists(('/dbfs/' + path)):
    print("path %s does not exist" % path)
    dbutils.fs.mkdirs(path)
    print("created path %s" % path)
  else:
    print("path %s exists"%path)

# 座標ファイルのダウンロード
dbutils.fs.rm(ANNOTATIONS_PATH, recurse=True)
dbutils.fs.mkdirs(ANNOTATIONS_PATH)

subprocess.call('wget https://raw.githubusercontent.com/baidu-research/NCRF/master/coords/tumor_train.txt',
                shell=True,
                cwd='/dbfs'+ANNOTATIONS_PATH)
subprocess.call('wget https://raw.githubusercontent.com/baidu-research/NCRF/master/coords/normal_train.txt',
                shell=True,
                cwd='/dbfs'+ANNOTATIONS_PATH)

print(dbutils.fs.head(ANNOTATIONS_PATH+'normal_train.txt', 111))

# COMMAND ----------

# MAGIC %md
# MAGIC データセットにあるスライドを見てみましょう。

# COMMAND ----------

display(dbutils.fs.ls(WSI_TIF_PATH))

# COMMAND ----------

import numpy as np
import openslide
import matplotlib.pyplot as plt

# COMMAND ----------

#f, axarr = plt.subplots(1,4,sharey=True)
#i=0
#for pid in ["normal_034","normal_036","tumor_044", "tumor_045"]:
#  path = '/dbfs/%s/%s.tif' %(WSI_TIF_PATH,pid)
#  slide = openslide.OpenSlide(path)
#  axarr[i].imshow(slide.get_thumbnail(np.array(slide.dimensions)//50))
#  axarr[i].set_title(pid)
#  i+=1
  
#display()

# COMMAND ----------

# MAGIC %md
# MAGIC レベルを指定することでスライドの領域を抽出することも可能です。
# MAGIC 
# MAGIC [OpenSlide Python — OpenSlide Python 1\.1\.2 documentation](https://openslide.org/api/python/#openslide.OpenSlide.level_count)
# MAGIC 
# MAGIC > The number of levels in the slide. Levels are numbered from 0 (highest resolution) to level_count - 1 (lowest resolution).
# MAGIC 
# MAGIC 0が最もズームインされた状態で、大きくなるほどズームアウトされます。

# COMMAND ----------

slide = openslide.OpenSlide('/dbfs/%s/normal_034.tif' %WSI_TIF_PATH)
image_datas=[]
region=[35034,131012] # 対象座標
size=[4000,4000] # 表示領域のサイズ
f, axarr = plt.subplots(2,3,sharex=True,sharey=True)

for level,ind in zip([0,1,2,3,4,5,6],[(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]):
  img = slide.read_region(region,level,size)
  axarr[ind].imshow(img)
  axarr[ind].set_title("level:%d"%level)

display()

# COMMAND ----------

# MAGIC %md
# MAGIC ここでは、パッチ抽出関数をpythonの関数としてラップします。この関数は、スライドの名称、抽出すべきパッチの中心xy座標、パッチに対応するラベル(0が正常、1が腫瘍)を引数とします。そして、抽出した領域を`path/<name>_<x coord>_<y coord>_LABEL<label 0/1>.jpg`の形式で保存します。このpython関数をSparkのUDF(ユーザー定義関数)として登録し、並列実行することでパッチを生成します。

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StringType

def process_patch(name, x_center, y_center, label, patch_base_path=None, drop_coords=False):
  """
  Generate a patch given coordinate centers and return patch information.
  name: name of the wsi file 
  x_center: x coordinate of the center of the patch
  y_center: y coordinate of the center of the patch
  label: label of the patch based on annotation data
  patch_base_path: specify if different than where the default working directory is (used for scoring)
  drop_coords: boolean, if True then the coordinates will not be included in the image file name (will be used in scoring)
  """

  if not patch_base_path:
    patch_base_path = '/dbfs/' + PATCH_PATH
    
  wsi_path = '/dbfs' + WSI_TIF_PATH + name + '.tif'
  _x = int(x_center)
  _y = int(y_center)

  x = _x - PATCH_SIZE // 2
  y = _y - PATCH_SIZE // 2
  
  if drop_coords:
    patch_path = "/%s/%s_LABEL%s.jpg" % (patch_base_path, name, str(label))
  else:
    patch_path = "/%s/%s_%s_%s_LABEL%s.jpg" % (patch_base_path, name, str(x_center), str(y_center), str(label))

  slide = openslide.OpenSlide(wsi_path)
  img = slide.read_region((x, y), LEVEL, (PATCH_SIZE, PATCH_SIZE)).convert('RGB')
  img.save(patch_path)
  
  return(patch_path)

# SparkのUDFとして関数を登録します
process_patch_udf = F.udf(process_patch, StringType())

# COMMAND ----------

# MAGIC %md
# MAGIC これで、注釈データに基づいて腫瘍・正常パッチの座標を含むデータフレームを作成できます。

# COMMAND ----------

# 正常パッチ座標をロードして label = 0 を割り当てます
df_coords_normal = (
  spark.read.csv(ANNOTATIONS_PATH+'/normal_train.txt').withColumn('label', F.lit(0))
)

# 腫瘍パッチ座標をロードして label = 1 を割り当てます
df_coords_tumor = (
  spark.read.csv(ANNOTATIONS_PATH+'/tumor_train.txt').withColumn('label', F.lit(1))
)

# 上記のパッチをUNIONで結合します
df_coords = df_coords_normal.union(df_coords_tumor).selectExpr('lower(_c0) as pid','_c1 as x_center', '_c2 as y_center', 'label')

display(df_coords)

# COMMAND ----------

# MAGIC %md
# MAGIC 確実に既存のWSIファイルの座標のみをパッチ生成関数に渡すように、既存のWSIファイルの一覧と座標のデータフレームをJOINします。

# COMMAND ----------

# WSI画像のパスを含むデータフレームを作成
df_wsi_path = (
  spark.createDataFrame(dbutils.fs.ls(WSI_TIF_PATH))
  .withColumn('name',F.regexp_replace('name', '.tif', ''))
  .withColumn('pid',F.lower(F.col('name')))
)

# 腫瘍・正常パッチの座標、ラベルのテーブルとJOIN
df_patch_info = (
  df_coords
  .join(df_wsi_path, on='pid')
  .selectExpr('pid','name','x_center','y_center','label')
)

display(df_patch_info)

# COMMAND ----------

# MAGIC %md
# MAGIC 座標情報、ラベルに対応する`jpg`ファイルを生成するためにパッチ生成関数を実行します。後段の分析を高速に行うために、生成されたパッチのパスを`csv`ファイルに出力します。
# MAGIC 
# MAGIC **注意**
# MAGIC - `MAX_N_PATCHES`の数のパッチを生成するため、並列処理の場合でも処理に若干の時間を要することにあらかじめご注意ください。
# MAGIC - 参考値： `MAX_N_PATCHES`が100,000の場合、r4.2xlargeの8台構成で約10分

# COMMAND ----------

patch_path_df = df_patch_info.sample(1.0,False).limit(MAX_N_PATCHES).\
                repartition(1024).select(process_patch_udf('name','x_center','y_center','label').alias('path'))

# パス一覧を格納したファイルを保存
patch_path_df.write.csv(BASE_PATH+'/patch_path.csv', header = True)

# COMMAND ----------

patch_path_df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ2: データのQC(品質管理)
# MAGIC 
# MAGIC パッチを作成後、データQCのために全ての画像をSparkデータフレームにロードします。

# COMMAND ----------

image_df = (
  spark.read.csv(BASE_PATH+'/patch_path.csv',header=True)
  .select('path', F.regexp_extract("path",r"LABEL(\d)",1).cast('int').alias('label'))
  .limit(MAX_N_PATCHES)
)

# COMMAND ----------

display(image_df)

# COMMAND ----------

# MAGIC %md
# MAGIC データセットにおける腫瘍・正常の分布を確認してみましょう。

# COMMAND ----------

display(image_df.select('label').groupBy('label').count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ3: トレーニング
# MAGIC 
# MAGIC データの前処理が終わったので、二値分類器のトレーニングの準備が整ったことになります。このタスクでは、転移学習アプローチをとります。ここでは、大量の腫瘍・正常パッチから特徴量を並列処理で抽出する際に、`pandas udf`を活用します。特徴量抽出には`InceptionV3`のアーキテクチャーと重み情報を使用し、以降ではデモンストレーションとして、分類を行うためにsparkmlのロジスティック回帰を使用します。
# MAGIC 
# MAGIC **参考情報**
# MAGIC - [転移学習とは \| メリット・デメリット・ファインチューニングの意味 \| Ledge\.ai](https://ledge.ai/transfer-learning/)
# MAGIC - [転移学習とは？ディープラーニングで期待の「転移学…｜Udemy メディア](https://udemy.benesse.co.jp/data-science/deep-learning/transfer-learning.html)
# MAGIC - [InceptionV3](https://keras.io/api/applications/inceptionv3/)
# MAGIC - [第10回　学習済みInception\-v3モデルを使った画像認識アプリケーションを作ろう｜Tech Book Zone Manatee](https://book.mynavi.jp/manatee/detail/id=77514)
# MAGIC 
# MAGIC > Googleによって開発されたInception-v3は、ILSVRCという大規模画像データセットを使った画像識別タスク用に1,000クラスの画像分類を行うよう学習されたモデルで、非常に高い精度の画像識別を達成しています。

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input 
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from pyspark.sql.functions import col, pandas_udf, PandasUDFType

# COMMAND ----------

# インセプションモデルのロード
model = InceptionV3(include_top=False)
model.summary() 

# COMMAND ----------

# 全てのワーカーノードで重み情報を利用できるように、重み情報をブロードキャストします
broadcaseted_model_weights = sc.broadcast(model.get_weights())
def model_fn():
  """
  Returns a InceptionV3 model with top layer removed and broadcasted pretrained weights.
  """
  model = InceptionV3(weights=None, include_top=False)
  model.set_weights(broadcaseted_model_weights.value)
  return model

# COMMAND ----------

from PIL import Image
import io
import numpy as np
import pandas as pd

def preprocess(content):
  """
  Preprocesses raw image bytes for prediction, making sure it works 
  """
  img = Image.open(content)
  arr = img_to_array(img)
  return preprocess_input(arr)

def featurize_series(model, content_series):
  """
  Featurize a pd.Series of raw images using the input model.
  :return: a pd.Series of image features
  """
  input = np.stack(content_series.map(preprocess))
  preds = model.predict(input)
  # いくかのレイヤーでは、アウトプットの特徴量は多次元のテンソルになります
  # Sparkデータフレームで格納しやすいように特徴量テンソルをベクトルに変換します
  output = [p.flatten() for p in preds]
  return pd.Series(output)

# COMMAND ----------

from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql import functions as F

@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_pudf(content_series_iter):
  '''
  This method is a Scalar Iterator pandas UDF wrapping our featurization function.
  The decorator specifies that this returns a Spark DataFrame column of type ArrayType(FloatType).
  
  :param content_series_iter: This argument is an iterator over batches of data, where each batch
                              is a pandas Series of image data.
  '''
  # pandas UDFのスカラーイテレータで一度だけモデルを読み込んで以降のデータバッチでは再利用します
  # これによって大きなモデルのロードに要するオーバーヘッドを削減します
  model = model_fn()
  for content_series in content_series_iter:
    yield featurize_series(model, content_series)

to_vector_udf = F.udf(lambda x: Vectors.dense(x), VectorUDT())

# COMMAND ----------

# Arrowのバッチサイズを指定します
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")

# COMMAND ----------

# MAGIC %md
# MAGIC 事前学習済みの[InceptionV3](https://arxiv.org/abs/1512.00567)モデルを用いて、画像データから特徴量を生成します。まず初めに、画像データセットを事前学習済みモデルによって新たな特徴量に変換し、特徴量を用いて二値分類器をトレーニングします。

# COMMAND ----------

# トレーニング用のデータセットとテスト用のデータセットを分割します
(train_img_df, test_img_df) = image_df.select('label',F.regexp_replace(F.col('path'),'dbfs:','/dbfs').alias('path')).randomSplit([0.7, 0.3], seed=42)

# COMMAND ----------

display(train_img_df)

# COMMAND ----------

train_df = train_img_df.repartition(128).select("label", featurize_pudf("path").alias("features")).select("label",to_vector_udf("features").alias('features_vec'))
test_df = test_img_df.repartition(128).select("label", featurize_pudf("path").alias("features")).select("label",to_vector_udf("features").alias('features_vec'))
print(train_df.count(),test_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC 以下の関数は、二値分類器をトレーニングし、MLflowを用いてモデルを保存します。加えて、各々のトレーニングの入力パラメーター、メトリクス(精度、トレーニング実行時刻)を記録します。

# COMMAND ----------

import mlflow
import mlflow.spark

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pyspark.sql.functions as F
import time

def train(max_iter, reg_param, elastic_net_param):
  start_time = time.time()

  with mlflow.start_run(run_name="binary-classifier") as run:
    
    lr = LogisticRegression(featuresCol='features_vec',maxIter=max_iter, regParam=reg_param, elasticNetParam=elastic_net_param)
    model = lr.fit(train_df)
  
    _df = model.transform(test_df)
    _predictions = _df.select("prediction", "label")
  
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
    area_under_roc=evaluator.evaluate(_predictions)
    elapsed_time = time.time() - start_time
  
    # 全てのパラメーターとメトリクス、モデル自身をMLflowを用いで記録します
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("reg_param", reg_param)
    mlflow.log_param("elastic_net_param", elastic_net_param)
  
    mlflow.log_metric("area_under_ROC", area_under_roc)
    mlflow.log_metric('elapsed_time', elapsed_time)
    mlflow.spark.log_model(model,'wsi-dt-model')
  
  return(area_under_roc)

# COMMAND ----------

# MAGIC %md
# MAGIC 以下の処理は生成した大量のパッチを用いてトレーニングを行いますので、推奨スペックのクラスターでない場合、処理に長い時間を要します。

# COMMAND ----------

# パラメーターを指定してトレーニングを実行します
max_iter, reg_param, elastic_net_param = 10, 0.1, 0.2
area_u_roc = train(max_iter, reg_param, elastic_net_param)
print('precision is %f'%area_u_roc)

# COMMAND ----------

# MAGIC %md
# MAGIC ノートブック上でMLflowのログを直接確認することができます。右上の`Experiment`ボタンを押してみてください。

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ4: スコアリング
# MAGIC 
# MAGIC トレーニングしたモデルを用いて、癌が転移している可能性がある領域を識別します。

# COMMAND ----------

# MAGIC %md
# MAGIC まず最初に、スライドと分析するセグメントの座標を受け取り、スコアリングに必要なパッチの生成に使用されるタイル(グリッド)の中心座標を格納したデータフレームを返却する関数を定義します。

# COMMAND ----------

from pyspark.sql.types import *

def generate_patch_grid_df(*args):
  x_min,x_max,y_min,y_max, slide_name = args
  
  x = np.array(range(x_min,x_max,PATCH_SIZE))
  y = np.array(range(y_min,y_max,PATCH_SIZE))
  xv, yv = np.meshgrid(x, y, indexing='ij')
  
  cSchema = StructType([
    StructField("x_center", IntegerType()),
    StructField("y_center", IntegerType()),
    StructField("i_j", StringType()),
  ])

  arr=[]
  for i in range(len(x)):
    for j in range(len(y)):
      x_center = int(xv[i,j].astype('int'))
      y_center = int(yv[i,j].astype('int'))
      arr+=[[x_center,y_center,"%d_%d"%(i,j)]]
  grid_size = xv.shape
  df = spark.createDataFrame(arr,schema=cSchema) 
  return(df,grid_size)

# COMMAND ----------

# MAGIC %md
# MAGIC スライド上で事前に指定されたセグメント上のグリッドに基づいてパッチを生成します。

# COMMAND ----------

PATCH_SIZE = 299
x_min, x_max, y_min, y_max = (23437, 53337, 135815, 165715) # 予測対象領域
path = f"{BASE_PATH}ml/patches" # パッチの出力先
dbutils.fs.mkdirs(path)

patch_path = f"/dbfs{path}" # パッチの出力先(DBFSプレフィクスあり)

print(path, patch_path)

# COMMAND ----------

name = "tumor_058" # 予測対象スライド
df, grid_size = generate_patch_grid_df(x_min, x_max, y_min, y_max,name) # パッチグリッドの生成
df = df.selectExpr("'%s' as name"%name, "x_center", "y_center", "i_j as label", "'%s' as patch_base_path"%patch_path)
display(df)

# COMMAND ----------

# パッチの作成
df.repartition(288).select(process_patch_udf('name', 'x_center', 'y_center', 'label', 'patch_base_path', F.lit("true"))).collect();

# COMMAND ----------

# MAGIC %md
# MAGIC 生成したタイルをデータフレームに再度読み込みます。このデータフレームのラベルはグリッド上のタイルの位置(例えば、ラベルが`i_j`の場合、i番目の行、j番目の列を意味します)を示していることに注意してください。このラベルは、あとのステップで腫瘍を含む確率に基づいてヒートマップを作成する際に使用します。

# COMMAND ----------

# 画像をデータフレームにロード
image_df = spark.read.format("image").load("file:" + patch_path)\
                                     .withColumn('label', F.regexp_extract("image.origin",r"LABEL(\d+_\d+)", 1))

image_df.repartition(100)

# COMMAND ----------

display(image_df)

# COMMAND ----------

#display(image_df.select("image.origin"))

# COMMAND ----------

# MAGIC %md
# MAGIC 分析対象のスライドからタイルを作成したので、モデルのURIを指定して学習済みモデルを読み込みます。特徴量を作成して、それぞれのタイルのスコアリングを行い、タイルに腫瘍が含まれる確率を記録していきます。

# COMMAND ----------

# MAGIC %md **注意**
# MAGIC - オリジナルのノートブックで使用されている`DeepImageFeaturizer`は、DBR 7.0以降では削除されているので、pandas UDFで特徴量抽出を行う必要があります。
# MAGIC 
# MAGIC [Deep Learning Pipelines \| Databricks on AWS](https://docs.databricks.com/applications/machine-learning/train-model/deep-learning-pipelines.html#transfer-learning)

# COMMAND ----------

# スコアリング対象の画像から特徴量を抽出してデータフレームを作成
image_df_with_path = image_df.select("image.origin", "label")
image_df_with_path = image_df_with_path.withColumn('dbfs_path', F.regexp_extract("origin", r"file\:(.*)", 1))

image_df_with_features = image_df_with_path.repartition(128).select("label", "dbfs_path", "origin", featurize_pudf("dbfs_path").alias("features")).select("label", "origin", to_vector_udf("features").alias('features_vec'))

# COMMAND ----------

#display(image_df_with_features.limit(1))

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
import mlflow.spark
import mlflow.pyfunc

# MLflowのランを確認してトレーニング済みモデルのURIを指定します
model_uri="dbfs:/databricks/mlflow-tracking/173527346111339/a50169ce2db049c5b9f331f87bf5ead6/artifacts/wsi-dt-model"
model = mlflow.spark.load_model(model_uri=model_uri)

# COMMAND ----------

# 予測の実施
get_p_udf = udf(lambda v:float(v[1]), FloatType())
predictions_df = model.transform(image_df_with_features).\
                       withColumn('p', get_p_udf('probability')).drop('features', 'features_vec', 'probability', 'rawPrediction')

# COMMAND ----------

display(predictions_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ5: スライド上に予測結果を可視化する
# MAGIC 
# MAGIC PythonのPILライブラリを用いて、予測結果に基づくヒートマップを生成します。

# COMMAND ----------

pid = "tumor_058"
patch_path = f"/dbfs{BASE_PATH}ml/patches/"

# COMMAND ----------

inp_arr=predictions_df.select("origin", "label", "p").collect()
pred_arr = [(x.label,x.p) for x in inp_arr]
height=width=299

# COMMAND ----------

from pyspark.sql import functions as F
from PIL import Image
def stitch_images(*args):
  patch_path,pid,height,width,scale_f,pred_arr = args
  path = patch_path+"/"+pid
  n_x,n_y=grid_size
  img_size = int(scale_f*width),int(scale_f*height)
  
  total_width = img_size[0]*n_x
  total_height = img_size[1]*n_y
  
  new_im = Image.new('RGB', (total_width, total_height))
  x_offset = 0
  y_offset = 0
  
  for ij,p in pred_arr:
      i = int(ij.split('_')[0])
      j = int(ij.split('_')[1])
      
      x_offset = i*img_size[0]
      y_offset = j*img_size[1]
      
      temp_matrix = (p , 0.0, 0.0, 0.0,
                     0.0, 56/255, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0 )
      
      pt = "{}_LABEL{}_{}.jpg".format(path,i,j)   
      img = Image.open(pt).resize(img_size).convert('RGB', temp_matrix)
      new_im.paste(img, (x_offset,y_offset))
      
      
  return(new_im)  

# COMMAND ----------

# 処理に10分程度時間を要します
patched_images = stitch_images(patch_path, pid, 299, 299, 0.2, pred_arr)

# COMMAND ----------

# MAGIC %md 予測結果において赤色が密集している箇所は、癌細胞が転移している確率が高いことを示しています。

# COMMAND ----------

slide = openslide.OpenSlide('/dbfs/%s/%s.tif' %(WSI_TIF_PATH,pid))
region= [x_min,y_min]

size=[2900,2900]
slide_segment= slide.read_region(region,3,size)

f, axarr = plt.subplots(1,3)
axarr[0].imshow(patched_images) # 予測結果の表示
axarr[1].imshow(slide_segment) # 予測範囲の元画像
axarr[1].set_xlim=3000
axarr[1].set_ylim=3000
axarr[2].imshow(slide.get_thumbnail(np.array(slide.dimensions)//50)) # スライド全体
axarr[2].axis('off')
f.set_figheight(12)
f.set_figwidth(12)
display()

# COMMAND ----------

# MAGIC %md
# MAGIC # END
