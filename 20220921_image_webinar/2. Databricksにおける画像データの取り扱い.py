# Databricks notebook source
# MAGIC %md
# MAGIC # Databricksにおける画像データの取り扱い
# MAGIC 
# MAGIC 本ノートブックでは、Databricksで画像データをどのように取り扱うのかをご説明します。
# MAGIC 
# MAGIC - PILによる画像の操作
# MAGIC - 画像データソースによる画像の読み込み
# MAGIC - バイナリーファイルデータソースによる画像の読み込みおよび保存
# MAGIC 
# MAGIC #### 参考資料
# MAGIC 
# MAGIC - [画像アプリケーションに対するDatabricksリファレンスソリューション \- Qiita](https://qiita.com/taka_yayoi/items/41be81691df3f7c8e6bf)
# MAGIC - [Databricksにおける画像の取り扱い \- Qiita](https://qiita.com/taka_yayoi/items/8d4b1b61699d68a34e58)
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2021/08/23</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>クラスター</td><td>8.4ML</td></tr>
# MAGIC </table>
# MAGIC <img style="margin-top:25px;" src="https://jixjiadatabricks.blob.core.windows.net/images/databricks-logo-small-new.png" width="140">

# COMMAND ----------

# MAGIC %md
# MAGIC ## PILを用いた画像データの読み込み
# MAGIC 
# MAGIC Databricksでは従来のPython環境同様に、PIL(Python Image Library)での画像操作を行うことができます。
# MAGIC 
# MAGIC #### 参考資料
# MAGIC - [初めてのPython画像処理 \- Qiita](https://qiita.com/uosansatox/items/4fa34e1d8d95d8783536)

# COMMAND ----------

from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np

# 画像の読み込み
img = Image.open('/dbfs/databricks-datasets/cctvVideos/train_images/label=0/LeftBagframe0004.jpg')
imshow(img)

# COMMAND ----------

# MAGIC %md
# MAGIC ぼかしを掛けてみます。

# COMMAND ----------

width, height = img.size
filter_size = 20
img2 = Image.new('RGB', (width - filter_size, height - filter_size))
img_pixels = np.array([[img.getpixel((x,y)) for x in range(width)] for y in range(height)])

for y in range(height - filter_size):
  for x in range(width - filter_size):
    # 位置(x,y)を起点に縦横フィルターサイズの小さい画像をオリジナル画像から切り取る            
    partial_img = img_pixels[y:y + filter_size, x:x + filter_size]
    # 小さい画像の各ピクセルの値を一列に並べる
    color_array = partial_img.reshape(filter_size ** 2, 3)
    # 各R,G,Bそれぞれの平均を求めて加工後画像の位置(x,y)のピクセルの値にセットする
    mean_r, mean_g, mean_b = color_array.mean(axis = 0)
    img2.putpixel((x,y), (int(mean_r), int(mean_g), int(mean_b)))

# COMMAND ----------

imshow(img2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 画像データソースによる画像データの読み込み
# MAGIC 
# MAGIC 画像データをSparkデータフレームに読み込むことで、分散処理が可能となります。[画像データソース](https://spark.apache.org/docs/latest/ml-datasource#image-data-source)は、詳細な画像の表現方式を抽象化し、画像データを読み込むための標準的なAPIを提供します。画像ファイルを読み込むには、データソースの`format`を`image`に指定します。
# MAGIC 
# MAGIC 画像データソースを用いることで、ネストされたディレクトリ構造(例えば、`/path/to/dir/**`のようなパス)をインポートすることができます。特定の画像に対しては、パーティションディレクトリ(`/path/to/dir/date=2018-01-02/category=automobile`のようなパス)のパスを指定することで、パーティションディスカバリーを使用することもできます。
# MAGIC 
# MAGIC **重要!**
# MAGIC > 画像データを生のバイト列としてSparkデータフレームに読む込む際には、[バイナリーファイルデータソース](https://docs.databricks.com/data/data-sources/binary-file.html)を使用することをお勧めします。画像データの取り扱いにおいてお勧めするワークフローに関しては、[画像アプリケーションに対するDatabricksリファレンスソリューション](https://qiita.com/taka_yayoi/items/41be81691df3f7c8e6bf)を参照ください。

# COMMAND ----------

# MAGIC %md ### セットアップ

# COMMAND ----------

# 画像ファイルパスの設定
sample_img_dir = "/databricks-datasets/cctvVideos/train_images/"

# COMMAND ----------

# MAGIC %md ### 画像データフレームの作成
# MAGIC 
# MAGIC Apache Sparkで提供される画像データソースを用いてデータフレームを作成します。画像データソースは、Hiveスタイルのパーティショニングをサポートしているので、以下のような構造で画像をアップロードした場合には、
# MAGIC 
# MAGIC * root_dir
# MAGIC   * label=0
# MAGIC     * image_001.jpg
# MAGIC     * image_002.jpg
# MAGIC     * ...
# MAGIC   * label=1
# MAGIC     * image_101.jpg
# MAGIC     * image_102.jpg
# MAGIC     * ...
# MAGIC 
# MAGIC 生成されるスキーマは以下のようになります(`image_df.printSchema()`コマンドを使用して表示します)。
# MAGIC 
# MAGIC ```
# MAGIC root
# MAGIC  |-- image: struct (nullable = true)
# MAGIC  |    |-- origin: string (nullable = true)
# MAGIC  |    |-- height: integer (nullable = true)
# MAGIC  |    |-- width: integer (nullable = true)
# MAGIC  |    |-- nChannels: integer (nullable = true)
# MAGIC  |    |-- mode: integer (nullable = true)
# MAGIC  |    |-- data: binary (nullable = true)
# MAGIC  |-- label: integer (nullable = true)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC 画像は"image"という列を持つデータフレームに読み込まれます。"image"は以下のフィールドを持つstruct型のカラムとなります。
# MAGIC 
# MAGIC ```
# MAGIC image: 画像データ全てを格納する構造体
# MAGIC   |-- origin: ソースURIを表現する文字列
# MAGIC   |-- height: 画素数による画像の高さ、整数値
# MAGIC   |-- width: 画素数による画像の幅、整数値
# MAGIC   |-- nChannels
# MAGIC   |-- mode
# MAGIC   |-- data
# MAGIC ```
# MAGIC 
# MAGIC **nChannels:** カラーチャンネルの数。典型的な値はグレースケールの画像場合は1、RGBのようなカラーイメージは3、アルファチャネルを持つカラーイメージは4となります。
# MAGIC 
# MAGIC **mode:** データフィールドを解釈するための整数値の情報を提供します。格納されているデータのデータ型とチャンネルオーダーを保持しています。フィールドの値は、以下のOpenCVタイプにマッピングされることを期待されます(強制はされません)。[OpenCV](https://opencv.org/)タイプは1、2、3、4のチャネル、画素値に対するいくつかのデータタイプを定義します。
# MAGIC 
# MAGIC OpenCVの値(データタイプ x チャンネルの数)に対するタイプのマッピングは以下の通りとなります。
# MAGIC 
# MAGIC ![](https://qiita-user-contents.imgix.net/https%3A%2F%2Fdatabricks.com%2Fwp-content%2Fuploads%2F2018%2F12%2FScreen-Shot-2018-12-10-at-9.19.49-AM.png?ixlib=rb-4.0.0&auto=format&gif-q=60&q=75&w=1400&fit=max&s=ee31af803213e15b55c7184e45a0b772)
# MAGIC 
# MAGIC **data:** バイナリー形式で格納された画像データです。画像データは、次元の形状(height, width, nChannels)の3次元の配列、modeで指定されるt型の配列値となります。配列は行優先(row-major)で格納されます。
# MAGIC 
# MAGIC Databricksの`display`関数は画像データの表示をサポートしています。詳細は[こちら](https://qiita.com/taka_yayoi/items/36a307e79e9433121c38#%E7%94%BB%E5%83%8F)を参照ください。

# COMMAND ----------

# 画像データソースを用いて画像データソースを作成します
image_df = spark.read.format("image").load(sample_img_dir)

# データフレームを表示します
display(image_df) 

# COMMAND ----------

# image_df.image.originを確認します
display(image_df.select("image.origin"))

# COMMAND ----------

# image_dfのスキーマを表示します
# ファイル構造の label=[0,1] に基づいてlabelカラムが生成されていることに注意してください
image_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC 画像データソースはSparkデータフレームを作成する過程で画像ファイルをデコードするのでデータサイズが増加し、以下のシナリオにおいては制限が生じます。
# MAGIC 
# MAGIC 1. データフレームの永続化: アクセスを容易にするためにDeltaテーブルでデータフレームを永続したい場合には、ディスク容量を節約するために生のバイト列を永続化すべきです。
# MAGIC 1. パーティションのシャッフル: デコードされた画像をシャッフルする際には、多くのディスク容量とネットワーク帯域が必要となるためシャッフルが遅くなります。画像のデコードは可能な限り後回しにすべきです。
# MAGIC 1. 他のデコード方法の利用: 画像データソースは、画像をデコードするためにjavaxの画像IOライブラリを使用します。このため、カスタムのデコードロジックや性能改善のために他の画像でコードライブラリを使用できません。
# MAGIC 
# MAGIC これらの制限は、画像データをロードする際に[バイナリーファイルデータソース](https://docs.databricks.com/data/data-sources/binary-file.html)を使用し、必要な時のみデコードすることで回避できます。

# COMMAND ----------

# MAGIC %md
# MAGIC ## バイナリーファイルデータソースによる画像の読み込みおよび保存
# MAGIC 
# MAGIC * `dbfs:/databricks-datasets`配下に格納されている [flowers dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers) を使用します。
# MAGIC * Sparkのテーブルに画像を格納するためにApache Sparkのバイナリーファイルデータソースを使います。
# MAGIC * 画像データとともに抽出したメタデータを格納します。
# MAGIC * データ管理をシンプルにするためにDelta Lakeを使用します。

# COMMAND ----------

import io
import numpy as np
import pandas as pd
import uuid
from pyspark.sql.functions import col, pandas_udf, regexp_extract
from PIL import Image

# COMMAND ----------

# MAGIC %md ### 花の画像データセット
# MAGIC 
# MAGIC TensorFlowで提供されているサンプルデータセットである [flowers dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers) を使用します。クラスごとに作成された5つのサブフォルダに花の写真が格納されています。このデータセットはDatabricksのサンプルデータセットとして提供されています。

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/flower_photos

# COMMAND ----------

# MAGIC %md ### バイナリーファイルデータソースを用いて画像をデータフレームにロード
# MAGIC 
# MAGIC Databricksランタイムはバイナリーファイルデータソースをサポートしています。これにより、1つの画像ファイルをデータフレームの1レコードに変換し、画像データとメタデータを一緒に保存することができます。
# MAGIC 
# MAGIC ```
# MAGIC images = spark.read.format("binaryFile")\ # バイナリーファイルデータソースの指定 
# MAGIC   .option("recursiveFileLookup", "true")\ # 再帰的にファイルを検索
# MAGIC   .option("pathGlobFilter", "*.jpg")\     # jpgのみを読み込み 
# MAGIC   .load("/databricks-datasets/flower_photos") \
# MAGIC   .repartition(4) # パーティション数を4に変更
# MAGIC ```

# COMMAND ----------

images = spark.read.format("binaryFile")\
  .option("recursiveFileLookup", "true")\
  .option("pathGlobFilter", "*.jpg")\
  .load("/databricks-datasets/flower_photos") \
  .repartition(4)

# COMMAND ----------

# MAGIC %md ### メタデータをデータフレームのカラムに変換します
# MAGIC 
# MAGIC 頻繁に使用するメタデータを `images` データフレームから抽出します:
# MAGIC * ファイルパスからラベルを抽出します
# MAGIC * ラベルのインデックスを追加します
# MAGIC * 画像サイズを抽出します
# MAGIC 
# MAGIC 上記処理を行うためのPandas UDF(ユーザー定義関数)を定義します。

# COMMAND ----------

def extract_label(path_col):
  """ビルトインのSQL関数を用いてファイルパスからラベルを抽出します"""
  return regexp_extract(path_col, "flower_photos/([^/]+)", 1)

# COMMAND ----------

def extract_size(content):
  """画像データから画像サイズを抽出します"""
  image = Image.open(io.BytesIO(content))
  return image.size

@pandas_udf("width: int, height: int")
def extract_size_udf(content_series):
  sizes = content_series.apply(extract_size)
  return pd.DataFrame(list(sizes))

# COMMAND ----------

# データフレームにUDFを適用します
images_with_label = images.select(
  col("path"),
  extract_label(col("path")).alias("label"),
  extract_size_udf(col("content")).alias("size"),
  col("content"))

# COMMAND ----------

# ラベルに対するインデックスを抽出します
labels = images_with_label.select(col("label")).distinct().collect()
label_to_idx = {label: index for index, (label, ) in enumerate(sorted(labels))}
num_classes = len(label_to_idx)

@pandas_udf("long")
def get_label_idx(labels):
  return labels.map(lambda label: label_to_idx[label])

# COMMAND ----------

df = images_with_label.withColumn("label_index", get_label_idx(col("label"))) \
  .select(col("path"), col("size"), col("label"), col("label_index"), col("content"))

# COMMAND ----------

# MAGIC %md ### データフレームをDeltaフォーマットで保存

# COMMAND ----------

# 画像データは圧縮済みなのでParquetの圧縮をオフにします
spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")
path = "/tmp/flowers/" + str(uuid.uuid4()) # ユニークなIDで保存場所のパスを作成します
df.write.format("delta").mode("overwrite").save(path)

# COMMAND ----------

# データフレームの中身を確認します
display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### クリーンアップ

# COMMAND ----------

dbutils.fs.rm(path, recurse=True)

# COMMAND ----------

# MAGIC %md　## Delta Lakeを用いた分散モデル推論
# MAGIC 
# MAGIC - Deltaテーブル`dbfs:/databricks-datasets/flowers/`からスタートします。これは画像を格納するDeltaテーブルを作成するノートブックから得られるアウトプットのコピーです。
# MAGIC - バッチによる予測を実行するためにスカラーイテレーターのPandasのUDF(ユーザー定義関数)を使用します。

# COMMAND ----------

import io

from tensorflow.keras.applications.imagenet_utils import decode_predictions
import pandas as pd
from pyspark.sql.functions import col, pandas_udf, PandasUDFType
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# COMMAND ----------

# MAGIC %md ### 入力を処理するデータセットを定義

# COMMAND ----------

class ImageNetDataset(Dataset):
  """
  標準的なImageNet前処理を用いて画像コンテンツをPyTorchデータセットへ変換
  """
  def __init__(self, contents):
    self.contents = contents

  def __len__(self):
    return len(self.contents)

  def __getitem__(self, index):
    return self._preprocess(self.contents[index])

  def _preprocess(self, content):
    """
    標準的なImageNet正規化を用いて入力画像コンテンツを前処理
    
    詳細は https://pytorch.org/docs/stable/torchvision/models.html
    """
    image = Image.open(io.BytesIO(content))
    transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image)

# COMMAND ----------

# MAGIC %md ### 推論タスクのためのPandas UDFの定義
# MAGIC 
# MAGIC 1対1のマッピングセマンティックを提供するPySparkのUDFには3種類あります。
# MAGIC 
# MAGIC - PySpark UDF: レコード -> レコード、データのシリアライゼーションの性能に課題があるためお勧めしません。
# MAGIC - スカラーPandas UDF: pandasのシリーズ/データフレーム -> pandasのシリーズ/データフレーム, バッチ間で状態が共有されません。
# MAGIC - **スカラーイテレーターPandas UDF**: いくつかの状態を初期化した後、バッチに進みます。
# MAGIC 
# MAGIC モデル推論にはスカラーイテレーターPandas UDFを用いることをお勧めします。
# MAGIC 
# MAGIC #### 参考資料
# MAGIC 
# MAGIC - [MobileNet\(v1,v2,v3\)を簡単に解説してみた \- Qiita](https://qiita.com/omiita/items/77dadd5a7b16a104df83)
# MAGIC - [MobileNetV2\(Tensorflow\)を触ってみた。 \| AI・人工知能、IoT、CPS、Mobile：システム開発・一括請負を行っております。](https://apprhythm.biz/archives/2981)

# COMMAND ----------

def imagenet_model_udf(model_fn):
  """
  予測を行うImageNetモデルをPandasのUDFにラップします。
  
  ご自身のユースケースに応じて以下のようなカスタマイゼーションが必要になるかもしれません。
    - 性能を改善するためにDataLoaderのbatch_sizeとnum_workersをチューニング
    - 高速化のためにGPUを使用
    - 予測タイプを変更
  """
  def predict(content_series_iter):
    model = model_fn()
    model.eval()
    for content_series in content_series_iter:
      dataset = ImageNetDataset(list(content_series))
      loader = DataLoader(dataset, batch_size=64)
      with torch.no_grad():
        for image_batch in loader:
          predictions = model(image_batch).numpy()
          predicted_labels = [x[0] for x in decode_predictions(predictions, top=1)]
          yield pd.DataFrame(predicted_labels)
  return_type = "class: string, desc: string, score:float"
  return pandas_udf(return_type, PandasUDFType.SCALAR_ITER)(predict)

# COMMAND ----------

# MobileNetV2をPandas UDFとしてラップします
mobilenet_v2_udf = imagenet_model_udf(lambda: models.mobilenet_v2(pretrained=True))

# COMMAND ----------

# MAGIC %md ### データフレームを用いた分散推論
# MAGIC 
# MAGIC - 必要となるカラムと、それぞれのカラムをどのように計算するのかを指定します。
# MAGIC - 命令的なRDDコードを指定する代わりにSparkに実行を最適化させます。
# MAGIC - Deltaテーブルの新規データに対して自動で推論を実行します。Deltaテーブルをストリームソースとしてロードするために`spark.readStream`を用い、予測結果を別のDeltaテーブルに書き込みます。

# COMMAND ----------

images = spark.read.format("delta") \
  .load("/databricks-datasets/flowers/delta") \
  .limit(100)  # デモ目的のため実行時間を短縮するためにサブセットを使用します
predictions = images.withColumn("prediction", mobilenet_v2_udf(col("content")))

# COMMAND ----------

display(predictions.select(col("path"), col("prediction")).limit(5))

# COMMAND ----------

# 画像と予測結果のみに絞って表示します
display(predictions.select(col("content"), col("prediction.desc"), col("prediction.score")))

# COMMAND ----------

# MAGIC %md
# MAGIC # END
