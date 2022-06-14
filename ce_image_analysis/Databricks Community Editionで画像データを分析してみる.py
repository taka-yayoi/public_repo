# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks Community Editionで画像データを分析してみる

# COMMAND ----------

# MAGIC %md
# MAGIC ## サンプルデータセットの確認
# MAGIC 
# MAGIC Databricksファイルシステム(DBFS)の`/databricks-datasets`にはさまざまなサンプルデータが格納されています。
# MAGIC 
# MAGIC 以下のセルではマジックコマンド`%fs`を指定しており、`ls`や`head`などを用いてファイルシステムにアクセスすることができます。

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /databricks-datasets

# COMMAND ----------

# MAGIC %md
# MAGIC ## データフレームにDeltaファイルをロード
# MAGIC 
# MAGIC 上のサンプルデータの中から花の画像を保存しているDelta形式のファイルをロードします。このデータには以下のカラムが含まれています。読み込んだデータフレームはSparkデータフレームとなります。Pythonの場合、PySparkというAPIを用いてこのデータフレームを操作します。
# MAGIC 
# MAGIC |カラム|説明|
# MAGIC |:--|:--|
# MAGIC |path|画像ファイルのパス|
# MAGIC |size|画像の幅と高さを持つ配列|
# MAGIC |label|画像のラベル|
# MAGIC |content|画像のバイナリ|
# MAGIC 
# MAGIC また、Delta形式のファイルにはバイナリーをカラムとして含めることができるので、画像データを用いた分析を簡便に行うことができます。

# COMMAND ----------

df = spark.read.format("delta").load("dbfs:/databricks-datasets/flowers/delta/")
display(df)

# COMMAND ----------

# 読み込んだデータフレームのスキーマを表示
df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## PySparkによる簡単な集計
# MAGIC 
# MAGIC データフレームにさまざまなメソッドを用いることで、SQLライクな集計を行うことができます。

# COMMAND ----------

display(df.groupBy("label")
        .count()
        .select(["label", "count"]))

# COMMAND ----------

# MAGIC %md
# MAGIC なお、一時ビューあるいはHiveメタストアにデータフレームを登録すると、直接SQLを実行することができます。以下の例では、データフレーム一時ビューを登録しています。マジックコマンド`%sql`を用いることで、セルに直接SQLを記述することができます。

# COMMAND ----------

df.createOrReplaceTempView("flower_photo")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT label, count(*) as count FROM flower_photo GROUP BY label

# COMMAND ----------

# MAGIC %md　## Delta Lakeを用いた分散モデル推論
# MAGIC 
# MAGIC 大量データに対する推論を行う際に、Sparkの並列分散処理を活用することで処理に要する時間を短縮することができます。このために、ScalarイテレーターのPandasのUDF(ユーザー定義関数)を使用します。
# MAGIC 
# MAGIC 必要なライブラリをimportします。Databricks機械学習ランタイムを用いると、tensorflow、PyTorchなど著名なライブラリが事前にインストールされています。

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
# MAGIC - Scalar Pandas UDF: pandasのシリーズ/データフレーム -> pandasのシリーズ/データフレーム, バッチ間で状態が共有されません。
# MAGIC - **ScalarイテレーターPandas UDF**: いくつかの状態を初期化した後、バッチに進みます。
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
# MAGIC - 上で定義したUDFを用いて、画像に対するラベルを推定します。

# COMMAND ----------

images = spark.read.format("delta") \
  .load("/databricks-datasets/flowers/delta") \
  .limit(100)  # デモ目的のため実行時間を短縮するためにサブセットを使用します

# 上のUDFを呼び出して予測を行います
predictions = images.withColumn("prediction", mobilenet_v2_udf(col("content")))

# 結果を表示します
display(predictions.select(col("path"), col("prediction")).limit(5))

# COMMAND ----------

# 画像と予測結果のみに絞って表示します
display(predictions.select(col("content"), col("prediction.desc"), col("prediction.score")))

# COMMAND ----------

# MAGIC %md
# MAGIC 精度に関しては改善の余地がありますが、事前学習済みモデルを用いることで簡単に画像に対するラベルを推定することができます。以下のようなメリットを感じていただけたら幸いです。
# MAGIC 
# MAGIC - Delta Lakeによる画像管理の簡素化
# MAGIC - Databricksノートブックで柔軟なプログラミング言語の使い分け
# MAGIC - Databricks機械学習ランタイムを用いてライブラリ管理をシンプルに
# MAGIC - Sparkによる推論の並列実行

# COMMAND ----------

# MAGIC %md
# MAGIC # END
