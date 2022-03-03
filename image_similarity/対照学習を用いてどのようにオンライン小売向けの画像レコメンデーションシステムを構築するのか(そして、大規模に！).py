# Databricks notebook source
# MAGIC %md
# MAGIC # 対照学習を用いてどのようにオンライン小売向けの画像レコメンデーションシステムを構築するのか(そして、大規模に！)
# MAGIC 
# MAGIC [eコマース向け類似画像レコメンデーションシステムの構築 \- Qiita](https://qiita.com/taka_yayoi/items/173d5228c1d08c5130ef)

# COMMAND ----------

# MAGIC %md
# MAGIC 本書では、類似するアイテムを検索するのにより適した革新的な機械学習アプローチである、類似学習を用いてトレーニングされたモデルを用いたレコメンデーションエンジンを構築するエンドツーエンドのプロセスを学びます。モデルをトレーニングするためにTensorflow_similarityライブラリを使用し、GPUクラスターでモデルのトレーニングをスケールさせるために、Spark、Horovod、Hypeoptを活用します。プロセスの全ての側面を記録、追跡するためにMLflowを用い、データのリネージュ、再現性を確保するためにDeltaを使用します。
# MAGIC 
# MAGIC ハイレベルでは、類似性モデルは対照学習を用いてトレーニングされれます。対照学習では、類似するアイテム間の距離が最小となり、似ていないアイテム間の距離が最大となるような埋め込み空間を機械学習モデル(適合アルゴリズム)が学習することがゴールとなります。このクイックスタートでは、さまざまな衣料品の約70,000点の画像から構成されるファッションMNISTデータセットを使用します。上の説明に基づいて、このラベル付データセットでトレーニングされる類似性モデルは、類似する商品(例えば、ブーツ同士)は集まり、異なる商品(例、ブーツとバンダナ)とは互いに離れ合う埋め込み空間を学習します。
# MAGIC 
# MAGIC これは以下のように図示することができます。

# COMMAND ----------

displayHTML("<img src='https://github.com/avisoori-databricks/Databricks_image_recommender/blob/main/images/simrec_embed.png?raw=true'")

# COMMAND ----------

# MAGIC %md 
# MAGIC 類似学習では、類似するアイテムが近くとなり、似ていないアイテムは遠くなるように機械学習モデルを学習させることがゴールとなります。教示あり類似学習では、アルゴリズムは学習に用いる画像ラベルのようなメタデータと、生の画像データにアクセスすることができます。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 目次
# MAGIC 
# MAGIC - **1 セットアップ**: クラスターの作成、必要となるライブラリのインストール、必要なモジュールのインポート、以降のタスクで必要となるデータをDeltaテーブルに取り込みます。
# MAGIC - **2 モデルトレーニング**: 類似性モデルのトレーニングを行います。オフィシャルのTensorflow Similarityリポジトリからサンプルを取得します。
# MAGIC - **3 HyperoptとSparkによるハイパーパラメーター探索のスケーリング**: スケーリングの方法の一つに、最適なモデルパフォーマンスにつながるベストなハイパーパラメーターの探索を分散させるというものがあります。
# MAGIC - **4 HorovodとSparkによるモデルトレーニングのスケーリング**: データセットが膨大な場合、単一モデルのトレーニングをクラスターにスケールさせることができます。
# MAGIC - **5 モデルのデプロイメント**: MLflowのモデルサービングを用いてRESTエンドポイントにモデルとアプリケーション文脈における後処理ロジックをデプロイします。

# COMMAND ----------

# MAGIC %md
# MAGIC ## セットアップ
# MAGIC 
# MAGIC クラスター設定の詳細(分散処理を行うために2台以上のワーカーノードを持つGPUクラスター)を指定します。T4 GPU(AWSの場合g4インスタンス)がこのタスクに適しています。

# COMMAND ----------

displayHTML("<img src='https://github.com/avisoori-databricks/Databricks_image_recommender/blob/main/images/simrec_gpu.png?raw=true'>")

# COMMAND ----------

# MAGIC %md
# MAGIC Tensorflow Similarityライブラリをインストールします:

# COMMAND ----------

# MAGIC %pip install tensorflow_similarity

# COMMAND ----------

# MAGIC %md
# MAGIC 必要なモジュールをインポートします。

# COMMAND ----------

import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt
import mlflow
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from pyspark.ml.feature import OneHotEncoder

import tensorflow_similarity as tfsim
from tensorflow_similarity.utils import tf_cap_memory
from tensorflow_similarity.layers import MetricEmbedding
from tensorflow_similarity.losses import MultiSimilarityLoss  
from tensorflow_similarity.models import SimilarityModel 
from tensorflow_similarity.samplers import MultiShotMemorySampler 
from tensorflow_similarity.samplers import select_examples 
from tensorflow_similarity.visualization import viz_neigbors_imgs 
from tensorflow_similarity.visualization import confusion_matrix 

# COMMAND ----------

# MAGIC %md
# MAGIC インストールされたTensorflowとTensorflow similarityライブラリのバージョンに注意します。

# COMMAND ----------

print('TensorFlow:', tf.__version__)
print('TensorFlow Similarity', tfsim.__version__)

# COMMAND ----------

# MAGIC %md 
# MAGIC Zalandoによるオフィシャルのfashion MNISTリポジトリ(ブログで言及されています)からデータを取得し、Deltaテーブルを作成するために以下の3つのセルを実行します。

# COMMAND ----------

# MAGIC %sh 
# MAGIC 
# MAGIC wget -O  test_labels.gz https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz
# MAGIC wget -O  test_images.gz https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz
# MAGIC wget -O  train_images.gz https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz
# MAGIC wget -O  train_labels.gz https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz
# MAGIC 
# MAGIC 
# MAGIC gunzip -dk *.gz
# MAGIC 
# MAGIC rm -r train_labels.gz test_labels.gz train_images.gz test_images.gz
# MAGIC 
# MAGIC ls

# COMMAND ----------

# MAGIC %md
# MAGIC 以下の関数は、上でダウンロードしたデータセットをトレーニング用画像、テスト用画像に対応するDeltaテーブルに変換するために https://pjreddie.com/projects/mnist-in-csv/ にあるものを変更したものとなります。
# MAGIC 
# MAGIC **注意**
# MAGIC - 以下のセルの1行目の`user`はご自身のユーザー名に変更してください。

# COMMAND ----------

datasets = [['test_images', 'test_labels','/FileStore/tables/takaaki.yayoi@databricks.com/delta/fmnist_test_data', 10000],  ['train_images', 'train_labels', '/FileStore/tables/takaaki.yayoi@databricks.com/delta/fmnist_train_data', 60000]]


def convert(imgf, labelf, outf, n):
  """この関数は画像のファイル名、ラベルのファイル名、出力パス、レコード数を受け取ります。画像データを読み込み、指定されたパスのDeltaテーブルに変換します。"""
  f = open(imgf, "rb")
  l = open(labelf, "rb")

  f.read(16)
  l.read(8)
  images = []

  for i in range(n):
    image = [ord(l.read(1))]
    for j in range(28*28):
      image.append(ord(f.read(1)))
    images.append(image)
        
  f.close()
  l.close()
  df = pd.DataFrame(images)
  sparkdf = spark.createDataFrame(df)
  sparkdf.write.format('delta').mode('overwrite').save(outf)

# COMMAND ----------

for dataset in datasets:
  convert(dataset[0], dataset[1], dataset[2], dataset[3])

# COMMAND ----------

# MAGIC %md
# MAGIC ファッションMNISTデータセットのクラスは以下の通りとなります。
# MAGIC 
# MAGIC - ラベル	説明
# MAGIC - 0	 T-shirt/top
# MAGIC - 1	 Trouser
# MAGIC - 2	 Pullover
# MAGIC - 3	 Dress
# MAGIC - 4	 Coat
# MAGIC - 5	 Sandal
# MAGIC - 6	 Shirt
# MAGIC - 7	 Sneaker
# MAGIC - 8	 Bag
# MAGIC - 9	 Ankle boot

# COMMAND ----------

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# COMMAND ----------

# MAGIC %md トレーニングデータセットとテストデータセットをDeltaテーブルから読み取ります。
# MAGIC 
# MAGIC **注意**
# MAGIC - 以下のセルの1行目の`user`はご自身のユーザー名に変更してください。

# COMMAND ----------

train = spark.read.format("delta").load("/FileStore/tables/takaaki.yayoi@databricks.com/delta/fmnist_train_data").toPandas().values
test = spark.read.format("delta").load("/FileStore/tables/takaaki.yayoi@databricks.com/delta/fmnist_test_data").toPandas().values

# COMMAND ----------

# MAGIC %md 
# MAGIC モデルのトレーニングプロセスに適した形式に変換するための関数を定義します。この関数は、お使いの画像データの形状、使用されるモデルアーキテクチャに依存します。

# COMMAND ----------

def get_dataset(train, test, rank=0, size=1):
  from tensorflow import keras
  import numpy as np
  
  np.random.shuffle(train)
  np.random.shuffle(test)

  x_train = train[:, 1:].reshape(-1, 28, 28)
  y_train = train[:, 0].astype(np.int32)
  x_test = test[:, 1:].reshape(-1, 28, 28)
  y_test = test[:, 0].astype(np.int32)

  x_train = x_train[rank::size]
  y_train = y_train[rank::size]
  x_test = x_test[rank::size]
  y_test = y_test[rank::size]

  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255.0
  x_test /= 255.0
  return (x_train, y_train), (x_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC モデルアーキテクチャを定義します。比較的シンプルなconvolutional neural networkアーキテクチャで、ロバストな性能を発揮できるところに、類似学習の美しさがあります。

# COMMAND ----------

def get_model():
    from tensorflow_similarity.layers import MetricEmbedding
    from tensorflow.keras import layers
    from tensorflow_similarity.models import SimilarityModel
    
    inputs = layers.Input(shape=(28, 28, 1))
    x = layers.experimental.preprocessing.Rescaling(1/255)(inputs)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPool2D(2, 2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPool2D(2, 2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(256, 3, activation='relu')(x)
    x = layers.MaxPool2D(2, 2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    # 検索時間をチューニングするためにエンべディングのサイズをチューニングします。小さいエンべディングは検索が高速になりますが、低い精度の結果となります。大きなエンべディングでは逆が成り立ちます。
    outputs = MetricEmbedding(256)(x)
    return SimilarityModel(inputs, outputs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## シングルGPUで機械学習モデルをトレーニングし、モデルのパラメーターとメトリクスをトラッキングします。

# COMMAND ----------

# データセットのクラスの総数は10となります
num_classes = 10

# COMMAND ----------

# MAGIC %md 
# MAGIC 上のアーキテクチャと、上で準備したデータセットを用いてモデルのトレーニングのための関数を定義します。

# COMMAND ----------

def train_model(train, test, learning_rate=0.001):
  """この関数は、Tensorflow Similarityを用いた類似性モデルに対する一般的なトレーニングロジックをカプセル化します"""
  from tensorflow import keras
  from tensorflow_similarity.losses import MultiSimilarityLoss   
  from tensorflow_similarity.samplers import MultiShotMemorySampler
  from tensorflow.keras.optimizers import Adam
  import mlflow
  mlflow.tensorflow.autolog()
  # Fashion MNISTデータセットのクラスの数は10です
  (x_train, y_train), (x_test, y_test) = get_dataset(train, test)
  classes = [2, 3, 1, 7, 9, 6, 8, 5, 0, 4]
  # モデルのトレーニングに使用するクラスの数。これは、類似性モデルは未知のクラスに対してもうまく汎化できるためです。 
  # このため、Fashion MNISTデータセットの10クラスのうち6クラスをモデルのトレーニングに使用します。
  num_classes_ = 6   
  class_per_batch = num_classes_
  example_per_class = 6  
  epochs = 10
  steps_per_epoch = 1000  

  sampler = MultiShotMemorySampler(x_train, y_train, 
                                   classes_per_batch=class_per_batch, 
                                   examples_per_class_per_batch=example_per_class,
                                   class_list=classes[:num_classes_],  
                                   steps_per_epoch=steps_per_epoch)
  model = get_model()
  distance = 'cosine' 
  loss = MultiSimilarityLoss(distance=distance)
  model.compile(optimizer=Adam(learning_rate), loss=loss)
  model.fit(sampler, epochs=epochs, validation_data=(x_test, y_test))
  return model 

# COMMAND ----------

model = train_model(train, test, learning_rate=0.001)

# COMMAND ----------

# MAGIC %md
# MAGIC ## HorovodとSparkを用いたGPUクラスターにまたがるトレーニング
# MAGIC 
# MAGIC 以下の`checkpoint_root`は適宜変更してください。

# COMMAND ----------

import os
import time

checkpoint_root = "/tmp/takaaki.yayoi@databricks.com/MNISTDemo/train"

# 既存のチェックポイントファイルを削除
dbutils.fs.rm((checkpoint_root), recurse=True)

# ディレクトリの作成
checkpoint_dir = '/dbfs{}/{}/'.format(checkpoint_root, time.time())
os.makedirs(checkpoint_dir)
print(checkpoint_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC Horovodを用いて単一のモデルを分散処理でトレーニングする関数を定義します。

# COMMAND ----------

def train_hvd(train, test, checkpoint_path, learning_rate=0.001):
  """この関数では、HorovodとSparkを用いて、モデルのトレーニングの分散に必要なロジックをカプセル化します。詳細は https://databricks.github.io/spark-deep-learning/index.html で確認することができます"""
  # この関数にモデルのトレーニングに必要となるロジックとインポート処理をカプセル化します
  import tensorflow as tf
  from tensorflow.keras import layers
  from tensorflow.keras.models import load_model
  from tensorflow.keras.optimizers import Adam
  # Horovodのフレーバー
  from tensorflow import keras
  import horovod.tensorflow.keras as hvd
  import mlflow
  
  from tensorflow_similarity.utils import tf_cap_memory
  from tensorflow_similarity.layers import MetricEmbedding  
  from tensorflow_similarity.losses import MultiSimilarityLoss   
  from tensorflow_similarity.models import SimilarityModel  
  from tensorflow_similarity.samplers import MultiShotMemorySampler  
  from tensorflow_similarity.samplers import select_examples  
  from tensorflow_similarity.visualization import viz_neigbors_imgs   
  from tensorflow_similarity.visualization import confusion_matrix  
  
  # Horovodの初期化
  hvd.init()
  
  batch_size = 128

  # ローカルのランクの処理に使用するためにGPUをピン留めします (プロセスあたり1GPU)
  # これらのステップはCPUクラスターではスキップされます
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    
  (x_train, y_train), (x_test, y_test) = get_dataset(train, test, hvd.rank(), hvd.size())
  classes = [2, 3, 1, 7, 9, 6, 8, 5, 0, 4]
  num_classes_ = 6  
  class_per_batch = num_classes_
  example_per_class = 6 
  epochs = 30
  steps_per_epoch = 1000 
  
  model = get_model()
  
  # GPU数に基づいて学習を調整
  optimizer = keras.optimizers.Adadelta(lr=learning_rate * hvd.size())

  # Horovod Distributed Optimizerを使用
  optimizer = hvd.DistributedOptimizer(optimizer)
  
  # ランク0から全ての他のプロセスに初期変数状態をブロードキャストするためのコールバックを作成
  callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0),]
  # これは、ランダムの重みでトレーニングをスタートした際、あるいは、チェックポイントから復帰した際に、全てのワーカーの初期化の一貫性を担保するために必要となります
 
  sampler = MultiShotMemorySampler(x_train, y_train, 
                                   classes_per_batch=class_per_batch, 
                                   examples_per_class_per_batch=example_per_class,
                                   class_list=classes[:num_classes_], 
                                   steps_per_epoch=steps_per_epoch)
  
  

  distance = 'cosine' 
  loss = MultiSimilarityLoss(distance=distance)
  
  model.compile(optimizer=Adam(learning_rate), loss=loss)
  
  # ワーカー間での競合を回避するためにワーカー0でのみチェックポイントを保存します
  if hvd.rank() == 0:
      mlflow.keras.autolog()

      callbacks.append(keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only = True))
      
  model.fit(sampler, callbacks = callbacks ,epochs=epochs, validation_data=(x_test, y_test))

# COMMAND ----------

# トレーニングプロセスを開始し、MLflowでメトリクス、パラメーターを記録します
with mlflow.start_run() as run:  
  from sparkdl import HorovodRunner
  from tensorflow_similarity.losses import MultiSimilarityLoss  # similarity lossに特化
  import mlflow

  checkpoint_path = checkpoint_dir + '/checkpoint-{epoch}.ckpt'
  learning_rate = 0.001
   
  
  # HorovodRunnerの実行
  hr = HorovodRunner(np=2, driver_log_verbosity='all')

  hr.run(train_hvd,  train  = train, test = test, checkpoint_path=checkpoint_path, learning_rate=learning_rate)
  
  distance = 'cosine' 

  loss = MultiSimilarityLoss(distance=distance)

  hvd_model = get_model()

  optimizer=Adam(learning_rate)

  hvd_model.compile(optimizer=Adam(learning_rate), loss=loss)

  hvd_model.load_weights(tf.train.latest_checkpoint(os.path.dirname(checkpoint_path)))

  (x_train, y_train), (x_test, y_test) = get_dataset(train, test)

  score = hvd_model.evaluate(x_test, y_test, verbose=0)

  mlflow.log_metric("loss", score)
  mlflow.log_param("lr", learning_rate)

  
mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC ## HyperoptとSparkを用いた分散ハイパーパラメーターの最適化

# COMMAND ----------

# MAGIC %md 
# MAGIC Hyperoptを使用するトレーニング関数を定義します。これは、これまでに定義したトレーニング関数と非常に似たものです。ここでは、クラスターで複数のモデルがトレーニングされ、あらゆる時点でそれぞれのエグゼキューターは固有のハイパーパラメーターの組み合わせを用いて単一のモデルをトレーニングします。

# COMMAND ----------

def train_hyperopt(space):
  """この関数はベイジアンサーチを行いたいハイパーパラメーター空間を表現するディクショナリーを受け付けます。Hyperoptに関する詳細は http://hyperopt.github.io/hyperopt/scaleout/spark/ で確認できます。"""
  from tensorflow import keras
  from tensorflow_similarity.losses import MultiSimilarityLoss  # similarity lossに特化
  from tensorflow_similarity.samplers import MultiShotMemorySampler
  from tensorflow.keras.optimizers import Adam
  import mlflow
  
  mlflow.tensorflow.autolog()

  (x_train, y_train), (x_test, y_test) = get_dataset(train, test )
  classes = [2, 3, 1, 7, 9, 6, 8, 5, 0, 4]
  num_classes_ = 7
  classes_per_batch = num_classes_
  examples_per_class = space['examples_per_class'] 
  epochs = 10
  steps_per_epoch = space['steps_per_epoch'] 

  sampler = MultiShotMemorySampler(x_train, y_train, 
                                   classes_per_batch=classes_per_batch, 
                                   examples_per_class_per_batch=examples_per_class,
                                   class_list=classes[:num_classes_], 
                                   steps_per_epoch=steps_per_epoch)
  model = get_model()
  distance = 'cosine' 
  loss = MultiSimilarityLoss(distance=distance)
  model.compile(optimizer=Adam(space["learning_rate"]), loss=loss)
  model.fit(sampler, epochs=epochs, validation_data=(x_test, y_test))
  return model.evaluate(x_test, y_test)

# COMMAND ----------

# MAGIC %md 
# MAGIC 必要なパッケージをインポートし、ディクショナリーとしてハイパーパラメーター検索空間を定義します。

# COMMAND ----------

import numpy as np
from hyperopt import fmin, hp, tpe, STATUS_OK, SparkTrials

space = {
  'steps_per_epoch': hp.choice('steps_per_epoch', np.arange(100, 2000, 250, dtype=int)),
  'examples_per_class' : hp.choice('examples_per_class',np.arange(5, 10, 1, dtype=int)),
  'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-1))
}

# COMMAND ----------

# MAGIC %md 
# MAGIC このクラスターでは2つのワーカーしかないので、並列度2を指定してSparktrialsのインスタンスを作成します。お使いのクラスターのワーカーの数に応じてこの値を変更します。

# COMMAND ----------

import mlflow
trials = SparkTrials(2)

# COMMAND ----------

# MAGIC %md 
# MAGIC ハイパーパラメーター検索アルゴリズムを定義し、探索をスタートします。上で定義したようにSparkTrialsを使用しているので、お使いのSparkクラスターに分散した処理が行われます。

# COMMAND ----------

algo=tpe.suggest
 
with mlflow.start_run():
  best_params = fmin(
    fn=train_hyperopt,
    space=space,
    algo=algo,
    max_evals=32,
    trials = trials,
  )

# COMMAND ----------

# MAGIC %md
# MAGIC 上のプロセスによってベストなパラメータが明らかになります。`steps_per_epoch == n`は、引き渡されたnumpyレンジのn番目のインデックスであることを意味し、Cmd41の探索空間に基づいて値を計算します。
# MAGIC 
# MAGIC |n|steps_per_epoch|
# MAGIC |:--|:--|
# MAGIC |0|100|
# MAGIC |1|350|
# MAGIC |2|600|
# MAGIC |3|850|
# MAGIC |4|1100|
# MAGIC |4|1350|

# COMMAND ----------

print(best_params)

# COMMAND ----------

# MAGIC %md 
# MAGIC クエリーに使用するインデックスを構築するためのモデルをトレーニングするために、これらのパラメーターを使用します。

# COMMAND ----------

# 最終モデルのトレーニング 
(x_train, y_train), (x_test, y_test) = get_dataset(train, test)
classes = [2, 3, 1, 7, 9, 6, 8, 5, 0, 4]
num_classes = 8  
classes_per_batch = num_classes
example_per_class = 4
epochs = 20
steps_per_epoch = 1350
learning_rate = 0.002186992406810626

sampler = MultiShotMemorySampler(x_train, y_train, 
                                 classes_per_batch=classes_per_batch, 
                                 examples_per_class_per_batch=example_per_class,
                                 class_list=classes[:num_classes],
                                 steps_per_epoch=steps_per_epoch)
tfsim_model = get_model()
distance = 'cosine' 
loss = MultiSimilarityLoss(distance=distance)
tfsim_model.compile(optimizer=Adam(learning_rate), loss=loss)
tfsim_model.fit(sampler, epochs=epochs, validation_data=(x_test, y_test))

# COMMAND ----------

# MAGIC %md 
# MAGIC モデルのアーキテクチャを確認します。

# COMMAND ----------

tfsim_model.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ## インデックスの構築

# COMMAND ----------

x_index, y_index = select_examples(x_train, y_train, classes, 20)
tfsim_model.reset_index()
tfsim_model.index(x_index, y_index, data=x_index)

# COMMAND ----------

# MAGIC %md 1つの画像を見てみましょう。

# COMMAND ----------

from matplotlib import pyplot as plt

sample_image = x_index[0]
sample_image = sample_image.reshape(1, sample_image.shape[0], sample_image.shape[1]) 
plt.imshow(sample_image[0], interpolation='nearest')
plt.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC クイックにサニティチェックを行いましょう。この画像に対応するラベルを確認します。

# COMMAND ----------

label  = y_index[0]
label
# 1 はTrouser(ズボン)となります: https://github.com/zalandoresearch/fashion-mnist

# COMMAND ----------

# MAGIC %md 
# MAGIC 指定された画像に対するレコメンデーションがどのようなものになるかをテストします。モデルオブジェクトは、このケースではインデックスに基づき、n個のapproximately nearest neighborを返却します。

# COMMAND ----------

# 最初に変数x_displayとy_displayに格納される返却オブジェクトの型を確認します
x_display, y_display = select_examples(x_test, y_test, classes, 1)

type(x_display), type(y_display)

# COMMAND ----------

# 画像の選択
x_display, y_display = select_examples(x_test, y_test, classes, 1)

# インデックスに基づき近隣のオブジェクトを検索します
nns = np.array(tfsim_model.lookup(x_display, k=5))

# 表示
for idx in np.argsort(y_display):
    viz_neigbors_imgs(x_display[idx], y_display[idx], nns[idx], 
                      fig_size=(16, 2), cmap='Greys')

# COMMAND ----------

# MAGIC %md
# MAGIC ## デプロイメント、クエリーのためのMLflow Pyfuncラッパークラスの作成

# COMMAND ----------

# MAGIC %md 
# MAGIC モデルを指定したディレクトリに保存します。これによって、モデル自身とクエリーに使用するインデックスを保存します。

# COMMAND ----------

tfsim_path = "models/tfsim.pth"

tfsim_model.save(tfsim_path)

# COMMAND ----------

# MAGIC %md
# MAGIC tensorflow_similarityモデルファイルを保存するために、ユニークな名前を割り当てて`artifacts`ディクショナリーを作成します。このディクショナリーは`mlflow.pyfunc.save_model`に引き渡され、モデルファイルを新規MLflowモデルのディレクトリにコピーします。

# COMMAND ----------

artifacts = {
    "tfsim_model": tfsim_path
}

# COMMAND ----------

# カスタムモデルクラスの定義
import mlflow.pyfunc
class TfsimWrapper(mlflow.pyfunc.PythonModel):
    """ モデルへの入力は、base64でエンコーディングされたバイト文字列(byte型)を含む単一の行、単一の列のpandasデータフレームです。この場合カラム名は'input'となります。"""
    """ モデルの出力はそれぞれの行が16進数に変化された文字列であるpandasデータフレームであり、bytesに変換される必要があり、np.frombuffer(...)を用いてnumpy arrayに変換され、(28, 28)にリシェイプされた上で、(必要であれば)可視化します。"""
    
    def load_context(self, context):
      import tensorflow_similarity as tfsim
      from tensorflow_similarity.models import SimilarityModel
      from tensorflow.keras import models
      import pandas as pd
      import numpy as np
      
      
      self.tfsim_model = models.load_model(context.artifacts["tfsim_model"])
      self.tfsim_model.load_index(context.artifacts["tfsim_model"])

    def predict(self, context, model_input):
      from PIL import Image
      import base64
      import io

      image = np.array(Image.open(io.BytesIO(base64.b64decode(model_input["input"][0].encode()))))    
      # モデルへの入力は (1, 28, 28) の形状である必要があります
      image_reshaped = image.reshape(-1, 28, 28)/255.0
      images = np.array(self.tfsim_model.lookup(image_reshaped, k=5))
      image_dict = {}
      for i in range(5):
        image_dict[i] = images[0][i].data.tostring().hex()
        
      return pd.DataFrame.from_dict(image_dict, orient='index')

# COMMAND ----------

from sys import version_info

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)

# COMMAND ----------

# MAGIC %md 
# MAGIC 必要な全ての依存関係と適切なpythonバージョンを含む新規MLflowモデルのためのConda環境を作成します。

# COMMAND ----------

import cloudpickle
conda_env = {
    'channels': ['defaults'],
    'dependencies': [
      'python={}'.format(PYTHON_VERSION),
      'pip',
      {
        'pip': [
          'mlflow',
          'tensorflow_similarity=={}'.format(tfsim.__version__),
          'tensorflow_cpu ==2.7.0',
          'cloudpickle=={}'.format(cloudpickle.__version__),
        ],
      },
    ],
    'name': 'tfsim_env'
}

# COMMAND ----------

# MLflowモデルを保存します
mlflow_pyfunc_model_path = "model/tfsim_mlflow_pyfunc_model"
mlflow.pyfunc.save_model(
        path=mlflow_pyfunc_model_path, python_model=TfsimWrapper(), artifacts=artifacts,
        conda_env=conda_env)

# COMMAND ----------

# MAGIC %md 
# MAGIC カスタムモデルの作成方法についてはこちらを参照ください: https://mlflow.org/docs/latest/models.html#custom-python-models 

# COMMAND ----------

# MAGIC %md
# MAGIC pyfuncラッパーのpredictメソッドが実際に動いていることをテストします。

# COMMAND ----------

img = "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAACN0lEQVR4nF3Sz2vaYBgH8DdKNypDZYO2hxWnVDGiESMxIUoStGKCMWgwEpVEtKIyxR848Qet1FEHPRShp0FPu4wNdtoOg3Wn3Xvb/7Npq0Z93hBe3g/PA3m/AUCrkG/xrlkAtKjlIQR086359ejhz93v+wcMfr7RMVf9MV8dpKnuz7vk6WQ2Fl9t2GGqd95tdduicqoOzwqN/hWxxv1gnPZ5vAjK8HyE8FEwglG2lVqJYEgQRUlMyUI5+fFvOJvA1q0kwpbYVEVMJVg+o0TlYorHGMMSE0ikwIfnYzEy6EcJNJgs47HV3Cjlm7QtVpfD5nScwLDHySs0Y1yiy2L78cWBIH6MwQM46ie+dgJh8slO/n0yz+4LMstyUZrj43L1O28djp9QHxOOPv9631EFlqE4qV6pfSuH8JdLPHa8UUdxNR4O00GKkXIZF2w52t+4wjJBu+0exGGHvUhEr53rFhk0qzLqQ2GnFyNSkgnSr2KBFoudvsso+WwmW8mfN4xgHdncAESqMSrCEAQpcErRAOk2MwMMQwSCATJEhDxkZE8L+xHVNIOSbjcM4yhV3wOaLUq47lebhayUrTfqt6YthABeyFXOykVVzsqcbAa6LeTSASZAzm8W9WOJgy0EoPchX8uXJFFQ8uXLwx3sTCeXF9Pm29ZgMOiZgF6bOX9ihVa9NO7k6oqqJJ5BO50Nnk4nJYKLRviG4fFP1qp/O51dXYza18Puzc2LnU/R2zHBTcJev81mNS7tP1M4itBUw7AYAAAAAElFTkSuQmCC"

data = {"input": [img] }
sample_image = pd.DataFrame.from_dict(data)

# COMMAND ----------

# `python_function`フォーマットでモデルをロードします
loaded_model = mlflow.pyfunc.load_model(mlflow_pyfunc_model_path)

# モデルを評価します
test_predictions = loaded_model.predict(sample_image)
print(test_predictions)

# COMMAND ----------

# MAGIC %md 
# MAGIC クエリーできるようにRESTエンドポイントにモデルをデプロイします。

# COMMAND ----------

# MAGIC %md
# MAGIC 最初にシグネチャを推定します。詳細はこちらを参照ください: https://www.mlflow.org/docs/latest/models.html#model-signature

# COMMAND ----------

from mlflow.models.signature import infer_signature
signature = infer_signature(sample_image, loaded_model.predict(sample_image))

# COMMAND ----------

# MAGIC %md
# MAGIC トレーニングしたモデルをロギングします。カスタムモデルラッパークラスを作成したので、これにはモデルとインデックスが含まれています。

# COMMAND ----------

mlflow.pyfunc.log_model(artifact_path=mlflow_pyfunc_model_path, python_model=TfsimWrapper(), artifacts=artifacts,
        conda_env=conda_env, signature = signature)

# COMMAND ----------

# MAGIC %md
# MAGIC このモデルをRESTエンドポイントにデプロイするためには、 https://docs.databricks.com/applications/mlflow/models.html の手順に従ってください。任意のMNSIT画像に対するレコメンデーションを生成する完全なサンプルに関しては、ブログ記事向けのGitリポジトリを参照ください。

# COMMAND ----------

# MAGIC %md 
# MAGIC DatabricksのデプロイメントUIでRESTエンドポイントをテストしたい場合には、以下のサンプルをお使いください(UIのBrowser入力ボックスに入力してテストします)。

# COMMAND ----------

[{"input":"iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAACN0lEQVR4nF3Sz2vaYBgH8DdKNypDZYO2hxWnVDGiESMxIUoStGKCMWgwEpVEtKIyxR848Qet1FEHPRShp0FPu4wNdtoOg3Wn3Xvb\\/7Npq0Z93hBe3g\\/PA3m\\/AUCrkG\\/xrlkAtKjlIQR086359ejhz93v+wcMfr7RMVf9MV8dpKnuz7vk6WQ2Fl9t2GGqd95tdduicqoOzwqN\\/hWxxv1gnPZ5vAjK8HyE8FEwglG2lVqJYEgQRUlMyUI5+fFvOJvA1q0kwpbYVEVMJVg+o0TlYorHGMMSE0ikwIfnYzEy6EcJNJgs47HV3Cjlm7QtVpfD5nScwLDHySs0Y1yiy2L78cWBIH6MwQM46ie+dgJh8slO\\/n0yz+4LMstyUZrj43L1O28djp9QHxOOPv9631EFlqE4qV6pfSuH8JdLPHa8UUdxNR4O00GKkXIZF2w52t+4wjJBu+0exGGHvUhEr53rFhk0qzLqQ2GnFyNSkgnSr2KBFoudvsso+WwmW8mfN4xgHdncAESqMSrCEAQpcErRAOk2MwMMQwSCATJEhDxkZE8L+xHVNIOSbjcM4yhV3wOaLUq47lebhayUrTfqt6YthABeyFXOykVVzsqcbAa6LeTSASZAzm8W9WOJgy0EoPchX8uXJFFQ8uXLwx3sTCeXF9Pm29ZgMOiZgF6bOX9ihVa9NO7k6oqqJJ5BO50Nnk4nJYKLRviG4fFP1qp\\/O51dXYza18Puzc2LnU\\/R2zHBTcJev81mNS7tP1M4itBUw7AYAAAAAElFTkSuQmCC"}]

# COMMAND ----------

# MAGIC %md
# MAGIC このUIは以下の手順に従ってアクセスすることもできます:
# MAGIC 
# MAGIC [Databricksにおけるモデルサービング \- Qiita](https://qiita.com/taka_yayoi/items/b5a5f83beb4c532cf921)

# COMMAND ----------

# MAGIC %md
# MAGIC REST APIエンドポイントを使用するアプリケーションをデプロイするためのコード、手順についてはこちらのリポジトリを参照ください: https://github.com/avisoori-databricks/Databricks_image_recommender/tree/main/recommender_app

# COMMAND ----------

# MAGIC %md
# MAGIC # END
