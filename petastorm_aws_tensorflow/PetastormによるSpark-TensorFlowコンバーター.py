# Databricks notebook source
# MAGIC %md # SparkからTensorFlowへのデータ変換の簡素化
# MAGIC 
# MAGIC このノートブックでは、Databricks上での以下のワークフローをデモします。
# MAGIC 1. Sparkによるデータロード
# MAGIC 1. petastrom`spark_dataset_converter`による、SparkデータフレームからTensorFlowデータセットへの変換
# MAGIC 1. シングルノードのTensorFlowモデルのトレーニングのためのデータ投入
# MAGIC 1. 分散ハイパーパラメーターチューニング関数へのデータ投入
# MAGIC 1. 分散TensorFlowモデルのトレーニングのためのデータ投入
# MAGIC 
# MAGIC このノートブックで用いる例は、[transfer learning tutorial from TensorFlow](https://www.tensorflow.org/tutorials/images/transfer_learning)をベースにしています。事前学習済みの[MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2)モデルを花のデータセットに適用します。
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### 要件
# MAGIC 1. Databricks Runtime 7.0 ML
# MAGIC 2. ノードタイプ：1つのドライバーおよび2つのワーカーノード。GPUインスタンスを使うことをお勧めします。
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2021/08/30</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>クラスター</td><td>8.3ML</td></tr>
# MAGIC </table>
# MAGIC <img style="margin-top:25px;" src="https://jixjiadatabricks.blob.core.windows.net/images/databricks-logo-small-new.png" width="140">

# COMMAND ----------

from pyspark.sql.functions import col

from petastorm.spark import SparkDatasetConverter, make_spark_converter

import io
import numpy as np
import tensorflow as tf
from PIL import Image
from petastorm import TransformSpec
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK

import horovod.tensorflow.keras as hvd
from sparkdl import HorovodRunner

# COMMAND ----------

IMG_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
NUM_EPOCHS = 5

# COMMAND ----------

# MAGIC %md ## 1. Sparkによるデータロード
# MAGIC 
# MAGIC ### 花のデータセット
# MAGIC 
# MAGIC サンプルデータセットとして、ここではTenforFlowチームによって提供される[flowers dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers)を使用します。これには、それぞれがクラスを表現する5つのサブディレクトリに花の写真が格納されています。これは、簡単にアクセスできるようにDatabricksデータセットのディレクトリ`dbfs:/databricks-datasets/flower_photos`に格納されています。
# MAGIC 
# MAGIC この例では、バイナリーファイルデータソースを用いて前処理済みの花のデータセットを保持する花のテーブルをロードします。ここでは約900のトレーニング画像、約100の検証画像から構成される花のデータセットの小規模なサブセットを使います。

# COMMAND ----------

from pyspark.sql.functions import col, pandas_udf, regexp_extract

df_sample = spark.read.format("delta").load("/databricks-datasets/flowers/delta") \
  .select(col("content"), col("label")) \
  .limit(1000)

# ラベルのインデックスを作成
labels = df_sample.select(col("label")).distinct().collect()
label_to_idx = {label: index for index, (label, ) in enumerate(sorted(labels))}
num_classes = len(label_to_idx)
#num_classes = df_sample.select("label").distinct().count()

@pandas_udf("long")
def get_label_idx(labels):
  return labels.map(lambda label: label_to_idx[label])

df = df_sample.withColumn("label_index", get_label_idx(col("label"))) \
  .select(col("label_index"), col("content")) # col("label"), 
  
df_train, df_val = df.randomSplit([0.9, 0.1], seed=12345)

# パーティションの数が最低でも分散トレーニングに必要となるワーカー数となっていることを確認してください
df_train = df_train.repartition(2)
df_val = df_val.repartition(2)

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Petastorm Sparkコンバーターを用いたSparkデータフレームのキャッシュ

# COMMAND ----------

# 中間データのためのDBFS FUSE上のキャッシュディレクトリを指定します
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, "file:///dbfs/tmp/petastorm/cache")

converter_train = make_spark_converter(df_train)
converter_val = make_spark_converter(df_val)

# COMMAND ----------

print(f"train: {len(converter_train)}, val: {len(converter_val)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. シングルノードのTensorFlowモデルのトレーニングのためのデータ投入
# MAGIC 
# MAGIC ### tensorflow.kerasからMobileNetV2モデルを取得します

# COMMAND ----------

# 最初にモデルをロードし、モデルの構造を調査します
MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet').summary()

# COMMAND ----------

def get_model(lr=0.001):

  # 事前学習済みのMobileNet V2モデルからベースモデルを作成します
  base_model = MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
  # 特徴量抽出レイヤーのパラメーターをフリーズします
  base_model.trainable = False
  
  # 転送学習のための新たな分類レイヤーを追加します
  global_average_layer = keras.layers.GlobalAveragePooling2D()
  prediction_layer = keras.layers.Dense(num_classes)
  
  model = keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
  ])
  return model

def get_compiled_model(lr=0.001):
  model = get_model(lr=lr)
  model.compile(optimizer=keras.optimizers.SGD(lr=lr, momentum=0.9),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

# COMMAND ----------

get_compiled_model().summary()

# COMMAND ----------

# MAGIC %md ### 画像の前処理
# MAGIC 
# MAGIC モデルにデータセットを投入する前に、生の画像バイトをデコードし、標準的なImageNet変換を適用する必要があります。Sparkデータフレーム上での変換は中間ファイルのサイズを増加させ、パフォーマンスを損なうためお勧めしません。代わりに、petastormの`TransformSpec`関数による変換をお勧めします。
# MAGIC 
# MAGIC あるいは、`dataset.map()`と`tf.map_fn()`を用いたコンバーターで返却されるTensorFlowデータセットに変換処理を適用することもできます。

# COMMAND ----------

def preprocess(content):
  """
  MobileNetV2 (ImageNet)向けの画像ファイルバイトの前処理
  """
  image = Image.open(io.BytesIO(content)).resize([224, 224])
  image_array = keras.preprocessing.image.img_to_array(image)
  return preprocess_input(image_array)

def transform_row(pd_batch):
  """
  この関数の入出力はpandasデータフレームとなります
  """
  pd_batch['features'] = pd_batch['content'].map(lambda x: preprocess(x))
  pd_batch = pd_batch.drop(labels='content', axis=1)
  return pd_batch

# `TransformSpec`の出力のシェイプはpetastormで自動で検知されないことに注意してください
# このため、`edit_fields`の新規カラムのシェイプと`selected_fields`の出力カラムの順序を指定する必要があります
transform_spec_fn = TransformSpec(
  transform_row, 
  edit_fields=[('features', np.float32, IMG_SHAPE, False)], 
  selected_fields=['features', 'label_index']
)

# COMMAND ----------

# MAGIC %md ### ローカルマシンでのトレーニングおよび評価
# MAGIC 
# MAGIC データセットを作成するために`converter.make_tf_dataset(...)`を使用します。

# COMMAND ----------

def train_and_evaluate(lr=0.001):
  model = get_compiled_model(lr)
  
  with converter_train.make_tf_dataset(transform_spec=transform_spec_fn, 
                                       batch_size=BATCH_SIZE) as train_dataset, \
       converter_val.make_tf_dataset(transform_spec=transform_spec_fn, 
                                     batch_size=BATCH_SIZE) as val_dataset:
    # tf.kerasはタプルのみを受け付けます、名前付きタプルは受け付けません
    train_dataset = train_dataset.map(lambda x: (x.features, x.label_index))
    steps_per_epoch = len(converter_train) // BATCH_SIZE

    val_dataset = val_dataset.map(lambda x: (x.features, x.label_index))
    validation_steps = max(1, len(converter_val) // BATCH_SIZE)
    
    print(f"steps_per_epoch: {steps_per_epoch}, validation_steps: {validation_steps}")

    hist = model.fit(train_dataset, 
                     steps_per_epoch=steps_per_epoch,
                     epochs=NUM_EPOCHS,
                     validation_data=val_dataset,
                     validation_steps=validation_steps,
                     verbose=2)
  return hist.history['val_loss'][-1], hist.history['val_accuracy'][-1] 
  
loss, accuracy = train_and_evaluate()
print("Validation Accuracy: {}".format(accuracy))

# COMMAND ----------

# MAGIC %md ## 4. 分散ハイパーパラメーターチューニング関数へのデータ投入
# MAGIC 
# MAGIC ハイパーパラメーターチューニングにHyperoptとSparkTrialsを使用します。

# COMMAND ----------

def train_fn(lr):
  import tensorflow as tf
  from tensorflow import keras
  loss, accuracy = train_and_evaluate()
  return {'loss': loss, 'status': STATUS_OK}

search_space = hp.loguniform('lr', -10, -4)

argmin = fmin(
  fn=train_fn,
  space=search_space,
  algo=tpe.suggest,
  max_evals=2,
  trials=SparkTrials(parallelism=2))

# COMMAND ----------

# 際的なハイパーパラメーターを参照します
argmin

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 分散TensorFlowモデルのトレーニングのためのデータ投入
# MAGIC 
# MAGIC 分散トレーニングにHorovodRunnerを用います。
# MAGIC 
# MAGIC 最後の不完全なバッチをハンドリングするのを避けるために、無限にバッチを生成するパラメーター`num_epochs=None`のデフォルト値を使用します。これは、全てのワーカーノードで確認されるステップごとのデータレコードの数が同じであることを保証する必要がある分散トレーニングシナリオにおいては有用です。それぞれのデータシャードの長さが同一でない場合に、特定の値を`num_epochs`に設定すると保証を実現できないため失敗する場合があります。

# COMMAND ----------

def train_and_evaluate_hvd(lr=0.001):
  
  hvd.init()  # Horovodの初期化
  
  # Horovod: ローカルランクを処理するために使うGPUをピン留めします (プロセスあたり1GPU)
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  if gpus:
      tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

  model = get_model(lr)
  
  # Horovod: GPUの数に応じて学習率を調整します
  optimizer = keras.optimizers.SGD(lr=lr * hvd.size(), momentum=0.9)
  dist_optimizer = hvd.DistributedOptimizer(optimizer)
  
  callbacks = [
    # Horovod: 初期変数の状態をランク0から他の全てのプロセスにブロードキャストします
    # これは、ランダムな重みからトレーニングを開始した、あるいは、チェックポイントから復旧した際に一貫性のある初期化を行うために必要です
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
  ]
  
  # TF 2.xでは、experimental_run_tf_function=Falseを設定する必要があります
  model.compile(optimizer=dist_optimizer, 
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                metrics=["accuracy"],
                experimental_run_tf_function=False)
    
  with converter_train.make_tf_dataset(transform_spec=transform_spec_fn, 
                                       cur_shard=hvd.rank(), shard_count=hvd.size(),
                                       batch_size=BATCH_SIZE) as train_dataset, \
       converter_val.make_tf_dataset(transform_spec=transform_spec_fn, 
                                     cur_shard=hvd.rank(), shard_count=hvd.size(),
                                     batch_size=BATCH_SIZE) as val_dataset:
    # tf.kerasはタプルのみを受け付け、名前付きタプルは受け付けません
    train_dataset = train_dataset.map(lambda x: (x.features, x.label_index))
    steps_per_epoch = len(converter_train) // (BATCH_SIZE * hvd.size())

    val_dataset = val_dataset.map(lambda x: (x.features, x.label_index))
    validation_steps = max(1, len(converter_val) // (BATCH_SIZE * hvd.size()))
    
    hist = model.fit(train_dataset, 
                     steps_per_epoch=steps_per_epoch,
                     epochs=NUM_EPOCHS,
                     validation_data=val_dataset,
                     validation_steps=validation_steps,
                     callbacks=callbacks,
                     verbose=2)

  return hist.history['val_loss'][-1], hist.history['val_accuracy'][-1]

# COMMAND ----------

hr = HorovodRunner(np=2)  # ワーカーノードが2つであるという前提を置いています
hr.run(train_and_evaluate_hvd)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
