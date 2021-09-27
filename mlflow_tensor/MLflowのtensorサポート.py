# Databricks notebook source
# MAGIC %md
# MAGIC # MLflowのtensorサポート
# MAGIC 
# MAGIC 本ノートブックでは、MLflowにおけるtensorサポートをデモします。
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2021/06/29</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>クラスター</td><td>8.3ML</td></tr>
# MAGIC </table>
# MAGIC <img style="margin-top:25px;" src="https://jixjiadatabricks.blob.core.windows.net/images/databricks-logo-small-new.png" width="140">
# MAGIC 
# MAGIC [MLflowでTensorの入力をサポートしました \- Qiita](https://qiita.com/taka_yayoi/items/3e439dc5df7257fd41db)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 準備

# COMMAND ----------

# このサンプルを実行するにはMLflow >=1.14.0 が必要となります
import tensorflow.keras
from tensorflow.keras.layers import Dense, Flatten, Dropout
import numpy as np
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature

# トレーニングデータを準備しましょう
(train_X, train_Y), (test_X, test_Y) = tensorflow.keras.datasets.mnist.load_data()
trainX, testX = train_X / 255.0, test_X / 255.0
trainY = tensorflow.keras.utils.to_categorical(train_Y)
testY = tensorflow.keras.utils.to_categorical(test_Y)

# COMMAND ----------

assert mlflow.__version__>="1.14.0", "This notebook requires MLflow>=1.14.0"

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルの定義

# COMMAND ----------

# モデルを定義しましょう
model = tensorflow.keras.models.Sequential(
    [
      Flatten(),
      Dense(128, activation="relu", name="layer1"),
      Dropout(0.2),
      Dense(10, activation='softmax')
    ]
)
opt = tensorflow.keras.optimizers.SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## トレーニング

# COMMAND ----------

# モデルをfitさせましょう
model.fit(trainX, trainY, epochs=2, batch_size=32, validation_data=(testX, testY))

# COMMAND ----------

# MAGIC %md
# MAGIC ## シグネチャの作成

# COMMAND ----------

# MLflowモデルレジストリに格納するためにtensor入力を用いてモデルのシグネチャを作成します
signature = infer_signature(testX, model.predict(testX))
# どのように見えるかを確認します
print(signature)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 入力サンプルの作成

# COMMAND ----------

# MLflowモデルレジストリに格納する入力サンプルを作成します
input_example = np.expand_dims(trainX[0], axis=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルをMLflowモデルレジストリに登録

# COMMAND ----------

# モデルをMLflowモデルレジストリに登録しましょう
model_name = 'mnist_example'
registered_model_name = "tensor-blog-post"
mlflow.keras.log_model(model, model_name, signature=signature, input_example=input_example, registered_model_name=registered_model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルのロード、予測

# COMMAND ----------

# モデルをロードしてサンプルの予測を実行しましょう
model_version = "1"
loaded_model = mlflow.pyfunc.load_model(f"models:/{registered_model_name}/{model_version}")
loaded_model.predict(input_example)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
