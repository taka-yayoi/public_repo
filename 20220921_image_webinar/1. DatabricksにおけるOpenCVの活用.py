# Databricks notebook source
# MAGIC %md
# MAGIC # DatabricksにおけるOpenCVの活用
# MAGIC 
# MAGIC Databricks上でOpenCVを活用して画像処理を行うことが可能です。なお、画像をノートブック上に表示する際には注意が必要です。
# MAGIC 
# MAGIC 準備:
# MAGIC - `opencv-python`をクラスターライブラリとしてインストールしてください。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/qiita-images/opencv_cluster_library.png)
# MAGIC 
# MAGIC 参考資料:
# MAGIC - [Python Image Processing on Azure Databricks \- Part 1, OpenCV Image Compare – Stochastic Coder](https://stochasticcoder.com/2018/06/06/python-image-processing-on-azure-databricks-part-1-opencv-image-compare/)
# MAGIC - [Pythonを用いた画像処理\(openCV,skimage\) \- Qiita](https://qiita.com/taka_baya/items/453e429b466ffaa702c9)
# MAGIC - [OpenCVを用いて、Jupyter Notebook上で画像を表示させたい（mac） \- Qiita](https://qiita.com/narumi-github/items/63715853394bd4d110d6)

# COMMAND ----------

import matplotlib.pyplot as plt
import cv2
import numpy as np

# COMMAND ----------

target = cv2.imread('/dbfs/databricks-datasets/cctvVideos/train_images/label=0/LeftBagframe0004.jpg', cv2.IMREAD_UNCHANGED)

# 画像の表示
plt.imshow(cv2.cvtColor(target, cv2.COLOR_BGR2RGB)) # OpenCV は色がGBR順なのでRGB順に並べ替える
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## グレースケール

# COMMAND ----------

# imreadの第二引数でグレースケールを指定
gry = cv2.imread('/dbfs/databricks-datasets/cctvVideos/train_images/label=0/LeftBagframe0004.jpg', cv2.IMREAD_GRAYSCALE)
#imshow(gry)
cv2.imwrite('/tmp/gray.jpg', gry)

# 画像の表示
plt.imshow(cv2.cvtColor(gry, cv2.COLOR_BGR2RGB)) # OpenCV は色がGBR順なのでRGB順に並べ替える
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## エッジ検出

# COMMAND ----------

# エッジ検出
canny_img = cv2.Canny(target, 50, 110)

# 画像の表示
plt.imshow(cv2.cvtColor(canny_img, cv2.COLOR_BGR2RGB)) # OpenCV は色がGBR順なのでRGB順に並べ替える
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## サイズ変更

# COMMAND ----------

# 画像の高さ幅を指定
width,height = 60, 60
# 画像をリサイズ
resized_img = cv2.resize(target, (width, height))

# 画像の表示
plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)) # OpenCV は色がGBR順なのでRGB順に並べ替える
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 処理画像のダウンロード

# COMMAND ----------

# MAGIC %md
# MAGIC 以下のコマンドでコピーを行いブラウザからダウンロードすることも可能です。ブラウザのURLで以下を指定します。
# MAGIC 
# MAGIC `https://<databricksホスト名>/file/shared_uploads/takaaki.yayoi@databricks.com/gray.jpg`

# COMMAND ----------

dbutils.fs.cp("file:/tmp/gray.jpg", "/FileStore/shared_uploads/takaaki.yayoi@databricks.com/gray.jpg")

# COMMAND ----------

# MAGIC %md
# MAGIC ノートブックに直接画像をレンダリングすることもできます。

# COMMAND ----------

displayHTML("<img src='/files/shared_uploads/takaaki.yayoi@databricks.com/gray.jpg'>")

# COMMAND ----------

# MAGIC %md
# MAGIC ## クリーンアップ

# COMMAND ----------

dbutils.fs.rm("/FileStore/shared_uploads/takaaki.yayoi@databricks.com/gray.jpg")

# COMMAND ----------

# MAGIC %md
# MAGIC # END
