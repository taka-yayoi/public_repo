# Databricks notebook source
# MAGIC %md # Databricksにおける画像の取り扱い
# MAGIC 
# MAGIC このノートブックでは、画像データソースをどのように使うのかを説明します。
# MAGIC 
# MAGIC 
# MAGIC #### データの利用権限
# MAGIC 
# MAGIC この画像はCAVIARメンバー – [EC Funded CAVIAR project/IST 2001 37540](http://groups.inf.ed.ac.uk/vision/CAVIAR/CAVIARDATA1/)による戦闘シーンの再現ビデオから得られたものです。これらの画像を生成するのに用いられたコードは[Identify Suspicious Behavior in Video with Databricks Runtime for Machine Learning](https://databricks.com/blog/2018/09/13/identify-suspicious-behavior-in-video-with-databricks-runtime-for-machine-learning.html)から参照することができます。
# MAGIC 
# MAGIC #### 参考資料
# MAGIC - [Databricksにおける画像の取り扱い \- Qiita](https://qiita.com/taka_yayoi/items/8d4b1b61699d68a34e58)
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2021/08/22</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>クラスター</td><td>DBR8.3</td></tr>
# MAGIC </table>
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/themes/databricks/assets/images/header_logo_2x.png" alt="logo" width="150"/> 

# COMMAND ----------

# MAGIC %md ## セットアップ

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
# MAGIC # END
