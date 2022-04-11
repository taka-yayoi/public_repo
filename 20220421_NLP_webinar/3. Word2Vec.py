# Databricks notebook source
# MAGIC %md
# MAGIC # Word2Vec
# MAGIC 
# MAGIC - [wget/curl large file from google drive \- Stack Overflow](https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive)
# MAGIC - [Kyubyong/wordvectors: Pre\-trained word vectors of 30\+ languages](https://github.com/Kyubyong/wordvectors)
# MAGIC - [Word2Vecの学習済み日本語モデルを読み込んで使う \- Qiita](https://qiita.com/omuram/items/6570973c090c6f0cb060)

# COMMAND ----------

# MAGIC %md
# MAGIC ## データのダウンロード

# COMMAND ----------

# MAGIC %pip install gdown

# COMMAND ----------

# MAGIC %sh
# MAGIC gdown 0B0ZXk88koS2KMzRjbnE4ZHJmcWM

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -al

# COMMAND ----------

# MAGIC %sh
# MAGIC unzip ja.zip

# COMMAND ----------

# MAGIC %md
# MAGIC ## gensimのインポート

# COMMAND ----------

import gensim
model = gensim.models.Word2Vec.load('ja.bin')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 分散表現の表示

# COMMAND ----------

model["日本"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## ベクトルが近しいものを表示

# COMMAND ----------

model.most_similar("日本")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 意味ベクトルの加算・減算
# MAGIC 
# MAGIC `姪` + `男性` - `女性` = ?

# COMMAND ----------

model.most_similar(positive=['姪', '男性'], negative=['女性'])

# COMMAND ----------

# MAGIC %md
# MAGIC # END
