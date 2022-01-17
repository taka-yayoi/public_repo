# Databricks notebook source
# MAGIC %md
# MAGIC # DatabricksにおけるMeCabの活用

# COMMAND ----------

# MAGIC %md ## initスクリプトの作成

# COMMAND ----------

import os

file_init="/dbfs/databricks/scripts/mecab_init_script.sh"
with open(file_init,"wt") as f1:
  f1.write("""#!/bin/bash
yes | apt-get install mecab
yes | apt-get install libmecab-dev
yes | apt-get install mecab-ipadic-utf8
yes | apt-get install swig
# using neologd
git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git > /dev/null 
echo yes | mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n > /dev/null 2>&1
echo "dicdir = "`mecab-config --dicdir`"/mecab-ipadic-neologd" > "/usr/local/etc/mecabrc"
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MeCabの動作確認

# COMMAND ----------

import MeCab

mecab = MeCab.Tagger("-Ochasen")
print(mecab.parse("今日はいい天気ですね。"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ダミーデータの作成

# COMMAND ----------

# sqlモジュールからPySparkのRowクラスをインポート
from pyspark.sql import *

tweet1 = Row(id='123456', tweet='テストのテキストデータです。これを分析することでどのような分析結果を得ることができるのでしょうか。')
tweet2 = Row(id='789012', tweet='感染者数が増えて心配')
tweet3 = Row(id='345678', tweet='ワクチンの効果があるから大丈夫')
tweet4 = Row(id='901234', tweet='海外旅行に行けるのはいつの日か')

tweets = [tweet1, tweet2, tweet3, tweet4]
df_tweet = spark.createDataFrame(tweets)

display(df_tweet)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spark UDFの定義
# MAGIC 
# MAGIC こちらのUDF(User Defined Function:ユーザー定義関数)は、形態素解析で抽出された名詞のみを半角スペースで連結した文字列を返却します。

# COMMAND ----------

def extract_words(text: str) -> str:
  """
  名詞のみを抽出するUDF
  :param text: Pyspark/SQLのカラム
  :return: 名詞を半角スペースで連結した文字列
  """
  
  word_str = ""
  
  mecab = MeCab.Tagger("-Ochasen")
  mecab.parseToNode('')
  node = mecab.parseToNode(text)
    
  while node:
    # 名詞のみを抽出
    if node.feature.split(",")[0] == "名詞":
      word = node.surface
      
      word_str = word_str + " " + word
      
    node = node.next
  
  return word_str

# COMMAND ----------

# Spark UDFとして関数を登録
spark.udf.register("extract_words", extract_words)

# COMMAND ----------

# MAGIC %md ## Spark UDFによる処理

# COMMAND ----------

# 上で作成したデータフレームを一時ビューとして登録
df_tweet.createOrReplaceTempView('tweets')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM tweets;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT tweet, extract_words(tweet) AS prosessed_tweet FROM tweets;

# COMMAND ----------

# MAGIC %md
# MAGIC ## おまけ：ワードクラウドによる可視化
# MAGIC 
# MAGIC テーブルを永続化することで<a target="_blank" href="/sql">Databricks SQL</a>でデータの可視化を行うことができます。

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS japanese_nlp_test

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE japanese_nlp_test.tweet_processed
# MAGIC AS 
# MAGIC SELECT tweet, extract_words(tweet) AS prosessed_tweet FROM tweets;

# COMMAND ----------

# MAGIC %md
# MAGIC # END
