# Databricks notebook source
# MAGIC %md
# MAGIC # はじめての自然言語処理(NLP)
# MAGIC 
# MAGIC - [Python, Janomeで日本語の形態素解析、分かち書き（単語分割） \| note\.nkmk\.me](https://note.nkmk.me/python-janome-tutorial/)
# MAGIC - [Advanced Natural Language Processing with Apache Spark NLP \- Databricks](https://databricks.com/session_na20/advanced-natural-language-processing-with-apache-spark-nlp)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ライブラリのインポート

# COMMAND ----------

import pandas as pd

from janome.tokenizer import Tokenizer

import matplotlib.pyplot as plt
from wordcloud import WordCloud

import string

# COMMAND ----------

# MAGIC %md
# MAGIC ## はじめての形態素解析

# COMMAND ----------

t = Tokenizer()

s = 'すもももももももものうち'

print(type(t.tokenize(s)))
# <class 'generator'>

print(type(t.tokenize(s).__next__()))
# <class 'janome.tokenizer.Token'>

for token in t.tokenize(s):
    print(token)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 大量テキストデータに対する処理
# MAGIC 
# MAGIC Amazonの商品レビューのテキストデータを使用します。
# MAGIC 
# MAGIC [https://s3\.amazonaws\.com/amazon\-reviews\-pds/tsv/index\.txt](https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt)

# COMMAND ----------

url = 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_multilingual_JP_v1_00.tsv.gz'
df = pd.read_csv(url, sep='\t')
df

# COMMAND ----------

# MAGIC %md
# MAGIC ### データ前処理
# MAGIC 
# MAGIC データには英語テキストも含まれているので、日本語のレビューに限定します。

# COMMAND ----------

def filter_by_ascii_rate(text, threshold=0.9):
    """
    アスキー文字の比率に基づいて日本語テキストをフィルタリング
    """
    ascii_letters = set(string.printable)
    rate = sum(c in ascii_letters for c in text) / len(text)
    return rate <= threshold

# COMMAND ----------

# 日本語テキストに限定
is_jp = df.review_body.apply(filter_by_ascii_rate)
filtered_df = df[is_jp]# is_jpカラムの追加
filtered_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### 形態素解析
# MAGIC 
# MAGIC 上記テキストデータを入力として形態素解析を行います。形態素解析の処理を関数にして、データフレームに`apply`を適用します。
# MAGIC 
# MAGIC [Word Cloudが作りたくなったら。 \- Qiita](https://qiita.com/kyoro1/items/59216cc09b56d5b5f760)

# COMMAND ----------

# janomeのトークナイザ
t = Tokenizer()

def tokenize(text):
    """
    janomeによる分かち書き(トークン化)
    """
    
    # 一般名詞のみを抽出
    noun_list = []
    for token in t.tokenize(text):
      split_token = token.part_of_speech.split(',')

      #and split_token[1] == '一般'
      if split_token[0] == '名詞' and len(token.surface) > 1:
        noun_list.append(token.surface)
  
    return noun_list

# COMMAND ----------

# MAGIC %md
# MAGIC pandasで上記データフレームに格納されているテキストデータ(約25万件)の形態素解析を行おうとすると非常に時間がかかります。このため、10件のみをサンプリングします。これに対する対策については後述します。

# COMMAND ----------

# サンプリング
filtered_df = filtered_df.head(n=10) 

# COMMAND ----------

filtered_df['tokens'] = filtered_df.review_body.apply(tokenize)
filtered_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 可視化
# MAGIC 
# MAGIC ワードクラウドで可視化します。事前に日本語フォントをインストールしておく必要があります。
# MAGIC 
# MAGIC [Word Clouds in Japanese](https://github.com/kairozu/Japanese-Word-Cloud/raw/master/kairozu_word_cloud.png)

# COMMAND ----------

def depict_word_cloud(noun_list):
    ## 名詞リストの要素を空白区切りにする(word_cloudの仕様)
    noun_space = ' '.join(map(str, noun_list))
    ## word cloudの設定(フォントの設定)
    wc = WordCloud(background_color="white", font_path=r"/usr/share/fonts/truetype/fonts-japanese-gothic.ttf", width=300,height=300)
    wc.generate(noun_space)
    ## 出力画像の大きさの指定
    plt.figure(figsize=(5,5))
    ## 目盛りの削除
    plt.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False,
                   length=0)
    ## word cloudの表示
    plt.imshow(wc)
    plt.show()

# COMMAND ----------

for i in range(len(filtered_df)):
  depict_word_cloud(filtered_df.iloc[i]['tokens'])

# COMMAND ----------

# MAGIC %md
# MAGIC # END
