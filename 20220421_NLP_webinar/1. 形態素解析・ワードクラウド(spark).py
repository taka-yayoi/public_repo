# Databricks notebook source
# MAGIC %md
# MAGIC # はじめての自然言語処理(NLP)
# MAGIC 
# MAGIC - [Python, Janomeで日本語の形態素解析、分かち書き（単語分割） \| note\.nkmk\.me](https://note.nkmk.me/python-janome-tutorial/)
# MAGIC - [Advanced Natural Language Processing with Apache Spark NLP \- Databricks](https://databricks.com/session_na20/advanced-natural-language-processing-with-apache-spark-nlp)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 大量テキストデータに対する処理
# MAGIC 
# MAGIC Amazonの商品レビューのテキストデータを使用します。
# MAGIC 
# MAGIC [https://s3\.amazonaws\.com/amazon\-reviews\-pds/tsv/index\.txt](https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt)

# COMMAND ----------

# MAGIC %md
# MAGIC ### データ準備
# MAGIC 
# MAGIC 初回のみ実行
# MAGIC 
# MAGIC ```
# MAGIC %sh
# MAGIC wget 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_multilingual_JP_v1_00.tsv.gz'
# MAGIC ```
# MAGIC 
# MAGIC ```
# MAGIC %fs
# MAGIC cp file:/databricks/driver/amazon_reviews_multilingual_JP_v1_00.tsv.gz dbfs:/FileStore/shared_uploads/takaaki.yayoi@databricks.com/
# MAGIC ```

# COMMAND ----------

file_path = "/FileStore/shared_uploads/takaaki.yayoi@databricks.com/amazon_reviews_multilingual_JP_v1_00.tsv.gz"

sdf = spark.read.format("csv").option("delimiter", "\t") \
                              .option("header", True) \
                              .load(file_path)
#display(sdf)

# データフレームをDeltaに保存します
path = f"/tmp/takaaki.yayoi@databricks.com/nlp/amazon_review.delta"
sdf.repartition(288).write.format('delta').mode("overwrite").option("path", path).saveAsTable("20210712_demo_takaakiyayoi.amazon_review")

# COMMAND ----------

# MAGIC %md
# MAGIC ### ライブラリのインポート

# COMMAND ----------

import pandas as pd

from janome.tokenizer import Tokenizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import string

# COMMAND ----------

sdf = spark.table("20210712_demo_takaakiyayoi.amazon_review")

# COMMAND ----------

# MAGIC %md
# MAGIC ### データ前処理
# MAGIC 
# MAGIC データには英語テキストも含まれているので、日本語のレビューに限定します。

# COMMAND ----------

from pyspark.sql.types import ArrayType, StringType, IntegerType, DoubleType
from pyspark.sql.functions import col
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import udf

# COMMAND ----------

# MAGIC %md
# MAGIC Apache Sparkを用いることで処理を並列分散することができます。Sparkのpandas UDF(User-defined Function:ユーザー定義関数)を活用して大量データに対する形態素解析処理を高速に処理します。
# MAGIC 
# MAGIC [DatabricksにおけるSpark pandasユーザー定義関数 \- Qiita](https://qiita.com/taka_yayoi/items/b65197128ee698d87910)

# COMMAND ----------

@pandas_udf(IntegerType())
def filter_by_ascii_rate(texts: pd.Series) -> pd.Series:
  threshold = 0.9 # テキストの9割が英数字の場合は除外
  ascii_letters = set(string.printable)
    
  filtered = []
  
  for text in texts:
      if text is None:
        filtered.append(0)
      else:
        rate = sum(c in ascii_letters for c in text) / len(text)
        filtered.append(int(rate <= threshold))
      
  return pd.Series(filtered)

# COMMAND ----------

# is_jpカラムの追加
filtered_sdf = sdf.withColumn("is_jp", filter_by_ascii_rate(col("review_body"))).filter("is_jp == 1")
filtered_sdf.write.format("noop").mode("append").save()

# COMMAND ----------

display(filtered_sdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 形態素解析
# MAGIC 
# MAGIC [Word Cloudが作りたくなったら。 \- Qiita](https://qiita.com/kyoro1/items/59216cc09b56d5b5f760)

# COMMAND ----------

@pandas_udf(ArrayType(StringType()))
def tokenize_base_form(texts: pd.Series) -> pd.Series:  
  from janome.tokenizer import Tokenizer
  t = Tokenizer()
  tokenized = []
  
  for text in texts:
  
    # 全てのトークンを抽出
    #tokens = [token.base_form for token in t.tokenize(text)]
    #tokenized.append(tokens)
  
    # 一般名詞のみを抽出
    noun_list = []
    for token in t.tokenize(text):
      split_token = token.part_of_speech.split(',')

      #and split_token[1] == '一般'
      if split_token[0] == '名詞' and len(token.surface) > 1:
        noun_list.append(token.surface)
  
    tokenized.append(noun_list)
  
  return pd.Series(tokenized)

# COMMAND ----------

tokenized_sdf = filtered_sdf.withColumn("tokens", tokenize_base_form(col("review_body")))
tokenized_sdf.write.format("noop").mode("append").save()

# COMMAND ----------

display(tokenized_sdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 可視化
# MAGIC 
# MAGIC ワードクラウドで可視化します。
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

noun_list = tokenized_sdf.take(10)
display(noun_list)

# COMMAND ----------

for i in range(len(noun_list)):
  depict_word_cloud(noun_list[i]['tokens'])

# COMMAND ----------

# MAGIC %md
# MAGIC # END
