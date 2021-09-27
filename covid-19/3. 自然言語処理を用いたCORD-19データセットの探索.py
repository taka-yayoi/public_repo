# Databricks notebook source
# ライブラリのインストール
%pip install wordcloud==1.5
%pip install bert-extractive-summarizer
%pip install sentencepiece

# COMMAND ----------

# MAGIC %md 
# MAGIC # 自然言語処理を用いたCORD-19データセットの探索
# MAGIC ### COVID-19 Open Research Dataset Challenge (CORD-19) 作業用ノートブック
# MAGIC 
# MAGIC このノートブックは、CORD-19データセットの分析を容易に始められるようにするための、 [COVID-19 Open Research Dataset Challenge (CORD-19)](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) に対する作業用ノートブックです。  
# MAGIC 
# MAGIC <img src="https://miro.medium.com/max/3648/1*596Ur1UdO-fzQsaiGPrNQg.png" width="900"/>
# MAGIC 
# MAGIC アトリビューション:
# MAGIC * このノートブックで使用されるデータセットのライセンスは、[downloaded dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/download)に含まれるメタデータcsvに記載されています。
# MAGIC * 2020-03-03のデータセットには以下が含まれています。
# MAGIC   * `comm_use_subset`: 商用利用のサブセット (PMCコンテンツを含む) -- 9000 論文(内3論文は空), 186Mb
# MAGIC   * `noncomm_use_subset`: 非商用利用のサブセット (PMCコンテンツを含む) -- 1973 論文(内1論文は空), 36Mb
# MAGIC   * `biorxiov_medrxiv`: bioRxiv/medRxiv サブセット (ピアレビューされていない準備稿) -- 803 論文, 13Mb
# MAGIC * DatabricksあるいはDatabricksコミュニティエディションを使用する際には、`/databricks-datasets/COVID/CORD-19`からデータセットのコピーを利用できます。
# MAGIC * このノートブックは[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/us/)のライセンスの下で共有することができます。
# MAGIC 
# MAGIC > **注意**<br>
# MAGIC > このノートブックを実行する前に「1. JSONデータセットの読み込み」を実行して、ファイルを準備してください。

# COMMAND ----------

# ユーザーごとに一意のパスになるようにユーザー名をパスに含めます
import re
from pyspark.sql.types import * 

# Username を取得
username_raw = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
# Username の英数字以外を除去し、全て小文字化。Username をファイルパスやデータベース名の一部で使用可能にするため。
username = re.sub('[^A-Za-z0-9]+', '', username_raw).lower()

print(username)

# COMMAND ----------

# MAGIC %md ## Parquetパス変数の設定
# MAGIC 
# MAGIC `/tmp/<ユーザー名>/COVID/CORD-19/2020-03-13/`にParquetフォーマットで保存されています。

# COMMAND ----------

# PythonにおけるPathの設定
comm_use_subset_pq_path = f"/tmp/{username}/COVID/CORD-19/2020-03-13/comm_use_subset.parquet"
noncomm_use_subset_pq_path = f"/tmp/{username}/COVID/CORD-19/2020-03-13/noncomm_use_subset.parquet"
biorxiv_medrxiv_pq_path = f"/tmp/{username}/COVID/CORD-19/2020-03-13/biorxiv_medrxiv/biorxiv_medrxiv.parquet"

# COMMAND ----------

# MAGIC %md ## Parquetファイルの読み込み

# COMMAND ----------

comm_use_subset = spark.read.format("parquet").load(comm_use_subset_pq_path)
noncomm_use_subset = spark.read.format("parquet").load(noncomm_use_subset_pq_path)
biorxiv_medrxiv = spark.read.format("parquet").load(biorxiv_medrxiv_pq_path)

# COMMAND ----------

# レコード数のカウント
comm_use_subset_cnt = comm_use_subset.count()
noncomm_use_subset_cnt = noncomm_use_subset.count()
biorxiv_medrxiv_cnt = biorxiv_medrxiv.count()

# 出力
print (f"comm_use_subset: {comm_use_subset_cnt}, noncomm_use_subset: {noncomm_use_subset_cnt}, biorxiv_medrxiv: {biorxiv_medrxiv_cnt}")

# COMMAND ----------

comm_use_subset.show(3)

# COMMAND ----------

comm_use_subset.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## すべてのタイトルからワードクラウドを作成してみましょう。
# MAGIC 
# MAGIC 事前に`wordcloud==1.5`をインストールしておく必要があります。

# COMMAND ----------

comm_use_subset.select("metadata.title").show(3)

# COMMAND ----------

from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 

def wordcloud_draw(text, color = 'white'):
    """
    テキストからストップワードを削除した後にワードクラウドをプロットします
    """
    cleaned_word = " ".join([word for word in text.split()])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=1000,
                      height=1000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(8, 8))
    plt.imshow(wordcloud)
    plt.axis('off')
    display(plt.show())

# COMMAND ----------

from pyspark.sql.functions import concat_ws, collect_list

all_title_df = comm_use_subset.agg(concat_ws(", ", collect_list(comm_use_subset['metadata.title'])).alias('all_titles'))
display(all_title_df)

# COMMAND ----------

wordcloud_draw(str(all_title_df.select('all_titles').collect()[0]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### ラウンド 2 - 意味のないキーワードの除外

# COMMAND ----------

def custom_wordcloud_draw(text, color = 'white'):
    """
    テキストからストップワードを削除した後にワードクラウドをプロットします
    """
    cleaned_word = " ".join([word for word in text.split()])
    wordcloud = WordCloud(stopwords= STOPWORDS.update(['using', 'based', 'analysis', 'study', 'research', 'viruses']),
                      background_color=color,
                      width=1000,
                      height=1000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(8, 8))
    plt.imshow(wordcloud)
    plt.axis('off')
    display(plt.show())

# COMMAND ----------

custom_wordcloud_draw(str(all_title_df.select('all_titles').collect()[0]))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## アブストラクトのサマリーを作成

# COMMAND ----------

# MAGIC %md
# MAGIC 次に、講義の要約に使用されたBERTでトレーニングされたsummarizerを使用します。
# MAGIC 
# MAGIC 詳細に関してはこちらの[論文](https://arxiv.org/abs/1906.04165)を参照してください。
# MAGIC 
# MAGIC ライブラリーを使うには、PyPiによる`bert-extractive-summarizer`のインストールが必要です。
# MAGIC 
# MAGIC [bert\-extractive\-summarizer · PyPI](https://pypi.org/project/bert-extractive-summarizer/)

# COMMAND ----------

from summarizer import Summarizer

# 最初のアブストラクトを取得
abstract1 = str(comm_use_subset.select("abstract.text").first())
abstract1

# COMMAND ----------

abstract2 = str(comm_use_subset.select("abstract.text").take(2)[1])
abstract2

# COMMAND ----------

# MAGIC %md
# MAGIC ### `min_length`パラメーターを用いたSummarizerモデルのトレーニング
# MAGIC 
# MAGIC トレーニングはMLflowによってトラッキングされます。右上の**Experiment**をクリックしてみて下さい。

# COMMAND ----------

model = Summarizer()
abstract1_summary = model(str(abstract1), min_length=20)

full_abstract1 = ''.join(abstract1_summary)
print(full_abstract1)

# COMMAND ----------

abstract2_summary = model(str(abstract2), min_length=20)

full_abstract2 = ''.join(abstract2_summary)
print(full_abstract2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### `max_length`パラメーターを用いたSummarizerモデルのトレーニング

# COMMAND ----------

summary_executive = model(str(abstract1), max_length=250)

full_exec_summary = ''.join(summary_executive)
print(full_exec_summary)

# COMMAND ----------

summary_executive2 = model(str(abstract2), max_length=250)

full_exec_summary2 = ''.join(summary_executive2)
print(full_exec_summary2)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
