# Databricks notebook source
# MAGIC %md
# MAGIC # Amazonレビューのレーティング予測(pandas)
# MAGIC 
# MAGIC [機械学習・深層学習による自然言語処理入門 \| マイナビブックス](https://book.mynavi.jp/ec/products/detail/id=113274)のソースコードを参考にしています。このノートブックの処理は全てpandasを使用しています。

# COMMAND ----------

import pandas as pd
import re
import string

from bs4 import BeautifulSoup
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# COMMAND ----------

# MAGIC %md
# MAGIC ## データの準備
# MAGIC 
# MAGIC Amazonの商品レビューのテキストデータを使用します。
# MAGIC 
# MAGIC [https://s3\.amazonaws\.com/amazon\-reviews\-pds/tsv/index\.txt](https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt)

# COMMAND ----------

def filter_by_ascii_rate(text, threshold=0.9):
    """
    アスキー文字の比率に基づいて日本語テキストをフィルタリング
    """
    ascii_letters = set(string.printable)
    rate = sum(c in ascii_letters for c in text) / len(text)
    return rate <= threshold


def load_dataset(filename, n=5000, state=6):
    """
    S3からアマゾンレビューを取得してデータフレームに格納
    """
    df = pd.read_csv(filename, sep='\t')

    # 日本語テキストに限定
    is_jp = df.review_body.apply(filter_by_ascii_rate)
    df = df[is_jp]

    # サンプリング
    df = df.sample(frac=1, random_state=state)  # シャッフルを実施
    grouped = df.groupby('star_rating')
    df = grouped.head(n=n) 
    # データの確認
    display(df)
    
    return df.review_body.values, df.star_rating.values

# データのロード(1000レコード)
url = 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_multilingual_JP_v1_00.tsv.gz'
x, y = load_dataset(url, n=1000)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ヘルパー関数
# MAGIC 
# MAGIC 自然言語処理におけるテキスト前処理を行う関数群

# COMMAND ----------

# janomeのトークナイザ
t = Tokenizer()

def tokenize(text):
    """
    janomeによる分かち書き(トークン化)
    """
    return t.tokenize(text, wakati=True)


def clean_html(html, strip=False):
    """
    HTMLタグの除去
    """
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(strip=strip)
    return text


def tokenize_base_form(text):
    """
    トークンの原型に統一
    """
    tokens = [token.base_form for token in t.tokenize(text)]
    return tokens


def normalize_number(text, reduce=False):
    """
    数字を0に統一
    """
    if reduce:
        normalized_text = re.sub(r'\d+', '0', text)
    else:
        normalized_text = re.sub(r'\d', '0', text)
    return normalized_text


def truncate(sequence, maxlen):
    """
    文字の切り詰め
    """
    return sequence[:maxlen]


def remove_url(html):
    """
    Aタグ除外
    """
    soup = BeautifulSoup(html, 'html.parser')
    for a in soup.findAll('a'):
        a.replaceWithChildren()
    return str(soup)

# COMMAND ----------

# MAGIC %md
# MAGIC トレーニングデータセット、テストデータセットに分割します。

# COMMAND ----------

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 予測モデルのトレーニング
# MAGIC 
# MAGIC 上で準備したデータを使ってトレーニング、精度評価を行います。
# MAGIC 
# MAGIC ここで使用する`CountVectorizer`はテキストに出現する単語のカウントを特徴量にして、機械学習モデルをトレーニングします。
# MAGIC 
# MAGIC - [fit、transform、fit\_transformの意味を、正規化の例で解説 \- 具体例で学ぶ数学](https://mathwords.net/fittransform)
# MAGIC - [fit と transform と fit\_transformの違いと学習する際の注意点 \- ごはんと飲み物は紙一重](https://twdlab.hatenablog.com/entry/2019/04/11/181329)

# COMMAND ----------

# CountVectorizer(前処理はトークン化のみ)
vectorizer = CountVectorizer(lowercase=None,
                                 tokenizer=tokenize,
                                 preprocessor=None)

# トレーニングデータのベクトル化
x_train_vec = vectorizer.fit_transform(x_train)

# COMMAND ----------

# トレーニングデータのベクトルの確認
pdf = pd.DataFrame(x_train_vec.toarray(), columns=vectorizer.get_feature_names())
pdf

# COMMAND ----------

# テストデータのベクトル化
x_test_vec = vectorizer.transform(x_test)

# ロジスティック回帰分類器のトレーニング
clf = LogisticRegression(solver='liblinear')
clf.fit(x_train_vec, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 予測の実施及び評価

# COMMAND ----------

# 予測
y_pred = clf.predict(x_test_vec)
# 精度の評価
score = accuracy_score(y_test, y_pred)
print('精度:{:.4f}'.format(score))

# COMMAND ----------

# MAGIC %md
# MAGIC # END
