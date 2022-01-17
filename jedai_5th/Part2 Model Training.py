# Databricks notebook source
# MAGIC %run ./utilities/Jedai_mission_impossible_common

# COMMAND ----------

# initialize
dbutils.fs.rm(f"{work_path}/datasets/gold/delta/_checkpoints", True)

# COMMAND ----------

spark.sql(f"USE {db_name_silver}")

# COMMAND ----------

# MAGIC %md
# MAGIC # MLトレーニングおよび分析
# MAGIC 
# MAGIC ここでは、奴らが作り出すだろう悪意のあるURLを生成するモデルを準備します。
# MAGIC 
# MAGIC - gTLD(例 .com, .org)とccTLD(例 .ru, cn, .uk, .ca)を除外してドメイン名を抽出
# MAGIC - モデルを構築

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alexaドメインリストの読み込み
# MAGIC - Alexaは人気でランキングされたインターネットのドメインのリストだから、これを参考に適切なURLのデータを準備します。

# COMMAND ----------

import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
alexa_dataframe = pd.read_csv(default_file_path + "/alexa_100k.txt");
display(alexa_dataframe)

# COMMAND ----------

# MAGIC %md
# MAGIC ## URLからドメイン名を抽出
# MAGIC - 分析には`.com`や`.org`は不要だから、URLの登録ドメイン、サブドメインからgTLDやccTLD(generic or country code top-level domain)を除外してドメイン名を抽出できるようにしておきます。
# MAGIC - トレーニングにはドメイン名が必要だから、そのための抽出関数を準備します。
# MAGIC - tldextractの結果は右のようになる: `ExtractResult(subdomain='forums.news', domain='cnn', suffix='com')` から、これを利用してドメイン名を抽出する関数を準備しています。

# COMMAND ----------

import tldextract
import numpy as np
def domain_extract(uri):
    ext = tldextract.extract(uri)
    if (not ext.suffix):
        return np.nan
    else:
        return ext.domain

spark.udf.register("domain_extract", domain_extract)

alexa_dataframe['domain'] = [ domain_extract(uri) for uri in alexa_dataframe['uri']]
del alexa_dataframe['uri']
del alexa_dataframe['rank']
display(alexa_dataframe)

# COMMAND ----------

# MAGIC %md
# MAGIC ## トレーニングデータにAlexaの合法なドメインを追加
# MAGIC 
# MAGIC 上で準備したデータに`legit`というラベルをつけてトレーニングデータを準備します。

# COMMAND ----------

# 空行などからNaNを持つ場合があります
alexa_dataframe = alexa_dataframe.dropna()
alexa_dataframe = alexa_dataframe.drop_duplicates()

# クラス(legit)の設定
alexa_dataframe['class'] = 'legit'

# データのシャッフル (トレーニング/テストにおいては重要です)
alexa_dataframe = alexa_dataframe.reindex(np.random.permutation(alexa_dataframe.index))
alexa_total = alexa_dataframe.shape[0]
print('Total Alexa domains %d' % alexa_total)
display(alexa_dataframe)

# COMMAND ----------

# MAGIC %md
# MAGIC DGAのデータも準備しておきます。これが怪しいサイトの教師データになる。IoC(Indicator of Compromise：セキュリティ侵害インジケーター)ということから、ラベルは`ioc`とします。

# COMMAND ----------

file_location = default_file_path + "/dga_domains_header.txt"

dga_dataframe = pd.read_csv(file_location, header=0);
# ブラックリストの値が大文字小文字の違いや.com/.org/.infoの違いであることに気づきました
dga_dataframe['domain'] = dga_dataframe.applymap(lambda x: x.split('.')[0].strip().lower())

# 空行などからNaNを持つ場合があります
dga_dataframe = dga_dataframe.dropna()
dga_dataframe = dga_dataframe.drop_duplicates()
dga_total = dga_dataframe.shape[0]
print('Total DGA domains %d' % dga_total)

# クラス(ioc)の設定
dga_dataframe['class'] = 'ioc'

print('Number of DGA domains: %d' % dga_dataframe.shape[0])
all_domains = pd.concat([alexa_dataframe, dga_dataframe], ignore_index=True)

# COMMAND ----------

# データセットからGDA検知結果をアウトプット
display(dga_dataframe)

# COMMAND ----------

# MAGIC %md
# MAGIC 怪しいドメインは、無意味な数字・アルファベットが並んでいますので、これを抽出するために文字列のエントロピーを活用します。
# MAGIC 
# MAGIC - いくつかの特徴量エンジニアリングを行い、データセットのエントロピー、長さの計算処理を追加します。
# MAGIC - 文字列長に対するユニークな文字列の数を比較することでエントロピー(文字列長に対してユニークな文字の数が多いほど高くなる)を計算します。

# COMMAND ----------

all_domains['length'] = [len(x) for x in all_domains['domain']]
all_domains = all_domains[all_domains['length'] > 6]

# COMMAND ----------

all_domains

# COMMAND ----------

import math
from collections import Counter
 
def entropy(s):
    p, lns = Counter(s), float(len(s))
    
    return -math.fsum( count/lns * math.log(count/lns, 2) for count in p.values()  )
  
all_domains['entropy'] = [entropy(x) for x in all_domains['domain']]

# COMMAND ----------

# 結果を表示します。エントロピーが高いほどDGAの可能性が高くなりますが、まだ十分ではありません。
display(all_domains)

# COMMAND ----------

# MAGIC %md
# MAGIC 文字列から計算できるエントロピーでもある程度判定できますが、まだ不安です。よし、辞書も活用して、ドメイン名のngram(数文字区切りの断片)を用いてマッチングをします。辞書にマッチしないほど怪しいドメインと考えられます。

# COMMAND ----------

# ここでは適切なドメインに対するn-gramの頻度分析を行うために追加の特徴量エンジニアリングを行います

y = np.array(all_domains['class'].tolist()) # 不思議ですがこれが必要となります 

import sklearn.ensemble
from sklearn import feature_extraction

# 3-5文字のngramから頻度マトリクスを生成
alexa_vc = sklearn.feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(3,5), min_df=1e-4, max_df=1.0)
counts_matrix = alexa_vc.fit_transform(alexa_dataframe['domain'])
alexa_counts = np.log10(counts_matrix.sum(axis=0).getA1())
ngrams_list = alexa_vc.get_feature_names()

# COMMAND ----------

# 辞書の単語をデータフレームにロードします
file_location = default_file_path + "/words.txt"
word_dataframe = pd.read_csv(file_location, header=0, sep=';');
word_dataframe = word_dataframe[word_dataframe['words'].map(lambda x: str(x).isalpha())]
word_dataframe = word_dataframe.applymap(lambda x: str(x).strip().lower())
word_dataframe = word_dataframe.dropna()
word_dataframe = word_dataframe.drop_duplicates()

# COMMAND ----------

word_dataframe

# COMMAND ----------

# 辞書のワードリストから3-5のngramから構成されるディクショナリーを作成します
dict_vc = sklearn.feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(3,5), min_df=1e-5, max_df=1.0)
counts_matrix = dict_vc.fit_transform(word_dataframe['words'])
dict_counts = np.log10(counts_matrix.sum(axis=0).getA1())
ngrams_list = dict_vc.get_feature_names()

# alexaのドメイン名、辞書とのngramレベルでのマッチ数を計算します
def ngram_count(domain):
    alexa_match = alexa_counts * alexa_vc.transform([domain]).T  # ベクトルの掛け算と転置です！
    dict_match = dict_counts * dict_vc.transform([domain]).T
    print('%s Alexa match:%d Dict match: %d' % (domain, alexa_match, dict_match))

# Examples:
ngram_count('beyonce')
ngram_count('dominos')
ngram_count('1cb8a5f36f')
ngram_count('zfjknuh38231')
ngram_count('bey6o4ce')
ngram_count('washington')

# COMMAND ----------

# MAGIC %md
# MAGIC これで辞書とのマッチ度合いも特徴量として追加できます。
# MAGIC 
# MAGIC - Alexaトップ100kリストとディクショナリーからn-gramを作成し、マッチング関数を作成して、テストサンプルを実行します。
# MAGIC - n-gramの詳細についてはこちら: https://blog.xrds.acm.org/2017/10/introduction-n-grams-need/

# COMMAND ----------

all_domains['alexa_grams']= alexa_counts * alexa_vc.transform(all_domains['domain']).T 
all_domains['word_grams']= dict_counts * dict_vc.transform(all_domains['domain']).T 

# COMMAND ----------

# MAGIC %md
# MAGIC ## n-gramのベクトル化モデルを構築
# MAGIC 
# MAGIC モデル構築にはベクトルが必要となります。

# COMMAND ----------

# Alexaに登録されているが、辞書やAlexaのドメインとの一致度合いが低い
weird_cond = (all_domains['class']=='legit') & (all_domains['word_grams']<3) & (all_domains['alexa_grams']<2)
weird = all_domains[weird_cond]
print(weird.shape[0])
all_domains.loc[weird_cond, 'class'] = 'weird'
print(all_domains['class'].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC 数字だけだとわかりにくいのでグラフを作成します。

# COMMAND ----------

display(all_domains)

# COMMAND ----------

# MAGIC %md
# MAGIC **Thinking Time #3**
# MAGIC 
# MAGIC 脅威ドメインを検知するモデルをトレーニングすることになるが、トレーニングしたモデルをどう管理したらいいのだろうか？
# MAGIC 
# MAGIC 1. ノートにメモる
# MAGIC 2. MLflowで自動ロギング
# MAGIC 3. Excel万能説

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルのトレーニング
# MAGIC 
# MAGIC ここまでの特徴量をまとめてモデルのトレーニングを行います。
# MAGIC 
# MAGIC - ドメイン名の不自然さに基づいてドメインのラベリングを行います。
# MAGIC - MLランタイムを使用していれば、パッケージは事前インストールされています。
# MAGIC - MLflowを用いることで繰り返し過程でのエクスペリメントをトラッキングできます。

# COMMAND ----------

from sklearn.model_selection import train_test_split
clf = sklearn.ensemble.RandomForestClassifier(n_estimators=20) # フォレストの木の数

# 学習に適さないweirdとラベリングされたデータを除外
not_weird = all_domains[all_domains['class'] != 'weird']
X = not_weird[['length', 'entropy', 'alexa_grams', 'word_grams']].values

# ラベル (scikit learnは分類ラベルに 'y' を使います)
y = np.array(not_weird['class'].tolist())

mlflow.sklearn.autolog()

with mlflow.start_run():
  # 80/20 スプリットでトレーニングします
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  # 予測パフォーマンスを見る前にまず全体に対してトレーニングを行います
  #clf.fit(X, y)

# COMMAND ----------

# MLflowを用いて我々のコンテンツライブラリにおけるモデルの位置を特定します
run_id = mlflow.search_runs()['run_id'][0]
model_uri = 'runs:/' + run_id + '/model'

# COMMAND ----------

# このランのランIDを取得します
run_id

# COMMAND ----------

# MAGIC %md
# MAGIC このモデルは後でも使うので、関数化して簡単に呼び出せるようにしておきます。
# MAGIC - 後でDFA予測を行うために使う予測関数を構築します。
# MAGIC - 予測関数に対して、事前/事後処理を追加します。

# COMMAND ----------

import mlflow.pyfunc

class vc_transform(mlflow.pyfunc.PythonModel):
    def __init__(self, alexa_vc, dict_vc, ctx):
        self.alexa_vc = alexa_vc
        self.dict_vc = dict_vc
        self.ctx = ctx

    def predict(self, context, model_input):
        _alexa_match = alexa_counts * self.alexa_vc.transform([model_input]).T  
        _dict_match = dict_counts * self.dict_vc.transform([model_input]).T
        _X = [len(model_input), entropy(model_input), _alexa_match, _dict_match]
        return str(self.ctx.predict([_X])[0])

# COMMAND ----------

# MAGIC %md
# MAGIC これでモデルが出来上がりました。例えば、いかにも怪しい`7ydbdehaaz`というドメイン名はどうなるでしょうか。

# COMMAND ----------

# トレーニングしたモデルを使用します 
vc_model = vc_transform(alexa_vc, dict_vc, clf)
vc_model.predict(mlflow.pyfunc.PythonModel, '7ydbdehaaz')

# COMMAND ----------

# MAGIC %md
# MAGIC # ニアリアルタイムのストリーミング分析
# MAGIC 
# MAGIC これでバッチ推論への対応は十分です。しかし、DNSの情報はリアルタイムで発生している。これに対応できるようにするにはどうすればいいでしょうか。
# MAGIC 
# MAGIC - 脅威インテリジェンスでデータを補強し、分析と補強を用いて脅威のあるアクティビティを検知します。

# COMMAND ----------

# MAGIC %md
# MAGIC **Thinking Time #4**
# MAGIC 
# MAGIC これでバッチ処理を用いて脅威のあるURLを特定することはできた。しかし、脅威には迅速なる対応が必要だ。いったいどうしたらいいのだろうか？
# MAGIC 
# MAGIC 1. 人が張り付いて処理を実行し続ける
# MAGIC 1. バッチ処理をスケジューリングする
# MAGIC 1. Apache Sparkのストリーミング処理を活用し、pDNSのストリーミングをニアリアルタイムで処理する

# COMMAND ----------

# DGAモデルをロードします。これは投入されるDNSイベントをエンリッチするために使用する事前学習済みモデルです。後のステップでこのモデルをどのようにトレーニングするのかを説明します
import mlflow
import mlflow.pyfunc

model_path = f'dbfs:{work_path}/tables/model'
loaded_model = mlflow.pyfunc.load_model(model_path)
spark.udf.register("ioc_detect", loaded_model.predict)

# COMMAND ----------

# pDNSのスキーマ定義 
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType, StringType, ArrayType

schema = StructType([
    StructField("rrname", StringType(), True),
    StructField("rrtype", StringType(), True),
    StructField("time_first", LongType(), True),
    StructField("time_last", LongType(), True),
    StructField("count", LongType(), True),
    StructField("bailiwick", StringType(), True),
    StructField("rdata", ArrayType(StringType(), True), True)
])


# COMMAND ----------

# テストデータセットのロード
df = spark.readStream.format("json").schema(schema).load(f"dbfs:{work_path}/tables/datasets/latest")

# COMMAND ----------

# テストデータセットに対する一時テーブルの作成 
df.createOrReplaceTempView("dns_latest")

# COMMAND ----------

#　ビジュアライズを通じた調査、1行目、3行目が怪しいことがわかります
display(df)

# COMMAND ----------

# MAGIC %sql -- ここでDGA検知を行います
# MAGIC -- DNSデータをスコアリングするためのビューを作成します
# MAGIC CREATE
# MAGIC OR REPLACE TEMPORARY VIEW dns_latest_scoring AS
# MAGIC SELECT
# MAGIC   rdata,
# MAGIC   count,
# MAGIC   rrname,
# MAGIC   bailiwick,
# MAGIC   rrtype,
# MAGIC   time_last,
# MAGIC   time_first,
# MAGIC   ioc_detect(domain_extract(rrname)) as isioc,
# MAGIC   domain_extract(dns_latest.rrname) domain
# MAGIC from
# MAGIC   dns_latest

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.1 DNSイベントストリームにおける脅威の発見
# MAGIC 
# MAGIC これでストリームに対してリアルタイムで脅威を検知する仕組みが整いました。ストリームデータに対する処理を行い、`ioc`と判別されているものはあるかを確認します。

# COMMAND ----------

# MAGIC %sql
# MAGIC Select * from dns_latest_scoring where isioc = 'ioc'

# COMMAND ----------

# MAGIC %md
# MAGIC 怪しいドメインが引っかかりました。URLHausのデータを使って詳細を確認してみましょう。
# MAGIC 
# MAGIC - PhishingあるいはTyposquating?
# MAGIC - ここでtyposquattingの検知を行います
# MAGIC - dnstwistを使うことで、怪しいgoogleeのようなドメインを検知できます

# COMMAND ----------

# MAGIC %sql
# MAGIC Select
# MAGIC   EnrichedTwistedDomainBrand.*
# MAGIC FROM
# MAGIC   dns_latest_scoring,
# MAGIC   EnrichedTwistedDomainBrand
# MAGIC Where
# MAGIC   EnrichedTwistedDomainBrand.dnstwisted_domain = dns_latest_scoring.domain

# COMMAND ----------

# MAGIC %md
# MAGIC 以下のラインではモデルを適用しています。
# MAGIC - 不正なドメインの検知
# MAGIC - アラート用テーブルの作成

# COMMAND ----------

dns_stream_iocs = spark.sql("Select * from dns_latest_scoring where isioc = 'ioc'")
dbutils.fs.rm(f'dbfs:{work_path}/datasets/gold/delta/DNS_IOC_Latest', True)

# 検知結果を保存できるようにしておこう。
dns_stream_iocs.writeStream.format("delta").outputMode("append").option("checkpointLocation", f"dbfs:{work_path}/datasets/gold/delta/_checkpoints/DNS_IOC_Latest").start(f"dbfs:{work_path}/datasets/gold/delta/DNS_IOC_Latest")

# COMMAND ----------

res = spark.sql(f"SELECT * FROM delta.`{work_path}/datasets/gold/delta/DNS_IOC_Latest`")
display(res)

# COMMAND ----------

# MAGIC %md
# MAGIC URLHausを用いて、このrrname(リソースリクエスト名)がなんなのかをチェックします。

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 不正なドメインを特定したので、我々が補強した脅威インテリジェンスフィードがこのドメインで活動しているのかをみてみましょう
# MAGIC select * from EnrichedThreatFeeds where EnrichedThreatFeeds.domain = domain_extract('ns1.asdklgb.cf.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent Tesla
# MAGIC 
# MAGIC ミッション成功です!!!
# MAGIC 
# MAGIC - ストリーミングDNSイベントに対してDGA検知モデルを適用しました。 
# MAGIC - DNSログから怪しいドメイン(ioc)を特定しました。
# MAGIC - URLHausを用いてiocデータを補強しました
# MAGIC - このDGAドメインがAgent Teslaに貢献していることを確認できます。
# MAGIC 
# MAGIC [さまざまな情報窃取で利用されるトロイの木馬型マルウェア Agent Tesla – Sophos News](https://news.sophos.com/ja-jp/2021/02/11/agent-tesla-amps-up-information-stealing-attacks-jp/)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
