# Databricks notebook source
# MAGIC %md
# MAGIC # 10分でわかるpandasからSparkのpandas APIへの移行

# COMMAND ----------

# MAGIC %md
# MAGIC ## pandasからSparkのpandas APIへの移行

# COMMAND ----------

# MAGIC %md
# MAGIC ### オブジェクトの作成

# COMMAND ----------

import numpy as np
import pandas as pd
import pyspark.pandas as ps

# COMMAND ----------

# pandasのシリーズの作成
pser = pd.Series([1, 3, 5, np.nan, 6, 8]) 
# Sparkのpandasシリーズの作成
psser = ps.Series([1, 3, 5, np.nan, 6, 8])
# pandasのシリーズを渡してSparkのpandasシリーズを作成
psser = ps.Series(pser)
psser = ps.from_pandas(pser)

# COMMAND ----------

pser

# COMMAND ----------

psser

# COMMAND ----------

psser.sort_index()

# COMMAND ----------

# pandasデータフレームの作成
pdf = pd.DataFrame({'A': np.random.rand(5),
                    'B': np.random.rand(5)})
# Sparkのpandasデータフレームの作成
psdf = ps.DataFrame({'A': np.random.rand(5),
                     'B': np.random.rand(5)})
# pandasデータフレームを渡してSparkのpandasデータフレームを作成
psdf = ps.DataFrame(pdf)
psdf = ps.from_pandas(pdf)

# COMMAND ----------

pdf

# COMMAND ----------

psdf.sort_index()

# COMMAND ----------

# MAGIC %md
# MAGIC ### データの参照

# COMMAND ----------

psdf.head(2)

# COMMAND ----------

psdf.describe()

# COMMAND ----------

psdf.sort_values(by='B')

# COMMAND ----------

psdf.transpose()

# COMMAND ----------

ps.get_option('compute.max_rows')

# COMMAND ----------

ps.set_option('compute.max_rows', 2000)
ps.get_option('compute.max_rows')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 選択

# COMMAND ----------

psdf['A']  # あるいは psdf.A

# COMMAND ----------

psdf[['A', 'B']]

# COMMAND ----------

psdf.loc[1:2]

# COMMAND ----------

psdf.iloc[:3, 1:2]

# COMMAND ----------

psser = ps.Series([100, 200, 300, 400, 500], index=[0, 1, 2, 3, 4])
# 以下のコメントアウトされている`psdf['C'] = psser`の行はエラーとなります。これは、Sparkのpandasが、
# コストの高いJOIN操作を伴う列の追加(異なるデータフレームやシリーズをSparkのpandasデータフレームに追加)を禁じているためです。
# 設定の`compute.ops_on_diff_frames`を`True`にすることで、この操作を有効化できます。
# 詳細に関しては、以下のブログ記事をご覧ください。
# https://databricks.com/blog/2020/03/31/10-minutes-from-pandas-to-koalas-on-apache-spark.html
# psdf['C'] = psser

# COMMAND ----------

# これらはオプションを管理するために必要になります
from pyspark.pandas.config import set_option, reset_option
set_option("compute.ops_on_diff_frames", True)
psdf['C'] = psser
# 今後起こり得る高コストなオペレーションを避けるためにデフォルトにリセットします
reset_option("compute.ops_on_diff_frames")
psdf

# COMMAND ----------

# MAGIC %md
# MAGIC ### SparkのpandasオブジェクトにPython関数をapply

# COMMAND ----------

psdf.apply(np.cumsum)

# COMMAND ----------

psdf.apply(np.cumsum, axis=1)

# COMMAND ----------

psdf.apply(lambda x: x ** 2)

# COMMAND ----------

def square(x) -> ps.Series[np.float64]:
    return x ** 2

# COMMAND ----------

psdf.apply(square)

# COMMAND ----------

# データのサイズが compute.shortcut_limit (1000) 以下なので適切に動作します
ps.DataFrame({'A': range(1000)}).apply(lambda col: col.max())

# COMMAND ----------

# データのサイズが compute.shortcut_limit (1000) より大きいのでうまく動きません
ps.DataFrame({'A': range(1001)}).apply(lambda col: col.max())

# COMMAND ----------

ps.set_option('compute.shortcut_limit', 1001)
ps.DataFrame({'A': range(1001)}).apply(lambda col: col.max())

# COMMAND ----------

# MAGIC %md
# MAGIC ### データのグルーピング

# COMMAND ----------

psdf.groupby('A').sum()

# COMMAND ----------

psdf.groupby(['A', 'B']).sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ### プロット

# COMMAND ----------

# ノートブック上でプロットするためにこれが必要となります
%matplotlib inline

# COMMAND ----------

speed = [0.1, 17.5, 40, 48, 52, 69, 88]
lifespan = [2, 8, 70, 1.5, 25, 12, 28]
index = ['snail', 'pig', 'elephant',
         'rabbit', 'giraffe', 'coyote', 'horse']
psdf = ps.DataFrame({'speed': speed,
                     'lifespan': lifespan}, index=index)
psdf.plot.bar()

# COMMAND ----------

psdf.plot.barh()

# COMMAND ----------

psdf = ps.DataFrame({'mass': [0.330, 4.87, 5.97],
                     'radius': [2439.7, 6051.8, 6378.1]},
                    index=['Mercury', 'Venus', 'Earth'])
psdf.plot.pie(y='mass')

# COMMAND ----------

psdf = ps.DataFrame({
    'sales': [3, 2, 3, 9, 10, 6, 3],
    'signups': [5, 5, 6, 12, 14, 13, 9],
    'visits': [20, 42, 28, 62, 81, 50, 90],
}, index=pd.date_range(start='2019/08/15', end='2020/03/09',
                       freq='M'))
psdf.plot.area()

# COMMAND ----------

psdf = ps.DataFrame({'pig': [20, 18, 489, 675, 1776],
                     'horse': [4, 25, 281, 600, 1900]},
                    index=[1990, 1997, 2003, 2009, 2014])
psdf.plot.line()

# COMMAND ----------

pdf = pd.DataFrame(
    np.random.randint(1, 7, 6000),
    columns=['one'])
pdf['two'] = pdf['one'] + np.random.randint(1, 7, 6000)
psdf = ps.from_pandas(pdf)
psdf.plot.hist(bins=12, alpha=0.5)

# COMMAND ----------

psdf = ps.DataFrame([[5.1, 3.5, 0], [4.9, 3.0, 0], [7.0, 3.2, 1],
                    [6.4, 3.2, 1], [5.9, 3.0, 2]],
                   columns=['length', 'width', 'species'])
psdf.plot.scatter(x='length',
                  y='width',
                  c='species')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sparkのpandas APIでサポートしていない機能と、ワークアラウンド

# COMMAND ----------

# MAGIC %md
# MAGIC ### 型変換を通じたpandas APIの直接の利用

# COMMAND ----------

psidx = psdf.index

# COMMAND ----------

# Index.to_list()はPandasNotImplementedErrorを引き起こします。
# これは全てのデータをクライアント(ドライバーノード)に集める必要があるため、Sparkのpandas APIではこれをサポートしていません。
# シンプルなワークアラウンドは、to_pandas()を用いてpandasに変換するというものです。
# 詳細を知りたい場合には、以下のブログ記事を参照ください。
# https://databricks.com/blog/2020/03/31/10-minutes-from-pandas-to-koalas-on-apache-spark.html
# psidx.to_list()

# COMMAND ----------

psidx.to_pandas().to_list()

# COMMAND ----------

# MAGIC %md
# MAGIC ### pandasオブジェクトのネイティブサポート
# MAGIC 
# MAGIC Timestamp()`などはまだ実装されていないため、pandasのものを利用する必要があります。

# COMMAND ----------

psdf = ps.DataFrame({'A': 1.,
                     'B': pd.Timestamp('20130102'),
                     'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                     'D': np.array([3] * 4, dtype='int32'),
                     'F': 'foo'})

# COMMAND ----------

psdf

# COMMAND ----------

# MAGIC %md
# MAGIC ### pandas関数の分散実行

# COMMAND ----------

i = pd.date_range('2018-04-09', periods=2000, freq='1D1min')
ts = ps.DataFrame({'A': ['timestamp']}, index=i)

# DataFrame.between_time()はSparkのpandas APIでは未実装です。
# シンプルなワークアラウンドはto_pandas()を使ってpandasに変換し、関数を適用するというものです。
# 詳細を知りたい場合には、以下のブログ記事を参照ください。
# https://databricks.com/blog/2020/03/31/10-minutes-from-pandas-to-koalas-on-apache-spark.html
# ts.between_time('0:15', '0:16')

# COMMAND ----------

ts.to_pandas().between_time('0:15', '0:16')

# COMMAND ----------

ts.pandas_on_spark.apply_batch(func=lambda pdf: pdf.between_time('0:15', '0:16'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sparkのpandas APIでのSQLの使用
# MAGIC 
# MAGIC `ps.sql`を用いてSQLを発行することができます。

# COMMAND ----------

psdf = ps.DataFrame({'year': [1990, 1997, 2003, 2009, 2014],
                     'pig': [20, 18, 489, 675, 1776],
                     'horse': [4, 25, 281, 600, 1900]})

# COMMAND ----------

ps.sql("SELECT * FROM {psdf} WHERE pig > 100")

# COMMAND ----------

pdf = pd.DataFrame({'year': [1990, 1997, 2003, 2009, 2014],
                    'sheep': [22, 50, 121, 445, 791],
                    'chicken': [250, 326, 589, 1241, 2118]})

# COMMAND ----------

ps.sql('''
    SELECT ps.pig, pd.chicken
    FROM {psdf} ps INNER JOIN {pdf} pd
    ON ps.year = pd.year
    ORDER BY ps.pig, pd.chicken''')

# COMMAND ----------

# MAGIC %md
# MAGIC ## PySparkとの連携

# COMMAND ----------

# MAGIC %md
# MAGIC ### PySparkデータフレームへの変換

# COMMAND ----------

psdf = ps.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
sdf = psdf.to_spark()
type(sdf)

# COMMAND ----------

sdf.show()

# COMMAND ----------

from pyspark.pandas import option_context
with option_context(
        "compute.default_index_type", "distributed-sequence"):
    psdf = sdf.to_pandas_on_spark()
type(psdf)

# COMMAND ----------

psdf

# COMMAND ----------

sdf.to_pandas_on_spark(index_col='A')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Spark実行計画の確認

# COMMAND ----------

from pyspark.pandas import option_context

with option_context(
        "compute.ops_on_diff_frames", True,
        "compute.default_index_type", 'distributed'):
    df = ps.range(10) + ps.range(10)
    df.spark.explain()

# COMMAND ----------

with option_context(
        "compute.ops_on_diff_frames", False,
        "compute.default_index_type", 'distributed'):
    df = ps.range(10)
    df = df + df
    df.spark.explain()

# COMMAND ----------

# MAGIC %md
# MAGIC ### データフレームのキャッシュ

# COMMAND ----------

with option_context("compute.default_index_type", 'distributed'):
    df = ps.range(10)
    new_df = (df + df).spark.cache()  # `(df + df)` は、ここでは `df` としてキャッシュされます
    new_df.spark.explain()

# COMMAND ----------

new_df.spark.unpersist()

# COMMAND ----------

with (df + df).spark.cache() as df:
    df.spark.explain()

# COMMAND ----------

# MAGIC %md
# MAGIC # END
