# Databricks notebook source
# MAGIC %md # 10分でpandasからKoalasに移行

# COMMAND ----------

# MAGIC %md ## pandasからKoalasへの移行

# COMMAND ----------

# MAGIC %md ### オブジェクトの作成

# COMMAND ----------

import numpy as np
import pandas as pd
import databricks.koalas as ks

# COMMAND ----------

# pandasのシリーズの作成
pser = pd.Series([1, 3, 5, np.nan, 6, 8]) 
# Koalasのシリーズの作成
kser = ks.Series([1, 3, 5, np.nan, 6, 8])
# pansasシリーズを渡してKoalasシリーズを作成
kser = ks.Series(pser)
kser = ks.from_pandas(pser)

# COMMAND ----------

pser

# COMMAND ----------

kser

# COMMAND ----------

kser.sort_index()

# COMMAND ----------

# pandasデータフレームの作成
pdf = pd.DataFrame({'A': np.random.rand(5),
                    'B': np.random.rand(5)})
# Koalasデータフレームの作成
kdf = ks.DataFrame({'A': np.random.rand(5),
                    'B': np.random.rand(5)})
# pandasデータフレームを渡してKoalasデータフレームを作成
kdf = ks.DataFrame(pdf)
kdf = ks.from_pandas(pdf)

# COMMAND ----------

pdf

# COMMAND ----------

kdf.sort_index()

# COMMAND ----------

# MAGIC %md ### データの参照

# COMMAND ----------

kdf.head(2)

# COMMAND ----------

kdf.describe()

# COMMAND ----------

kdf.sort_values(by='B')

# COMMAND ----------

kdf.transpose()

# COMMAND ----------

# MAGIC %md ### オプションの設定
# MAGIC 
# MAGIC [Options and settings — Koalas 1\.7\.0 documentation](https://koalas.readthedocs.io/en/latest/user_guide/options.html)
# MAGIC 
# MAGIC **compute.max_rows**
# MAGIC > `compute.max_rows`は現在のKoalasデータフレームのリミットを設定します。Noneに設定した場合、入力長に制限を設けません。リミットが設定されている場合は、データをドライバーノードに集め、pandas APIを使用するショートカットの処理を行います。リミットが設定されていない場合には、PySparkによって処理が行われます。デフォルトは1000です。

# COMMAND ----------

from databricks.koalas.config import set_option, get_option
ks.get_option('compute.max_rows')

# COMMAND ----------

ks.set_option('compute.max_rows', 2000)
ks.get_option('compute.max_rows')

# COMMAND ----------

# MAGIC %md ### 列の選択

# COMMAND ----------

kdf['A']  # あるいは kdf.A

# COMMAND ----------

kdf[['A', 'B']]

# COMMAND ----------

kdf.loc[1:2]

# COMMAND ----------

kdf.iloc[:3, 1:2]

# COMMAND ----------

# MAGIC %md
# MAGIC 以下の`kdf['C'] = kser`の行はエラーとなります。これは、Koalasが、コストの高いJOIN操作を伴う列の追加(異なるデータフレームやシリーズをKoalasのデータフレームに追加)を禁じているためです。設定の`compute.ops_on_diff_frames`を`True`にすることで、この操作を有効化できます。
# MAGIC 
# MAGIC 詳細を知りたい場合には、こちらの[ブログ記事](https://databricks.com/blog/2020/03/31/10-minutes-from-pandas-to-koalas-on-apache-spark.html)を参照ください。

# COMMAND ----------

kser = ks.Series([100, 200, 300, 400, 500], index=[0, 1, 2, 3, 4])
#kdf['C'] = kser # エラーになります

# COMMAND ----------

# Those are needed for managing options
from databricks.koalas.config import set_option, reset_option
set_option("compute.ops_on_diff_frames", True)
kdf['C'] = kser
# Reset to default to avoid potential expensive operation in the future
reset_option("compute.ops_on_diff_frames")
kdf

# COMMAND ----------

# MAGIC %md ### KoalasオブジェクトへのPython関数の適用

# COMMAND ----------

kdf.apply(np.cumsum)

# COMMAND ----------

kdf.apply(np.cumsum, axis=1)

# COMMAND ----------

kdf.apply(lambda x: x ** 2)

# COMMAND ----------

def square(x) -> ks.Series[np.float64]:
    return x ** 2

# COMMAND ----------

kdf.apply(square)

# COMMAND ----------

# MAGIC %md
# MAGIC [Options and settings — Koalas 1\.7\.0 documentation](https://koalas.readthedocs.io/en/latest/user_guide/options.html#available-options)
# MAGIC 
# MAGIC **compute.shortcut_limit**
# MAGIC > `compute.shortcut_limit`はショートカットのリミットを設定します。指定された行数とスキーマを用いて計算を行います。データフレームの長さがこのリミットより大きい場合、PySparkを使用して計算を行います。

# COMMAND ----------

# データサイズが compute.shortcut_limit (1000) 以下なので正しく動作します
ks.DataFrame({'A': range(1000)}).apply(lambda col: col.max())

# COMMAND ----------

# データサイズが compute.shortcut_limit (1000) より大きいため正しく動作しません
ks.DataFrame({'A': range(1001)}).apply(lambda col: col.max())

# COMMAND ----------

# compute.shortcut_limit を変更します
ks.set_option('compute.shortcut_limit', 1001)
ks.DataFrame({'A': range(1001)}).apply(lambda col: col.max())

# COMMAND ----------

# MAGIC %md ### データのグルーピング

# COMMAND ----------

# A列でのグルーピング
kdf.groupby('A').sum()

# COMMAND ----------

# A、B列でのグルーピング
kdf.groupby(['A', 'B']).sum()

# COMMAND ----------

# MAGIC %md ### データのプロット

# COMMAND ----------

# ノートブック上でプロットするためにインラインの設定が必要です
%matplotlib inline

# COMMAND ----------

speed = [0.1, 17.5, 40, 48, 52, 69, 88]
lifespan = [2, 8, 70, 1.5, 25, 12, 28]
index = ['snail', 'pig', 'elephant',
         'rabbit', 'giraffe', 'coyote', 'horse']
kdf = ks.DataFrame({'speed': speed,
                   'lifespan': lifespan}, index=index)
kdf.plot.bar()

# COMMAND ----------

kdf.plot.barh()

# COMMAND ----------

kdf = ks.DataFrame({'mass': [0.330, 4.87, 5.97],
                    'radius': [2439.7, 6051.8, 6378.1]},
                   index=['Mercury', 'Venus', 'Earth'])
kdf.plot.pie(y='mass')

# COMMAND ----------

kdf = ks.DataFrame({
    'sales': [3, 2, 3, 9, 10, 6, 3],
    'signups': [5, 5, 6, 12, 14, 13, 9],
    'visits': [20, 42, 28, 62, 81, 50, 90],
}, index=pd.date_range(start='2019/08/15', end='2020/03/09',
                       freq='M'))
kdf.plot.area()

# COMMAND ----------

kdf = ks.DataFrame({'pig': [20, 18, 489, 675, 1776],
                    'horse': [4, 25, 281, 600, 1900]},
                   index=[1990, 1997, 2003, 2009, 2014])
kdf.plot.line()

# COMMAND ----------

kdf = pd.DataFrame(
    np.random.randint(1, 7, 6000),
    columns=['one'])
kdf['two'] = kdf['one'] + np.random.randint(1, 7, 6000)
kdf = ks.from_pandas(kdf)
kdf.plot.hist(bins=12, alpha=0.5)

# COMMAND ----------

kdf = ks.DataFrame([[5.1, 3.5, 0], [4.9, 3.0, 0], [7.0, 3.2, 1],
                    [6.4, 3.2, 1], [5.9, 3.0, 2]],
                   columns=['length', 'width', 'species'])
kdf.plot.scatter(x='length',
                 y='width',
                 c='species',
                 colormap='viridis')

# COMMAND ----------

# MAGIC %md ## Koalasに無い機能、及びワークアラウンド

# COMMAND ----------

# MAGIC %md ### pandasオブジェクトのネイティブサポート
# MAGIC 
# MAGIC `ks.Timestamp()`などはまだ実装されていないため、pandasのものを利用する必要があります。

# COMMAND ----------

kdf = ks.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'F': 'foo'})

# COMMAND ----------

kdf

# COMMAND ----------

# MAGIC %md ### pandas関数の分散実行
# MAGIC 
# MAGIC `DataFrame.between_time()`はKoalasでは未実装です。シンプルなワークアラウンドは`to_pandas()`を用いてpandasデータフレームに変換し、関数を適用するというものです。

# COMMAND ----------

i = pd.date_range('2018-04-09', periods=2000, freq='1D1min')
ts = ks.DataFrame({'A': ['timestamp']}, index=i)
#ts.between_time('0:15', '0:16') # エラーとなります

# COMMAND ----------

ts.to_pandas().between_time('0:15', '0:16')

# COMMAND ----------

ts.map_in_pandas(func=lambda pdf: pdf.between_time('0:15', '0:16'))

# COMMAND ----------

# MAGIC %md ### KoalasでのSQLの使用
# MAGIC 
# MAGIC `ks.sql`を用いてSQLを発行することができます。

# COMMAND ----------

kdf = ks.DataFrame({'year': [1990, 1997, 2003, 2009, 2014],
                    'pig': [20, 18, 489, 675, 1776],
                    'horse': [4, 25, 281, 600, 1900]})

# COMMAND ----------

ks.sql("SELECT * FROM {kdf} WHERE pig > 100")

# COMMAND ----------

pdf = pd.DataFrame({'year': [1990, 1997, 2003, 2009, 2014],
                    'sheep': [22, 50, 121, 445, 791],
                    'chicken': [250, 326, 589, 1241, 2118]})

# COMMAND ----------

ks.sql('''
    SELECT ks.pig, pd.chicken
    FROM {kdf} ks INNER JOIN {pdf} pd
    ON ks.year = pd.year
    ORDER BY ks.pig, pd.chicken''')

# COMMAND ----------

# MAGIC %md ## PySparkとの連携

# COMMAND ----------

# MAGIC %md ### PySpark DataFrameとの変換

# COMMAND ----------

kdf = ks.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
sdf = kdf.to_spark()
type(sdf)

# COMMAND ----------

sdf.show()

# COMMAND ----------

from databricks.koalas import option_context
with option_context(
        "compute.default_index_type", "distributed-sequence"):
    kdf = sdf.to_koalas()
type(kdf)

# COMMAND ----------

kdf

# COMMAND ----------

sdf.to_koalas(index_col='A')

# COMMAND ----------

# MAGIC %md ### Spark実行計画の確認

# COMMAND ----------

from databricks.koalas import option_context

with option_context(
        "compute.ops_on_diff_frames", True,
        "compute.default_index_type", 'distributed'):
    df = ks.range(10) + ks.range(10)
    df.explain()

# COMMAND ----------

with option_context(
        "compute.ops_on_diff_frames", False,
        "compute.default_index_type", 'distributed'):
    df = ks.range(10)
    df = df + df
    df.explain()

# COMMAND ----------

# MAGIC %md ### データフレームのキャッシュ

# COMMAND ----------

with option_context("compute.default_index_type", 'distributed'):
    df = ks.range(10)
    new_df = (df + df).cache()  # `(df + df)` は `df` としてキャッシュされます
    new_df.explain()

# COMMAND ----------

new_df.unpersist()

# COMMAND ----------

with (df + df).cache() as df:
    df.explain()

# COMMAND ----------

# MAGIC %md
# MAGIC # END
