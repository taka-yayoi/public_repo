# Databricks notebook source
# MAGIC %md
# MAGIC # 大規模特徴量エンジニアリング
# MAGIC 
# MAGIC **注意**
# MAGIC - クラスターライブラリとして、PyPI経由で`datasketch`をインストールしてください。

# COMMAND ----------

# MAGIC %md
# MAGIC StoreSalesなどモジュール化された特徴量クラス

# COMMAND ----------

# MAGIC %run ./feature_dict

# COMMAND ----------

# MAGIC %md
# MAGIC ベースのデータフレームに特徴量追加する`append_features`

# COMMAND ----------

# MAGIC %run ./factory

# COMMAND ----------

# MAGIC %md
# MAGIC ヘルパー関数

# COMMAND ----------

# MAGIC %run ./functions

# COMMAND ----------

# MAGIC %md 
# MAGIC ## TPC-DSからソーステーブルの準備
# MAGIC 
# MAGIC このデモノートブックを実行する前に、[tpcds_datagenノートブック]($../tpcds_datagen)を用いてTCP-DSからソースDeltaテーブルを作成する必要があります。

# COMMAND ----------

# tpcds_datagenノートブックを用いて生成したdeltaテーブル/格納場所
store_sales_delta_path = "/tmp/takaaki.yayoi@databricks.com/sales_store_tpcds"
src_df = spark.table(f"delta.`{store_sales_delta_path}`")

# COMMAND ----------

display(src_df)

# COMMAND ----------

features = StoreSales()
# total_sales に i_category と month_id を掛け合わせて特徴量を増幅
fv_months = features.total_sales.multiply("i_category", ["Music", "Home", "Shoes"]).multiply("month_id", [200012, 200011, 200010])
# ベースのデータフレームに追加
# 以下の例ではcustomer_idでグルーピングしています
df = append_features(src_df, [features.collector], fv_months)

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## One hotエンコーディング

# COMMAND ----------

src_df = spark.createDataFrame([(1, "iphone"), (2, "samsung"), (3, "htc"), (4, "vivo")], ["user_id", "device_type"])
encode = Feature(_name="device_type_encode", _base_col=f.lit(1), _negative_value=0)
onehot_encoder = encode.multiply("device_type", ["iphone", "samsung", "htc", "vivo"])
df = append_features(src_df, ["device_type"], onehot_encoder)

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## MinHash
# MAGIC 
# MAGIC Minhashの実装は https://pypi.org/project/datasketch/ からインストールする必要があります。
# MAGIC 
# MAGIC Minhashは膨大なカウントを推定するのには有用ですが、ここではMinHashがどのように適用できるのかをデモするために小規模なソースデータセットのみを使用します。

# COMMAND ----------

from hashlib import sha1
from datasketch.minhash import MinHash
from datasketch.lean_minhash import LeanMinHash
import numpy as np

@pandas_udf("binary")
def create_minhash(v: pd.Series) -> bytearray:
  mh = MinHash(num_perm=64)
  for val in v:
    if val: mh.update(val.encode('utf8'))
  lean_minhash = LeanMinHash(mh)
  buf = bytearray(lean_minhash.bytesize())
  lean_minhash.serialize(buf)
  return buf

# COMMAND ----------

@pandas_udf("integer")
def minhash_cardinality(mh: pd.Series) -> pd.Series:
  result = []
  for buf in mh:
    lmh = LeanMinHash.deserialize(buf)
    result.append(round(lmh.count()))
  return pd.Series(result)

# COMMAND ----------

@pandas_udf("integer")
def minhash_intersect(mh1: pd.Series, mh2: pd.Series) -> pd.Series:
  result = []
  N = len(mh1)
  for i in range(N):
    buf1, buf2 = mh1[i], mh2[i]
    lmh1 = LeanMinHash.deserialize(buf1)
    lmh2 = LeanMinHash.deserialize(buf2)
    jac = lmh1.jaccard(lmh2)
    lmh1.merge(lmh2)
    intersect_cnt = jac * lmh1.count()
    result.append(round(intersect_cnt))
  return pd.Series(result)

# COMMAND ----------

src_df = spark.table(f"delta.`{store_sales_delta_path}`")

# COMMAND ----------

src_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC エクササイズとして、`create_minhash` がどのように `BaseFeature`の`_agg_func`として定義されるのかを確認することができます。

# COMMAND ----------

import pyspark.sql.functions as f
grocery_1m_trans_df = src_df.groupby("ss_customer_sk").agg(
  create_minhash(f.when(f.col("i_category")=="Music", f.concat("ss_ticket_number", "d_date")).otherwise(None)).alias("music_trans_minhash"),
  create_minhash(f.when(f.col("month_id")==200012, f.concat("ss_ticket_number", "d_date")).otherwise(None)).alias("trans_1m_minhash")
)

# COMMAND ----------

result_df = grocery_1m_trans_df.select("ss_customer_sk", minhash_intersect("music_trans_minhash", "trans_1m_minhash").alias("total_trans_music_1m"))

# COMMAND ----------

display(result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
