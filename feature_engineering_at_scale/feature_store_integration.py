# Databricks notebook source
# MAGIC %md
# MAGIC # 特徴量テーブルとのインテグレーション

# COMMAND ----------

# MAGIC %md
# MAGIC モジュール化された特徴量クラス

# COMMAND ----------

# MAGIC %run ./core_feature_factory/feature_dict

# COMMAND ----------

# MAGIC %run ./core_feature_factory/factory

# COMMAND ----------

# MAGIC %run ./core_feature_factory/functions

# COMMAND ----------

# MAGIC %md 
# MAGIC ## TPC-DSからソーステーブルの準備
# MAGIC 
# MAGIC このデモノートブックを実行する前に、tpcds_datagenノートブックを用いてTCP-DSからソースDeltaテーブルを作成する必要があります。

# COMMAND ----------

#store_sales_delta_path = "tpcds_datagenノートブックを用いて生成したdeltaテーブル/格納場所"
#e.g. store_sales_delta_path = "/tmp/takaaki.yayoi@databricks.com/sales_store_tpcds"
store_sales_delta_path = "/tmp/takaaki.yayoi@databricks.com/sales_store_tpcds"

# COMMAND ----------

src_df = spark.table(f"delta.`{store_sales_delta_path}`")

# COMMAND ----------

features = StoreSales()
fv_months = features.total_sales.multiply("i_category", ["Music", "Home", "Shoes"]).multiply("month_id", [200012, 200011, 200010])
df = append_features(src_df, [features.collector], fv_months)

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 特徴量の作成

# COMMAND ----------

from databricks.feature_store import feature_table

@feature_table
def compute_customer_features(data):
  features = StoreSales()
  fv_months = features.total_sales.multiply("i_category", ["Music", "Home", "Shoes"]).multiply("month_id", [200012, 200011, 200010])
  df = append_features(src_df, [features.collector], fv_months)
  return df

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()
customer_features_df = compute_customer_features(src_df)

# COMMAND ----------

display(customer_features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 特徴量テーブルへの登録

# COMMAND ----------

from databricks import feature_store

fs = feature_store.FeatureStoreClient()
fs.create_feature_table(
    name="taka_jumpstart_db.customer_features",
    keys=["customer_id"],
    features_df=customer_features_df,
    description="customer feature table",
)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from taka_jumpstart_db.customer_features

# COMMAND ----------

# MAGIC %md
# MAGIC # END
