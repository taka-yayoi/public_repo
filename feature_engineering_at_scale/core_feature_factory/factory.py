# Databricks notebook source
# MAGIC %md
# MAGIC # ベースのデータフレームに特徴量追加する`append_features`

# COMMAND ----------

# MAGIC %run ./feature

# COMMAND ----------

from pyspark.sql.dataframe import DataFrame
from pyspark.sql.column import Column
from typing import List, Union
from collections import OrderedDict

def append_features(df: DataFrame, groupby: List[Union[Column, Feature]], features: FeatureVector):
    features = features.to_list()
    if groupby and isinstance(groupby[0], Feature):
      groupby = [g.assembled_column for g in groupby]
    df = df.repartition(*groupby)
    agg_cols = []
    non_agg_cols = OrderedDict()

    for feature in features:
        if feature.agg_func:
            agg_cols.append(feature.assembled_column)
        else:
            non_agg_cols[feature.name] = feature.assembled_column

    if agg_cols:
        df = df.groupBy(*groupby).agg(*agg_cols)
    for fn, col in non_agg_cols.items():
        df = df.withColumn(fn, col)
    return df

# COMMAND ----------


