# Databricks notebook source
# MAGIC %md
# MAGIC # 特徴量のモジュール化

# COMMAND ----------

# MAGIC %run ./feature

# COMMAND ----------

from pyspark.sql.column import Column
# from core_feature_factory.feature import Feature
import pyspark.sql.functions as f
from pyspark.sql.session import SparkSession
from collections.abc import Mapping
import copy


class ImmutableDictBase(Mapping):
    def __init__(self):
        pass

    def __getitem__(self, item):
        return copy.deepcopy(self._dct[item])

    def __iter__(self):
        return iter(self._dct)

    def __len__(self):
        return len(self._dct)


class CommonFeatures(ImmutableDictBase):
    def __init__(self):
        self._dct["customer_id"] = Feature(_name="customer_id", _base_col=f.col("ss_customer_sk"))
        self._dct["trans_id"] = Feature(_name="trans_id", _base_col=f.concat("ss_ticket_number","d_date"))

    @property
    def collector(self):
        return self._dct["customer_id"]

    @property
    def trans_id(self):
        return self._dct["trans_id"]


class Filters(ImmutableDictBase):
    def __init__(self):
        self._dct["valid_sales"] = f.col("ss_net_paid") > 0

    @property
    def valid_sales(self):
        return self._dct["valid_sales"]


class StoreSales(CommonFeatures, Filters):
    def __init__(self):
        self._dct = dict()
        CommonFeatures.__init__(self)
        Filters.__init__(self)

        self._dct["total_trans"] = Feature(_name="total_trans",
                                           _base_col=self.trans_id,
                                           _filter=[],
                                           _negative_value=None,
                                           _agg_func=f.countDistinct)

        self._dct["total_sales"] = Feature(_name="total_sales",
                                           _base_col=f.col("ss_net_paid").cast("float"),
                                           _filter=self.valid_sales,
                                           _negative_value=0,
                                           _agg_func=f.sum)
        
        
    @property
    def total_sales(self):
        return self._dct["total_sales"]

    @property
    def total_trans(self):
        return self._dct["total_trans"]
      
    @property
    def trans_sketch(self):
        return self._dct["trans_sketch"]

# COMMAND ----------


