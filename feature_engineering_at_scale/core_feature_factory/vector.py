# Databricks notebook source
# MAGIC %md
# MAGIC # 特徴量ベクトル

# COMMAND ----------

# MAGIC %run ./feature

# COMMAND ----------

# MAGIC %run ./feature_dict

# COMMAND ----------

# MAGIC %run ./functions

# COMMAND ----------

from typing import List
# from core_feature_factory.feature_dict import Features
import pyspark.sql.functions as f
# from core_feature_factory.functions import avg_func, stdev_func

class FeatureVector:

    def __init__(self, features: List[Feature] = None):
        if not features:
          self._features = []
        else:
          self._features = features

    def __add__(self, other):
        """
        Overrides default add so that two feature vectors can be added to form a new feature vector.
        e.g. fv1 = fv2 + fv3 in which fv1 contains all metrics from both fv2 and fv2
        :param other:
        :return:
        """
        return FeatureVector(self.metrics + other.metrics)
      
    @classmethod
    def create_by_names(cls, feature_collection: ImmutableDictBase,  feature_names: List[str]):
        feat_list = [feature_collection[fn] for fn in feature_names]
        return FeatureVector(feat_list)

    def multiply(self, multiplier_col: str, multiplier_values: List[str]):
        feats = FeatureVector()
        for feature in self._features:
            fv = feature.multiply(multiplier_col, multiplier_values)
            feats += fv
        return feats

    def create_stats(self, base_name: str, stats=["min", "max", "avg", "stdev"]):
        cols = [f.col(feat.name) for feat in self._features]
        fl = []
        for stat in stats:
            if stat == "min":
                fl.append(Feature(_name=base_name + "_min", _base_col=f.array_min(f.array(cols))))
            elif stat == "max":
                fl.append(Feature(_name=base_name + "_max", _base_col=f.array_max(f.array(cols))))
            elif stat == "avg":
                fl.append(Feature(_name=base_name + "_avg", _base_col=avg_func(f.array(cols))))
            elif stat == "stdev":
                fl.append(Feature(_name=base_name + "_stdev", _base_col=stdev_func(f.array(cols))))
        return FeatureVector(fl)
      


class FeatureVector2D:
    def __init__(self, features: dict):
        self._feature_dict = features

    def to_list(self):
        lst = []
        for fn, fv in self._feature_dict.items():
            lst += fv._features
        return lst

    def create_stats(self, stats=["min", "max", "avg", "stdev"]):
        dct = dict()
        for fn, fv in self._feature_dict.items():
            dct[fn+"_stats"] = fv.create_stats(fn, stats)
        return FeatureVector2D(dct)

# COMMAND ----------


