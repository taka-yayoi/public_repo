# Databricks notebook source
# MAGIC %md
# MAGIC # ベース特徴量の定義
# MAGIC 
# MAGIC リファレンス実装においては、特徴量はFeatureクラスとして実装されます。Featureクラスのメソッドとして、オペレーションが実装されます。

# COMMAND ----------

import pyspark.sql.functions as f
from typing import List
from pyspark.sql.column import Column
from functools import reduce


class Feature:
    def __init__(self,
                 _name: str,
                 _base_col: Column,
                 _filter=[],
                 _negative_value=None,
                 _agg_func=None):
        """

        :param _name: 特徴量の名前
        :param _base_col: filterがtrueあるいはfilterがない場合、特徴量を導出するカラム
            base_col は複数のカラムになる場合があることに注意してください。例 base_col = F.concat(F.col("POS_TRANS_ID"), F.col("TRANS_DATE"), F.col("STORE_ID"), F.col("POS_TILL_NUM")
        :param _filter: base_col あるいは nagative_value を選択する条件
        :param _negative_value: フィルターがfalseの場合に特徴量が導出される値
        :param _agg_func: base_col あるいは negative_value から特徴量を計算する際の集計関数
        """
        self.name = _name
        self.base_col = _base_col.assembled_column if isinstance(_base_col, Feature) else _base_col
        self.filter = _filter if type(_filter) is list else [_filter]
        self.negative_value = _negative_value if _negative_value != "" else None
        self.output_alias = _name
        self.agg_func = _agg_func
        # self.aggs = []
        self._assemble_column()

    def _assemble_column(self):
        if (self.base_col is not None) and (len(self.filter) > 0) and (self.agg_func is not None):
            self.assembled_column = self.agg_func(f.when(self._assemble_filter(), self.base_col).otherwise(self.negative_value)).alias(self.output_alias)
            # self._assemble_aggs()
        elif (self.base_col is not None) and (len(self.filter) == 0) and (self.agg_func is not None):
            self.assembled_column = self.agg_func(self.base_col).alias(self.output_alias)
            # self._assemble_aggs()
        elif (self.base_col is not None) and (len(self.filter) > 0) and (self.agg_func is None):
            self.assembled_column = f.when(self._assemble_filter(), self.base_col).otherwise(self.negative_value).alias(self.output_alias)
        else:
            self.assembled_column = self.base_col.alias(self.output_alias)


    def _assemble_filter(self):
        if len(self.filter) == 1:
            return self.filter[0]
        else:
            final_filter = reduce((lambda x, y: x & y), self.filter)
            return final_filter

    # def _assemble_aggs(self):
    #     self.aggs.append(self.agg_func(self.output_alias).alias(self.output_alias))

    def _equals(self, that):
        this_expr = self.assembled_column._jc.toString()
        that_expr = that.assembled_column._jc.toString()
        return this_expr == that_expr
    
    def clone():
        return Feature(_name=self.name,
                       _base_col = self.base_col,
                       _filter = self.filter,
                       _negative_value = self.negative_value,
                       _agg_func = self.agg_func
                      )

    def multiply(self, col: str, multipliers: List[str]):
        feats = []
        for mult in multipliers:
            feat = Feature(_name=self.name+"_"+str(mult),
                           _base_col=self.base_col,
                           _agg_func=self.agg_func,
                           _filter=self.filter+[f.col(col)==mult],
                           _negative_value= self.negative_value)
            feats.append(feat)
        return FeatureVector(feats)
    

# COMMAND ----------

from typing import List
import pyspark.sql.functions as f

class FeatureVector:

    def __init__(self, features: List[Feature] = None):
        if not features:
          self._features = []
        else:
          self._features = features

    def __add__(self, other):
        """
        新規特徴量ベクトルを形成するために2つの特徴量ベクトルを加算できるようにデフォルトのaddを上書き
        例 fv1 = fv2 + fv3 では fv1 は fv2 と fv3 の両方の特徴量全てを持ちます
        :param other:
        :return:
        """
        return FeatureVector(self._features + other._features)
      
    @classmethod
    def create_by_names(cls, feature_collection,  feature_names: List[str]):
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
    
    def to_cols(self):
        return [f.col(feat.name) for feat in self._features]

    def to_list(self):
        return self._features[:]
