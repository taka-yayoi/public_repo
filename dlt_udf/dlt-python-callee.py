# Databricks notebook source
from pyspark.sql.types import IntegerType

def square(i: int) -> int:
    """
    指定されたパラメーターを二乗するシンプルなUDF
    :param i: Pyspark/SQLのカラム
    :return: パラメーターの二乗
    """
    return i * i

spark.udf.register("makeItSquared", square, IntegerType()) # Spark SQLで使用する二乗計算udfを登録

# COMMAND ----------

from pyspark.sql.types import BooleanType

def is_even(i: int) -> bool:
    """
    偶数かどうかを判定するシンプルなUDF
    :param i: Pyspark/SQLのカラム
    :return: 偶数の場合True
    """
    
    if i % 2 == 0:
      return True
    else:
      return False
    
spark.udf.register("passOnlyEven", is_even, BooleanType()) # Spark SQLで使用する偶数判定udfを登録
