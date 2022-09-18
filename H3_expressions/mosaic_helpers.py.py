# Databricks notebook source
# MAGIC %md Install [Mosaic](https://databrickslabs.github.io/mosaic/index.html) from Databricks Labs
# MAGIC 
# MAGIC > Provides some convenience functions to include working with Kepler.

# COMMAND ----------

# MAGIC %pip install databricks-mosaic==0.2.1 --quiet

# COMMAND ----------

# MAGIC %md _Workaround for `mosaic_kepler` magic rendering issue in Mosaic 0.2.1 (will be fixed in 0.3+)_
# MAGIC 
# MAGIC > main function to call `as_kepler(table_name, feature_name, feature_type, limit_cnt=1000)`

# COMMAND ----------

import re

import h3
import pandas as pd

from keplergl import KeplerGl
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, conv, lower

from mosaic.api.accessors import st_astext
from mosaic.api.functions import st_centroid2D
from mosaic.config import config
from mosaic.resources import mosaic_logo_b64str
from mosaic.utils.kepler_config import mosaic_kepler_config


def displayKepler(map_instance, height, width):
    decoded = (
        map_instance._repr_html_()
        .decode("utf-8")
        .replace(".height||400", f".height||{height}")
        .replace(".width||400", f".width||{width}")
    )
    ga_script_redacted = re.sub(
        r"\<script\>\(function\(i,s,o,g,r,a,m\).*?GoogleAnalyticsObject.*?(\<\/script\>)",
        "",
        decoded,
        flags=re.DOTALL,
    )
    async_script_redacted = re.sub(
        r"s\.a\.createElement\(\"script\",\{async.*?\}\),",
        "",
        ga_script_redacted,
        flags=re.DOTALL,
    )
    config.notebook_utils.displayHTML(async_script_redacted)


def get_spark_df(table_name):
    try:
        table = config.ipython_hook.ev(table_name)
        if isinstance(table, pd.DataFrame):  # pandas dataframe
            data = config.mosaic_spark.createDataFrame(table)
        elif isinstance(table, DataFrame):  # spark dataframe
            data = table
        else:  # table name
            data = config.mosaic_spark.read.table(table)
    except NameError:
        try:
            data = config.mosaic_spark.read.table(table_name)
        except:
            raise Exception(f"Table name reference invalid.")
    return data


def set_centroid(pandas_data, feature_type, feature_name):
    if feature_type == "h3":
        centroid = h3.h3_to_geo(pandas_data[feature_name][0])
        # set to centroid of a geom
        mosaic_kepler_config["config"]["mapState"]["latitude"] = centroid[0]
        mosaic_kepler_config["config"]["mapState"]["longitude"] = centroid[1]
    elif feature_type == "geometry":
        tmp_sdf = config.mosaic_spark.createDataFrame(pandas_data.iloc[:1])
        centroid = (
            tmp_sdf.select(st_centroid2D(col(feature_name)))
            .limit(1)
            .collect()[0][0]
        )
        # set to centroid of a geom
        mosaic_kepler_config["config"]["mapState"]["latitude"] = centroid[1]
        mosaic_kepler_config["config"]["mapState"]["longitude"] = centroid[0]


def as_kepler(table_name:str, feature_name:str, feature_type:str, limit_cnt:int=1000) -> None:
    """
    Modified slightly since not a magic.
    - working on , color_col:str=None, height_col:str=None
    """
    data = get_spark_df(table_name)
    feature_col_dt = [dt for dt in data.dtypes if dt[0] == feature_name][0]

    if feature_type == "h3":
        if feature_col_dt[1] == "bigint":
            data = data.withColumn(
                feature_name, lower(conv(col(feature_name), 10, 16))
            )
    elif feature_type == "geometry":
        data = data.withColumn(feature_name, st_astext(col(feature_name)))
    else:
        raise Exception(f"Unsupported geometry type: {feature_type}.")

    allowed_schema = [
        field.name
        for field in data.schema.fields
        if field.dataType.typeName() in ["string", "long", "integer", "double"]
    ]
    data = data.select(*allowed_schema)
    pandas_data = data.limit(limit_cnt).toPandas()

    set_centroid(pandas_data, feature_type, feature_name)
    
    conf = mosaic_kepler_config
    conf['config']['visState']['layers'][0]['config']['dataId']=f"{table_name}-{feature_name}-{feature_type}-{limit_cnt}"

    m1 = KeplerGl(config=conf)
    m1.add_data(data=pandas_data, name=table_name)
    
    displayKepler(m1, 800, 1200)

# COMMAND ----------

# MAGIC %md _Initialize Mosaic_

# COMMAND ----------

from mosaic import enable_mosaic
enable_mosaic(spark, dbutils)
