# Databricks notebook source
# MAGIC %md # タクシーデータセットによるFeature Storeのデモ - 入力データセットの作成
# MAGIC 
# MAGIC 本ノートブックではFeature Storeデモノートブックで使用するタクシーデータを作成します。
# MAGIC 
# MAGIC 以下のステップでデータを作成します。
# MAGIC 
# MAGIC 1. `nyc_yellow_taxi_with_zips_<ユーザー名文字列>`データベースの作成
# MAGIC 1. `dbfs:/databricks-datasets/nyctaxi`から[NYC Taxi Data](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)のサブセットを読み込み
# MAGIC 1. カラム名の変更、不要なカラムの削除
# MAGIC 1. 緯度経度をZIPコードに変換するUDFを適用し、データフレームにZIPコードのカラムを追加。この変換は[Extracting ZIP codes from longitude and latitude in PySpark](https://medium.com/@m.majidpour/extracting-zip-codes-from-longitude-and-latitude-in-pyspark-eafbcfef474c)をベースにしています。
# MAGIC 1. `nyc_yellow_taxi_with_zips_<ユーザー名文字列>`データベースの`feature_store_taxi_example`テーブルに結果を書き込みます。
# MAGIC 
# MAGIC **要件**
# MAGIC - Databricks Runtime for Machine Learning 8.3以降
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2021/06/22</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>クラスター</td><td>8.3ML</td></tr>
# MAGIC </table>
# MAGIC <img style="margin-top:25px;" src="https://jixjiadatabricks.blob.core.windows.net/images/databricks-logo-small-new.png" width="140">

# COMMAND ----------

# MAGIC %pip install uszipcode

# COMMAND ----------

# MAGIC %md ## セットアップおよびインストール
# MAGIC 
# MAGIC オリジナルのデータセットの緯度経度をZIPコードに変換するために、Pythonパッケージ`uszipcode`を使用します。[PyPI](https://pypi.org/project/uszipcode/)から利用できます。
# MAGIC 
# MAGIC **注意**
# MAGIC - Feature StoreのテーブルはDelta Lakeで管理されます。
# MAGIC - データベース名が他のユーザーのものと重複すると期待しない動作をする場合があります。
# MAGIC - これを避けるために、次のセルではユーザー名をキーとしたデータベースを作成します。

# COMMAND ----------

import re
from pyspark.sql.types import * 

# ログインIDからUsernameを取得
username_raw = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
# Username の英数字以外を除去し、全て小文字化。Username をファイルパスやデータベース名の一部で使用可能にするため。
username = re.sub('[^A-Za-z0-9]+', '', username_raw).lower()

# データベース名
db_name = f"nyc_yellow_taxi_with_zips_{username}"

# Hiveメタストアのデータベースの準備:データベースの作成
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
# Hiveメタストアのデータベースの選択
spark.sql(f"USE {db_name}")

print("database name: " + db_name)

# COMMAND ----------

# MAGIC %md ## データ処理のためのヘルパー関数の定義

# COMMAND ----------

from uszipcode import SearchEngine
import sqlite3
import pandas as pd
from pyspark.sql.functions import udf, col
from pyspark.sql.types import IntegerType
import math
from urllib import request
import os

BAD_ZIPCODE_VALUE = 'bad_zipcode'
file_location = "dbfs:/databricks-datasets/nyctaxi/tripdata/yellow/"
file_type = "csv"
target_year = 2016

def push_zipcode_data_to_executors():
  # githubから直接ダウンロード(デフォルトのダウンロード箇所が変更される可能性があるため)
  target_dir = '/tmp/db/'
  target_file = os.path.join(target_dir, 'simple_db.sqlite')
  remote_url = 'https://github.com/MacHu-GWU/uszipcode-project/files/5183256/simple_db.log'
  os.makedirs(target_dir, exist_ok=True)
  request.urlretrieve(remote_url, target_file)
  # zipデータベースをpandasデータフレームに読み込みます
  search = SearchEngine(db_file_dir=target_dir)
  conn = sqlite3.connect(target_file)
  pdf = pd.read_sql_query('''select  zipcode, lat, lng, radius_in_miles, 
                          bounds_west, bounds_east, bounds_north, bounds_south from 
                          simple_zipcode''',conn)
  return sc.broadcast(pdf)
  
# 緯度経度からZIPコードを検索するUDFを定義
@udf('string')
def get_zipcode(lat, lng):
    if lat is None or lng is None:
      return BAD_ZIPCODE_VALUE
    dist_btwn_lat_deg = 69.172
    dist_btwn_lon_deg = math.cos(lat) * 69.172
    radius = 5
    lat_degr_rad = abs(radius / dist_btwn_lat_deg)
    lon_degr_rad = abs(radius / dist_btwn_lon_deg)
    lat_lower = lat - lat_degr_rad
    lat_upper = lat + lat_degr_rad
    lng_lower = lng - lon_degr_rad
    lng_upper = lng + lon_degr_rad
    pdf = zipcodes_broadcast_df.value
    try:
        out = pdf[(pdf['lat'].between(lat_lower, lat_upper)) & (pdf['lng'].between(lng_lower, lng_upper))]
        dist = [None]*len(out)
        for i in range(len(out)):
            dist[i] = (out['lat'].iloc[i]-lat)**2 + (out['lng'].iloc[i]-lng)**2
        zip = out['zipcode'].iloc[dist.index(min(dist))]
    except:
        zip = BAD_ZIPCODE_VALUE
    return zip
  
def get_data_files(yyyy, months):
  data_files = []
  for mm in months:
    mm = str(mm) if mm >= 10 else f"0{mm}"
    month_data_files = list(filter(lambda file_name: f"{yyyy}-{mm}" in file_name,
                           [f.path for f in dbutils.fs.ls(file_location)]))
    data_files += month_data_files
  return data_files
  
def load_data(data_files, sample=1.0):
  df = (spark.read.format("csv")
        .option("inferSchema", "true")
        .option("header", "true")
        .option("ignoreLeadingWhiteSpace", "true")
        .option("ignoreTrailingWhiteSpace", "true")
        .option("sep", ",")
        .load(data_files)
      ).sample(False, sample, 123)
  
  # カラム名変更、型変換、カラムのフィルタリング
  column_allow_list = { 
    "pickup_datetime": ["tpep_pickup_datetime", "timestamp"],
    "tpep_pickup_datetime": ["tpep_pickup_datetime", "timestamp"],
    
    # 型変換
    "dropoff_datetime": ["tpep_dropoff_datetime", "timestamp"],
    "tpep_dropoff_datetime": ["tpep_dropoff_datetime", "timestamp"],
    
    "pickup_zip": ["pickup_zip", "integer"],
    "dropoff_zip": ["dropoff_zip", "integer"],
    "trip_distance": ["trip_distance", "double"],
    "fare_amount": ["fare_amount", "double"],
    "pickup_latitude": ["pickup_latitude", "double"],
    "pickup_longitude": ["pickup_longitude", "double"],
    "dropoff_latitude": ["dropoff_latitude", "double"],
    "dropoff_longitude": ["dropoff_longitude", "double"],
  }
  columns = []
  for orig in df.columns:
    orig_lower = orig.lower()
    if orig_lower in column_allow_list:
      new_name, data_type = column_allow_list[orig_lower]
      columns.append(col(orig).cast(data_type).alias(new_name.lower()))
  
  return df.select(columns)  

def annotate_zipcodes(df):
  to_zip = lambda lat, lng:  get_zipcode(col(lat).astype("double"), col(lng).astype("double"))
  # ZIPコードカラムの追加、中間カラムの削除
  df = (df
          .withColumn('pickup_zip', to_zip("pickup_latitude", "pickup_longitude"))
          .withColumn('dropoff_zip', to_zip("dropoff_latitude", "dropoff_longitude"))
          .drop('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')
         )
  # 不正データの除外
  df = df.filter(df.pickup_zip != BAD_ZIPCODE_VALUE)
  df = df.filter(df.dropoff_zip != BAD_ZIPCODE_VALUE)
  
  # ZIPコードをintに変換
  df = df.withColumn("pickup_zip", df["pickup_zip"].cast(IntegerType()))
  df = df.withColumn("dropoff_zip", df["dropoff_zip"].cast(IntegerType()))
  return df

def write_to_table(df, database, table):
  (df.write
   .format("delta")
   .mode("overwrite")
   .option("overwriteSchema", "true")
   .saveAsTable(f"{database}.{table}"))

# COMMAND ----------

# MAGIC %md ## データベースの作成

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name};")

# COMMAND ----------

# MAGIC %md ## データフレームの作成およびテーブルへの書き込み

# COMMAND ----------

# UDFの処理を高速化するためにZIPコードデータを読み込んでデータフレームをエグゼキューターにブロードキャストします
zipcodes_broadcast_df = push_zipcode_data_to_executors()

# 2016年の最初の2ヶ月のデータファイルを生成します
data_files = get_data_files(target_year,months=[1,2])

# 処理の高速化のために小規模なサンプルを読み込みます
df = load_data(data_files, sample=.001)

# リパーティション -- デフォルトではこのデータセットは単一のパーティションとなっています  
# データセットは上で小規模なものになっていますので、少数のパーティション数を指定します
df = df.repartition(6)

# 緯度経度をZIPコードに変換します 
df_with_zip = annotate_zipcodes(df)

# データフレームをDeltaテーブルに書き込みます
write_to_table(df_with_zip, database=db_name, table="nyc_yellow_taxi_with_zips")

# COMMAND ----------

# MAGIC %md
# MAGIC # END
