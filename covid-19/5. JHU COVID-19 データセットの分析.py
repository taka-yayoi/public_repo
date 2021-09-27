# Databricks notebook source
# ライブラリのインストール
%pip install altair
%pip install vega-datasets

# COMMAND ----------

# MAGIC %md # 5. JHU COVID-19 データセットの分析
# MAGIC 
# MAGIC このノートブックは[2019 Novel Coronavirus COVID-19 (2019-nCoV) Data Repository by Johns Hopkins CSSE](https://github.com/CSSEGISandData/COVID-19)に対する簡単な処理、分析を行うためのものです。データは定期的に`/databricks-datasets/COVID/CSSEGISandData/`で更新されるので、直接データにアクセスすることができます。
# MAGIC 
# MAGIC [Altair: Declarative Visualization in Python — Altair 4\.1\.0 documentation](https://altair-viz.github.io/index.html)

# COMMAND ----------

# ユーザーごとに一意のパスになるようにユーザー名をパスに含めます
import re
from pyspark.sql.types import * 
import os

# Username を取得
username_raw = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
# Username の英数字以外を除去し、全て小文字化。Username をファイルパスやデータベース名の一部で使用可能にするため。
username = re.sub('[^A-Za-z0-9]+', '', username_raw).lower()

print(username)

os.environ['username']=''.join(username)

# COMMAND ----------

# 標準ライブラリ
import io

# 外部ライブラリ
import requests
import numpy as np
import pandas as pd
import altair as alt
from vega_datasets import data

# 地理情報
topo_usa = 'https://vega.github.io/vega-datasets/data/us-10m.json'
topo_wa = 'https://raw.githubusercontent.com/deldersveld/topojson/master/countries/us-states/WA-53-washington-counties.json'
topo_king = 'https://raw.githubusercontent.com/johan/world.geo.json/master/countries/USA/WA/King.geo.json'

# COMMAND ----------

# MAGIC %md ## `jhu_daily` テーブルの作成
# MAGIC * ソース: `/databricks-datasets/COVID/CSSEGISandData/csse_covid_19_data/csse_covid_19_daily_reports/`
# MAGIC * COVID-19の日次レポートを格納

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType, DateType, TimestampType
schema = StructType([
  StructField('FIPS', IntegerType(), True), 
  StructField('Admin2', StringType(), True),
  StructField('Province_State', StringType(), True),  
  StructField('Country_Region', StringType(), True),  
  StructField('Last_Update', TimestampType(), True),  
  StructField('Lat', DoubleType(), True),  
  StructField('Long_', DoubleType(), True),
  StructField('Confirmed', IntegerType(), True), 
  StructField('Deaths', IntegerType(), True), 
  StructField('Recovered', IntegerType(), True), 
  StructField('Active', IntegerType(), True),   
  StructField('Combined_Key', StringType(), True),  
  #StructField('Incident_Rate', DoubleType(), True),  
  #StructField('Case_Fatality_Ratio', DoubleType(), True),  
  StructField('process_date', DateType(), True),    
])

# 上記スキーマに基づいて空のSparkデータフレームを作成します
jhu_daily = spark.createDataFrame([], schema)

# COMMAND ----------

# MAGIC %md ## それぞれのファイルに対するループ処理
# MAGIC 
# MAGIC 以下のコードスニペットは、各ファイルに以下の処理を行います。
# MAGIC * 日付を特定するためにファイル名を抽出
# MAGIC * 時間と共にスキーマが変化しているので、それぞれのスキーマに応じてロジックを切り替えてデータを追加
# MAGIC 
# MAGIC > **注意**<br>
# MAGIC > データが日々更新されているため、スキーマを修正する必要性が出てくる場合があります。

# COMMAND ----------

import os
import pandas as pd
import glob
from pyspark.sql.functions import input_file_name, lit, col

# すべてのcsvファイルの一覧を作成
globbed_files = glob.glob("/dbfs/databricks-datasets/COVID/CSSEGISandData/csse_covid_19_data/csse_covid_19_daily_reports/*.csv") 
#globbed_files = glob.glob("/dbfs/databricks-datasets/COVID/CSSEGISandData/csse_covid_19_data/csse_covid_19_daily_reports/04*.csv")

file_total = len(globbed_files)
i = 0
for csv in globbed_files:
  i = i + 1 

  # ファイル名
  source_file = csv[5:200]
  process_date = csv[100:104] + "-" + csv[94:96] + "-" + csv[97:99]
  
  print(f"{i}/{file_total} {source_file} {process_date}")

  # 一時データフレームにデータを読み込み
  df_tmp = spark.read.option("inferSchema", True).option("header", True).csv(source_file)
  df_tmp.createOrReplaceTempView("df_tmp")

  # スキーマの取得
  schema_txt = ' '.join(map(str, df_tmp.columns)) 
    
  # 3種類のスキーマ (2020-05-27時点) 
  schema_01 = "Province/State Country/Region Last Update Confirmed Deaths Recovered" # 01-22-2020 〜 02-29-2020
  schema_02 = "Province/State Country/Region Last Update Confirmed Deaths Recovered Latitude Longitude" # 03-01-2020 〜 03-21-2020
  schema_03 = "FIPS Admin2 Province_State Country_Region Last_Update Lat Long_ Confirmed Deaths Recovered Active Combined_Key" # 03-22-2020 以降
  schema_04 = "FIPS Admin2 Province_State Country_Region Last_Update Lat Long_ Confirmed Deaths Recovered Active Combined_Key Incident_Rate Case_Fatality_Ratio" # 2020-05-27時点で発見
  schema_05 = "FIPS Admin2 Province_State Country_Region Last_Update Lat Long_ Confirmed Deaths Recovered Active Combined_Key Incident_Rate Case-Fatality_Ratio" # 2020-05-27時点で発見
  schema_06 = "FIPS Admin2 Province_State Country_Region Last_Update Lat Long_ Confirmed Deaths Recovered Active Combined_Key Incidence_Rate Case-Fatality_Ratio" # 2020-05-27時点で発見
    
  # スキーマのタイプに基づいてデータを追加
  if (schema_txt == schema_01):
    df_tmp = (df_tmp
                .withColumn("FIPS", lit(None).cast(IntegerType()))
                .withColumn("Admin2", lit(None).cast(StringType()))
                .withColumn("Province_State", col("Province/State"))
                .withColumn("Country_Region", col("Country/Region"))
                .withColumn("Last_Update", col("Last Update"))
                .withColumn("Lat", lit(None).cast(DoubleType()))
                .withColumn("Long_", lit(None).cast(DoubleType()))
                .withColumn("Active", lit(None).cast(IntegerType()))
                .withColumn("Combined_Key", lit(None).cast(StringType()))
                #.withColumn("Incident_Rate", lit(None).cast(DoubleType()))
                #.withColumn("Case_Fatality_Ratio", lit(None).cast(DoubleType()))
                .withColumn("process_date", lit(process_date))
                .select("FIPS", 
                        "Admin2", 
                        "Province_State", 
                        "Country_Region", 
                        "Last_Update", 
                        "Lat", 
                        "Long_", 
                        "Confirmed", 
                        "Deaths", 
                        "Recovered", 
                        "Active", 
                        "Combined_Key", 
                        #"Incident_Rate", 
                        #"Case_Fatality_Ratio", 
                        "process_date")
               )
    jhu_daily = jhu_daily.union(df_tmp)
  elif (schema_txt == schema_02):
    df_tmp = (df_tmp
                .withColumn("FIPS", lit(None).cast(IntegerType()))
                .withColumn("Admin2", lit(None).cast(StringType()))
                .withColumn("Province_State", col("Province/State"))
                .withColumn("Country_Region", col("Country/Region"))
                .withColumn("Last_Update", col("Last Update"))
                .withColumn("Lat", col("Latitude"))
                .withColumn("Long_", col("Longitude"))
                .withColumn("Active", lit(None).cast(IntegerType()))
                .withColumn("Combined_Key", lit(None).cast(StringType()))
                #.withColumn("Incident_Rate", lit(None).cast(DoubleType()))
                #.withColumn("Case_Fatality_Ratio", lit(None).cast(DoubleType()))
                .withColumn("process_date", lit(process_date))
                .select("FIPS", 
                        "Admin2", 
                        "Province_State", 
                        "Country_Region", 
                        "Last_Update", 
                        "Lat", 
                        "Long_", 
                        "Confirmed", 
                        "Deaths", 
                        "Recovered", 
                        "Active", 
                        "Combined_Key", 
                        #"Incident_Rate", 
                        #"Case_Fatality_Ratio", 
                        "process_date")
               )
    jhu_daily = jhu_daily.union(df_tmp)

  elif (schema_txt == schema_03):
    df_tmp = (df_tmp
                #.withColumn("Incident_Rate", lit(None).cast(DoubleType()))
                #.withColumn("Case_Fatality_Ratio", lit(None).cast(DoubleType()))
                .withColumn("process_date", lit(process_date))
                .select("FIPS", 
                        "Admin2", 
                        "Province_State", 
                        "Country_Region", 
                        "Last_Update", 
                        "Lat", 
                        "Long_", 
                        "Confirmed", 
                        "Deaths", 
                        "Recovered", 
                        "Active", 
                        "Combined_Key", 
                        #"Incident_Rate", 
                        #"Case_Fatality_Ratio", 
                        "process_date")
             )
    jhu_daily = jhu_daily.union(df_tmp)
    
  elif (schema_txt == schema_04):
    df_tmp = (df_tmp.withColumn("process_date", lit(process_date))
                   .select("FIPS", 
                        "Admin2", 
                        "Province_State", 
                        "Country_Region", 
                        "Last_Update", 
                        "Lat", 
                        "Long_", 
                        "Confirmed", 
                        "Deaths", 
                        "Recovered", 
                        "Active", 
                        #"Combined_Key", 
                        #"Incident_Rate", 
                        "Case_Fatality_Ratio", 
                        "process_date")
             )
    
    jhu_daily = jhu_daily.union(df_tmp)
    
  elif (schema_txt == schema_05):
    df_tmp = (df_tmp.withColumn("process_date", lit(process_date))
                   #.withColumn("Case_Fatality_Ratio", col("Case-Fatality_Ratio"))
                    .select("FIPS", 
                        "Admin2", 
                        "Province_State", 
                        "Country_Region", 
                        "Last_Update", 
                        "Lat", 
                        "Long_", 
                        "Confirmed", 
                        "Deaths", 
                        "Recovered", 
                        "Active", 
                        "Combined_Key", 
                        #"Incident_Rate", 
                        #"Case_Fatality_Ratio", 
                        "process_date")
             )
    
    jhu_daily = jhu_daily.union(df_tmp)    

  elif (schema_txt == schema_06):
    df_tmp = (df_tmp.withColumn("process_date", lit(process_date))
                   #.withColumn("Incident_Rate", col("Incidence_Rate"))
                   #.withColumn("Case_Fatality_Ratio", col("Case-Fatality_Ratio"))
              .select("FIPS", 
                        "Admin2", 
                        "Province_State", 
                        "Country_Region", 
                        "Last_Update", 
                        "Lat", 
                        "Long_", 
                        "Confirmed", 
                        "Deaths", 
                        "Recovered", 
                        "Active", 
                        "Combined_Key", 
                        #"Incident_Rate", 
                        #"Case_Fatality_Ratio", 
                        "process_date")
             )
    
    jhu_daily = jhu_daily.union(df_tmp)    
    
  else:
    print(f"Schema may have changed: {schema_txt}")
    raise
  
  # 進捗表示
  #print("%s | %s" % (process_date, schema_txt))

# COMMAND ----------

jhu_daily.createOrReplaceTempView("jhu_daily")
display(jhu_daily)

# COMMAND ----------

#%sh
#rm -fR /dbfs/tmp/$username/COVID/jhu_daily/

# COMMAND ----------

# # jhu_dailyテーブルの保存
# file_path = f'/tmp/{username}/COVID/jhu_daily/'
# jhu_daily.repartition(4).write.format("parquet").save(file_path)

# COMMAND ----------

# MAGIC %md ## 2019年の人口推定値のダウンロード

# COMMAND ----------

# MAGIC %sh mkdir -p /dbfs/tmp/$username/COVID/population_estimates_by_county/ && wget -O /dbfs/tmp/$username/COVID/population_estimates_by_county/co-est2019-alldata.csv https://raw.githubusercontent.com/databricks/tech-talks/master/datasets/co-est2019-alldata.csv && ls -al /dbfs/tmp/$username/COVID/population_estimates_by_county/

# COMMAND ----------

map_popest_county = spark.read.option("header", True).option("inferSchema", True).csv(f"/tmp/{username}/COVID/population_estimates_by_county/co-est2019-alldata.csv")
map_popest_county.createOrReplaceTempView("map_popest_county")
fips_popest_county = spark.sql("select State * 1000 + substring(cast(1000 + County as string), 2, 3) as fips, STNAME, CTYNAME, census2010pop, POPESTIMATE2019 from map_popest_county")
fips_popest_county.createOrReplaceTempView("fips_popest_county")

# COMMAND ----------

# MAGIC %md ## 人口推定値の取り込み
# MAGIC 
# MAGIC 人口推定値を取り込むために `jhu_daily_pop` 一時テーブルを作成します。3/22より前のデータは `FIPS` 情報を含んでいないため、推定人口値を取り込む際にはデータセットを3/22以降に限定します。

# COMMAND ----------

jhu_daily_pop = spark.sql("""
SELECT f.FIPS, f.Admin2, f.Province_State, f.Country_Region, f.Last_Update, f.Lat, f.Long_, f.Confirmed, f.Deaths, f.Recovered, f.Active, f.Combined_Key, f.process_date, p.POPESTIMATE2019 
  FROM jhu_daily f
    JOIN fips_popest_county p
      ON p.fips = f.FIPS
""")
jhu_daily_pop.createOrReplaceTempView("jhu_daily_pop")

# COMMAND ----------

# MAGIC %md ## 最初の探索的データ分析(Exploratory Data Analysis)

# COMMAND ----------

# MAGIC %md ### NY郡とKing郡における感染者数、死者数の検証

# COMMAND ----------

# MAGIC %sql
# MAGIC select process_date, Admin2, Confirmed, Deaths, Recovered, Active from jhu_daily where Province_State in ('New York') and Admin2 in ('New York City')

# COMMAND ----------

# MAGIC %sql
# MAGIC select process_date, Admin2, Confirmed, Deaths, Recovered, Active from jhu_daily where Province_State in ('Washington') and Admin2 in ('King')

# COMMAND ----------

# MAGIC %md ### NY郡とKing郡における人口に対する感染者数、死者数の比率の検証

# COMMAND ----------

# MAGIC %sql
# MAGIC select process_date, Admin2, 100000.*Confirmed/POPESTIMATE2019 as Confirmed_per100K, 100000.*Deaths/POPESTIMATE2019 as Deaths_per100K, Recovered, Active from jhu_daily_pop where Province_State in ('New York') and Admin2 in ('New York City')

# COMMAND ----------

# MAGIC %sql
# MAGIC select process_date, Admin2, 100000.*Confirmed/POPESTIMATE2019 as Confirmed_per100K, 100000.*Deaths/POPESTIMATE2019 as Deaths_per100K, Recovered, Active from jhu_daily_pop where Province_State in ('Washington') and Admin2 in ('King')

# COMMAND ----------

# MAGIC %md ## 郡ごとのCOVID-19感染者数、死者数

# COMMAND ----------

# `usa`データフレームの作成
df_usa = spark.sql("select fips, cast(100000.*Confirmed/POPESTIMATE2019 as int) as confirmed_per100K, cast(100000.*Deaths/POPESTIMATE2019 as int) as deaths_per100K, recovered, active, lat, long_, admin2 as county, province_state as state, process_date, cast(replace(process_date, '-', '') as integer) as process_date_num from jhu_daily_pop where lat is not null and long_ is not null and fips is not null and (lat <> 0 and long_ <> 0)")
df_usa.createOrReplaceTempView("df_usa")

# pandasデータフレームに変換
pdf_usa = df_usa.toPandas()
pdf_usa['confirmed_per100K'] = pdf_usa['confirmed_per100K'].astype('int32')
pdf_usa['deaths_per100K'] = pdf_usa['deaths_per100K'].astype('int32')

# COMMAND ----------

def map_usa_cases(curr_date):
  # altairの地形情報を取得
  us_states = alt.topo_feature(topo_usa, 'states')
  us_counties = alt.topo_feature(topo_usa, 'counties')

  # 州の境界線
  base_states = alt.Chart(us_states).mark_geoshape().encode(
    stroke=alt.value('lightgray'), fill=alt.value('white')
  ).properties(
    width=1200,
    height=960,
  ).project(
    type='albersUsa',
  )


  # 郡ごとの感染者数
  base_counties = alt.Chart(us_counties).mark_geoshape().encode(
    color=alt.Color('confirmed:Q', scale=alt.Scale(type='log'), title='Confirmed'),
  ).transform_lookup(
    lookup='id',
    from_=alt.LookupData(pdf_usa[(pdf_usa['confirmed'] > 0) & (pdf_usa['process_date'] == curr_date)], 'fips', ['confirmed'])  
  )

  # 緯度経度に基づく死者数
  points = alt.Chart(pdf_usa[(pdf_usa['deaths'] > 0) & (pdf_usa['process_date'] == curr_date)]).mark_point(opacity=0.75, filled=True).encode(
    longitude='long_:Q',
    latitude='lat:Q',
    size=alt.Size('sum(deaths):Q', scale=alt.Scale(type='symlog'), title='deaths'),
    color=alt.value('#BD595D'),
    stroke=alt.value('brown'),
    tooltip=[
      alt.Tooltip('state', title='state'), 
      alt.Tooltip('county', title='county'), 
      alt.Tooltip('confirmed', title='confirmed'),
      alt.Tooltip('deaths', title='deaths'),       
    ],
  ).properties(
    # 図のタイトル
    title=f'COVID-19 Confirmed Cases and Deaths by County {curr_date}'
  )

  # グラフの表示
  return (base_states + base_counties + points)

# COMMAND ----------

def map_usa_cases(curr_date):
  # altairの地形情報を取得
  us_states = alt.topo_feature(topo_usa, 'states')
  us_counties = alt.topo_feature(topo_usa, 'counties')

  # 州の境界線
  base_states = alt.Chart(us_states).mark_geoshape().encode(
    stroke=alt.value('lightgray'), fill=alt.value('white')
  ).properties(
    width=1200,
    height=960,
  ).project(
    type='albersUsa',
  )


  # 郡ごとの感染者数
  base_counties = alt.Chart(us_counties).mark_geoshape().encode(
    color=alt.Color('confirmed_per100K:Q', scale=alt.Scale(domain=(1, 7500), type='log'), title='Confirmed per 100K'),
  ).transform_lookup(
    lookup='id',
    from_=alt.LookupData(pdf_usa[(pdf_usa['confirmed_per100K'] > 0) & (pdf_usa['process_date'] == curr_date)], 'fips', ['confirmed_per100K'])  
  )

  # 緯度経度に基づく死者数
  points = alt.Chart(pdf_usa[(pdf_usa['deaths_per100K'] > 0) & (pdf_usa['process_date'] == curr_date)]).mark_point(opacity=0.75, filled=True).encode(
    longitude='long_:Q',
    latitude='lat:Q',
     size=alt.Size('deaths_per100K:Q', scale=alt.Scale(domain=(1, 1000), type='log'), title='deaths_per100K'),
     #size=alt.Size('deaths_per100K:Q', title='deaths_per100K'),
     color=alt.value('#BD595D'),
     stroke=alt.value('brown'),
    tooltip=[
      alt.Tooltip('state', title='state'), 
      alt.Tooltip('county', title='county'), 
      alt.Tooltip('confirmed_per100K', title='confirmed'),
      alt.Tooltip('deaths_per100K', title='deaths'),       
    ],
  ).properties(
    # 図のタイトル
    title=f'COVID-19 Confirmed Cases and Deaths by County (by 100K) {curr_date}'
  )

   # グラフの表示
  return (base_states + base_counties + points)

# COMMAND ----------

# 最初の日 (2020-03-22)
map_usa_cases('2020-03-22')

# COMMAND ----------

# 最新日 (2020-04-14)
map_usa_cases('2020-04-14')

# COMMAND ----------

# MAGIC %md
# MAGIC # END
