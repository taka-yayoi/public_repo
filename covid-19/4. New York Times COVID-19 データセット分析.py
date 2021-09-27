# Databricks notebook source
# ライブラリのインストール
%pip install altair
%pip install vega-datasets

# COMMAND ----------

# MAGIC %md # 4. New York Times COVID-19 データセット分析
# MAGIC 
# MAGIC このノートブックは[New York Times COVID-19 データセット](https://github.com/nytimes/covid-19-data)に対する簡単な処理、分析を行うためのものです。データは定期的に`/databricks-datasets/COVID/covid-19-data/`で更新されるので、直接データにアクセスすることができます。

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

# 地形データ
topo_usa = 'https://vega.github.io/vega-datasets/data/us-10m.json'
topo_wa = 'https://raw.githubusercontent.com/deldersveld/topojson/master/countries/us-states/WA-53-washington-counties.json'
topo_king = 'https://raw.githubusercontent.com/johan/world.geo.json/master/countries/USA/WA/King.geo.json'

# COMMAND ----------

# MAGIC %md ## 郡のFIPS(Federal Information Processing Series)のダウンロード、緯度経度への変換
# MAGIC 
# MAGIC [Understanding Geographic Identifiers \(GEOIDs\)](https://www.census.gov/programs-surveys/geography/guidance/geo-identifiers.html)

# COMMAND ----------

# MAGIC %sh mkdir -p /dbfs/tmp/$username/COVID/map_fips/ && wget -O /dbfs/tmp/$username/COVID/map_fips/countyfips_lat_long.csv https://raw.githubusercontent.com/dennyglee/tech-talks/master/datasets/countyfips_lat_long.csv && ls -al /dbfs/tmp/$username/COVID/map_fips/

# COMMAND ----------

# 郡のFIPSと中心の緯度経度のマッピングの作成
map_fips = spark.read.option("header", True).option("inferSchema", True).csv(f"/tmp/{username}/COVID/map_fips/countyfips_lat_long.csv")
map_fips = (map_fips
              .withColumnRenamed("STATE", "state")
              .withColumnRenamed("COUNTYNAME", "county")
              .withColumnRenamed("LAT", "lat")
              .withColumnRenamed("LON", "long_"))
map_fips.createOrReplaceTempView("map_fips")

# COMMAND ----------

map_fips_dedup = spark.sql("""select fips, min(state) as state, min(county) as county, min(long_) as long_, min(lat) as lat from map_fips group by fips""")
map_fips_dedup.createOrReplaceTempView("map_fips_dedup")

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

# MAGIC %md ## `nyt_daily`テーブルの作成
# MAGIC * ソース: `/databricks-datasets/COVID/covid-19-data/`
# MAGIC * 日次のCOVID-19レポートを格納

# COMMAND ----------

nyt_daily = spark.read.option("inferSchema", True).option("header", True).csv("/databricks-datasets/COVID/covid-19-data/us-counties.csv")
nyt_daily.createOrReplaceTempView("nyt_daily")
display(nyt_daily)

# COMMAND ----------

# MAGIC %md ## 特定の郡におけるCOVID-19感染者数、死者数
# MAGIC 
# MAGIC 教育機関の閉鎖の時期における2週間枠に注目します。
# MAGIC * ワシントン州のトップ10の郡 (3/13/2020)
# MAGIC * ニューヨーク州のトップ10の郡 (3/18/2020)
# MAGIC 
# MAGIC クエリーでは、2019年の国勢調査の人口推定値を使用します。

# COMMAND ----------

# ワシントン州の2週間ウィンドウ
wa_state_window = spark.sql("""
SELECT date, 100 + datediff(date, '2020-03-06T00:00:00.000+0000') as day_num, county, fips, cases, deaths, 100000.*cases/population_estimate AS cases_per_100Kpop, 100000.*deaths/population_estimate AS deaths_per_100Kpop
  from (
SELECT CAST(f.date AS date) AS date, f.county, f.fips, SUM(f.cases) AS cases, SUM(f.deaths) AS deaths, MAX(p.POPESTIMATE2019) AS population_estimate 
  FROM nyt_daily f 
    JOIN fips_popest_county p
      ON p.fips = f.fips
 WHERE f.state = 'Washington' 
   AND date BETWEEN '2020-03-06T00:00:00.000+0000' AND '2020-03-20T00:00:00.000+0000'
 GROUP BY f.date, f.county, f.fips
) a""")
wa_state_window.createOrReplaceTempView("wa_state_window")

# ニューヨーク州の2週間ウィンドウ
ny_state_window = spark.sql("""
SELECT date, 100 + datediff(date, '2020-03-11T00:00:00.000+0000') as day_num, county, fips, cases, deaths, 100000.*cases/population_estimate AS cases_per_100Kpop, 100000.*deaths/population_estimate AS deaths_per_100Kpop
  FROM (
SELECT CAST(f.date AS date) AS date, f.county, p.fips, SUM(f.cases) as cases, SUM(f.deaths) as deaths, MAX(p.POPESTIMATE2019) AS population_estimate  
  FROM nyt_daily f 
    JOIN fips_popest_county p
      ON p.fips = coalesce(f.fips, 36061)
 WHERE f.state = 'New York' 
   AND date BETWEEN '2020-03-11T00:00:00.000+0000' AND '2020-03-25T00:00:00.000+0000'
 GROUP BY f.date, f.county, p.fips
) a""")
ny_state_window.createOrReplaceTempView("ny_state_window")

# ニューヨーク州の2週間ウィンドウ (1週前)
ny_state_window_m1 = spark.sql("""
SELECT date, 100 + datediff(date, '2020-03-06T00:00:00.000+0000') as day_num, county, fips, cases, deaths, 100000.*cases/population_estimate AS cases_per_100Kpop, 100000.*deaths/population_estimate AS deaths_per_100Kpop
  FROM (
SELECT CAST(f.date AS date) AS date, f.county, p.fips, SUM(f.cases) as cases, SUM(f.deaths) as deaths, MAX(p.POPESTIMATE2019) AS population_estimate  
  FROM nyt_daily f 
    JOIN fips_popest_county p
      ON p.fips = coalesce(f.fips, 36061)
 WHERE f.state = 'New York' 
   AND date BETWEEN '2020-03-06T00:00:00.000+0000' AND '2020-03-20T00:00:00.000+0000'
 GROUP BY f.date, f.county, p.fips
) a""")
ny_state_window_m1.createOrReplaceTempView("ny_state_window_m1")

# COMMAND ----------

# MAGIC %md ### ワシントン州、ニューヨーク州の郡におけるCOVID-19感染者数

# COMMAND ----------

# DBTITLE 1,ワシントン州における3/6 - 3/20での感染者数 - 教育機関の閉鎖 : 3/13
# MAGIC %sql
# MAGIC SELECT f.date, f.county, f.cases 
# MAGIC   FROM wa_state_window f
# MAGIC   JOIN (
# MAGIC       SELECT county, sum(cases) as cases FROM wa_state_window GROUP BY county ORDER BY cases DESC LIMIT 10
# MAGIC     ) x ON x.county = f.county

# COMMAND ----------

# DBTITLE 1,ニューヨーク州における 3/11 - 3/25 での感染者数 - 教育機関の閉鎖 : 3/18
# MAGIC %sql
# MAGIC SELECT f.date, f.county, f.cases 
# MAGIC   FROM ny_state_window f
# MAGIC   JOIN (
# MAGIC       SELECT county, sum(cases) as cases FROM ny_state_window GROUP BY county ORDER BY cases DESC LIMIT 10
# MAGIC     ) x ON x.county = f.county

# COMMAND ----------

# DBTITLE 1,ニューヨーク州における  3/6 - 3/20 での感染者数 - 教育機関の閉鎖 : 3/18
# MAGIC %sql
# MAGIC SELECT f.date, f.county, f.cases 
# MAGIC   FROM ny_state_window_m1 f
# MAGIC   JOIN (
# MAGIC       SELECT county, sum(cases) as cases FROM ny_state_window_m1 GROUP BY county ORDER BY cases DESC LIMIT 10
# MAGIC     ) x ON x.county = f.county

# COMMAND ----------

# MAGIC %md ## ワシントン州、ニューヨーク州の郡における10万人あたりのCOVID-19感染者数
# MAGIC 
# MAGIC 上のグラフで注意すべきことは、感染者数の総数だけでは人口密度を考慮していないため、ワシントン州とニューヨーク州を比較するのは困難であるということです。完璧ではありませんが、事前の策として人口推定値に対する割合でデータを検証することにします。
# MAGIC 
# MAGIC 人口に対するパーセンテージを見てみましょう。使用されている数字は2019年の国勢調査における郡ごとの人口推定値です。
# MAGIC 
# MAGIC **感染者のパーセンテージではなく、感染者数のトップ10の郡を検証していることに注意してください。**

# COMMAND ----------

# DBTITLE 1,ワシントン州における 3/6 - 3/20 での10万人あたりの感染者数 - 教育機関の閉鎖 : 3/13
# MAGIC %sql
# MAGIC SELECT f.date, f.county, f.cases_per_100Kpop 
# MAGIC   FROM wa_state_window f
# MAGIC   JOIN (
# MAGIC       SELECT county, sum(cases) as cases FROM wa_state_window GROUP BY county ORDER BY cases DESC LIMIT 10  
# MAGIC     ) x ON x.county = f.county

# COMMAND ----------

# DBTITLE 1,ニューヨーク州における 3/11 - 3/25 での感染者数 - 教育機関の閉鎖 : 3/18
# MAGIC %sql
# MAGIC SELECT f.date, f.county, f.cases_per_100Kpop 
# MAGIC   FROM ny_state_window f
# MAGIC   JOIN (
# MAGIC       SELECT county, sum(cases) as cases FROM ny_state_window GROUP BY county ORDER BY cases DESC LIMIT 10
# MAGIC     ) x ON x.county = f.county

# COMMAND ----------

# DBTITLE 1,ニューヨーク州における 3/6 - 3/20 での感染者数 - 教育機関の閉鎖 : 3/18
# MAGIC %sql
# MAGIC SELECT f.date, f.county, f.cases_per_100Kpop 
# MAGIC   FROM ny_state_window_m1 f
# MAGIC   JOIN (
# MAGIC       SELECT county, sum(cases) as cases FROM ny_state_window_m1 GROUP BY county ORDER BY cases DESC LIMIT 10
# MAGIC     ) x ON x.county = f.county

# COMMAND ----------

# MAGIC %md ## コロプレスマップによる州ごとの感染者数の可視化
# MAGIC 
# MAGIC 郡の中心の緯度経度を取得するために`map_fips_dedup`とjoinします。
# MAGIC 
# MAGIC [Choropleth map \- Wikipedia](https://en.wikipedia.org/wiki/Choropleth_map#:~:text=A%20choropleth%20map%20(from%20Greek,each%20area%2C%20such%20as%20population)

# COMMAND ----------

# 日付番号と中心地の緯度経度の抽出
wa_daynum = spark.sql("""select f.fips, f.county, f.date, f.day_num, cases as confirmed, cast(f.cases_per_100Kpop as int) as confirmed_per100K, deaths, cast(f.deaths_per_100Kpop as int) as deaths_per100K, m.lat, m.long_ from wa_state_window f join map_fips_dedup m on m.fips = f.fips""")
wa_daynum.createOrReplaceTempView("wa_daynum")
ny_daynum = spark.sql("""select cast(f.fips as int) as fips, f.county, f.date, f.day_num, cases as confirmed, cast(f.cases_per_100Kpop as int) as confirmed_per100K, deaths, cast(f.deaths_per_100Kpop as int) as deaths_per100K, m.lat, m.long_ from ny_state_window f join map_fips_dedup m on m.fips = f.fips""")
ny_daynum.createOrReplaceTempView("ny_daynum")
ny_daynum_m1 = spark.sql("""select cast(f.fips as int) as fips, f.county, f.date, f.day_num, cases as confirmed, cast(f.cases_per_100Kpop as int) as confirmed_per100K, deaths, cast(f.deaths_per_100Kpop as int) as deaths_per100K, m.lat, m.long_ from ny_state_window_m1 f join map_fips_dedup m on m.fips = f.fips""")
ny_daynum_m1.createOrReplaceTempView("ny_daynum_m1")

# COMMAND ----------

# 地理情報の取得
topo_usa = 'https://vega.github.io/vega-datasets/data/us-10m.json'
topo_wa = 'https://raw.githubusercontent.com/deldersveld/topojson/master/countries/us-states/WA-53-washington-counties.json'
topo_ny = 'https://raw.githubusercontent.com/deldersveld/topojson/master/countries/us-states/NY-36-new-york-counties.json'
us_counties = alt.topo_feature(topo_usa, 'counties')
wa_counties = alt.topo_feature(topo_wa, 'cb_2015_washington_county_20m')
ny_counties = alt.topo_feature(topo_ny, 'cb_2015_new_york_county_20m')

# COMMAND ----------

# ワシントン州の検証
confirmed_wa = wa_daynum.select("fips", "day_num", "date", "confirmed", "confirmed_per100K", "county").where("confirmed > 0").toPandas()
confirmed_wa['date'] = confirmed_wa['date'].astype(str)
deaths_wa = wa_daynum.select("lat", "long_", "day_num", "date", "deaths", "deaths_per100K", "county").where("deaths > 0").toPandas()
deaths_wa['date'] = deaths_wa['date'].astype(str)

# ニューヨーク州の検証
confirmed_ny = ny_daynum.select("fips", "day_num", "date", "confirmed", "confirmed_per100K", "county").where("confirmed > 0").toPandas()
confirmed_ny['date'] = confirmed_ny['date'].astype(str)
deaths_ny = ny_daynum.select("lat", "long_", "day_num", "date", "deaths", "deaths_per100K", "county").where("deaths > 0").toPandas()
deaths_ny['date'] = deaths_ny['date'].astype(str)

# 一週前のニューヨーク州の検証
confirmed_ny_m1 = ny_daynum_m1.select("fips", "day_num", "date", "confirmed", "confirmed_per100K", "county").where("confirmed > 0").toPandas()
confirmed_ny_m1['date'] = confirmed_ny_m1['date'].astype(str)
deaths_ny_m1 = ny_daynum_m1.select("lat", "long_", "day_num", "date", "deaths", "deaths_per100K", "county").where("deaths > 0").toPandas()
deaths_ny_m1['date'] = deaths_ny_m1['date'].astype(str)

# COMMAND ----------

# 州ごとのコロプレスマップを可視化する関数
def map_state(curr_day_num, state_txt, state_counties, confirmed, confirmed_min, confirmed_max, deaths, deaths_min, deaths_max):
  # date_strの取得
  date_str = confirmed[confirmed['day_num'] == 101]['date'].head(1).item()
  
  # 州
  base_state = alt.Chart(state_counties).mark_geoshape(
      fill='white',
      stroke='lightgray',
  ).properties(
      width=800,
      height=600,
  ).project(
      type='mercator'
  )

  # 郡
  base_state_counties = alt.Chart(us_counties).mark_geoshape(
  ).transform_lookup(
    lookup='id',
    from_=alt.LookupData(confirmed[(confirmed['confirmed_per100K'] > 0) & (confirmed['day_num'] == curr_day_num)], 'fips', ['confirmed_per100K', 'confirmed', 'county', 'date', 'fips'])  
  ).encode(
     color=alt.Color('confirmed_per100K:Q', scale=alt.Scale(type='log', domain=[confirmed_min, confirmed_max]), title='Confirmed per 100K'),
    tooltip=[
      alt.Tooltip('fips:O'),
      alt.Tooltip('confirmed:Q'),
      alt.Tooltip('confirmed_per100K:Q'),
      alt.Tooltip('county:N'),
      alt.Tooltip('date:N'),
    ],
  )

  # 緯度経度に基づく死者数
  points = alt.Chart(deaths[(deaths['deaths_per100K'] > 0) & (deaths['day_num'] == curr_day_num)]).mark_point(opacity=0.75, filled=True).encode(
    longitude='long_:Q',
    latitude='lat:Q',
    size=alt.Size('sum(deaths_per100K):Q', scale=alt.Scale(type='symlog', domain=[deaths_min, deaths_max]), title='Deaths per 100K'),
    color=alt.value('#BD595D'),
    stroke=alt.value('brown'),
    tooltip=[
      alt.Tooltip('lat'),
      alt.Tooltip('long_'),
      alt.Tooltip('deaths'),
      alt.Tooltip('county:N'),      
      alt.Tooltip('date:N'),      
    ],
  ).properties(
    # 図のタイトルの更新
    title=f'COVID-19 {state_txt} Confirmed Cases and Deaths per 100K by County [{curr_day_num}, {date_str}]'
  )

  return (base_state + base_state_counties + points)

# COMMAND ----------

# MAGIC %md 
# MAGIC | 要因 | ワシントン | ニューヨーク | 
# MAGIC | ------- | -- | -- | 
# MAGIC | 教育機関の閉鎖| 3/13/2020 | 3/18/2020 |
# MAGIC | Day 00 | 3/6/2020 | 3/11/2020 |
# MAGIC | Day 14 | 3/20/2020 | 3/25/2020 | 
# MAGIC | 最大感染者数 | 794 | 20011 |
# MAGIC | 最大死者数 | 68 | 280 |
# MAGIC | 10万人あたり最大感染者数 | 50.55 | 1222.97 | 
# MAGIC | 10万人あたり最大死者数 | 3.27 | 17.11 |

# COMMAND ----------

# MAGIC %md ### ワシントン州 (10万人あたりの感染者数、死者数)

# COMMAND ----------

map_state(101, 'WA', wa_counties, confirmed_wa, 1, 60, deaths_wa, 1, 5)

# COMMAND ----------

map_state(107, 'WA', wa_counties, confirmed_wa, 1, 60, deaths_wa, 1, 5)

# COMMAND ----------

map_state(114, 'WA', wa_counties, confirmed_wa, 1, 60, deaths_wa, 1, 5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ニューヨーク州 (10万人あたりの感染者数、死者数)

# COMMAND ----------

map_state(101, 'NY', ny_counties, confirmed_ny, 1, 1500, deaths_ny, 1, 20)

# COMMAND ----------

map_state(107, 'NY', ny_counties, confirmed_ny, 1, 1500, deaths_ny, 1, 20)

# COMMAND ----------

map_state(114, 'NY', ny_counties, confirmed_ny, 1, 1500, deaths_ny, 1, 20)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## コロプレスマップによるワシントン州、ニューヨーク州の郡の感染者数、死者数の可視化(スライダー付き)

# COMMAND ----------

# 州のコロプレスマップの可視化関数
def map_state_slider(state_txt, state_counties, confirmed, confirmed_min, confirmed_max, deaths, deaths_min, deaths_max, state_fips):
  # day_numによる感染日のPivot
  confirmed_pv = confirmed[['fips', 'day_num', 'confirmed']].copy()
  confirmed_pv['fips'] = confirmed_pv['fips'].astype(str)
  confirmed_pv['day_num'] = confirmed_pv['day_num'].astype(str)
  confirmed_pv['confirmed'] = confirmed_pv['confirmed'].astype('int64')
  confirmed_pv = confirmed_pv.pivot_table(index='fips', columns='day_num', values='confirmed', fill_value=0).reset_index()

  # day_numによる死亡日のPivot
  deaths_pv = deaths[['lat', 'long_', 'day_num', 'deaths']].copy()
  deaths_pv['day_num'] = deaths_pv['day_num'].astype(str)
  deaths_pv['deaths'] = deaths_pv['deaths'].astype('int64')
  deaths_pv = deaths_pv.pivot_table(index=['lat', 'long_'], columns='day_num', values='deaths', fill_value=0).reset_index()

  # スライダーのカラム名の抽出
  column_names = confirmed_pv.columns.tolist()

  # 最初の要素の削除 (`fips`)
  column_names.pop(0)

  # intへの変換
  column_values = [None] * len(column_names)
  for i in range(0, len(column_names)): column_values[i] = int(column_names[i]) 
  
  # max_rowsの無効化
  alt.data_transformers.disable_max_rows()

  # 地理情報
  us_states = alt.topo_feature(topo_usa, 'states')
  us_counties = alt.topo_feature(topo_usa, 'counties')

  # 州、郡の境界線
  base_state = alt.Chart(state_counties).mark_geoshape(
      fill='white',
      stroke='lightgray',
  ).properties(
      width=800,
      height=600,
  ).project(
      type='mercator'
  )

  # スライダーの設定
  min_day_num = column_values[0]
  max_day_num = column_values[len(column_values)-1]
  slider = alt.binding_range(min=min_day_num, max=max_day_num, step=1)
  slider_selection = alt.selection_single(fields=['day_num'], bind=slider, name="day_num", init={'day_num':min_day_num})


  # 州ごとの感染者数
  base_state_counties = alt.Chart(us_counties).mark_geoshape(
      stroke='black',
      strokeWidth=0.05
  ).transform_lookup(
      lookup='id',
      from_=alt.LookupData(confirmed_pv, 'fips', column_names)  
  ).transform_fold(
      column_names, as_=['day_num', 'confirmed']
  ).transform_calculate(
      state_id = "(datum.id / 1000)|0",
      day_num = 'parseInt(datum.day_num)',
      confirmed = 'isValid(datum.confirmed) ? datum.confirmed : -1'
  ).encode(
      color = alt.condition(
          'datum.confirmed > 0',      
          alt.Color('confirmed:Q', scale=alt.Scale(domain=(confirmed_min, confirmed_max), type='symlog')),
          alt.value('white')
        )  
  ).properties(
    # 図のタイトルの更新
    title=f'COVID-19 WA State Confirmed Cases by County'
  ).transform_filter(
      (alt.datum.state_id)==state_fips
  ).transform_filter(
      slider_selection
  )

  # 緯度経度に基づく死者数
  points = alt.Chart(
    deaths_pv
  ).mark_point(
    opacity=0.75, filled=True
  ).transform_fold(
    column_names, as_=['day_num', 'deaths']
  ).transform_calculate(
      day_num = 'parseInt(datum.day_num)',
      deaths = 'isValid(datum.deaths) ? datum.deaths : -1'  
  ).encode(
    longitude='long_:Q',
    latitude='lat:Q',
    size=alt.Size('deaths:Q', scale=alt.Scale(domain=(deaths_min, deaths_max), type='symlog'), title='deaths'),
    color=alt.value('#BD595D'),
    stroke=alt.value('brown'),
  ).add_selection(
      slider_selection
  ).transform_filter(
      slider_selection
  )

  # 感染者数(郡レベル)、死者数(点)
  return (base_state + base_state_counties + points) 

# COMMAND ----------

# MAGIC %md 
# MAGIC | 要因 | ワシントン | ニューヨーク | 
# MAGIC | ------- | -- | -- | 
# MAGIC | 教育機関の閉鎖| 3/13/2020 | 3/18/2020 |
# MAGIC | Day 00 | 3/6/2020 | 3/11/2020 |
# MAGIC | Day 14 | 3/20/2020 | 3/25/2020 | 
# MAGIC | 最大感染者数 | 794 | 20011 |
# MAGIC | 最大死者数 | 68 | 280 |
# MAGIC | 10万人あたり最大感染者数 | 50.55 | 1222.97 | 
# MAGIC | 10万人あたり最大死者数 | 3.27 | 17.11 |

# COMMAND ----------

map_state_slider('WA', wa_counties, confirmed_wa, 1, 800, deaths_wa, 1, 70, 53)

# COMMAND ----------

map_state_slider('NY', ny_counties, confirmed_ny, 1, 21000, deaths_ny, 1, 300, 36)

# COMMAND ----------

# MAGIC %md ## ワシントン州、ニューヨーク州の郡におけるCOVID-19感染者数、死者数(地図、グラフ)

# COMMAND ----------

# map_state_graph
def map_state_graph(state_txt, state_counties, confirmed, confirmed_min, confirmed_max, deaths, deaths_min, deaths_max, state_fips):
  
  # 感染者数のpivot (日付による)
  confirmed_pv2 = confirmed[['fips', 'date', 'confirmed']].copy()
  confirmed_pv2['fips'] = confirmed_pv2['fips'].astype(str)
  confirmed_pv2['date'] = confirmed_pv2['date'].astype(str)
  confirmed_pv2['confirmed'] = confirmed_pv2['confirmed'].astype('int64')
  confirmed_pv2 = confirmed_pv2.pivot_table(index='fips', columns='date', values='confirmed', fill_value=0).reset_index()

  # 死者数のpivot
  deaths_pv2 = deaths[['lat', 'long_', 'date', 'deaths']].copy()
  deaths_pv2['date'] = deaths_pv2['date'].astype(str)
  deaths_pv2['deaths'] = deaths_pv2['deaths'].astype('int64')
  deaths_pv2 = deaths_pv2.pivot_table(index=['lat', 'long_'], columns='date', values='deaths', fill_value=0).reset_index()

  # スライドバーのためのカラム取得
  column_names2 = confirmed_pv2.columns.tolist()

  # 最初の要素の削除 (`fips`)
  column_names2.pop(0)

  # 日付の選択
  pts = alt.selection(type="single", encodings=['x'])

  # 州
  base_state = alt.Chart(state_counties).mark_geoshape(
      fill='white',
      stroke='lightgray',
  ).properties(
      width=800,
      height=600,
  ).project(
      type='mercator'
  )

  # 州、郡
  base_state_counties = alt.Chart(us_counties).mark_geoshape(
    stroke='black',
    strokeWidth=0.05,
  ).transform_lookup(
    lookup='id',
   from_=alt.LookupData(confirmed_pv2, 'fips', column_names2)
   ).transform_fold(
     column_names2, as_=['date', 'confirmed']
  ).transform_calculate(
      state_id = "(datum.id / 1000)|0",
      date = 'datum.date',
      confirmed = 'isValid(datum.confirmed) ? datum.confirmed : -1'
  ).encode(
       color = alt.condition(
          'datum.confirmed > 0',      
          alt.Color('confirmed:Q', scale=alt.Scale(domain=(confirmed_min, confirmed_max), type='symlog')),
          alt.value('white')
        )  
  ).transform_filter(
    pts
  ).transform_filter(
      (alt.datum.state_id)==state_fips
  )

  # 棒グラフ
  bar = alt.Chart(confirmed).mark_bar().encode(
      x='date:N',
      y='confirmed_per100K:Q',
      color=alt.condition(pts, alt.ColorValue("steelblue"), alt.ColorValue("grey"))
  ).properties(
      width=800,
      height=200,
      title='Confirmed Cases per 100K'
  ).add_selection(pts)

  # 死者数
  points = alt.Chart(deaths).mark_point(opacity=0.75, filled=True).encode(
    longitude='long_:Q',
    latitude='lat:Q',
    size=alt.Size('sum(deaths):Q', scale=alt.Scale(domain=[deaths_min, deaths_max]), title='Deaths'),
    color=alt.value('#BD595D'),
    stroke=alt.value('brown'),
    tooltip=[
      alt.Tooltip('lat'),
      alt.Tooltip('long_'),
      alt.Tooltip('deaths'),
      alt.Tooltip('county:N'),      
      alt.Tooltip('date:N'),      
    ],
  ).properties(
    # 図のタイトルの更新
    title=f'COVID-19 Confirmed Cases and Deaths by County'
  ).transform_filter(
      pts
  )

  return (base_state + base_state_counties + points) & bar

# COMMAND ----------

map_state_graph('WA', wa_counties, confirmed_wa, 1, 800, deaths_wa, 1, 70, 53)

# COMMAND ----------

map_state_graph('NY', ny_counties, confirmed_ny, 1, 21000, deaths_ny, 1, 300, 36)

# COMMAND ----------

map_state_graph('NY', ny_counties, confirmed_ny_m1, 1, 4500, deaths_ny, 1, 70, 36)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
