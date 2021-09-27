-- Databricks notebook source
-- MAGIC %md ## 人口 対 販売価格

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC このノートブックのデフォルト言語はSQLなので、以下のいくつかのセルでは`%python`マジックコマンドを使用します。

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # オプションを指定してCSVデータソースからデータを読み込みます:
-- MAGIC #  - 最初の行はヘッダーです
-- MAGIC #  - 自動的にデータのスキーマを推測します
-- MAGIC data = spark.read.csv("/databricks-datasets/samples/population-vs-price/data_geo.csv", header="true", inferSchema="true") 
-- MAGIC data.cache()  # 高速に再利用できるようにキャッシュします
-- MAGIC data = data.dropna() # 欠損値を持つ行を削除します

-- COMMAND ----------

-- MAGIC %python
-- MAGIC data.take(10)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC display(data)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # SQLでアクセスできるようにテーブル(一時ビュー)として登録します
-- MAGIC data.createOrReplaceTempView("data_geo")

-- COMMAND ----------

-- MAGIC %md #### 州の上にカーソルを移動すると2015年の住宅販売価格の中央値が表示されます

-- COMMAND ----------

select `State Code`, `2015 median sales price` from data_geo

-- COMMAND ----------

-- MAGIC %md ## 2015年の住宅販売価格の中央値におけるトップ10の都市

-- COMMAND ----------

select City, `2014 Population estimate`/1000 as `2014 Population Estimate (1000s)`, `2015 median sales price` as `2015 Median Sales Price (1000s)` from data_geo order by `2015 median sales price` desc limit 10;

-- COMMAND ----------

-- MAGIC %md ## ワシントン州の2014年人口の推定値

-- COMMAND ----------

select City, `2014 Population estimate` from data_geo where `State Code` = 'WA';

-- COMMAND ----------

-- MAGIC %md ## 2015年の販売価格の中央値をボックスプロット
-- MAGIC 
-- MAGIC ボックスプロットは平均値と価格の変動幅を表示します。

-- COMMAND ----------

select `State Code`, `2015 median sales price` from data_geo order by `2015 median sales price` desc;

-- COMMAND ----------

-- MAGIC %md ## 2015年の販売各中央値を州ごとにヒストグラムとして表示

-- COMMAND ----------

select `State Code`, `2015 median sales price` from data_geo order by `2015 median sales price` desc;

-- COMMAND ----------

-- MAGIC %md ## 30万ドル以上の販売価格中央値において、Quantile Plotを表示
-- MAGIC 
-- MAGIC Quantile Plotは分布を表示(この場合、都市に渡る販売価格の分布)するのに役立ち、分布の偏りをハイライトすることにも使えます。

-- COMMAND ----------

select `State Code`, `2015 median sales price` from data_geo where `2015 median sales price` >= 300;

-- COMMAND ----------

-- MAGIC %md ## 2015年の販売価格中央値が30万ドル以上の都市

-- COMMAND ----------

select `City`, `State Code`, `2015 median sales price` from data_geo where `2015 median sales price` >= 300 limit 20;

-- COMMAND ----------

-- MAGIC %md ## 2015年の販売価格中央値のQ-Qプロット
-- MAGIC 
-- MAGIC Q-Qプロットも分布を表示する手法です。詳細は[Q\-Qプロット \- Wikipedia](https://ja.wikipedia.org/wiki/Q-Q%E3%83%97%E3%83%AD%E3%83%83%E3%83%88)を参照ください。

-- COMMAND ----------

select `State Code`, case when `2015 median sales price` >= 300 then '>=300K' when `2015 median sales price` < 300 then '< 300K' end as `Category`, `2015 median sales price` from data_geo order by `2015 median sales price` desc;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # END
