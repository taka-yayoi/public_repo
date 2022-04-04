# Databricks notebook source
# MAGIC %md
# MAGIC # Pythonにおけるチャートとグラフのタイプ
# MAGIC 
# MAGIC このノートブックでは、Databricksに組み込まれている様々なチャートとグラフをカバーします。
# MAGIC 
# MAGIC このノートブックにおけるビジュアライゼーションで表示されるテストデータの生成にPythonを使用していますが、これらのチャートやグラフの設定方法は全てのノートブックに対して適用されます。

# COMMAND ----------

# MAGIC %md  
# MAGIC ## テーブルビュー
# MAGIC 
# MAGIC **テーブルビュー**はデータを参照する最も基本的な方法です。テーブルビューでは最初の1000行のみが表示されます。

# COMMAND ----------

from pyspark.sql import Row

array = map(lambda x: Row(key="k_%04d" % x, value = x), range(1, 5001))
largeDataFrame = spark.createDataFrame(sc.parallelize(array))
largeDataFrame.registerTempTable("largeTable")
display(spark.sql("select * from largeTable"))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Plot Optionsによる設定
# MAGIC 
# MAGIC - ほとんどのグラフタイプにおけるX軸として通常表示される制御変数を**Keys**セクションに指定します。ほとんどのグラフでは、キーに対して約1000の値をプロットできます。繰り返しになりますが、これはグラフによって異なります。
# MAGIC -  通常Y軸に表示される観測変数を**Values**セクションに指定します。多くのグラフタイプにおいては観測される数値になる傾向があります。
# MAGIC - データを分割する方法を**Series groupings**で指定します。棒グラフにおいては、それぞれの系列グルーピングには異なる色と、それぞれの系列グルーピングを示す凡例が与えられます。多くのグラフタイプでは、10以下のユニークな値を持つ系列グルーピングのみを取り扱うことができます。
# MAGIC 
# MAGIC **いくつかのグラフタイプではより多くのオプションを指定することができ、適用できる際にはこれらも議論します。**

# COMMAND ----------

# MAGIC %md  
# MAGIC ## ピボットテーブル
# MAGIC 
# MAGIC **Pivot table**はテーブル形式でデータを参照するもう一つの方法です。
# MAGIC 
# MAGIC テーブルで生の結果を表示するのではなく、テーブルに格納されているデータのソート、合計や平均を自動で計算します。
# MAGIC 
# MAGIC - ピボットテーブルの詳細に関しては右を参照ください: http://en.wikipedia.org/wiki/Pivot_table
# MAGIC - ピボットテーブルに対しては、キー、系列グルーピング、値のフィールドを指定することができます。
# MAGIC - **Key**は最初のカラムとなり、ピボットテーブルではキーあたり1行が存在します。
# MAGIC - **Series Grouping**に対して、個々のユニークな値のカラムを追加することができます。
# MAGIC - テーブルのセルには**Values**フィールドが含まれます。値は集計関数を用いて結合できるように数値である必要があります。
# MAGIC - ピボットテーブルのセルには、オリジナルのテーブルの複数行から計算された値が入ります。
# MAGIC   - オリジナルの行を結合するための方法として**SUM**、**AVG**、**MIN**、**MAX**、**COUNT**を選択します。
# MAGIC - セルの値を計算するために、Databricksのクラウドのサーバーサイドでピボット処理が行われます。 

# COMMAND ----------

# MAGIC %md
# MAGIC ピボットテーブルを作成するには、下のグラフアイコンをクリックして**Pivot**を選択します。

# COMMAND ----------

# Plot Optionsボタンをクリックして、どのようにピボットテーブルが設定されるのかを見てみましょう。
from pyspark.sql import Row

largePivotSeries = map(lambda x: Row(key="k_%03d" % (x % 200), series_grouping = "group_%d" % (x % 3), value = x), range(1, 5001))
largePivotDataFrame = spark.createDataFrame(sc.parallelize(largePivotSeries))
largePivotDataFrame.registerTempTable("table_to_be_pivoted")
display(spark.sql("select * from table_to_be_pivoted"))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ピボットテーブルの別の考え方として、キーと系列グルーピングでオリジナルのテーブルをグルーピングし、(キー、系列グルーピング、集計関数)タプルを出力するのではなく、スキーマがキーと、系列グルーピングそれぞれのユニークな値であるテーブルを出力するというものがあります。
# MAGIC 
# MAGIC - 上のピボットテーブルの全てのデータを含みますが結果のスキーマが異なる、以下のgroup_by文の結果をご覧ください。

# COMMAND ----------

# MAGIC %sql select key, series_grouping, sum(value) from table_to_be_pivoted group by key, series_grouping order by key, series_grouping

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 棒グラフ
# MAGIC 
# MAGIC **Bar Chart**はピボットテーブルのビジュアルなグラフであり、データを可視化する基本的な方法となります。
# MAGIC - 以下のグラフを設定するために **Plot Options...** を使用します。
# MAGIC - **Key**はX軸に表示される*Year*となります。
# MAGIC - **Series groupings**は*Product*であり、それぞれに対して異なる色が割り当てられます。
# MAGIC - **Values**はY軸に表示される*salesAmount*となります。
# MAGIC - 集計方法として**Sum**が選択され、ピボットのために行が加算されます。

# COMMAND ----------

from pyspark.sql import Row
salesEntryDataFrame = spark.createDataFrame(sc.parallelize([
  Row(category="fruits_and_vegetables", product="apples", year=2012, salesAmount=100.50),
  Row(category="fruits_and_vegetables", product="oranges", year=2012, salesAmount=100.75),
  Row(category="fruits_and_vegetables", product="apples", year=2013, salesAmount=200.25),
  Row(category="fruits_and_vegetables", product="oranges", year=2013, salesAmount=300.65),
  Row(category="fruits_and_vegetables", product="apples", year=2014, salesAmount=300.65),
  Row(category="fruits_and_vegetables", product="oranges", year=2015, salesAmount=100.35),
  Row(category="butcher_shop", product="beef", year=2012, salesAmount=200.50),
  Row(category="butcher_shop", product="chicken", year=2012, salesAmount=200.75),
  Row(category="butcher_shop", product="pork", year=2013, salesAmount=400.25),
  Row(category="butcher_shop", product="beef", year=2013, salesAmount=600.65),
  Row(category="butcher_shop", product="beef", year=2014, salesAmount=600.65),
  Row(category="butcher_shop", product="chicken", year=2015, salesAmount=200.35),
  Row(category="misc", product="gum", year=2012, salesAmount=400.50),
  Row(category="misc", product="cleaning_supplies", year=2012, salesAmount=400.75),
  Row(category="misc", product="greeting_cards", year=2013, salesAmount=800.25),
  Row(category="misc", product="kitchen_utensils", year=2013, salesAmount=1200.65),
  Row(category="misc", product="cleaning_supplies", year=2014, salesAmount=1200.65),
  Row(category="misc", product="cleaning_supplies", year=2015, salesAmount=400.35)
]))
salesEntryDataFrame.registerTempTable("test_sales_table")
display(spark.sql("select * from test_sales_table"))

# COMMAND ----------

# MAGIC %md **小技:** チャートのそれぞれのバーの上にマウスカーソルを移動すると正確な値を確認することができます。

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 折れ線グラフ
# MAGIC 
# MAGIC **Line Graph**はデータセットのトレンドをハイライトする別のピボットテーブルグラフのサンプルとなります。
# MAGIC - 以下のグラフを設定するために **Plot Options...** を使用します。
# MAGIC - **Key**はX軸に表示される*Year*となります。
# MAGIC - **Series groupings**は*Category*であり、それぞれに対して異なる色が割り当てられます。
# MAGIC - **Values**はY軸に表示される*salesAmount*となります。
# MAGIC - 集計方法として**Sum**を選択します。

# COMMAND ----------

# MAGIC %sql select cast(string(year) as date) as year, category, salesAmount from test_sales_table

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 円グラフ
# MAGIC 
# MAGIC **Pie Chart**は、データセットの値のパーセンテージを参照できるピボットテーブルのグラフタイプとなります。
# MAGIC * **注意:** 上のサンプルとは異なり、キーと系列グルーピングがスイッチされます。
# MAGIC - 以下のグラフを設定するために **Plot Options...** を使用します。
# MAGIC - **Key**はX軸に表示される*Year*となります。
# MAGIC - **Series groupings**は*Category*であり、それぞれに対して異なる色が割り当てられます。
# MAGIC - **Values**はY軸に表示される*salesAmount*となります。
# MAGIC - 集計方法として**Sum**を選択します。

# COMMAND ----------

# MAGIC %sql select * from test_sales_table

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 地図グラフ
# MAGIC 
# MAGIC **Map Graph**は地図上にデータを可視化することができます。
# MAGIC 
# MAGIC - 以下のグラフを設定するために **Plot Options...** を使用します。
# MAGIC - **Keys**には位置を示すフィールドを含める必要があります。
# MAGIC - **Series groupings**はWorld Mapグラフでは常に無視されます。
# MAGIC - **Values**には、数値を持つ1つのフィールドを含める必要があります。
# MAGIC - 同じ位置のキーを持つ複数行が存在するので、一つのキーに対して値を組み合わせるために"Sum", "Avg", "Min", "Max", "COUNT"を選択します。
# MAGIC - 異なる値に対して地図上の異なる色が割り当てられ、レンジは常に均等な間隔となります。
# MAGIC 
# MAGIC **ティップ:** 値が均等に分布していない場合、グラフに対してスムーシング関数を適用してください。

# COMMAND ----------

from pyspark.sql import Row
stateRDD = spark.createDataFrame(sc.parallelize([
  Row(state="MO", value=1), Row(state="MO", value=10),
  Row(state="NH", value=4),
  Row(state="MA", value=8),
  Row(state="NY", value=4),
  Row(state="CA", value=7)
]))
stateRDD.registerTempTable("test_state_table")
display(spark.sql("Select * from test_state_table"))

# COMMAND ----------

# MAGIC %md
# MAGIC 世界地図にグラフをプロットするには、キーとして[country codes in ISO 3166-1 alpha-3 format](http://en.wikipedia.org/wiki/ISO_3166-1_alpha-3)を使用します。

# COMMAND ----------

from pyspark.sql import Row
worldRDD = spark.createDataFrame(sc.parallelize([
  Row(country="USA", value=1000),
  Row(country="JPN", value=23),
  Row(country="GBR", value=23),
  Row(country="FRA", value=21),
  Row(country="TUR", value=3)
]))
display(worldRDD)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 散布図
# MAGIC 
# MAGIC **Scatter Plot**を用いることで、二つの変数に相関があるかどうかを確認することができます。
# MAGIC - 以下のグラフを設定するために **Plot Options...** を使用します。
# MAGIC - **Keys**はグラフのポイントの色と凡例に使用されます。
# MAGIC - **Series groupings**は無視されます。
# MAGIC - **Values**には、最低2つのフィールドを含める必要があります。グラフは値としてa、b、cを持っています。
# MAGIC - 結果のプロットの対角線は変数のカーネル密度となります。
# MAGIC - 行には常にY軸の変数を持ち、カラムはX軸の変数を持ちます。

# COMMAND ----------

from pyspark.sql import Row
scatterPlotRDD = spark.createDataFrame(sc.parallelize([
  Row(key="k1", a=0.2, b=120, c=1), Row(key="k1", a=0.4, b=140, c=1), Row(key="k1", a=0.6, b=160, c=1), Row(key="k1", a=0.8, b=180, c=1),
  Row(key="k2", a=0.2, b=220, c=1), Row(key="k2", a=0.4, b=240, c=1), Row(key="k2", a=0.6, b=260, c=1), Row(key="k2", a=0.8, b=280, c=1),
  Row(key="k1", a=1.8, b=120, c=1), Row(key="k1", a=1.4, b=140, c=1), Row(key="k1", a=1.6, b=160, c=1), Row(key="k1", a=1.8, b=180, c=1),
  Row(key="k2", a=1.8, b=220, c=2), Row(key="k2", a=1.4, b=240, c=2), Row(key="k2", a=1.6, b=260, c=2), Row(key="k2", a=1.8, b=280, c=2),
  Row(key="k1", a=2.2, b=120, c=1), Row(key="k1", a=2.4, b=140, c=1), Row(key="k1", a=2.6, b=160, c=1), Row(key="k1", a=2.8, b=180, c=1),
  Row(key="k2", a=2.2, b=220, c=3), Row(key="k2", a=2.4, b=240, c=3), Row(key="k2", a=2.6, b=260, c=3), Row(key="k2", a=2.8, b=280, c=3)
]))
display(scatterPlotRDD)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 散布図のLOESSフィット曲線
# MAGIC 
# MAGIC [LOESS](https://en.wikipedia.org/wiki/Local_regression)は、お使いの散布図におけるデータのトレンドを説明するスムース推定曲線を生成するために、データに対する局所的な回帰を実行する方法です。データポイントの近傍点で曲線を内挿することで、これを実現します。LOESSフィット曲線は、プロットをスムースにするために使用する近隣のポイントの数を指定する帯域パラメーターで制御されます。高い帯域パラメーター(1に近い値)は非常にスムースな曲線を提供しますが、一般的なトレンドを捉えられない場合があります。一方、低い帯域パラメーター(0に近い値)はあまりプロットをスムースにしません。
# MAGIC 
# MAGIC LOESSフィット曲線は、散布図で利用できます。ここでは、お使いのサンプル図に対するLOESSフィットをどのように作成するのかを示します。
# MAGIC 
# MAGIC **注意:** データセットに5000ポイント以上ある場合、LOESSフィットは最初の5000ポイントを用いて計算されます。

# COMMAND ----------

import numpy as np
import math

# 散布図のデータポイントを作成します
np.random.seed(0)
points = sc.parallelize(range(0,1000)).map(lambda x: (x/100.0, 4 * math.sin(x/100.0) + np.random.normal(4,1))).toDF()

# COMMAND ----------

# MAGIC %md 
# MAGIC テーブル表示の左下にあるコントロールを用いてこのデータを散布図にします。
# MAGIC 
# MAGIC ![plot-menu-pick-scatter](https://docs.databricks.com/_static/images/notebooks/plot-menu-pick-scatter.png)
# MAGIC 
# MAGIC *Plot Options*を選択した際にLOESSフィットオプションにアクセスすることができます。
# MAGIC 
# MAGIC ![screen shot 2015-10-13 at 3 43 16 pm](https://cloud.githubusercontent.com/assets/7594753/10472058/d7ce763c-71d0-11e5-91b2-4d90e9a704c9.png)
# MAGIC 
# MAGIC ノイジーなデータに対して曲線がどのように適用されるのかを見るために、帯域パラメーターを変えて実験することができます。
# MAGIC 
# MAGIC 変更を受け入れると、散布図に対するLOESSフィットを確認することができます！

# COMMAND ----------

display(points)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## ヒストグラム
# MAGIC 
# MAGIC **Histogram**を用いることで値の分布を確認することができます。
# MAGIC 
# MAGIC - 以下のグラフを設定するために **Plot Options...** を使用します。
# MAGIC - **Values**には1つのフィールドを含める必要があります。
# MAGIC - **Series groupings**は常に無視されます。
# MAGIC - **Keys**では2フィールドまで指定することができます。
# MAGIC   - キーが指定されない場合、1つのヒストグラムが出力されます。
# MAGIC   - 2つのフィールドが指定された場合、ヒストグラムが格子状に表示されます。
# MAGIC - **Aggregation**は適用できません。
# MAGIC - **Number of bins**はヒストグラムプロットでのみ表示される特殊なオプションであり、ヒストグラムのビンの数を制御します。
# MAGIC - ビンはヒストグラムに対してサーバーサイドで計算されるので、テーブルの全ての行がプロットされます。

# COMMAND ----------

from pyspark.sql import Row
# プロットされた値の正確な値をヒストグラムから読み取るためにはエントリーにマウスカーソルを移動します
histogramRDD = spark.createDataFrame(sc.parallelize([
  Row(key1="a", key2="x", val=0.2), Row(key1="a", key2="x", val=0.4), Row(key1="a", key2="x", val=0.6), Row(key1="a", key2="x", val=0.8), Row(key1="a", key2="x", val=1.0), 
  Row(key1="b", key2="z", val=0.2), Row(key1="b", key2="x", val=0.4), Row(key1="b", key2="x", val=0.6), Row(key1="b", key2="y", val=0.8), Row(key1="b", key2="x", val=1.0), 
  Row(key1="a", key2="x", val=0.2), Row(key1="a", key2="y", val=0.4), Row(key1="a", key2="x", val=0.6), Row(key1="a", key2="x", val=0.8), Row(key1="a", key2="x", val=1.0), 
  Row(key1="b", key2="x", val=0.2), Row(key1="b", key2="x", val=0.4), Row(key1="b", key2="x", val=0.6), Row(key1="b", key2="z", val=0.8), Row(key1="b", key2="x", val=1.0)]))
display(histogramRDD)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Quantileプロット
# MAGIC 
# MAGIC **Quantile plot**を用いることで、指定された分位数に対する値が何かを確認することができます。
# MAGIC - Quantile Plotの詳細については、 http://en.wikipedia.org/wiki/Normal_probability_plot を参照ください。
# MAGIC - 以下のグラフを設定するために **Plot Options...** を使用します。
# MAGIC - **Values**には1つのフィールドを含める必要があります。
# MAGIC - **Series groupings**は常に無視されます。
# MAGIC - **Keys**では2フィールドまで指定することができます。
# MAGIC   - キーが指定されない場合、1つのヒストグラムが出力されます。
# MAGIC   - 2つのフィールドが指定された場合、Quantileプロットが格子状に表示されます。
# MAGIC - **Aggregation**は適用できません。
# MAGIC - Quantileは現状サーバーサイドでは計算されませんので、プロットには1000行のみが反映されます。

# COMMAND ----------

from pyspark.sql import Row
quantileSeries = map(lambda x: Row(key="key_%01d" % (x % 4), grouping="group_%01d" % (x % 3), otherField=x, value=x*x), range(1, 5001))
quantileSeriesRDD = spark.createDataFrame(sc.parallelize(quantileSeries))
display(quantileSeriesRDD)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Q-Qプロット
# MAGIC 
# MAGIC Q-Qプロットは、フィールドの値の分布を表示します。
# MAGIC - Q-Qプロットの詳細については、 http://en.wikipedia.org/wiki/Q%E2%80%93Q_plot を参照ください。
# MAGIC * **Value**は1つあるいは2つのフィールドを含める必要があります。
# MAGIC * **Series Grouping**は常に無視されます。
# MAGIC - **Keys**では2フィールドまで指定することができます。
# MAGIC   - キーが指定されない場合、1つのヒストグラムが出力されます。
# MAGIC   - 2つのフィールドが指定された場合、Quantileプロットが格子状に表示されます。
# MAGIC - **Aggregation**は適用できません。
# MAGIC - Q-Qプロットは現状サーバーサイドでは計算されませんので、プロットには1000行のみが反映されます。

# COMMAND ----------

from pyspark.sql import Row
qqPlotSeries = map(lambda x: Row(key="key_%03d" % (x % 5), grouping="group_%01d" % (x % 3), value=x, value_squared=x*x), range(1, 5001))
qqPlotRDD = spark.createDataFrame(sc.parallelize(qqPlotSeries))

# COMMAND ----------

# MAGIC %md 
# MAGIC Valuesに対して1つのフィールドのみが指定された場合、Q-Qプロットは単に正規分布とフィールドの分布を比較します。

# COMMAND ----------

display(qqPlotRDD)

# COMMAND ----------

# MAGIC %md 
# MAGIC Valuesに2つのフィールドを指定すると、2つのフィールドの分布を比較します。

# COMMAND ----------

display(qqPlotRDD)

# COMMAND ----------

# MAGIC %md 
# MAGIC Q-Qプロットで格子プロットを作成するには、最大2つのキーを設定します。

# COMMAND ----------

display(qqPlotRDD)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ボックスプロット
# MAGIC 
# MAGIC **Box plot**は期待される値のレンジが何であるのか、外れ値があるのかに対するアイデアを与えてくれます。
# MAGIC 
# MAGIC * ボックスプロットに関しては http://en.wikipedia.org/wiki/Box_plot をご覧ください。
# MAGIC * **Value**には1つのフィールドを含める必要があります。
# MAGIC * **Series Grouping**は常に無視されます。
# MAGIC * **Keys**を追加することができます。
# MAGIC   * キーのそれぞれの値に対して一つのボックス・ウィスカープロットが作成されます。
# MAGIC * **Aggregation**は適用できません。
# MAGIC * ボックスプロットは現状サーバーサイドでは計算されませんので、プロットには1000行のみが反映されます。
# MAGIC * ボックスの上にマウスカーソルを移動すると、ボックスプロットの中央値が表示されます。

# COMMAND ----------

from pyspark.sql import Row
import random
# ボックスの上にマウスカーソルを移動すると、ボックスプロットの中央値が表示されます。
boxSeries = map(lambda x: Row(key="key_%01d" % (x % 2), grouping="group_%01d" % (x % 3), value=random.randint(0, x)), range(1, 5001))
boxSeriesRDD = spark.createDataFrame(sc.parallelize(boxSeries))
display(boxSeriesRDD)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
