# Databricks notebook source
# MAGIC %md # ipywidgetsのサンプル
# MAGIC 
# MAGIC このノートブックではDatabricksノートブックでどのようにインタラクティブなipywidgetsを使用するのかを説明します。このサンプルでは、Databricksにビルトインされているデータセットの一つを使用します。
# MAGIC 
# MAGIC ipywidgetsの詳細に関しては、[ipywidgetsのドキュメント](https://ipywidgets.readthedocs.io/en/7.7.0/index.html)を参照ください。
# MAGIC 
# MAGIC このノートブックでは、データサイエンティストがどのように新たなデータセットをブラウズするのかをウォークスルーします。ipywidgetsのサンプルにスキップするにはセル10に移動してください。
# MAGIC 
# MAGIC ## 要件
# MAGIC 
# MAGIC Databricksランタイム11.0以降

# COMMAND ----------

# MAGIC %md The bike-sharingデータセットには、日付、気候、登録ユーザーによって、あるいはカジュアルにレンタルされた自転車の数を含む日毎のデータ2年分が含まれています。 

# COMMAND ----------

sparkDF = spark.read.csv("/databricks-datasets/bikeSharing/data-001/day.csv", header="true", inferSchema="true")
display(sparkDF)

# COMMAND ----------

# MAGIC %md 
# MAGIC データセットを探索する一般的な方法は、関係性を見出すために変数をプロットすることです。次のセルでは、日毎にレンタルされた自転車の総数と、その日に記録された気温の散布図を作成します。

# COMMAND ----------

pdf = sparkDF.toPandas()
pdf.plot.scatter(x='temp', y='cnt')

# COMMAND ----------

# MAGIC %md
# MAGIC 異なる説明変数と目的変数を簡単に参照できるようにするために、関数を作成することもできます。

# COMMAND ----------

def f(x ='temp', y = 'cnt'):
  pdf.plot.scatter(x=x, y=y)

# COMMAND ----------

# MAGIC %md 
# MAGIC これで、関係性をプロットするために任意の2つのカラム名を指定することができます。

# COMMAND ----------

f('hum', 'casual')

# COMMAND ----------

# MAGIC %md 
# MAGIC ipywidgetsを用いることで、ご自身のプロットにインタラクティブなコントローラを追加することができます。`@interact`[デコレーター](https://ipywidgets.readthedocs.io/en/latest/examples/Using%20Interact.html)を用いることで、1行のコードでインタラクティブなウィジェットを定義することができます。
# MAGIC 
# MAGIC 以下のセルを実行した後、y軸に`cnt`、x軸にデフォルト値である`temp`を持つプロットが表示されます。`fit=True`のデフォルト値のため、プロットには回帰曲線が含まれます。
# MAGIC 
# MAGIC x軸に異なる値を選択するために、上のプロットのwidgetセレクターを使用したり、回帰曲線の表示のオンオフを行うことができます。ウィジェット用いて選択を行うと、即座にプロットに反映されます。
# MAGIC 
# MAGIC 異なるタイプのipywidgetsに関しては、[ipywidgetsのドキュメント](https://ipywidgets.readthedocs.io/en/7.7.0/index.html)をご覧ください。

# COMMAND ----------

import ipywidgets as widgets
import seaborn as sns 
from ipywidgets import interact

# このコードでは、リスト ['temp', 'atemp', 'hum', 'windspeed'] によってドロップダウンメニューウィジェットが作成されます。 
# 変数にTrue/Falseを設定すると (`fit=True`) チェックボックスウィジェットが作成されます。
@interact(column=['temp', 'atemp', 'hum', 'windspeed'], fit=True)
def f(column='temp', fit=True):
  sns.lmplot(x=column, y='cnt', data=pdf, fit_reg=fit)

# COMMAND ----------

# MAGIC %md 
# MAGIC 次のセルでは、ドロップダウンメニューを用いることで、ヒストグラムに任意の気候変数をプロットすることができます。
# MAGIC また、ヒストグラムのビンの数を指定するためにスライダーを使うことができます。

# COMMAND ----------

# このコードでは、 `(bins=(2, 20, 2)` が2から20の間で2ごとの値を指定できる整数値スライダーウイジェットを定義します。
@interact(bins=(2, 20, 2), value=['temp', 'atemp', 'hum', 'windspeed'])
def plot_histogram(bins, value):
  pdf = sparkDF.toPandas()
  pdf.hist(column=value, bins=bins)

# COMMAND ----------


