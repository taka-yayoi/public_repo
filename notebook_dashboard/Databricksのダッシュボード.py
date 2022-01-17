# Databricks notebook source
# MAGIC %md # Databricksのダッシュボード
# MAGIC 
# MAGIC ダッシュボードを用いることで、ノートブックの出力結果のグラフやビジュアライゼーションを組織内で公開し、プレゼンテーション形式で共有することができます。このノートブックでは、ダッシュボードの作成、編集、削除の方法を説明します。
# MAGIC 
# MAGIC [Databricksのダッシュボード \- Qiita](https://qiita.com/taka_yayoi/items/4b200d08ab3d0863d40a)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 使ってみる
# MAGIC 
# MAGIC ダッシュボードはエレメントから構成されます。エレメントはノートブックセルのアウトプットによって生成されます。ここで構築するダッシュボードで使用するために、いくつかのエレメント(セル)を作成してみましょう。最初のセルでは、``displayHTML()``関数を用いてダッシュボードのタイトルを作成します。

# COMMAND ----------

displayHTML("""<font size="6" color="red" face="sans-serif">バイクシェアリング分析ダッシュボード</font>""")

# COMMAND ----------

# MAGIC %md Markdownを使ってタイトル、ラベルを作成することもできます。
# MAGIC 
# MAGIC ```
# MAGIC %md ## ダッシュボードのラベル
# MAGIC ```
# MAGIC 
# MAGIC これは以下のようにレンダリングされます:
# MAGIC 
# MAGIC ## ダッシュボードのラベル

# COMMAND ----------

# MAGIC %md 
# MAGIC Databricksがホストするサンプルデータ(Databricksデータセット)に含まれるバイクシェアリングのデータセットを表示するダッシュボードを作成します。

# COMMAND ----------

# この例では、`dbfs`(Databricksファイルシステム)にあるバイクシェアリングデータセットを使用します。
df = spark.read.format("csv").option("inferSchema", "true").option("header", "true").load("dbfs:/databricks-datasets/bikeSharing/data-001/day.csv")

df.registerTempTable("bikeshare")

# COMMAND ----------

# MAGIC %md 
# MAGIC 季節ごとのバイクの条件(風速、温度、湿度)を示すグラフを作成します。また、グラフを適切に説明するラベルを示すセルを作成します。

# COMMAND ----------

# MAGIC %md **季節ごとのバイクの条件**

# COMMAND ----------

display(spark.sql("SELECT season, MAX(temp) as temperature, MAX(hum) as humidity, MAX(windspeed) as windspeed FROM bikeshare GROUP BY season ORDER BY SEASON"))

# COMMAND ----------

# MAGIC %md ## ダッシュボードの作成
# MAGIC 
# MAGIC 表示すべきエレメントを作成したので、これらを用いてダッシュボードを作成します。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **View**メニューを開き**+ New Dashboard**を選択します。
# MAGIC 
# MAGIC ![dashboard demo](https://docs.databricks.com/_static/images/dashboards/dashboard-demo-0.png)
# MAGIC 
# MAGIC ダッシュボードに名前をつけます。

# COMMAND ----------

# MAGIC %md 
# MAGIC デフォルトでは、ここまでで作成した全てのセルが新規ダッシュボードには含まれています。好きなようにセルを並び替えたり、サイズを変更したりすることができます。**View**メニューから**Standard**を選択することでノートブックに戻ってくることができます。
# MAGIC 
# MAGIC ![something](https://docs.databricks.com/_static/images/dashboards/dashboard-demo-2.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ダッシュボードに新規セルは自動では現れません。右にあるダッシュボードアイコンを選択し、ダッシュボード名の隣にあるチェックボックスを選択することで、エレメントを手動で追加することができます。
# MAGIC 
# MAGIC ![something](https://docs.databricks.com/_static/images/dashboards/dashboard-demo-3.png)
# MAGIC 
# MAGIC 同じようにして、セルを追加・削除することができます。

# COMMAND ----------

# MAGIC %md ## ダッシュボードへのグラフの追加
# MAGIC 
# MAGIC さらにグラフをダッシュボードに追加することができます。最初に、データセットにおける全ての月におけるバイクの条件のグラフを追加します。そして、ラベルを追加し、セルをダッシュボードに追加します。

# COMMAND ----------

# MAGIC %md **バイク条件の平均**

# COMMAND ----------

display(spark.sql("SELECT mnth as month, AVG(temp) as temperature, AVG(hum) as humidity, AVG(windspeed) as windspeed FROM bikeshare GROUP BY month ORDER BY month"))

# COMMAND ----------

# MAGIC %md 
# MAGIC 平均は見やすいですが、ここでは極端な条件に注意を払いたいと考えます。月毎の極端なバイクの条件のグラフを作成します。そして、ラベルを作成し、ダッシュボードに追加します。

# COMMAND ----------

# MAGIC %md 
# MAGIC **極端なバイクの条件**

# COMMAND ----------

# MAGIC %sql SELECT mnth as month, MAX(temp) as max_temperature, MAX(hum) as max_humidity, MAX(windspeed) as max_windspeed FROM bikeshare GROUP BY mnth ORDER BY mnth

# COMMAND ----------

# MAGIC %md 
# MAGIC ここまでで作成したダッシュボードを整理します。新規ダッシュボードの上部に移動します。それぞれのタイルのサイズを変更するために左下、あるいいは右下を選択します。マークダウンのセルは、ダッシュボードのそれぞれのセクションのラベルになります。
# MAGIC 
# MAGIC ![something](https://docs.databricks.com/_static/images/dashboards/dashboard-demo-4.png)

# COMMAND ----------

# MAGIC %md ## ダッシュボードをプレゼンテーションとして表示
# MAGIC 
# MAGIC 右側にある**Present Dashboard**を選択することで、ダッシュボードをプレゼンテーションモードで表示することができます。
# MAGIC 
# MAGIC ![something](https://docs.databricks.com/_static/images/dashboards/dashboard-demo-5.png)

# COMMAND ----------

# MAGIC %md ## ダッシュボードの編集
# MAGIC 
# MAGIC ダッシュボードビューでダッシュボードを編集することができます。ダッシュボードを表示し、上述したように編集を行います。

# COMMAND ----------

# MAGIC %md ## ダッシュボードの削除
# MAGIC 
# MAGIC ダッシュボードビューでダッシュボードを削除できます。ダッシュボードを表示し、右側の**Delete this dashboard**をクリックします。
