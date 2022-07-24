# Databricks notebook source
# MAGIC %md 
# MAGIC # Apache Spark 3を用いた高精細需要予測
# MAGIC 
# MAGIC このノートブックの目的は、Databricksの分散処理能力を活用することで、効率的な方法で店舗レベルの大量のきめ細かい予測をどのように行うのかを説明することです。これは、Spark 2.x向けに以前開発されたノートブックのSpark 3.xのアップデート版です。
# MAGIC 
# MAGIC ノートブックの**アップデート**マークには、Spark 3.xあるいはDatabricksプラットフォームの新機能を利用するためにコードに変更がされたことを意味します。

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC このエクササイズでは、需要予測で人気を高めているライブラリである[FBProphet](https://facebook.github.io/prophet/)を用い、Databricksランタイム7.1以降のクラスターに関連づけられているノートブックセッションにロードします。
# MAGIC 
# MAGIC ![](https://databricks.com/wp-content/uploads/2020/01/FB-Prophet-logo.png)
# MAGIC 
# MAGIC **アップデート** Databricks 7.1では、`%pip`マジックコマンドを用いて[ノートブックスコープライブラリ](https://docs.databricks.com/dev-tools/databricks-utils.html#library-utilities)としてインストールすることができます。

# COMMAND ----------

# DBTITLE 1,必要ライブラリのインストール
# MAGIC %pip install pystan==2.19.1.1  # 参考 https://github.com/facebook/prophet/commit/82f3399409b7646c49280688f59e5a3d2c936d39#comments
# MAGIC %pip install fbprophet==0.6

# COMMAND ----------

# MAGIC %md ## Step 1: データの検証
# MAGIC 
# MAGIC 使用するトレーニングデータセットとして、10店舗における50商品5年分の店舗・商品のユニット売り上げデータを使用します。このデータセットは過去のKaggleのコンペティションの一部として公開されており、[こちら](https://www.kaggle.com/c/demand-forecasting-kernels-only/data)からダウンロードすることができます。
# MAGIC 
# MAGIC ダンロードしたあとは、*train.csv.zip*ファイルを解凍し、[こちら](https://docs.databricks.com/data/databricks-file-system.html#!#user-interface)で説明されているファイルのインポートステップに従って、解凍したCSVを `/FileStore/demand_forecast/train/` にアップロードします。Databricksからデータセットにアクセスできるようになったので、モデルを準備するためにデータを探索します。

# COMMAND ----------

# DBTITLE 1,データセットへのアクセス
from pyspark.sql.types import *

# トレーニングデータセットの構造
train_schema = StructType([
  StructField('date', DateType()),
  StructField('store', IntegerType()),
  StructField('item', IntegerType()),
  StructField('sales', IntegerType())
  ])

# トレーニングファイルをデータフレームに読み込み
train = spark.read.csv(
  'dbfs:/FileStore/demand_forecast/train/train.csv', 
  header=True, 
  schema=train_schema
  )

# データフレームにクエリーを実行できるように一時ビューを作成
train.createOrReplaceTempView('train')

# データの表示
display(train)

# COMMAND ----------

train.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 年ごとのトレンドの参照
# MAGIC 
# MAGIC 需要予測を行う際、多くのケースにおいて我々は一般的なトレンドや季節性に着目します。ユニットの売上における年次のトレンドを見るところから探索をスタートしましょう。

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   year(date) as year, 
# MAGIC   sum(sales) as sales
# MAGIC FROM train
# MAGIC GROUP BY year(date)
# MAGIC ORDER BY year;

# COMMAND ----------

# MAGIC %md
# MAGIC 店舗に関わらず、トータルユニットセールスにおいては共通した上昇志向のトレンドがありことが明確となっています。これらの店舗に関連する市場に対して、より多くの知識があれば、予測期間を通じてアプローチするに値する最大の成長キャパシティがあるのかを判断したいと考えるかもしれません。しかし、そのような知識がなくても、このデータセットをクイックに確認することで、我々のゴールが数日先、数ヶ月先、あるいは数年先の予測なのであれば、問題はないと安心でき、このタイムスパンでは線形の成長が継続することを期待できます。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 月ごとのトレンドの表示
# MAGIC 
# MAGIC 次に季節性を検証しましょう。各年の月ごとのデータを集計すると、全体的なセールスの増加傾向とともに年間の季節的なパターンを確認することができます。

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT 
# MAGIC   TRUNC(date, 'MM') as month,
# MAGIC   SUM(sales) as sales
# MAGIC FROM train
# MAGIC GROUP BY TRUNC(date, 'MM')
# MAGIC ORDER BY month;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 曜日ごとのトレンドの表示
# MAGIC 
# MAGIC 曜日レベルでデータを集計すると、日曜日にピークがあって月曜日に減少し、日曜日に向けて一定の割合で増加していく傾向を確認することができます。このパターンは、五年間を通じて非常に安定したパターンであるように見えます。
# MAGIC 
# MAGIC **アップデート** Spark 3の一部は[Proleptic Gregorian calendar](https://databricks.com/blog/2020/07/22/a-comprehensive-look-at-dates-and-timestamps-in-apache-spark-3-0.html)に移行したので、`CAST(DATE_FORMAT(date, 'u')`における'u'オプションは削除されました。 同様のアウトプットを得るために`E`を使用します。

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   YEAR(date) as year,
# MAGIC   (
# MAGIC     CASE
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Sun' THEN 0
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Mon' THEN 1
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Tue' THEN 2
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Wed' THEN 3
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Thu' THEN 4
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Fri' THEN 5
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Sat' THEN 6
# MAGIC     END
# MAGIC   ) % 7 as weekday,
# MAGIC   AVG(sales) as sales
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     date,
# MAGIC     SUM(sales) as sales
# MAGIC   FROM train
# MAGIC   GROUP BY date
# MAGIC  ) x
# MAGIC GROUP BY year, weekday
# MAGIC ORDER BY year, weekday;

# COMMAND ----------

# MAGIC %md 
# MAGIC これで、データに対する基本的なパターンに慣れ親しんだので、どのように予測モデルを構築するのかを探索していきましょう。

# COMMAND ----------

# MAGIC %md ## Step 2: 単一の予測モデルの構築
# MAGIC 
# MAGIC 店舗と商品の個々の組み合わせの予測を生成しようとする前に、FBProphetの使い方に慣れるという目的のみにおいても、単一の予測モデルを構築することは有益と言えるでしょう。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 単一の商品-店舗の組み合わせのデータの取得
# MAGIC 
# MAGIC 最初のステップは、モデルをトレーニングする履歴データを構成することとなります。

# COMMAND ----------

# データをdate(ds)レベルに集計するクエリー
sql_statement = '''
  SELECT
    CAST(date as date) as ds,
    sales as y
  FROM train
  WHERE store=1 AND item=1
  ORDER BY ds
  '''

# Pandasデータフレームとしてデータセットを構成
history_pd = spark.sql(sql_statement).toPandas()

# 欠損値のあるレコードの削除
history_pd = history_pd.dropna()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prophetライブラリのインポート
# MAGIC 
# MAGIC 次にfbprophetライブラリをインポートします。使用する際に多くの情報が出力されるのでログの出力を抑制しますが、実際の環境においいてはログの設定をチューニングする必要があります。

# COMMAND ----------

from fbprophet import Prophet
import logging

# fbprophetのインフォーメーションメッセージを無効化
logging.getLogger('py4j').setLevel(logging.ERROR)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Prophetモデルのトレーニング
# MAGIC 
# MAGIC データを確認した結果に基いて、全体的な成長パターンをlinearに設定し、週と年の季節性パターンを有効化すべきです。また、季節性のパターンがセールスにおける全体的な成長に合わせて成長するように見えるので、季節性モードを`multiplicative`(増加)に設定しても構いません。

# COMMAND ----------

# モデルパラメーターの設定
model = Prophet(
  interval_width=0.95,
  growth='linear',
  daily_seasonality=False,
  weekly_seasonality=True,
  yearly_seasonality=True,
  seasonality_mode='multiplicative'
  )

# 履歴データに対してモデルをフィッティング
model.fit(history_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 予測の実施
# MAGIC 
# MAGIC これでモデルをトレーニングできたので、90日の予測にモデルを使用しましょう。

# COMMAND ----------

# 履歴データと最新の日付以降90日の両方を含むデータセットを定義
future_pd = model.make_future_dataframe(
  periods=90, 
  freq='d', 
  include_history=True
  )

# データセットに対して予測を実施
forecast_pd = model.predict(future_pd)

display(forecast_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 予測コンポーネントの検証
# MAGIC 
# MAGIC モデルはどのような性能なのでしょうか？作成したモデルにおける一般的なトレンド、季節性のトレンドをグラフとして確認することができます。

# COMMAND ----------

trends_fig = model.plot_components(forecast_pd)
#display(trends_fig)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 履歴データ vs. 予測の参照
# MAGIC 
# MAGIC そして、ここでは実際のデータと将来の予測値を含む予測データを確認することができますが、読みやすいように過去一年の履歴データに限定してグラフに表示しています。

# COMMAND ----------

predict_fig = model.plot( forecast_pd, xlabel='date', ylabel='sales')

# 過去1年分 + 90日の予測を表示するように図を調整
xlim = predict_fig.axes[0].get_xlim()
new_xlim = ( xlim[1]-(180.0+365.0), xlim[1]-90.0)
predict_fig.axes[0].set_xlim(new_xlim)

#display(predict_fig)

# COMMAND ----------

# MAGIC %md **注意** このビジュアライゼーションは若干ビジーです。Bartosz Mikulskiが[これをブレークダウンした素晴らしい説明](https://www.mikulskibartosz.name/prophet-plot-explained/)を公開しており、チェックする価値があるものです。簡単に言うと、黒い点が実際の値を表現しており、暗い青の線は予測値、明るい青の帯は95%の信頼区間を表現しています。

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 評価メトリクスの計算
# MAGIC 
# MAGIC 視覚による調査は有用ですが、予測を評価するより良い方法はMean Absolute Errorを計算するというものです。我々の環境では、実際の値に対する予測値のMean Absolute ErrorやRoot Mean Squared Errorを計算します。
# MAGIC 
# MAGIC **アップデート** pandasの機能変更によって、日付文字列を適切なデータ型に変換するために*pd.to_datetime*を使用する必要があります。

# COMMAND ----------

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from datetime import date

# 比較のために過去の実績値と予測値を取得
actuals_pd = history_pd[ history_pd['ds'] < date(2018, 1, 1) ]['y']
predicted_pd = forecast_pd[ forecast_pd['ds'] < pd.to_datetime('2018-01-01') ]['yhat']

# 評価メトリクスの計算
mae = mean_absolute_error(actuals_pd, predicted_pd)
mse = mean_squared_error(actuals_pd, predicted_pd)
rmse = sqrt(mse)

# メトリクスの表示
print( '\n'.join(['MAE: {0}', 'MSE: {1}', 'RMSE: {2}']).format(mae, mse, rmse) )

# COMMAND ----------

# MAGIC %md 
# MAGIC FBProphetでは、あなたの予測結果が時間の経過を通じてどれだけ良い結果を出しているのかを評価するための[別の方法](https://facebook.github.io/prophet/docs/diagnostics.html)を提供しています。予測モデルを構築する際にこれらの活用を検討することを強くお勧めしますが、ここではスケーリングの課題にフォーカスしているので、説明はスキップします。

# COMMAND ----------

# MAGIC %md ## Step 3: 予測処理をスケール
# MAGIC 
# MAGIC 我々の保持するメカニズムを活用して、個々の店舗と商品の組み合わせに対する大量の高精細モデルと予測を行うという元々の目標に取り組みましょう。セールスデータを店舗・商品・日付の粒度に構成するところからスタートします。
# MAGIC 
# MAGIC **注意**: このデータセットのデータはすでにこのレベルの粒度に集計されていますが、ここではデータが期待している構造になっていることを確実するために、明示的に集計を行います。

# COMMAND ----------

# MAGIC %md
# MAGIC ### すべての店舗・商品の組み合わせに対応するデータの取得

# COMMAND ----------

sql_statement = '''
  SELECT
    store,
    item,
    CAST(date as date) as ds,
    SUM(sales) as y
  FROM train
  GROUP BY store, item, ds
  ORDER BY store, item, ds
  '''

store_item_history = (
  spark
    .sql( sql_statement )
    .repartition(sc.defaultParallelism, ['store', 'item'])
  ).cache()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 予測アウトプットのスキーマを定義
# MAGIC 
# MAGIC 店舗・商品・日付レベルにデータを集計したら、データをどのようにFBProphetに渡すのかを検討する必要があります。我々のゴールが、それぞれの店舗と商品の組み合わせに対応するモデルの構築であれば、構成した店舗・商品のサブセットを引渡し、そのサブセットでモデルをトレーニングし、店舗・商品レベルの予測結果を受け取る必要があります。予測を実行した店舗と商品のIDを維持できるように、予測結果がこのような構造のデータセットで返却されてほしいので、Prophetモデルによって生成されるフィールドから適切なものに出力を限定します。

# COMMAND ----------

from pyspark.sql.types import *

result_schema =StructType([
  StructField('ds',DateType()),
  StructField('store',IntegerType()),
  StructField('item',IntegerType()),
  StructField('y',FloatType()),
  StructField('yhat',FloatType()),
  StructField('yhat_upper',FloatType()),
  StructField('yhat_lower',FloatType())
  ])

# COMMAND ----------

# MAGIC %md 
# MAGIC ### モデルをトレーニングし予測を行う関数の定義
# MAGIC 
# MAGIC モデルをトレーニングし予測行うために、Pandasの関数を活用します。店舗と商品の組み合わせに対応するデータのサブセットを受け取るために、この関数を活用します。上のセルで定義したフォーマットで予測結果が返却されます。
# MAGIC 
# MAGIC **アップデート** Spark 3.0では、このpandas UDFの機能をpandas functionsで置き換えました。非推奨となったpandas UDFの文法はまだサポートされていますが、将来的にはサポートされなくなります。新たに整理されたpandas functions APIに関しては、[ドキュメント](https://databricks.com/blog/2020/05/20/new-pandas-udfs-and-python-type-hints-in-the-upcoming-release-of-apache-spark-3-0.html)を参照してください。

# COMMAND ----------

def forecast_store_item( history_pd: pd.DataFrame ) -> pd.DataFrame:
  
  # 以前と同様にモデルをトレーニング
  # --------------------------------------
  # 欠損値の除外 (日付・店舗・商品レベルでは発生頻度が高まります)
  history_pd = history_pd.dropna()
  
  # モデルの設定
  model = Prophet(
    interval_width=0.95,
    growth='linear',
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative'
    )
  
  # モデルのトレーニング
  model.fit( history_pd )
  # --------------------------------------
  
  # 以前と同様に予測を実施
  # --------------------------------------
  # 予測の実施
  future_pd = model.make_future_dataframe(
    periods=90, 
    freq='d', 
    include_history=True
    )
  forecast_pd = model.predict( future_pd )  
  # --------------------------------------
  
  # 期待する結果セットの構成
  # --------------------------------------
  # 予測結果から適切なフィールドを取得
  f_pd = forecast_pd[ ['ds','yhat', 'yhat_upper', 'yhat_lower'] ].set_index('ds')
  
  # 履歴から適切なフィールドを取得
  h_pd = history_pd[['ds','store','item','y']].set_index('ds')
  
  # 履歴と予測をjoin
  results_pd = f_pd.join( h_pd, how='left' )
  results_pd.reset_index(level=0, inplace=True)
  
  # 入力データセットから店舗と商品を取得
  results_pd['store'] = history_pd['store'].iloc[0]
  results_pd['item'] = history_pd['item'].iloc[0]
  # --------------------------------------
  
  # 期待されるデータセットを返却
  return results_pd[ ['ds', 'store', 'item', 'y', 'yhat', 'yhat_upper', 'yhat_lower'] ]  

# COMMAND ----------

# MAGIC %md 
# MAGIC 我々の関数の中にはいろいろな物が含まれていますが、このコードでモデルをトレーニングし、予測結果を生成する最初の2つのブロックと、このノートブックの前半のセルとを比較すると、非常に類似していることがわかります。コードに追加されているのは、必要な結果セットを構成する部分のみであり、これは標準的なPandasデータフレームの操作によって構成されています。

# COMMAND ----------

# MAGIC %md 
# MAGIC ### それぞれの店舗・商品の組み合わせに対して予測関数を適用
# MAGIC 
# MAGIC 予測を行うために上で構築したpandas関数を呼び出しましょう。このために、履歴データセットを店舗と商品でグルーピングします。そして、それぞれのグループに上の関数を適用し、データ管理のために本日の日付を *training_date* として追加します。
# MAGIC 
# MAGIC **アップデート** 上のアップデートの記述の通り、pandas関数を呼び出すためにpandas UDFではなく`applyInPandas()`を使用します。

# COMMAND ----------

from pyspark.sql.functions import current_date

results = (
  store_item_history
    .groupBy('store', 'item')
      .applyInPandas(forecast_store_item, schema=result_schema)
    .withColumn('training_date', current_date() )
    )

results.createOrReplaceTempView('new_forecasts')

display(results)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 予測結果の永続化
# MAGIC 
# MAGIC 多くの場合、予測結果をレポートで活用したいと考えるので、クエリー可能なテーブル構造として保存します。

# COMMAND ----------

import re
from pyspark.sql.types import * 

# Username を取得
username_raw = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
# Username の英数字以外を除去し、全て小文字化
username = re.sub('[^A-Za-z0-9]+', '', username_raw).lower()

# ユーザー固有のデータベース名を生成します
db_name = f"databricks_handson_{username}"

# データベースの準備
spark.sql(f"DROP DATABASE IF EXISTS {db_name} CASCADE")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
spark.sql(f"USE {db_name}")

# データベースを表示
print(f"database_name: {db_name}")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 予測テーブルの作成
# MAGIC create table if not exists forecasts (
# MAGIC   date date,
# MAGIC   store integer,
# MAGIC   item integer,
# MAGIC   sales float,
# MAGIC   sales_predicted float,
# MAGIC   sales_predicted_upper float,
# MAGIC   sales_predicted_lower float,
# MAGIC   training_date date
# MAGIC   )
# MAGIC using delta
# MAGIC partitioned by (training_date);
# MAGIC 
# MAGIC -- データのロード
# MAGIC insert into forecasts
# MAGIC select 
# MAGIC   ds as date,
# MAGIC   store,
# MAGIC   item,
# MAGIC   y as sales,
# MAGIC   yhat as sales_predicted,
# MAGIC   yhat_upper as sales_predicted_upper,
# MAGIC   yhat_lower as sales_predicted_lower,
# MAGIC   training_date
# MAGIC from new_forecasts;

# COMMAND ----------

# MAGIC %md 
# MAGIC ### それぞれの予測結果を評価するために同じテクニックを適用
# MAGIC 
# MAGIC しかし、それぞれの予測はどれだけ良いものなのでしょうか？以下のようにpandas関数のテクニックを使うことで、店舗・商品レベルの予測に対する評価メトリクスを計算することができます。

# COMMAND ----------

# 期待する結果セットのスキーマ
eval_schema =StructType([
  StructField('training_date', DateType()),
  StructField('store', IntegerType()),
  StructField('item', IntegerType()),
  StructField('mae', FloatType()),
  StructField('mse', FloatType()),
  StructField('rmse', FloatType())
  ])

# メトリクスを計算する関数の定義
def evaluate_forecast( evaluation_pd: pd.DataFrame ) -> pd.DataFrame:
  
  # 入力データセットの店舗と商品を取得
  training_date = evaluation_pd['training_date'].iloc[0]
  store = evaluation_pd['store'].iloc[0]
  item = evaluation_pd['item'].iloc[0]
  
  # 評価メトリクスの計算
  mae = mean_absolute_error( evaluation_pd['y'], evaluation_pd['yhat'] )
  mse = mean_squared_error( evaluation_pd['y'], evaluation_pd['yhat'] )
  rmse = sqrt( mse )
  
  # 結果セットの構成
  results = {'training_date':[training_date], 'store':[store], 'item':[item], 'mae':[mae], 'mse':[mse], 'rmse':[rmse]}
  return pd.DataFrame.from_dict( results )

# メトリクスの計算
results = (
  spark
    .table('new_forecasts')
    .filter('ds < \'2018-01-01\'') # 評価を保有している履歴データの期間に限定
    .select('training_date', 'store', 'item', 'y', 'yhat')
    .groupBy('training_date', 'store', 'item')
    .applyInPandas(evaluate_forecast, schema=eval_schema)
    )

results.createOrReplaceTempView('new_forecast_evals')

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 評価メトリクスの永続化
# MAGIC この場合も、予測ごとのメトリクスをレポートしたいと考えるので、これらをクエリー可能なテーブルに永続化します。

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC create table if not exists forecast_evals (
# MAGIC   store integer,
# MAGIC   item integer,
# MAGIC   mae float,
# MAGIC   mse float,
# MAGIC   rmse float,
# MAGIC   training_date date
# MAGIC   )
# MAGIC using delta
# MAGIC partitioned by (training_date);
# MAGIC 
# MAGIC insert into forecast_evals
# MAGIC select
# MAGIC   store,
# MAGIC   item,
# MAGIC   mae,
# MAGIC   mse,
# MAGIC   rmse,
# MAGIC   training_date
# MAGIC from new_forecast_evals;

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 予測結果の可視化
# MAGIC 
# MAGIC ここまでで、それぞれの店舗・商品の組み合わせに対する予測結果を生成し、それぞれの基本的な評価メトリクスを計算しました。この予測データを確認するために、シンプルなクエリー(ここでは店舗1から店舗3における商品1に限定しています)を実行することができます。

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   store,
# MAGIC   date,
# MAGIC   sales_predicted,
# MAGIC   sales_predicted_upper,
# MAGIC   sales_predicted_lower
# MAGIC FROM forecasts a
# MAGIC WHERE item = 1 AND
# MAGIC       store IN (1, 2, 3) AND
# MAGIC       date >= '2018-01-01' AND
# MAGIC       training_date=current_date()
# MAGIC ORDER BY store

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 評価メトリクスの取得
# MAGIC 
# MAGIC そして、これらのそれぞれに対して、予測の信頼性評価の助けとなる指標を取得します。

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   store,
# MAGIC   mae,
# MAGIC   mse,
# MAGIC   rmse
# MAGIC FROM forecast_evals a
# MAGIC WHERE item = 1 AND
# MAGIC       training_date=current_date()
# MAGIC ORDER BY store

# COMMAND ----------

# MAGIC %md
# MAGIC ## クリーンアップ

# COMMAND ----------

display(spark.sql("DROP TABLE IF EXISTS forecasts"))
display(spark.sql("DROP TABLE IF EXISTS forecast_evals"))

# COMMAND ----------

# MAGIC %md
# MAGIC # END
