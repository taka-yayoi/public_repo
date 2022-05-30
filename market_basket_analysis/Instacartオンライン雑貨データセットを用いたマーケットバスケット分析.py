# Databricks notebook source
# MAGIC %md
# MAGIC # Instacartオンライン雑貨データセットを用いたマーケットバスケット分析
# MAGIC ## Instacartの顧客はどの商品を再び買うのでしょうか？
# MAGIC 
# MAGIC [Databricksのレイクハウスプラットフォーム](https://databricks.com/jp/product/data-lakehouse)におけるマーケットバスケット分析のコンセプトを説明するために、Instacartの300万のオーダーデータセット、[3 Million Instacart Orders, Open Sourced](https://www.instacart.com/datasets/grocery-shopping-2017)を使用します。
# MAGIC 
# MAGIC > “The Instacart Online Grocery Shopping Dataset 2017”, Accessed from https://www.instacart.com/datasets/grocery-shopping-2017 on 01/17/2018. 
# MAGIC この匿名化されたデータセットには、200,000以上のInstacartユーザーによる300万もの雑貨注文のサンプルが含まれています。
# MAGIC ユーザーごとに4から100の注文があり、注文ごとの購入の順番も含まれています。また、注文があった週と時間も含まれており、注文間の相対的な時間間隔も含まれています。
# MAGIC 
# MAGIC 綿密に計画された雑貨品のリストに基づいて買い物するか、自分の直感に基づいてみたものを購入するのか、いずれにしてもユニークな食べ物の習慣はあなたが誰であるのかを定義します。Instacartの雑貨注文と配送アプリは、あなたの個人的好みと必要な時には特産品で冷蔵庫と食料品置き場を容易に満たすことを狙いとしています。Instacartのアプリで商品を選択した後に、自身の注文のパーソナルレビューを行い、店舗内ショッピングやデリバリーを提供します。
# MAGIC 
# MAGIC このノートブックでは、どの商品を再び購入するのかをどのように予測し、初回に試すことをレコメンドするのかを説明します。
# MAGIC <img src="https://s3.us-east-2.amazonaws.com/databricks-dennylee/media/buy+it+again+or+recommend.png" width="1100"/>
# MAGIC 
# MAGIC 
# MAGIC *Source: [3 Million Instacart Orders, Open Sourced](https://tech.instacart.com/3-million-instacart-orders-open-sourced-d40d29ead6f2)*

# COMMAND ----------

# DBTITLE 1,データエンジニアリングパイプライン
# MAGIC %md
# MAGIC ![](https://s3.us-east-2.amazonaws.com/databricks-dennylee/media/data-engineering-pipeline-3.png)
# MAGIC 
# MAGIC 一般的にデータエンジニアリングパイプラインはこれらのコンポーネントから構成されます。
# MAGIC 
# MAGIC * **データの取り込み**: ソースシステムからデータの持ち込み。多くの場合ETLプロセスが関係します(このデモではシンプルさのためこのステップをスキップします)。
# MAGIC * **データの探索**: クレンジングされたデータが入手できたので、いくつかのビジネス洞察を得るためにデータを探索します。
# MAGIC * **MLモデルのトレーニング**: 頻出パターンマイニングのためにFP-growthを実行します。
# MAGIC * **アソシエーションルールのレビュー**: 生成されたアソシエーションルールをレビューします。

# COMMAND ----------

# MAGIC %md ## データの取り込み

# COMMAND ----------

# DBTITLE 0,Ingest Data
# MAGIC %md
# MAGIC はじめに、[3 Million Instacart Orders, Open Sourced](https://www.instacart.com/datasets/grocery-shopping-2017)をダウンロードし、`dbfs`にアップロードします。詳細についてはデータの[インポート](https://qiita.com/taka_yayoi/items/4fa98b343a91b8eaa480)をご覧ください。
# MAGIC 
# MAGIC `dbutils filesystem (fs)`のクエリーコマンドを実行すると6つのファイルが表示されます。
# MAGIC 
# MAGIC * `Orders`: 3.4M 行, 206K ユーザー
# MAGIC * `Products`: 50K 行
# MAGIC * `Aisles`: 134 行 
# MAGIC * `Departments`: 21 行
# MAGIC * `order_products__SET`: 30M+ 行 ここでは `SET` は以下のように定義されます:
# MAGIC   * `prior`: 3.2M の事前の注文
# MAGIC   * `train`: トレーニングデータセットのための 131K の注文
# MAGIC   
# MAGIC リファレンス: [The Instacart Online Grocery Shopping Dataset 2017 Data Descriptions](https://gist.github.com/jeremystan/c3b39d947d9b88b3ccff3147dbcf6c6b)
# MAGIC 
# MAGIC ### 重要
# MAGIC データをアップロードした場所に応じて、格納場所(以下の例では `/FileStore/shared_uploads/takaaki.yayoi@databricks.com/market_basket/` を使用しています)を**編集**する必要があります。
# MAGIC 
# MAGIC ### 注意
# MAGIC 
# MAGIC 2022/5/30時点では上記リンク先のデータセットが利用できないため、Kaggleからデータをダウンロードしています。
# MAGIC 
# MAGIC https://www.kaggle.com/competitions/instacart-market-basket-analysis/data

# COMMAND ----------

# DBTITLE 1,取り込みデータの確認
# MAGIC %fs ls /FileStore/shared_uploads/takaaki.yayoi@databricks.com/market_basket/

# COMMAND ----------

# DBTITLE 1,`orders.csv`ファイルの確認
# MAGIC %fs head dbfs:/FileStore/shared_uploads/takaaki.yayoi@databricks.com/market_basket/orders.csv

# COMMAND ----------

# DBTITLE 1,データフレームの作成
dataset_path = "/FileStore/shared_uploads/takaaki.yayoi@databricks.com/market_basket"

# データのインポート
aisles = spark.read.csv(f"{dataset_path}/aisles.csv", header=True, inferSchema=True)
departments = spark.read.csv(f"{dataset_path}/departments.csv", header=True, inferSchema=True)
order_products_prior = spark.read.csv(f"{dataset_path}/order_products__prior.csv", header=True, inferSchema=True)
order_products_train = spark.read.csv(f"{dataset_path}/order_products__train.csv", header=True, inferSchema=True)
orders = spark.read.csv(f"{dataset_path}/orders.csv", header=True, inferSchema=True)
products = spark.read.csv(f"{dataset_path}/products.csv", header=True, inferSchema=True)

# 一時テーブルの作成
aisles.createOrReplaceTempView("aisles")
departments.createOrReplaceTempView("departments")
order_products_prior.createOrReplaceTempView("order_products_prior")
order_products_train.createOrReplaceTempView("order_products_train")
orders.createOrReplaceTempView("orders")
products.createOrReplaceTempView("products")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 探索的データ分析(EDA)
# MAGIC 
# MAGIC Spark SQLを用いてInstacartデータを探索します。

# COMMAND ----------

# DBTITLE 1,最も忙しい曜日
# MAGIC %sql
# MAGIC select 
# MAGIC   count(order_id) as total_orders, 
# MAGIC   (case 
# MAGIC      when order_dow = '0' then 'Sunday'
# MAGIC      when order_dow = '1' then 'Monday'
# MAGIC      when order_dow = '2' then 'Tuesday'
# MAGIC      when order_dow = '3' then 'Wednesday'
# MAGIC      when order_dow = '4' then 'Thursday'
# MAGIC      when order_dow = '5' then 'Friday'
# MAGIC      when order_dow = '6' then 'Saturday'              
# MAGIC    end) as day_of_week 
# MAGIC   from orders  
# MAGIC  group by order_dow 
# MAGIC  order by total_orders desc

# COMMAND ----------

# DBTITLE 1,時間ごとの注文のブレークダウン
# MAGIC %sql
# MAGIC select 
# MAGIC   count(order_id) as total_orders, 
# MAGIC   order_hour_of_day as hour 
# MAGIC   from orders 
# MAGIC  group by order_hour_of_day 
# MAGIC  order by order_hour_of_day

# COMMAND ----------

# DBTITLE 1,部門ごとの売り上げが最大の商品
# MAGIC %sql
# MAGIC select countbydept.*
# MAGIC   from (
# MAGIC   -- productsテーブルから部門ごとのレコード数をカウントし、カウントでソート(高いとこから低いところに)してみましょう。
# MAGIC   select department_id, count(1) as counter
# MAGIC     from products
# MAGIC    group by department_id
# MAGIC    order by counter asc 
# MAGIC   ) as maxcount
# MAGIC inner join (
# MAGIC   -- エクササイズを繰り返しますが、今回はdeptとprodカウントの完全なリストを得るために、productsとdepartmentsテーブルをjoinします。
# MAGIC   select
# MAGIC     d.department_id,
# MAGIC     d.department,
# MAGIC     count(1) as products
# MAGIC     from departments d
# MAGIC       inner join products p
# MAGIC          on p.department_id = d.department_id
# MAGIC    group by d.department_id, d.department 
# MAGIC    order by products desc
# MAGIC   ) countbydept 
# MAGIC   -- 商品カウントをマッチさせることで2つのクエリーの結果を組み合わせます。
# MAGIC   on countbydept.products = maxcount.counter

# COMMAND ----------

# DBTITLE 1,人気アイテムトップ10
# MAGIC %sql
# MAGIC select count(opp.order_id) as orders, p.product_name as popular_product
# MAGIC   from order_products_prior opp, products p
# MAGIC  where p.product_id = opp.product_id 
# MAGIC  group by popular_product 
# MAGIC  order by orders desc 
# MAGIC  limit 10

# COMMAND ----------

# DBTITLE 1,部門ごとの棚のスペース
# MAGIC %sql
# MAGIC select d.department, count(distinct p.product_id) as products
# MAGIC   from products p
# MAGIC     inner join departments d
# MAGIC       on d.department_id = p.department_id
# MAGIC  group by d.department
# MAGIC  order by products desc
# MAGIC  limit 10

# COMMAND ----------

# MAGIC %md ### 買い物バスケットの整理及び参照

# COMMAND ----------

# DBTITLE 1,買い物バスケットの整理
# 買い物バスケットごとにデータを整理
from pyspark.sql.functions import collect_set, col, count
rawData = spark.sql("select p.product_name, o.order_id from products p inner join order_products_train o where o.product_id = p.product_id")
baskets = rawData.groupBy('order_id').agg(collect_set('product_name').alias('items'))
baskets.createOrReplaceTempView('baskets')

# COMMAND ----------

# DBTITLE 1,買い物バスケットの参照
display(baskets)

# COMMAND ----------

# MAGIC %md ## MLモデルのトレーニング
# MAGIC 
# MAGIC 互いに関連づけられているアイテムの頻度(ピーナッツバターとジャムなど)を理解するために、マーケットバスケット分析のためのアソシエーションルールマイニングを使用します。[Spark MLlib](http://spark.apache.org/docs/latest/mllib-guide.html)では頻度パターンマイニング(FPM)に関連する2つのアルゴリズムを実装しています。[FP-growth](https://spark.apache.org/docs/latest/mllib-frequent-pattern-mining.html#fp-growth) と [PrefixSpan](https://spark.apache.org/docs/latest/mllib-frequent-pattern-mining.html#prefixspan)です。違いは、FP-growthはアイテムセット内の順番を使用せず、PrefixSpanはアイテムセットに順序がある場合のシーケンシャルパターンマイニングのために設計されているということです。このユースケースでは順番の情報は重要ではないので、ここではFP-growthを使用します。
# MAGIC 
# MAGIC > ここでは`Scala API`を使用するので `setMinConfidence` を設定できることに注意してください。

# COMMAND ----------

# DBTITLE 1,FP-growthの使用
# MAGIC %scala
# MAGIC import org.apache.spark.ml.fpm.FPGrowth
# MAGIC 
# MAGIC // アイテムの抽出
# MAGIC val baskets_ds = spark.sql("select items from baskets").as[Array[String]].toDF("items")
# MAGIC 
# MAGIC // FPGrowthの使用
# MAGIC val fpgrowth = new FPGrowth().setItemsCol("items").setMinSupport(0.001).setMinConfidence(0)
# MAGIC val model = fpgrowth.fit(baskets_ds)

# COMMAND ----------

# DBTITLE 1,最頻アイテムセット
# MAGIC %scala
# MAGIC // 頻出するアイテムセットの表示
# MAGIC val mostPopularItemInABasket = model.freqItemsets
# MAGIC mostPopularItemInABasket.createOrReplaceTempView("mostPopularItemInABasket")

# COMMAND ----------

# MAGIC %sql
# MAGIC select items, freq from mostPopularItemInABasket where size(items) > 2 order by freq desc limit 20

# COMMAND ----------

# MAGIC %md ## アソシエーションルールのレビュー
# MAGIC 
# MAGIC `freqItemSets`に加え、`FP-growth`は`アソシエーションルール`を生成します。例えば、あるお客様が*ピーナッツバター*を購入する場合、同時に購入するであろう商品は*ジャム*となります。詳細に関しては、Susan Liの良いリファレンスである[A Gentle Introduction on Market Basket Analysis — Association Rules](https://towardsdatascience.com/a-gentle-introduction-on-market-basket-analysis-association-rules-fa4b986a40ce)をご覧ください。

# COMMAND ----------

# DBTITLE 1,生成されたアソシエーションルールの参照
# MAGIC %scala
# MAGIC // 生成されたアソシエーションルールの表示
# MAGIC val ifThen = model.associationRules
# MAGIC ifThen.createOrReplaceTempView("ifThen")

# COMMAND ----------

# MAGIC %sql
# MAGIC select antecedent as `antecedent (if)`, consequent as `consequent (then)`, confidence from ifThen order by confidence desc limit 20

# COMMAND ----------

# MAGIC %md
# MAGIC # END
