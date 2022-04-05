# Databricks notebook source
# MAGIC %md
# MAGIC # Labelbox Connector for Databricksのチュートリアルノートブック

# COMMAND ----------

# MAGIC %md
# MAGIC #### 前提条件
# MAGIC 
# MAGIC 1. このチュートリアルノートブックではAPIキーが必要となります。[Labelbox Account](app.labelbox.com)にログインし、[API Key](https://app.labelbox.com/account/api-keys)を作成してください。
# MAGIC 1. 以下のいくつかのセルではLabelbox SDKとコネクターライブラリをインストールします。このインストールはノートブックスコープであるため、クラスターの他の部分には影響を与えません。
# MAGIC 1. 最新のLTSバージョンのDatabricksを実行していることを確認してください。
# MAGIC 
# MAGIC #### ノートブックの概要
# MAGIC 
# MAGIC このノートブックでは以下のステップをガイドします。
# MAGIC 
# MAGIC 1. SDK経由でLabelboxに接続
# MAGIC 1. Databricks上の非構造化データのテーブルからラベリング用のデータセットを作成
# MAGIC 1. プログラムを用いて、Labelbox上にオントロジー、ラベリングプロジェクトをセットアップ
# MAGIC 1. サンプルのラベル付きプロジェクトからブロンズ、シルバーのアノテーションテーブルをロード
# MAGIC 1. 追加のセルでは、動画のアノテーションの取り扱い、Labelbox DiagnosticsとCatalogの使い方を説明しています。
# MAGIC 
# MAGIC ノートブックの最後では追加のドキュメントへのリンクが提供されています。

# COMMAND ----------

# MAGIC %md
# MAGIC Databricks/Labelboxコネクターを試していただきありがとうございます！あなた、あるいは組織のどなたかがDatabricks Partner Connectを通じてLabelboxのトライアルにサインアップしました。非構造化データワークフローを支援するために、LabelboxとDatabricksを組み合わせられるのかを説明するために、このノートブックはお使いのSharedディレクトリにロードされました。
# MAGIC 
# MAGIC Labelboxは、お使いのデータレイクからさまざまな非構造化データ([画像](https://labelbox.com/product/image)、[動画](https://labelbox.com/product/video)、[テキスト](https://labelbox.com/product/text)、そして、[地図タイル画像](https://docs.labelbox.com/docs/tiled-imagery-editor))にアノテーションを迅速に付与するために使われます。そして、Databricks向けLabelboxコネクターを用いることで、簡単にアノテーションをAI/MLと分析ワークフローのためのレイクハウス環境に戻すことが可能となります。
# MAGIC 
# MAGIC ワークフローの動画を視聴したいのであれば、[Data & AI Summit Demo](https://databricks.com/session_na21/productionizing-unstructured-data-for-ai-and-analytics)をご覧ください。
# MAGIC 
# MAGIC <img src="https://labelbox.com/static/images/partnerships/collab-chart.svg" alt="example-workflow" width="800"/>
# MAGIC 
# MAGIC 質問やコメントがある場合には、[ecosystem+databricks@labelbox.com](mailto:ecosystem+databricks@labelbox.com)にご連絡ください。

# COMMAND ----------

# DBTITLE 1,LabelboxライブラリとDatabricks向けLabelboxコネクターのインストール
# MAGIC %pip install labelbox labelspark

# COMMAND ----------

# MAGIC %md
# MAGIC ## SDKの設定
# MAGIC 
# MAGIC LabelboxとDatabricksのライブラリがインストールできたので、SDKを設定する必要があります。[こちら](https://app.labelbox.com/account/api-keys)からAPIキーを作成する必要があります。DatabricksシークレットAPIを用いてキーを格納することも可能です。SDKは環境変数`LABELBOX_API_KEY`の読み込みを試行します。

# COMMAND ----------

from labelbox import Client, Dataset
from labelbox.schema.ontology import OntologyBuilder, Tool, Classification, Option
import databricks.koalas as pd
import labelspark

API_KEY = "<APIキー>" 

if not(API_KEY):
  raise ValueError("Go to Labelbox to get an API key")
  
client = Client(API_KEY)

# COMMAND ----------

# MAGIC %md
# MAGIC ## シードデータの取り込み
# MAGIC 
# MAGIC 次に、デモ用データセットをSparkテーブルにロードし、URLを通じて簡単にアセットをLabelboxにロードする方法を理解することができます。シンプルにするために、LabelboxからデータセットIDを取得することができ、みなさまが使用できるようにこれらのURLをSparkテーブルにロードします(なので、このデモノートブックを実行するためにデータを見つけ出す必要はありません)。以下では、Labelboxトライアルに含まれる"Example Nature Dataset"を取得します。
# MAGIC 
# MAGIC また、LabelboxはAWS、Azure、GCPのクラウドストレージをネイティブでサポートしています。[Delegated Access](https://docs.labelbox.com/docs/iam-delegated-access)を通じてLabelboxをお使いのストレージに接続することができ、アノテーションするためにこれらのアセットを簡単にロードすることができます。詳細については、こちらの[動画](https://youtu.be/wlWo6EmPDV4)をご覧ください。

# COMMAND ----------

sample_dataset = next(client.get_datasets(where=(Dataset.name == "Example Nature Dataset")))
sample_dataset.uid

# COMMAND ----------

# ディレクトリをパースし、画像URLのSparkテーブルを作成することができます
SAMPLE_TABLE = "sample_unstructured_data"

tblList = spark.catalog.listTables()

if not any([table.name == SAMPLE_TABLE for table in tblList]):
   
  df = pd.DataFrame([
    {
      "external_id": dr.external_id,
      "row_data": dr.row_data
    } for dr in sample_dataset.data_rows()
  ]).to_spark()
  df.registerTempTable(SAMPLE_TABLE)
  print(f"Registered table: {SAMPLE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC いくつかのデモ用画像のファイル名とURLを含む一時テーブル"sample_unstructured_data"を手に入れたことになります。Databricks向けLabelboxコネクターを用いてLabelboxとこのテーブルを共有します！

# COMMAND ----------

display(sqlContext.sql(f"select * from {SAMPLE_TABLE} LIMIT 5"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ラベリングプロジェクトの作成
# MAGIC 
# MAGIC プロジェクトはチームがラベルを作成する場所となります。プロジェクトにはラベル付けされるアセットのデータセットと、ラベリングの推論を設定するためのオントロジーを必要となります。

# COMMAND ----------

# MAGIC %md
# MAGIC ### ステップ 1: データセットの作成
# MAGIC 
# MAGIC [Labelbox Connector for Databricks](https://pypi.org/project/labelspark/)は、2つのカラムを持つSparkテーブルを期待します。最初のカラムは"external_id"で、2つ目のカラムは"row_data"となります。
# MAGIC 
# MAGIC `external_id`は"birds.jpg"や"my_video.mp4"のようなファイル名となります。
# MAGIC 
# MAGIC `row_data`はファイルに対するURLパスとなります。Labelboxはラベリングを行う際、ユーザーのマシンにローカルにレンダリングするので、ラベリング担当はアセットにアクセスする権限を必要とします。 
# MAGIC 
# MAGIC サンプル: 
# MAGIC 
# MAGIC | external_id | row_data                             |
# MAGIC |-------------|--------------------------------------|
# MAGIC | image1.jpg  | https://url_to_your_asset/image1.jpg |
# MAGIC | image2.jpg  | https://url_to_your_asset/image2.jpg |
# MAGIC | image3.jpg  | https://url_to_your_asset/image3.jpg |

# COMMAND ----------

unstructured_data = spark.table(SAMPLE_TABLE)

demo_dataset = labelspark.create_dataset(client, unstructured_data, "Databricks Demo Dataset")

# COMMAND ----------

print("Open the dataset in the App")
print(f"https://app.labelbox.com/data/{demo_dataset.uid}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### ステップ 2: プロジェクトの作成
# MAGIC 
# MAGIC 使用するオントロジー(次に実施します)を構築するためにlabebox SDKを使用します。app.labelbox.comのウェブサイトを通じて、プロジェクト全体をセットアップすることができます。
# MAGIC 
# MAGIC [ontology creation documentation.](https://docs.labelbox.com/docs/configure-ontology)をチェックしてください。

# COMMAND ----------

# 新規プロジェクトの作成
project_demo = client.create_project(name="Labelbox and Databricks Example")
project_demo.datasets.connect(demo_dataset)  # キューにデータセットを追加

ontology = OntologyBuilder()

tools = [
  Tool(tool=Tool.Type.BBOX, name="Frog"),
  Tool(tool=Tool.Type.BBOX, name="Flower"),
  Tool(tool=Tool.Type.BBOX, name="Fruit"),
  Tool(tool=Tool.Type.BBOX, name="Plant"),
  Tool(tool=Tool.Type.SEGMENTATION, name="Bird"),
  Tool(tool=Tool.Type.SEGMENTATION, name="Person"),
  Tool(tool=Tool.Type.SEGMENTATION, name="Sleep"),
  Tool(tool=Tool.Type.SEGMENTATION, name="Yak"),
  Tool(tool=Tool.Type.SEGMENTATION, name="Gemstone"),
]
for tool in tools: 
  ontology.add_tool(tool)

conditions = ["clear", "overcast", "rain", "other"]

weather_classification = Classification(
    class_type=Classification.Type.RADIO,
    instructions="what is the weather?", 
    options=[Option(value=c) for c in conditions]
)  
ontology.add_classification(weather_classification)


# エディターのセットアップ
for editor in client.get_labeling_frontends():
    if editor.name == 'Editor':
        project_demo.setup(editor, ontology.asdict()) 

print("Project Setup is complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### ステップ 3: データラベリングの実施

# COMMAND ----------

print("Open the project to start labeling")
print(f"https://app.labelbox.com/projects/{project_demo.uid}/overview")

# COMMAND ----------

raise ValueError("Go label some data before continuing")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ラベル・アノテーションのエクスポート
# MAGIC 
# MAGIC Labelboxでラベルを作成した後は、Databricksでモデルのトレーニング、分析に使用するためにこれらをエクスポートすることができます。

# COMMAND ----------

LABEL_TABLE = "exported_labels"

# COMMAND ----------

labels_table = labelspark.get_annotations(client, project_demo.uid, spark, sc)
labels_table.registerTempTable(LABEL_TABLE)
display(labels_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Labelboxの他の機能
# MAGIC 
# MAGIC [<h3>モデルアシストラベリング</h3>](https://docs.labelbox.com/docs/model-assisted-labeling)
# MAGIC 
# MAGIC 非構造化データの初期セットに対してモデルをトレーニングした後で、モデルアシストラベリングのワークフローをサポートするために、そのモデルをLabelboxにプラグインすることができます。モデルの出力を確認し、訂正を行い、簡単二歳トレーニングすることができます！モデルアシストラベリングを活用することで、将来的には50%以上のラベリングのコストを削減することができます。
# MAGIC 
# MAGIC <img src="https://files.readme.io/4c65e12-model-assisted-labeling.png" alt="MAL" width="800"/>
# MAGIC 
# MAGIC [<h3>カタログ</h3>](https://docs.labelbox.com/docs/catalog)
# MAGIC 
# MAGIC Labelboxでデータセットとアノテーションを作成したら、カタログで容易にデータセットを参照し、新たなものを整理することができます。類似性検索を用いて画像を検索するために、お使いのモデルのエンべディングを使用します。
# MAGIC 
# MAGIC <img src="https://files.readme.io/14f82d4-catalog-marketing.jpg" alt="Catalog" width="800"/>
# MAGIC 
# MAGIC [<h3>モデル診断</h3>](https://labelbox.com/product/model-diagnostics)
# MAGIC 
# MAGIC 大規模な実験の予測結果を容易に可視化するための能力を用いることで、LabelboxはMLFlowのエクスペリメントトラッキングを補完します。モデル診断を用いることで、お使いのモデルの弱点の領域を迅速に特定し、適切なデータを収集し、次のイテレーションでモデルを改善することができます。
# MAGIC 
# MAGIC <img src="https://images.ctfassets.net/j20krz61k3rk/4LfIELIjpN6cou4uoFptka/20cbdc38cc075b82f126c2c733fb7082/identify-patterns-in-your-model-behavior.png" alt="Diagnostics" width="800"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 追加情報
# MAGIC 
# MAGIC Labelbox Connector for Databricksを使っている際には、Labelbox SDKを使いたい(例えば、プログラムによるオントロジーの作成)と考えることでしょう。これらのリソースは、みなさまがLabelbox Python SDKに慣れ親しむ助けになることでしょう。
# MAGIC * SDKの動作原理を学ぶために[ドキュメント](https://labelbox.com/docs/python-api)を参照ください。
# MAGIC * インタラクティブなチュートリアルを試すために、[notebook examples](https://github.com/Labelbox/labelspark/tree/master/notebooks)をチェックしてください。
# MAGIC * [API reference](https://labelbox.com/docs/python-api/api-reference)を参照ください。
# MAGIC 
# MAGIC 質問やコメントがある場合には、[ecosystem+databricks@labelbox.com](mailto:ecosystem+databricks@labelbox.com)にご連絡ください。

# COMMAND ----------

# MAGIC %md
# MAGIC Copyright Labelbox, Inc. 2021. The source in this notebook is provided subject to the [Labelbox Terms of Service](https://docs.labelbox.com/page/terms-of-service).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC |Library Name|Library license | Library License URL | Library Source URL |
# MAGIC |---|---|---|---|
# MAGIC |Labelbox Python SDK|Apache-2.0 License |https://github.com/Labelbox/labelbox-python/blob/develop/LICENSE|https://github.com/Labelbox/labelbox-python
# MAGIC |Labelbox Connector for Databricks|Apache-2.0 License |https://github.com/Labelbox/labelspark/blob/master/LICENSE|https://github.com/Labelbox/labelspark
# MAGIC |Python|Python Software Foundation (PSF) |https://github.com/python/cpython/blob/master/LICENSE|https://github.com/python/cpython|
# MAGIC |Apache Spark|Apache-2.0 License |https://github.com/apache/spark/blob/master/LICENSE|https://github.com/apache/spark|
