# Databricks notebook source
# MAGIC %md 
# MAGIC # イベントログに対するクエリー
# MAGIC 
# MAGIC それぞれのDelta Live Tablesのパイプラインごとにイベントログが作成され、メンテナンスされます。イベントログには、監査ログ、データ品質チェック、パイプラインの進捗状況、データリネージュを含むパイプラインに関する全ての情報が含まれています。このイベントログを用いて、お使いのパイプラインの状態を追跡、理解、モニタリングすることができます。イベントログは`/events`に格納されており、APIあるいはDelta Live Tables UIでアクセスすることができます。イベントログはDeltaテーブルとして格納されており、さらに複雑な分析を行うためにDatabricksノートブックからアクセスすることができます。このノートブックでは、イベントログから有用なデータをどのようにクエリーするのかを示すシンプルな例をデモします。
# MAGIC 
# MAGIC このノートブックでは、Databricksランタイム8.1以降で利用できるJSON SQL関数を使用しています。

# COMMAND ----------

# MAGIC %md
# MAGIC ## Storage locationテキストボックスの作成

# COMMAND ----------

# ノートブック上部に入力ボックスを作成します
dbutils.widgets.text('storage', 'dbfs:/pipelines/production-data', 'Storage Location')

# COMMAND ----------

# MAGIC %md
# MAGIC ## イベントログの参照
# MAGIC 
# MAGIC このノートブックのサンプルでは、イベントログに対するクエリーをシンプルにするために`event_log_raw`というビューを使用します。ビュー`event_log_raw`を作成するには、以下を実行します。
# MAGIC 
# MAGIC 1. **Storage Location**テキストボックスにイベントログのパスを入力します。パイプライン設定の`storage`でパスを確認することができます。
# MAGIC 1. `event_log_raw`ビューを作成するために以下のコマンドを実行します。

# COMMAND ----------

# MAGIC %md
# MAGIC ### イベントログビューの作成

# COMMAND ----------

# DBTITLE 0,Create the event log view
# Storage Location入力ボックスをパイプラインのストレージロケーションで置き換えます。これは、パイプライン設定の'storage'で設定を確認することができます。
storage_location = dbutils.widgets.get('storage')
event_log_path = storage_location + "/system/events"

# クエリーを簡単に行えるようにイベントログを一時ビューに読み込みます。
event_log = spark.read.format('delta').load(event_log_path)
event_log.createOrReplaceTempView("event_log_raw")

# COMMAND ----------

# DBTITLE 0,Event log schema
# MAGIC %md
# MAGIC ### イベントログのスキーマ
# MAGIC 
# MAGIC イベントログのトップレベルスキーマを示します。
# MAGIC 
# MAGIC | フィールド | 説明 |
# MAGIC |:--|:--|
# MAGIC | id  | パイプラインのID  |
# MAGIC | sequence  | イベントを識別し並び替えるためのメタデータを含むJSONドキュメント  |
# MAGIC | origin  | イベントの起源を示すメタデータを含むJSONドキュメント。例えば、クラウドプロバイダー、リージョン、ユーザーID、パイプラインID  |
# MAGIC |  timestamp | イベントが記録された時刻  |
# MAGIC | message  | イベントを説明する判読可能なメッセージ  |
# MAGIC | level  | イベントタイプ。例えば、INFO、WARN、ERROR、METRICS  |
# MAGIC | error  | エラーが発生した場合、エラーを説明する詳細メッセージ  |
# MAGIC | details  | イベントの詳細を含むJSONドキュメント。イベントを分析する際に用いられる主要なフィールド  |
# MAGIC | event_type  | イベントタイプ  |

# COMMAND ----------

# MAGIC %md
# MAGIC ### イベントログレコードのサンプルの参照

# COMMAND ----------

# DBTITLE 0,View a sample of event log records
# MAGIC %sql
# MAGIC SELECT * FROM event_log_raw LIMIT 100

# COMMAND ----------

# MAGIC %md
# MAGIC ## 監査ログ
# MAGIC 
# MAGIC データパイプラインの一般的、重要なユースケースには、ユーザーが実行したアクションの監査ログの作成があります。ユーザーアクションに対する情報を含むイベントは、イベントタイプ`user_action`となります。以下の例では、アクションを行なったユーザー名、タイムスタンプ、アクションのタイプを検索しています。

# COMMAND ----------

# MAGIC %md
# MAGIC ### ユーザーイベント監査のサンプルクエリー

# COMMAND ----------

# DBTITLE 0,Example query for user events auditing
# MAGIC %sql
# MAGIC SELECT timestamp, details:user_action:action, details:user_action:user_name FROM event_log_raw WHERE event_type = 'user_action'

# COMMAND ----------

# MAGIC %md
# MAGIC ### パイプラインアップデートの詳細
# MAGIC 
# MAGIC パイプライン実行のそれぞれのインスタンスは*update*と呼ばれます。以下の例では、最後のパイプライン実行となる最新のアップデートの情報を抽出しています。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 最新のパイプラインアップデートのIDの取得

# COMMAND ----------

# DBTITLE 0,Get the ID of the most recent pipeline update
# クエリーで使えるように最新のアップデートIDをSpark設定として保存します
latest_update_id = spark.sql("SELECT origin.update_id FROM event_log_raw WHERE event_type = 'create_update' ORDER BY timestamp DESC LIMIT 1").collect()[0].update_id
spark.conf.set('latest_update.id', latest_update_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## リネージュ
# MAGIC 
# MAGIC リネージュはUIにグラフとして表示されます。企業におけるコンプライアンスに対するレポートの作成やデータの依存関係の追跡を行うといったタスクを実行するために、この情報をプログラムから抽出することができます。リネージュに関する情報を含むイベントは、イベントタイプ`flow_definition`となっています。`flow_definition`オブジェクトのフィールドには、データセット間の関係を推定するために必要な情報が含まれています。

# COMMAND ----------

# MAGIC %md
# MAGIC ### パイプラインリネージュに対するサンプルクエリー

# COMMAND ----------

# DBTITLE 0,Example query for pipeline lineage
# MAGIC %sql
# MAGIC SELECT details:flow_definition.output_dataset, details:flow_definition.input_datasets FROM event_log_raw WHERE event_type = 'flow_definition' AND origin.update_id = '${latest_update.id}'

# COMMAND ----------

# MAGIC %md
# MAGIC ## データ品質
# MAGIC 
# MAGIC イベントログにはデータ品質に関するメトリクスが含まれています。データ品質に関する情報を含むイベントは、イベントタイプ`flow_progress`となります。以下のクエリーでは、それぞれのデータセットに定義された個々のデータ品質ルールを通過したレコード数、失敗したレコード数を抽出します。

# COMMAND ----------

# MAGIC %md
# MAGIC ### データ品質に対するサンプルクエリー

# COMMAND ----------

# DBTITLE 0,Example query for data quality
# MAGIC %sql
# MAGIC SELECT
# MAGIC   row_expectations.dataset as dataset,
# MAGIC   row_expectations.name as expectation,
# MAGIC   SUM(row_expectations.passed_records) as passing_records,
# MAGIC   SUM(row_expectations.failed_records) as failing_records
# MAGIC FROM
# MAGIC   (
# MAGIC     SELECT
# MAGIC       explode(
# MAGIC         from_json(
# MAGIC           details :flow_progress :data_quality :expectations,
# MAGIC           "array<struct<name: string, dataset: string, passed_records: int, failed_records: int>>"
# MAGIC         )
# MAGIC       ) row_expectations
# MAGIC     FROM
# MAGIC       event_log_raw
# MAGIC     WHERE
# MAGIC       event_type = 'flow_progress'
# MAGIC       AND origin.update_id = '${latest_update.id}'
# MAGIC   )
# MAGIC GROUP BY
# MAGIC   row_expectations.dataset,
# MAGIC   row_expectations.name

# COMMAND ----------

# MAGIC %md
# MAGIC # END
