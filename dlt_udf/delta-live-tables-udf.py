# Databricks notebook source
# MAGIC %md
# MAGIC # Delta Live TablesにおけるUDFの活用
# MAGIC 
# MAGIC このノートブックでは、Pythonノートブックで定義された[Python UDF]($./dlt-python-callee)を[Python SQL]($./dlt-sql-caller)、[PySpark]($./dlt-python-caller)から呼び出すサンプルを説明します。
# MAGIC 
# MAGIC - [Delta Live Tablesクックブック \- Qiita](https://qiita.com/taka_yayoi/items/184d872392ac4fb7fb38#sql%E3%81%A7python-udf%E3%82%92%E4%BD%BF%E3%81%86)
# MAGIC - [Delta Live Tablesユーザーガイド \- Qiita](https://qiita.com/taka_yayoi/items/6726ad1edfa92d5cd0e9#%E4%B8%8D%E6%AD%A3%E3%83%AC%E3%82%B3%E3%83%BC%E3%83%89%E3%82%92%E4%BF%9D%E6%8C%81)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 環境の設定
# MAGIC 
# MAGIC 以下のセルを実行するところからスタートします。`$mode`は、現状は再帰的にソースデータ、ディスティネーションデータを削除します。これによって、新たなDLTパイプライン、新たなデータ処理を行うエンドツーエンドのデモの再利用が可能になります。
# MAGIC 
# MAGIC **注意**: このノートブックは添付されているSQL、Pythonノートブックのいずれでも動作しますが、このセットアップスクリプトはロードしたデータに対する新規パイプラインをデプロイする前に実行してください。

# COMMAND ----------

# MAGIC %run ./Includes/setup $mode="reset"

# COMMAND ----------

storage_location.split(':')[1]

# COMMAND ----------

# MAGIC %md
# MAGIC ## テーブルに対するクエリー
# MAGIC 
# MAGIC このパイプラインでは、ストレージロケーションとして`storage_location`変数にパスを割り当てているので、全てのテーブルは指定した場所に書き込まれます。
# MAGIC 
# MAGIC 以下の関数を用いてテーブルとビューに対してクエリーを発行することができます。初回のパイプラインの処理が成功した後でクエリーを実行することができます。

# COMMAND ----------

def query_table(table_name):
    return spark.sql(f"SELECT * FROM delta.`{storage_location}/tables/{table_name}`")

# COMMAND ----------

display(query_table("squared_even"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## クリーンアップ
# MAGIC 
# MAGIC クリーンアップした後はパイプラインも削除してください。これによって、自動的に2つの関連づけられたジョブを削除します。 (ジョブ名は: **Pipelines - `<your_pipeline_name>`**と**Pipeline Maintenance - `<your_pipeline_name>`**となります)
# MAGIC 
# MAGIC 以下のセルを実行することで、デモで用いたデータファイル、テーブル、ログを削除します。

# COMMAND ----------

# MAGIC %run ./Includes/setup $mode="cleanup"

# COMMAND ----------

# MAGIC %md
# MAGIC # END
