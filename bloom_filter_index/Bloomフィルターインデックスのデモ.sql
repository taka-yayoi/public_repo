-- Databricks notebook source
-- MAGIC %md # Bloomフィルターインデックスのデモ
-- MAGIC 
-- MAGIC Bloomフィルターは、偽陽性の確率を用いてセットにキーがあるかどうか、セットにアイテムが _あるかもしれない_ ことを教えてくれます。
-- MAGIC インデックスで使用すると、Bloomフィルターは他のテクニックでは高速化できない様なフィールドに対して「藁山から針を探す」ようなクエリーの高速化に役立ちます。
-- MAGIC 
-- MAGIC このノートブックでは以下のことを行います:
-- MAGIC 
-- MAGIC - テーブルをセットアップし、テーブルにBloomフィルターを設定し、テーブルをハッシュ値で埋めます。
-- MAGIC - 「藁山から針を探す」クエリーを実行します:
-- MAGIC   - インデックスが作成されていないカラム
-- MAGIC   - インデックスが作成されているカラム
-- MAGIC   - 結果が0になることが予想されるインデックスが作成されたカラム(藁山に針がない場合にどうなるか)
-- MAGIC - サンプルデータセットのクリーンアップ

-- COMMAND ----------

-- MAGIC %md ## テーブル、インデックスの作成、データのロード

-- COMMAND ----------

-- DBTITLE 1,Bloomフィルターインデックスの有効化
SET spark.databricks.io.skipping.bloomFilter.enabled = true;

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import re
-- MAGIC from pyspark.sql.types import * 
-- MAGIC 
-- MAGIC # Username を取得
-- MAGIC username_raw = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
-- MAGIC # Username の英数字以外を除去し、全て小文字化。Username をファイルパスやデータベース名の一部で使用可能にするため。
-- MAGIC username = re.sub('[^A-Za-z0-9]+', '', username_raw).lower()
-- MAGIC 
-- MAGIC # データベース名
-- MAGIC db_name = f"bloom_filter_{username}"
-- MAGIC 
-- MAGIC # パス
-- MAGIC data_path = f"dbfs:/tmp/{username_raw}/bloom_test"
-- MAGIC 
-- MAGIC # Hiveメタストアのデータベースの準備:データベースの作成
-- MAGIC spark.sql(f"DROP DATABASE IF EXISTS {db_name} CASCADE")
-- MAGIC spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
-- MAGIC # Hiveメタストアのデータベースの選択
-- MAGIC spark.sql(f"USE {db_name}")
-- MAGIC 
-- MAGIC print("database name: " + db_name)

-- COMMAND ----------

-- DBTITLE 1,テーブルの作成
-- MAGIC %python
-- MAGIC spark.sql(f"""
-- MAGIC CREATE OR REPLACE TABLE bloom_test (
-- MAGIC   id   BIGINT NOT NULL,
-- MAGIC   str1 STRING NOT NULL,
-- MAGIC   sha  STRING NOT NULL,
-- MAGIC   sha1 STRING NOT NULL,
-- MAGIC   sha2_256 STRING NOT NULL,
-- MAGIC   row_hash_too_big STRING NOT NULL,
-- MAGIC   row_hash STRING NOT NULL
-- MAGIC )
-- MAGIC USING DELTA
-- MAGIC LOCATION '{data_path}'
-- MAGIC """)

-- COMMAND ----------

-- DBTITLE 1,データを追加する前にインデックスを作成
CREATE BLOOMFILTER INDEX
ON TABLE bloom_test
FOR COLUMNS(sha OPTIONS (fpp=0.1, numItems=50000000))

-- COMMAND ----------

-- DBTITLE 1,データ生成
TRUNCATE TABLE bloom_test;

WITH sample (
  SELECT
    id,
    'windows.exe' as str1,
    monotonically_increasing_id() mono_id,
    hash(id) hash,
    sha (cast(id % 50000000 as string)) sha,
    sha1(cast(id % 50000000 as string)) sha1,
    sha2(cast(id as string), 256)    sha2_256
  from
    RANGE(0, 1000000000, 1, 448)  -- start, end, step, numPartitions
)
INSERT INTO bloom_test 
SELECT id, 
  str1, 
  sha,
  sha1,
  sha2_256,
  sha2(concat_ws('||',id, str1, mono_id, hash, sha, sha1, sha2_256),512) row_hash_too_big,
  sha2(concat_ws('||',id, str1, mono_id, hash, sha, sha1, sha2_256),256) row_hash
FROM sample
-- LIMIT 20000

-- COMMAND ----------

SET spark.databricks.delta.optimize.maxFileSize = 1610612736;
OPTIMIZE bloom_test
ZORDER BY id

-- COMMAND ----------

-- MAGIC %md ## 物理テーブルとインデックスを確認

-- COMMAND ----------

-- MAGIC %python
-- MAGIC display(dbutils.fs.ls(data_path))

-- COMMAND ----------

-- MAGIC %python
-- MAGIC display(dbutils.fs.ls(f"{data_path}/_delta_index"))

-- COMMAND ----------

DESCRIBE EXTENDED bloom_test

-- COMMAND ----------

-- MAGIC %md ## テストクエリーの実行

-- COMMAND ----------

-- DBTITLE 1,同じファイルに存在しないであろうハッシュ値の検索
SELECT * FROM bloom_test WHERE id in ( 0, 1, 999999998, 999999999)

-- COMMAND ----------

-- DBTITLE 1,Bloomフィルターのインデックスが無いカラムに対するクエリー
SELECT count(*) FROM bloom_test WHERE sha1 = 'b2f9544427aed7b712b209fffc756c001712b7ca'

-- COMMAND ----------

-- DBTITLE 1,Bloomフィルターのインデックスがあるカラムに対するクエリー
SELECT count(*) FROM bloom_test WHERE sha = 'b6589fc6ab0dc82cf12099d1c2d40ab994e8410c'

-- COMMAND ----------

-- DBTITLE 1,Bloomフィルターのインデックスに対して存在しないものを検索
SELECT count(*) FROM bloom_test WHERE sha = 'b6589fc6ab0dc82cf12099d1c2d40ab994e8410_'

-- COMMAND ----------

-- MAGIC %md ## クリーンアップ

-- COMMAND ----------

-- DBTITLE 1,すべてを実行した際にクリーンアップしない様に停止
-- MAGIC %python dbutils.notebook.exit(0)

-- COMMAND ----------

DROP BLOOMFILTER INDEX ON TABLE bloom_test FOR COLUMNS(sha);

-- COMMAND ----------

DROP TABLE IF EXISTS bloom_test

-- COMMAND ----------

-- MAGIC %python
-- MAGIC dbutils.fs.rm(data_path, True)

-- COMMAND ----------

-- MAGIC %md ## 結果
-- MAGIC 
-- MAGIC ### パフォーマンス
-- MAGIC | Metric | Value |
-- MAGIC | --- | --- |
-- MAGIC | Bloomフィルターなし | ~21 s |
-- MAGIC | Bloomフィルターあり | ~13 s |
-- MAGIC | セットに存在しなアイテムでBloomフィルターの列を検索 | ~9 s|
-- MAGIC 
-- MAGIC ### クラスター設定:
-- MAGIC 
-- MAGIC | Metric | Value|
-- MAGIC | --- | --- |
-- MAGIC | ワーカー数 | 4 |
-- MAGIC | ワーカー |16 GB, 8 core  |
-- MAGIC | ドライバー | 16 GB, 8 core |
-- MAGIC | Databricksランタイム | 7.1.x-scala2.12 |
-- MAGIC     
-- MAGIC ### データ:
-- MAGIC | Metric | Value  |
-- MAGIC |---|---|
-- MAGIC | レコード数|1,000,000,000  | 
-- MAGIC |'needle'のインスタンス | 20 | 
-- MAGIC | Delta Parquetデータファイル数 | 140  |
-- MAGIC | Delta Parquetインデックスファイル数  |140 |
-- MAGIC | データファイルサイズ | 1.5GB | 
-- MAGIC | Deltaインデックスファイルサイズ | 30MB| 
-- MAGIC | 合計バイト数 | 150943777871, ~151GB |
-- MAGIC 
-- MAGIC 
-- MAGIC ### Bloomフィルターのパラメーター
-- MAGIC | Parameter | Value |
-- MAGIC | --- | --- |
-- MAGIC | FPP | 0.1, 0.1% |
-- MAGIC | numItems |  50,000,000 - 50k distinct items|
