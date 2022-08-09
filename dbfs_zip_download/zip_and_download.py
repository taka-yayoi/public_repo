# Databricks notebook source
# MAGIC %md
# MAGIC # DBFS上のファイルを圧縮してダウンロードする
# MAGIC 
# MAGIC 処理の結果生成される複数のファイルを簡単にダウンロードできるように圧縮を行い、zipファイルへのリンクを表示するサンプルです。
# MAGIC 
# MAGIC **参考資料**
# MAGIC - [Databricksユーティリティ \- Qiita](https://qiita.com/taka_yayoi/items/3717623187859809515d)
# MAGIC - [Databricksにおけるzipファイルの取り扱い \- Qiita](https://qiita.com/taka_yayoi/items/0197d5c985089255f16a)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ファイルの準備
# MAGIC 
# MAGIC ダミーデータを準備します。以下のパスは適宜変更してください。

# COMMAND ----------

# 処理の結果が格納されるパスです
output_file_path = "/tmp/takaaki.yayoi@databricks.com/processed"

# ブラウザからダウンロードできるようにFileStore配下のパスとします
download_file_path = "/FileStore/shared_upload/takaaki.yayoi@databricks.com/viz_results"

# COMMAND ----------

# datetimeモジュールを使用
import datetime

t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, 'JST')
now = datetime.datetime.now(JST)
#print(repr(now))

# YYYYMMDDhhmmss形式に書式化
directory_name = now.strftime('%Y%m%d-%H%M%S')
#print(d)

target_path = f"{output_file_path}/{directory_name}"
print("target path:", target_path)

# 日付の入ったディレクトリを作成
dbutils.fs.mkdirs(target_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ワインデータセットをコピーします。

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/databricks-datasets/wine-quality/

# COMMAND ----------

dbutils.fs.cp("dbfs:/databricks-datasets/wine-quality/winequality-red.csv", target_path)
dbutils.fs.cp("dbfs:/databricks-datasets/wine-quality/winequality-white.csv", target_path)
dbutils.fs.cp("dbfs:/databricks-datasets/wine-quality/README.md", target_path)

# COMMAND ----------

# ファイルが格納されていることを確認
display(dbutils.fs.ls(target_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ファイル圧縮
# MAGIC 
# MAGIC DBFSの実態はオブジェクトストレージなので、ランダムアクセスが必要となるファイル圧縮をその場で行うことができません。このため、クラスターのローカルストレージに一旦コピーして圧縮を行う必要があります。
# MAGIC 
# MAGIC `dbutils.fs`でパスを指定する際にローカルストレージを参照するには、パスの先頭に`file:`をつけます。
# MAGIC 
# MAGIC **参考資料**
# MAGIC - [Databricksファイルシステム\(DBFS\) \- Qiita](https://qiita.com/taka_yayoi/items/897264c486e179d72247)
# MAGIC - [Databricksにおけるファイルシステム \- Qiita](https://qiita.com/taka_yayoi/items/e16c7272a7feb5ec9a92)
# MAGIC - [Databricksを使い始めたときに感じる疑問 \+ ベストプラクティス \- Qiita](https://qiita.com/taka_yayoi/items/8718c7c7d922e6f942bc)

# COMMAND ----------

# DBFSからローカルストレージにコピー
dbutils.fs.cp(target_path, f"file:/tmp/{directory_name}", True)

# COMMAND ----------

# ローカルストレージを参照
display(dbutils.fs.ls(f"file:/tmp/{directory_name}"))

# COMMAND ----------

# MAGIC %md
# MAGIC `shutil`を使って圧縮を行います。
# MAGIC 
# MAGIC [PythonでZIPファイルを圧縮・解凍するzipfile \| note\.nkmk\.me](https://note.nkmk.me/python-zipfile/)

# COMMAND ----------

# ディレクトリを圧縮
import shutil
shutil.make_archive(f"/tmp/{directory_name}", format='zip', root_dir=f'/tmp/{directory_name}')

# COMMAND ----------

# MAGIC %md
# MAGIC ブラウザからダウンロードできるように、DBFSの`/FileStore`にファイルをコピーします。

# COMMAND ----------

dbutils.fs.cp(f"file:/tmp/{directory_name}.zip", f"{download_file_path}/{directory_name}.zip")

# COMMAND ----------

display(dbutils.fs.ls(download_file_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ダウンロードリンクの生成

# COMMAND ----------

import re

# FileStoreをfilesに置き換えます
download_url_path = re.sub("FileStore", "files", download_file_path)
print(download_url_path)

# COMMAND ----------

displayHTML (f"""
<a href='{download_url_path}/{directory_name}.zip'>{directory_name}.zip</a>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## クリーンアップ
# MAGIC 
# MAGIC 出力用のパスを空にしておきます。

# COMMAND ----------

dbutils.fs.rm(output_file_path, True)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
