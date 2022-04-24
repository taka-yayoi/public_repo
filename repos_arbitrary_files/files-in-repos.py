# Databricks notebook source
# MAGIC %md # Databricks Reposにおけるファイルの取り扱い
# MAGIC 
# MAGIC このノートブックでは、Databricks Reposにおける任意のファイルの取り扱い方法を説明します。一般的なユースケースは:
# MAGIC 
# MAGIC - カスタムのPython、Rのモジュール。これらのモジュールをRepoに取り込むことで、Repoのノートブックは`import`文を用いてこれらの関数にアクセスすることができます。
# MAGIC - Repoの`requirements.txt`ファイルで環境を定義。ノートブックの環境を作成し、パッケージをインストールするには、ノートブックから`pip install -r requirements.txt`を実行するだけです。
# MAGIC - Repoに小規模データファイルを格納。これは開発やユニットテストで有用です。Repoのデータファイルの最大サイズは100MBです。Databricks Reposでは小さいファイル(< 10 MB)のエディタを提供します。
# MAGIC 
# MAGIC ## Reposとは
# MAGIC 
# MAGIC GithubなどのリポジトリとDatabricksワークスペースを同期する機能`Repositories`を略して**Repos**と呼んでいます。このため、ここでは、個々のリポジトリはRepoと呼称します。
# MAGIC 
# MAGIC ## このノートブックの使い方
# MAGIC 
# MAGIC このノートブックを使うには、以下のRepo([AWS](https://docs.databricks.com/repos.html#clone-a-remote-git-repository)|[Azure](https://docs.microsoft.com/azure/databricks/repos#clone-a-remote-git-repository)|[GCP](https://docs.gcp.databricks.com/repos.html#clone-a-remote-git-repository))をワークスペースにクローンします:
# MAGIC - https://github.com/databricks/files_in_repos

# COMMAND ----------

# MAGIC %md ## Pythonモジュールを取り扱う
# MAGIC 
# MAGIC 現在の作業ディレクトリ(`/Workspace/Repos/<username>/<repo_name>`)は自動的にPythonパスに追加されます。現在のディレクトリ、サブディレクトリにある任意のモジュールをインポートすることができます。

# COMMAND ----------

# パスの表示
import sys
print("\n".join(sys.path))

# COMMAND ----------

from sample import n_to_mth
n_to_mth(3, 4)

# COMMAND ----------

from utils.sample2 import cube_root
cube_root(8)

# COMMAND ----------

# MAGIC %md 
# MAGIC 他のリポジトリからモジュールをインポートするには、Pythonパスにそれらを追加します。例えば、Pythonモジュール`lib.py`を持つ`supplemental_files`というRepoがある場合には、次のセルのようにすることでインポートすることができます。

# COMMAND ----------

import sys
import os

# 以下のコマンドでは <username> をご自身のDatabricksユーザー名で置き換えてください
sys.path.append(os.path.abspath('/Workspace/Repos/<username>/supplemental_files'))

# これで、supplemental_files RepoからPythonモジュールをインポートすることができます
# import lib

# COMMAND ----------

# MAGIC %md ## 自動リロード
# MAGIC 
# MAGIC 長方形の面積を計算する関数`rectangle`を追加するために`sample.py`を編集したとします。以下のコマンドを実行することでモジュールは自動でリロードされます。

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# このコマンドを動作させるためには、`sample.py`を編集して、以下の関数を追加する必要があります:
# def rectangle(a, b):
#    return a * b

# そして、このセルを実行します
from sample import rectangle
rectangle(5, 4)

# COMMAND ----------

# MAGIC %md ## `requirements.txt`ファイルからのパッケージのインストール

# COMMAND ----------

pip install -r requirements.txt

# COMMAND ----------

# MAGIC %md ## 小さいデータファイルの取り扱い
# MAGIC 
# MAGIC Repoに小さいデータファイルを格納することができ、開発やユニットテストが便利になります。Repoにおけるデータファイルの最大サイズは100MBです。Databricks Reposは小さいファイル(< 10 MB)のエディタを提供します。Python、シェルコマンド、pandas、Koalas、PySparkでデータファイルを読み込むことができます。

# COMMAND ----------

# MAGIC %md ### Pythonによるファイルの参照

# COMMAND ----------

import csv
with open('data/winequality-red.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)

# COMMAND ----------

# MAGIC %md ### シェルコマンドによるファイルの参照

# COMMAND ----------

# MAGIC %sh head -10 data/winequality-red.csv

# COMMAND ----------

# MAGIC %md ### pandasによるファイルのロード

# COMMAND ----------

import pandas as pd

df= pd.read_csv("data/winequality-red.csv")
display(df)

# COMMAND ----------

# MAGIC %md ### Koalasによるファイルのロード
# MAGIC 
# MAGIC Koalasでは絶対ファイルパスが必要です。

# COMMAND ----------

import os
import databricks.koalas as ks

df= ks.read_csv(f"file:{os.getcwd()}/data/winequality-red.csv") # Koalasでは "file:" プレフィクスと絶対ファイルパスが必要です
display(df)

# COMMAND ----------

# MAGIC %md ### PySparkによるファイルのロード
# MAGIC 
# MAGIC PySparkでは絶対ファイルパスが必要です。

# COMMAND ----------

import os

df=spark.read.csv(f"file:{os.getcwd()}/data/winequality-red.csv", header=True) # PySparkでは "file:" プレフィクスと絶対ファイルパスが必要です
display(df)

# COMMAND ----------

# MAGIC %md ## 制限
# MAGIC 
# MAGIC プログラムからファイルに書き込みを行うことはできません。

# COMMAND ----------

# MAGIC %md
# MAGIC # END
