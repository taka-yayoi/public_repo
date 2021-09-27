# Databricks notebook source
# MAGIC %md # Databricksにおけるライブラリの取り扱い
# MAGIC 
# MAGIC 本ノートブックでは、Databricksにおけるライブラリの形態及び利用方法をご説明します。
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2021/02/10</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>クラスター</td><td>7.4ML</td></tr>
# MAGIC </table>
# MAGIC <img style="margin-top:25px;" src="https://sajpstorage.blob.core.windows.net/workshop20210205/databricks-logo-small-new.png" width="140">
# MAGIC 
# MAGIC [Libraries — Databricks Documentation](https://docs.databricks.com/libraries/index.html)

# COMMAND ----------

# MAGIC %md ## Databricksにおけるライブラリの形態
# MAGIC 
# MAGIC Databricksにおいては、以下の3つのライブラリの形態があります。
# MAGIC 
# MAGIC 1. ワークスペースライブラリ
# MAGIC 2. クラスターライブラリ
# MAGIC 3. ノートブックスコープライブラリ

# COMMAND ----------

# MAGIC %md ### 1. ワークスペースライブラリ
# MAGIC 
# MAGIC ワークスペースライブラリは、後述するクラスターライブラリをインストールする際にローカルのリポジトリとして動作します。お客様固有のカスタムライブラリを保持したり、環境を標準化する目的で特定バージョンのライブラリを保持するために用いられます。
# MAGIC 
# MAGIC ノートブックやジョブで使用する前にワークスペースライブラリをインストールしておく必要があります。
# MAGIC 
# MAGIC `Shared`フォルダにあるワークスペースライブラリは、ワークスペースの全ユーザから利用できます。一方、ユーザーフォルダ内のワークスペースライブラリは、そのユーザーからのみ利用可能です。

# COMMAND ----------

# MAGIC %md #### ワークスペースライブラリの作成
# MAGIC 
# MAGIC 1. Workspace上で、Create > Libraryを選択します。<br>
# MAGIC ![](https://docs.databricks.com/_images/create-library.png)
# MAGIC 
# MAGIC 2. ダイアログが表示されますので、ライブラリソースに従って設定します。<br>
# MAGIC ![](https://docs.databricks.com/_images/select-library-aws.png)
# MAGIC 
# MAGIC PyPIの場合、`<library>==<version>`という形式で、バージョン番号を指定できます。

# COMMAND ----------

# MAGIC %md ### 2. クラスターライブラリ
# MAGIC 
# MAGIC クラスターライブラリは、クラスター上で動作する全てのノートブックで利用できるライブラリです。クラスターライブラリはPyPIやMavenなどから直接インストールすることもできますし、上述のワークスペースライブラリからインストールすることも可能です。

# COMMAND ----------

# MAGIC %md #### クラスターライブラリのインストール
# MAGIC 
# MAGIC クラスターライブラリをインストールするには以下の2つの選択肢があります。
# MAGIC 
# MAGIC 1. 既にインストールされているワークスペースライブラリからクラスターにインストールする
# MAGIC 2. 特定のクラスターにのみライブラリをインストールする
# MAGIC 
# MAGIC **注意** クラスターにライブラリをインストールする際、クラスターにノートブックがアタッチされていると、当該のノートブックからライブラリが参照できない場合があります。その場合には、ノートブックのdetach/attachを行なってください。

# COMMAND ----------

# MAGIC %md #### ワークスペースライブラリからのインストール
# MAGIC 
# MAGIC **注意** Databricks Runtime 7.2以降では、ワークスペースライブラリからクラスターにインストールされた順にライブラリが処理されます。ライブラリ間に依存関係がある場合には、クラスターにインストールする順番に注意してください。
# MAGIC 
# MAGIC インストールは、クラスターUIもしくはライブラリUIから実行することができます。
# MAGIC 
# MAGIC ##### クラスターUIからのインストール
# MAGIC 
# MAGIC 1. サイドバーのClustersアイコン![](https://docs.databricks.com/_images/clusters-icon.png)をクリックします。
# MAGIC 2. クラスター名をクリックします
# MAGIC 3. **Libraries**タブをクリックします
# MAGIC 4. **Install New**をクリックします
# MAGIC 5. Library Sourceのボタンから**Workspace**を選択します
# MAGIC 6. ワークスペースライブラリを選択します
# MAGIC 7. **Install**をクリックします
# MAGIC 8. 全てのクラスターにライブラリをインストールする際には、libraryをクリックし、**Install automatically on all clusters**チェックボックスをチェックし、**Confirm**をクリックします
# MAGIC 
# MAGIC ##### ライブラリUIからのインストール
# MAGIC 
# MAGIC 1. ワークスペースライブラリを選択します
# MAGIC 2. ライブラリをインストールするクラスターを選択します

# COMMAND ----------

# MAGIC %md ### 3. ノートブックスコープライブラリ
# MAGIC 
# MAGIC ノートブックスコープライブラリは、あなたが実行しているノートブックのセッション内でのみ有効となるライブラリです。ノートブックスコープライブラリは、同じクラスタで動作している他のノートブックには影響を及ぼしません。このライブラリは永続化されず、セッションの都度インストールする必要があります。
# MAGIC 
# MAGIC ノートブック特有のライブラリが必要な場合に、ノートブックスコープライブラリを使用します。<br>
# MAGIC 
# MAGIC - **Databricks Runtime ML 6.4以降**においては、`%pip`、`%conda`のマジックコマンドを用いてインストールすることで、ノートブックスコープライブラリとなります。**Databricks Runtime 7.1以降**では、`%pip`を使用します。
# MAGIC - ライブラリユーティリティを用いても、ノートブックスコープライブラリをインストールできますが、`%pip`と互換性が無いことにご注意ください。Databricksでは、`%pip`を使用することを推奨しています。
# MAGIC - ライブラリユーティリティは、**Databricks Runtime**でのみサポートされており、**Databricks ML Runtime**ではサポートされていません。

# COMMAND ----------

# MAGIC %md #### ノートブックスコープライブラリの要件
# MAGIC マジックコマンドによるライブラリのインストールは、Databricks Runtime 7.1、Databricks Runtime 7.1 ML以降でのみサポートされています。

# COMMAND ----------

# MAGIC %md 
# MAGIC #### ドライバーノードに関する留意点
# MAGIC ノートブックスコープライブラリのインストールは、クラスタ上の全ワーカーノードにライブラリをインストールするため、ドライバーノードにおいて多くのトラフィックが発生する場合があります。10以上のノードを持つクラスターを構成する際には、ドライバーノードが以下のスペックを満たすことをお勧めしています。<br><br>
# MAGIC 
# MAGIC - 100CPUノードのクラスターの場合、`i3.8xlarge`をお使いください
# MAGIC - 10GPUノードのクラスターの場合、`p2.xlarge`をお使いください

# COMMAND ----------

# MAGIC %md #### `%pip`、`%conda`の使い分け
# MAGIC 
# MAGIC [Anaconda \| Understanding Conda and Pip](https://www.anaconda.com/blog/understanding-conda-and-pip)
# MAGIC 
# MAGIC Databricks Runtimeでは、`%pip`を使ってノートブックスコープライブラリをインストールできます。また、Databricks Runtime MLでは、`%pip`に加えて`%conda`を使用することもできます。Databricksでは、特別な理由がない限り`%pip`を使用することを推奨しています。
# MAGIC 
# MAGIC **注意**
# MAGIC - ノートブックの一番最初に`%pip`、`%conda`を記述してください。`%pip`、`%conda`が実行された後に、ノートブックの内部状態がリセットされます。変数や関数を宣言した後に`%pip`、`%conda`を実行するとそれらは消去されてしまいます。
# MAGIC - 一つのノートブックで`%pip`、`%conda`を混在させなくてはいけない場合には、競合を避けるために以下のリンク先をご一読ください。
# MAGIC 
# MAGIC [Notebook\-scoped Python libraries — Databricks Documentation](https://docs.databricks.com/libraries/notebooks-python-libraries.html#pip-conda-interactions)

# COMMAND ----------

# MAGIC %md #### `%pip`によるライブラリのインストール

# COMMAND ----------

# MAGIC %pip install matplotlib

# COMMAND ----------

# MAGIC %md #### `%pip`によるライブラリのアンインストール

# COMMAND ----------

# MAGIC %pip uninstall -y matplotlib

# COMMAND ----------

# MAGIC %md #### インストールされているライブラリの一覧取得

# COMMAND ----------

# MAGIC %pip freeze > /dbfs/tmp/requirements.txt

# COMMAND ----------

# MAGIC %fs head /tmp/requirements.txt

# COMMAND ----------

# MAGIC %md #### requirementファイルによるライブラリのインストール

# COMMAND ----------

# インストールして問題がないかご確認ください
#%pip install -r /dbfs/tmp/requirements.txt

# COMMAND ----------

# MAGIC %md #### `%conda`によるライブラリのインストール

# COMMAND ----------

# MAGIC %md
# MAGIC **注意** Anaconda, Inc.は2020年9月に[利用規約](https://www.anaconda.com/terms-of-service)をアップデートしています。新たな利用規約によりcommercial licenseが必要になる場合があります。[こちら](https://www.anaconda.com/blog/anaconda-commercial-edition-faq)で詳細を確認してください。この変更を受けて、Databricks Runtime ML 8.0において、Condaパッケージマネージャのデフォルトチャネル設定を削除しています。`%conda`でライブラリをインストールする際には、利用規約に適合していることを確認の上、チャネルを指定してください。

# COMMAND ----------

# MAGIC %conda install boto3 -c conda-forge

# COMMAND ----------

# MAGIC %md #### `%conda`によるライブラリのアンインストール

# COMMAND ----------

# MAGIC %conda uninstall boto3

# COMMAND ----------

# MAGIC %conda list

# COMMAND ----------

# MAGIC %md #### 環境の保存及び再利用

# COMMAND ----------

# MAGIC %conda env export -f /dbfs/tmp/myenv.yml

# COMMAND ----------

#%conda env update -f /dbfs/tmp/myenv.yml

# COMMAND ----------

# MAGIC %md ### Pythonにおけるライブラリインストールの選択肢
# MAGIC 
# MAGIC `%conda`は`%pip`と同じように動作しますが、`%conda`におけるURLの指定は`--channel`で行います。
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>Pythonパッケージソース</th><th>`%pip`によるノートブックスコープライブラリ</th><th>ライブラリユーティリティによるノートブックスコープライブラリ</th><th>クラスターライブラリ</th></tr>
# MAGIC   <tr><td>PyPI</td><td>`%pip install`を使用</td><td>`dbutils.library.installPyPI`を使用</td><td>sourceにPyPIを指定</td></tr>
# MAGIC   <tr><td>プライベートPyPI</td><td>`--index-url`オプションとともに`%pip install`を使用</td><td>repoオプションとともに`dbutils.library.installPyPI`を使用</td><td>未サポート</td></tr>
# MAGIC   <tr><td>GitHubのようなVCS</td><td>パッケージ名にリポジトリURLを指定して`%pip install`を使用</td><td>未サポート</td><td>パッケージ名にリポジトリURLを指定してsourceにPyPIを指定</td></tr>
# MAGIC   <tr><td>プライベートVCS</td><td>パッケージ名にリポジトリURL(Basic認証含む)を指定して`%pip install`を使用</td><td>未サポート</td><td>未サポート</td></tr>
# MAGIC   <tr><td>DBFS</td><td>`%pip install`を使用</td><td>`dbutils.library.install(dbfs_path)`を使用</td><td>sourceにDBFS/S3を指定</td></tr>
# MAGIC </table>

# COMMAND ----------

# MAGIC %md ## libifyによるカスタムライブラリの利用
# MAGIC 
# MAGIC ローカルでカスタムライブラリを構築している場合、カスタムライブラリをDatabricksに移行したいというケースが想定されます。その際の手順としては以下の二つがあります。
# MAGIC 
# MAGIC 1. wheelを作成して、ワークスペースライブラリとしてインストール<br>
# MAGIC [Python でパッケージを開発して配布する標準的な方法 \- Qiita](https://qiita.com/propella/items/803923b2ff02482242cd)
# MAGIC 2. [libify · PyPI](https://pypi.org/project/libify/)をインストールし、ノートブック間のimportを実現する
# MAGIC 
# MAGIC ここでは、2の方法をデモで説明します。importされるノートブックは`my_lib/my_module_1`となります。
# MAGIC 
# MAGIC **注意** libifyで呼び出す・呼び出されるノートブック名に日本語を含めることはできません。

# COMMAND ----------

# 事前にlibifyをクラスターライブラリとしてインストールしてください
import libify

# COMMAND ----------

# caller
mod1 = libify.importer(globals(), 'my_lib/my_module_1')

# COMMAND ----------

mod1.test()

# COMMAND ----------

# MAGIC %md ## END
