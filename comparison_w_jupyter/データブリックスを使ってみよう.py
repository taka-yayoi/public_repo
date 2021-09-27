# Databricks notebook source
# MAGIC %md
# MAGIC # データブリックスを使ってみよう
# MAGIC 
# MAGIC AI/機械学習の世界で注目を浴びているデータブリックスをご存知ですか？
# MAGIC 
# MAGIC > 「最近聞くけど調べる時間がないなー。Jupyter notebookみたいなもの？何が嬉しいの？」
# MAGIC 
# MAGIC 大丈夫です！このノートブックではデータブリックスを初めて触る方向けに、メリットを含めてデータブリックスの使い方をご説明します。
# MAGIC 
# MAGIC **ハンズオン資料**
# MAGIC - [ハンズオン事前準備資料](https://sajpstorage.blob.core.windows.net/handson20210629/2021-06-29_Databricks_Workshop_prep.pdf)
# MAGIC - [pandasとSparkの比較ノートブック](https://sajpstorage.blob.core.windows.net/handson20210629/pandasとSparkの比較.html)
# MAGIC 
# MAGIC **参考資料**
# MAGIC - [Databricksで分析業務がはかどっている話 \- Qiita](https://qiita.com/paulxll/items/1962e806cdc8a96068c1)
# MAGIC - [Databricksクイックスタートガイド \- Qiita](https://qiita.com/taka_yayoi/items/125231c126a602693610)
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2021/06/29</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>クラスター</td><td>8.3ML</td></tr>
# MAGIC </table>
# MAGIC <img style="margin-top:25px;" src="https://jixjiadatabricks.blob.core.windows.net/images/databricks-logo-small-new.png" width="140">

# COMMAND ----------

# MAGIC %md
# MAGIC ## データブリックスとは？
# MAGIC 
# MAGIC まず初めに、データブリックスを説明させてください。データブリックスはApache Sparkを実行するためのマネージドプラットフォームです。つまり、Sparkを利用するために、複雑なクラスター管理の考え方や、面倒なプラットフォームの管理手順を学ぶ必要がないということです。
# MAGIC 
# MAGIC また、データブリックスは、Sparkを利用したワークロードを円滑にするための機能も提供しています。GUIでの操作を好む方向けに、マウスのクリックで操作できるプラットフォームとなっています。しかし、UIに加えて、データ処理のワークロードをジョブで自動化したい方向けには、洗練されたAPIも提供しています。エンタープライズでの利用に耐えるために、データブリックスにはロールベースのアクセス制御や、使いやすさを改善するためだけではなく、管理者向けにコストや負荷軽減のための最適化が図られています。

# COMMAND ----------

# MAGIC %md
# MAGIC ## データブリックスの提供機能
# MAGIC 
# MAGIC データ・AIに関する作業を効率化するために、データブリックスは様々な機能を提供していますが、以下の3つに分類されます。
# MAGIC 1. 大量データを高速処理するための基盤
# MAGIC 2. 生産性高くデータ・AIに関する作業を行うためのワークスペース
# MAGIC 3. 機械学習モデルを管理するための仕組み
# MAGIC 
# MAGIC このワークショップでは主に2.ワークスペースをご説明します。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/handson20210629/databricks_capability.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ### コミュニティエディションの制限
# MAGIC 
# MAGIC フルバージョンのデータブリックスと比較してコミュニティエディションには以下の制限があります。
# MAGIC <br><br>
# MAGIC - 作成できるクラスターは15GB RAM、2 Core CPUのシングルノードのみ
# MAGIC - ワークスペースに追加できるユーザー数は最大3名まで
# MAGIC - クラスターのリージョンはus-westのみ
# MAGIC - 使用できない機能
# MAGIC   - ジョブのスケジュール機能
# MAGIC   - クラスターのオートスケーリング機能
# MAGIC   - Git連携
# MAGIC   - MLflowの一部モデル管理機能(レジストリ、RESTサービング)
# MAGIC   - REST APIによるワークスペースの制御
# MAGIC   - セキュリティ、ロールベースのアクセス制御、監査、シングルサインオン
# MAGIC   - BIツール連携のサポート
# MAGIC   
# MAGIC **参考資料**
# MAGIC - [IBJP: Community Editionで始めるDatabricks \- Databricks](https://databricks.com/jp/international-blogs/get-started-with-databricks-community-edition-jp)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 用語集
# MAGIC 
# MAGIC データブリックスには理解すべきキーとなるコンセプトがあります。これらの多くは画面左側のサイドバーにアイコンとして表示されています。これらは、エンドユーザーであるあなたのために用意された基本的なツールとなります。これらはウェブアプリケーションのUIからも利用できますし、REST APIからも利用できます。<br><br>
# MAGIC 
# MAGIC - **Workspaces(ワークスペース)**
# MAGIC   - Databricksで行う作業はワークスペース上で実施することになります。あなたのコンピューターのフォルダー階層のように、**notebooks(ノートブック)**や**libraries(ライブラリ)**を保存でき、それらを他のユーザーと共有することができます。ワークスペースはデータとは別となっており、データを格納するために用いるものではありません。これらは単に**notebooks(ノートブック)**や**libraries(ライブラリ)**を保存するためのものであり、データを操作するためにこれらを使用することになります。
# MAGIC - **Notebooks(ノートブック)**
# MAGIC   - ノートブックはコマンドを実行するためのセルの集合です。セルには以下の言語を記述することができます： `Scala`、`Python`、`R`、`SQL`、`Markdown`。ノートブックにはデフォルトの言語が存在しますが、セルレベルで言語を指定することも可能です。これは、セルの先頭に`%[言語名]`十することで実現できます。例えば、`%python`です。すぐにこの機能を利用することになります。
# MAGIC   - コマンドを実行するためには、ノートブックは**cluster(クラスター)**に接続される必要がありますが、永久につながっている必要はありません。これによって、ウェブ経由で共有したり、ローカルマシンにダウンロードすることができます。
# MAGIC   - ノートブックのデモ動画を[こちら](http://www.youtube.com/embed/MXI0F8zfKGI)から参照できます。    
# MAGIC   - **Dashboards(ダッシュボード)**
# MAGIC     - **Dashboards(ダッシュボード)**は**notebooks(ノートブック)**から作成することができ、ダッシュボードを生成したコードを非表示にして結果のみを表示する手段として利用することができます。
# MAGIC   - **Notebooks(ノートブック)**は、ワンクリックで**jobs(ジョブ)**としてスケジュールすることができ、データパイプラインの実行、機械学習の更新、ダッシュボードの更新などを行うことができます。
# MAGIC - **Libraries(ライブラリ)**
# MAGIC   - ライブラリは、あなたが問題を解決するために必要となる追加機能を提供するパッケージあるいはモジュールです。これらは、ScalaやJava jarによるカスタムライブラリ、Pythonのeggsやカスタムのパッケージとなります。あなた自身の手でライブラリを記述し手動でアップロードできますし、pypiやmavenなどのパッケージ管理ユーティリティ経由で直接インストールすることもできます。
# MAGIC - **Tables(テーブル)**
# MAGIC   - テーブルはあなたとあなたのチームが分析に使うことになる構造化データです。テーブルはいくつかの場所に存在します。テーブルはAmazon S3やAzure Blob Storageに格納できますし、現在使用しているクラスターにも格納できます。あるいは、メモリーにキャッシュすることも可能です。詳細は[Databricksにおけるデータベースおよびテーブル \- Qiita](https://qiita.com/taka_yayoi/items/e7f6982dfbee7fc84894)を参照ください。
# MAGIC - **Clusters(クラスター)**
# MAGIC   - クラスターは、あなたが単一のコンピューターとして取り扱うことのできるコンピューター群です。Databricksにおいては、効果的に20台のコンピューターを1台としてと扱えることを意味します。クラスターを用いることで、あなたのデータに対して**notebooks(ノートブック)**や**libraries(ライブラリ)**のコードを実行することができます。これらのデータはS3に格納されている構造化データかもしれませんし、作業しているクラスターに対して**table(テーブル)**としてアップロードされた非構造化データかもしれません。
# MAGIC   - クラスターにはアクセス制御機能があることに注意してください。
# MAGIC   - クラスターのデモ動画は[こちら](http://www.youtube.com/embed/2-imke2vDs8)となります。
# MAGIC - **Jobs(ジョブ)**
# MAGIC   - ジョブによって、既存の**cluster(クラスター)**あるいはジョブ専用のクラスター上で実行をスケジュールすることができます。実行対象は**notebooks(ノートブック)**、jars、pythonスクリプトとなります。ジョブは手動で作成できますが、REST API経由でも作成できます。
# MAGIC   - ジョブのデモ動画は[こちら](<http://www.youtube.com/embed/srI9yNOAbU0)となります。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 基本的な使い方
# MAGIC 
# MAGIC 画面左の**サイドバー**からDatabricksの主要な機能にアクセスします。
# MAGIC 
# MAGIC サイドバーのコンテンツは選択するペルソナ(**Data Science & Engineering**、**Machine Learning**、**SQL**)によって決まります。
# MAGIC 
# MAGIC - **Data Science & Engineering:** PythonやR、SQLを用いてノートブックを作成し実行するペルソナ
# MAGIC - **Machine Learning:** モデル管理や特徴量ストアを活用してノートブックを作成するペルソナ
# MAGIC - **SQL:** BIを行うペルソナ

# COMMAND ----------

# MAGIC %md
# MAGIC ### サイドバーの使い方
# MAGIC 
# MAGIC - デフォルトではサイドバーは畳み込まれた状態で表示され、アイコンのみが表示されます。サイドバー上にカーソルを移動すると全体を表示することができます。
# MAGIC - ペルソナを変更するには、Databricksロゴの直下にあるアイコンからペルソナを選択します。
# MAGIC 
# MAGIC ![](https://docs.databricks.com/_images/change-persona.gif)
# MAGIC 
# MAGIC - 次回ログイン時に表示されるペルソナを固定するには、ペルソナの隣にある![](https://docs.databricks.com/_images/persona-pin.png)をクリックします。再度クリックするとピンを削除することができます。
# MAGIC - サイドバーの一番下にある**Menu options**で、サイドバーの表示モードを切り替えることができます。Auto(デフォルト)、Expand(展開)、Collapse(畳み込み)から選択できます。
# MAGIC - 機械学習に関連するページを開く際には、ペルソナは自動的に**Machine Learning**に切り替わります。

# COMMAND ----------

# MAGIC %md
# MAGIC ### データブリックスのヘルプリソース
# MAGIC 
# MAGIC データブリックスには、Apache Sparkとデータブリックスを効果的に使うために学習する際に、助けとなる様々なツールが含まれています。データブリックスには膨大なApache Sparkのドキュメントが含まれており。Webのどこからでも利用可能です。リソースには大きく二つの種類があります。Apace Sparkとデータブリックスの使い方を学ぶためのものと、基本を理解した方が参照するためのリソースです。
# MAGIC 
# MAGIC これらのリソースにアクセスするには、画面右上にあるクエスチョンマークをクリックします。サーチメニューでは、以下のドキュメントを検索することができます。
# MAGIC 
# MAGIC ![img](https://sajpstorage.blob.core.windows.net/handson20210629/help_menu.png)
# MAGIC 
# MAGIC - **Help Center(ヘルプセンター)**
# MAGIC   - [Help Center \- Databricks](https://help.databricks.com/s/)にアクセスして、ドキュメント、ナレッジベース、トレーニングなどのリソースにアクセスできます。
# MAGIC - **Release Notes(リリースノート)**
# MAGIC   - 定期的に実施される機能アップデートの内容を確認できます。
# MAGIC - **Documentation(ドキュメント)**
# MAGIC   - マニュアルにアクセスできます。
# MAGIC - **Knowledge Base(ナレッジベース)**
# MAGIC   - 様々なノウハウが蓄積されているナレッジベースにアクセスできます。
# MAGIC - **Databricks Status(ステータス)**
# MAGIC   - データブリックスの稼働状況を表示します。
# MAGIC - **Feedback(フィードバック)**
# MAGIC   - 製品に対するフィードバックを投稿できます。
# MAGIC     
# MAGIC <hr>
# MAGIC データブリックスを使い始める方向けに資料をまとめたDatabricksクイックスタートガイドもご活用ください。
# MAGIC - [Databricksクイックスタートガイド \- Qiita](https://qiita.com/taka_yayoi/items/125231c126a602693610)

# COMMAND ----------

# MAGIC %md
# MAGIC ### クラスターの作成
# MAGIC <head>
# MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
# MAGIC   <style>
# MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
# MAGIC   </style>
# MAGIC </head>
# MAGIC 
# MAGIC <div style='line-height:1.5rem; padding-top: 10px;'>
# MAGIC (1) 左のサイドバーの **Compute** を右クリック。新しいタブもしくはウィンドウを開く。<br>
# MAGIC (2) クラスタページにおいて **Create Cluster** をクリック。<br>
# MAGIC (3) クラスター名を **[自分の名前(takaakiなど)]** とする。<br>
# MAGIC (4) 最後に **Create Cluster** をクリックすると、クラスターが起動 !
# MAGIC </div>
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/handson20210629/create_cluster.png)
# MAGIC 
# MAGIC ### ノートブックを作成したクラスターに紐付けて、run all コマンドを実行
# MAGIC <head>
# MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
# MAGIC   <style>
# MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
# MAGIC   </style>
# MAGIC </head>
# MAGIC 
# MAGIC <div style='line-height:1.5rem; padding-top: 10px;'>
# MAGIC (1) ノートブックに戻ります。<br>
# MAGIC (2) 左上の ノートブック メニューバーから、 **<img src="http://docs.databricks.com/_static/images/notebooks/detached.png"/></a> > [自分の名前]** を選択。<br>
# MAGIC (3) クラスターが <img src="http://docs.databricks.com/_static/images/clusters/cluster-starting.png"/></a> から <img src="http://docs.databricks.com/_static/images/clusters/cluster-running.png"/></a> へ変更となったら  **<img src="http://docs.databricks.com/_static/images/notebooks/run-all.png"/></a> Run All** をクリック。<br>
# MAGIC </div>
# MAGIC 
# MAGIC **本ハンズオンでは操作の流れが分かるようにステップ毎に実行していきます。**

# COMMAND ----------

# MAGIC %md
# MAGIC ## データブリックスの良いところ
# MAGIC 
# MAGIC 以降では、主にJupyter notebookと比較したデータブリックスのメリットをご説明します。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/handson20210629/databricks_merits.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 探索的データ分析(EDA)、ETL、機械学習モデル構築、モデル配備、BIまでを一つのプラットフォームで実行できます
# MAGIC Jupyter notebookは主にEDAと機械学習モデル構築に焦点を置いていますが、データブリックスでは分析の前段のデータ加工からデータパイプラインの構築、モデルの構築からデプロイ、ひいては経営層向けに提示するBIレポート作成までもカバーしています。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/demo20210421-spark-introduction/lakehouse.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ### インフラの管理が簡単です
# MAGIC 
# MAGIC 事前準備で体験いただいたように、クリックだけの操作でSparkクラスターを起動することができます。フルバージョンのデータブリックスでは、任意のスペックのクラスターを構成でき、オートスケーリングを活用することができます。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/handson20210629/infra.png)
# MAGIC 
# MAGIC **参考情報**
# MAGIC - [Databricksにおけるクラスター作成 \- Qiita](https://qiita.com/taka_yayoi/items/d36a469a1e0c0cebaf1b)
# MAGIC - [Databricksにおけるクラスター管理 \- Qiita](https://qiita.com/taka_yayoi/items/991aae376c089df58504)
# MAGIC - [Databricksクラスター設定のベストプラクティス \- Qiita](https://qiita.com/taka_yayoi/items/ef3dc37143e7b77b50ad)
# MAGIC - [DatabricksにおけるPythonライブラリ管理 \- Qiita](https://qiita.com/taka_yayoi/items/d3a46efdc1ad01a581d0)
# MAGIC - [Databricksにおけるジョブ管理 \- Qiita](https://qiita.com/taka_yayoi/items/b3275a1983c51a8bbe1a)

# COMMAND ----------

# MAGIC %md
# MAGIC ### データベースのインストールが不要です
# MAGIC 
# MAGIC データブリックスでは最初からデータベースとしてHiveメタストアを利用できます。以下ではSQLを用いてデータベースを操作してみます。`%sql`に関しては後ほどご説明します。
# MAGIC 
# MAGIC **参考情報**
# MAGIC - [Databricksにおけるデータベースおよびテーブル \- Qiita](https://qiita.com/taka_yayoi/items/e7f6982dfbee7fc84894)

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW DATABASES;

# COMMAND ----------

# MAGIC %md
# MAGIC Databricks環境の`/databricks-datasets`配下には様々なサンプルデータが格納されています。定期的に更新されるCOVID-19データセットも含まれています。以下のセルではサンプルデータの一覧を表示しています。`%fs`に関しては後ほどご説明します。
# MAGIC 
# MAGIC **参考情報**
# MAGIC - [DatabricksにおけるCOVID\-19データセットの活用: データコミュニティで何ができるのか \- Qiita](https://qiita.com/taka_yayoi/items/3d62c4dbdc0e39e4772c)

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /databricks-datasets

# COMMAND ----------

# MAGIC %sql
# MAGIC -- テーブルが存在する場合には削除します
# MAGIC DROP TABLE IF EXISTS diamonds;
# MAGIC 
# MAGIC -- OPTIONSでcsvのファイルパス(path)、ヘッダーがあること(header)を指定し、USINGでファイルフォーマットを指定してdiamondsテーブを作成します
# MAGIC CREATE TABLE diamonds(_c0 INT,
# MAGIC   carat DOUBLE,
# MAGIC   cut STRING,
# MAGIC   color STRING,
# MAGIC   clarity STRING, 
# MAGIC   depth DOUBLE,
# MAGIC   table INT,
# MAGIC   price INT,
# MAGIC   x DOUBLE,
# MAGIC   y DOUBLE,
# MAGIC   z DOUBLE)
# MAGIC USING csv
# MAGIC OPTIONS (path "/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv", header "true");

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM diamonds

# COMMAND ----------

# MAGIC %sql
# MAGIC -- テーブルのメタデータを確認します
# MAGIC DESCRIBE EXTENDED diamonds

# COMMAND ----------

# MAGIC %md
# MAGIC データベースおよびテーブルは左のサイドバーの**Data**からも参照できます。

# COMMAND ----------

# MAGIC %md
# MAGIC ### リモートワークでのコラボレーションが簡単です
# MAGIC 
# MAGIC 遠隔地であってもリアルタイム、時間差でのコミュニケーションを容易に行うことができます。
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/handson20210629/collaboration.png)
# MAGIC 
# MAGIC <img style="margin-top:25px;" src="https://psajpstorage.blob.core.windows.net/commonfiles/Collaboration001.gif" width="1000">

# COMMAND ----------

# MAGIC %md
# MAGIC ### ノートブックのバージョンが自動で管理されます
# MAGIC 
# MAGIC データブリックスはノートブックに対する基本的なバージョン管理機能を提供します。画面右上の**Revision history**から過去のノートブックを参照できます。それぞれのバージョンに対して、コメントの追加、復旧、バージョンの削除、バージョン履歴の削除を行うことができます。Githubのリポジトリと連携することもできます。
# MAGIC 
# MAGIC **参考資料**
# MAGIC - [Databricksノートブックを使う(バージョン管理) \- Qiita](https://qiita.com/taka_yayoi/items/dfb53f63aed2fbd344fc#%E3%83%90%E3%83%BC%E3%82%B8%E3%83%A7%E3%83%B3%E7%AE%A1%E7%90%86)
# MAGIC - [Databricks ReposによるGit連携 \- Qiita](https://qiita.com/taka_yayoi/items/b89f199ff0d3a4c16140)

# COMMAND ----------

# MAGIC %md
# MAGIC ### プログラミング言語を柔軟に切り替えることができます
# MAGIC 
# MAGIC Jupyter NotebookでもR用のカーネルをインストールすることで、ノートブックでRを利用することは可能です。データブリックスのノートブックではPython/R/Scala/SQLをネイティブでサポートしています。今お使いのノートブック名の右に`(Python)`と表示されているのは、このノートブックのデフォルト言語がPythonであることを意味しています。クリックすることでデフォルト言語を変更することができます。
# MAGIC 
# MAGIC ノートブックレベルでの言語指定に加えて、セル単位で言語が指定できます。上で`%sql`というマジックコマンドを指定しましたが、ノートブックはデフォルト言語がPythonのままで、当該セルの言語をSQLに切り替えました。これによって、「データベースを操作するときには直接SQLを記述したい」、「ここはPythonのAPIを使った方が便利だ」というケースに柔軟に対応できます。
# MAGIC 
# MAGIC 言語指定以外のマジックコマンドには以下のようなものがあります。
# MAGIC - `%sh`: ノートブック上でシェルコードを実行できます。
# MAGIC - `%fs`: dbutilsファイルシステムコマンドを実行できます。詳細は[Databricks CLI(英語)](https://docs.databricks.com/data/databricks-file-system.html#dbfs-dbutils)を参照ください。
# MAGIC - `%md`: マークダウン言語を用いてテキスト、画像、数式などと言った様々なドキュメンテーションを行うことができます。
# MAGIC 
# MAGIC **参考資料**
# MAGIC - [Databricksノートブックを使う(混成言語) \- Qiita](https://qiita.com/taka_yayoi/items/dfb53f63aed2fbd344fc#%E6%B7%B7%E6%88%90%E8%A8%80%E8%AA%9E)
# MAGIC - [Databricksにおけるファイルシステム \- Qiita](https://qiita.com/taka_yayoi/items/e16c7272a7feb5ec9a92)

# COMMAND ----------

# ノートブックのデフォルト言語はPythonです

# 上で使用したダイアモンドデータセットを読み込みます
dataPath = "/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv"

# inferSchemaはデータを読み込んで自動的にカラムの型を識別するオプションです。データを読み込む分のコストがかかります。
diamonds = spark.read.format("csv")\
  .option("header","true")\
  .option("inferSchema", "true")\
  .load(dataPath)
  
# display関数でデータフレームを表示します
display(diamonds)

# COMMAND ----------

# MAGIC %md
# MAGIC `%r`マジックコマンドを指定することで、Rも使用できます。

# COMMAND ----------

# MAGIC %r
# MAGIC library(SparkR)
# MAGIC 
# MAGIC # データソースからSparkデータフレームを作成
# MAGIC diamondsDF <- read.df("/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv", source = "csv", header="true", inferSchema = "true")
# MAGIC head(diamondsDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ### データを簡単に可視化できます
# MAGIC 
# MAGIC Jupyter notebookでは可視化を行う際にはmatplotlibなどを用いてグラフを生成する必要がありますが、データブリックスには可視化機能がビルトインされています。SQLでレコードを取得したり、`display`関数でデータフレームを表示した際に現れるグラフボタンを押すことで簡単に可視化を行えます。
# MAGIC 
# MAGIC ![](https://docs.databricks.com/_images/display-charts.png)
# MAGIC 
# MAGIC **参考資料**
# MAGIC - [Databricksにおけるデータの可視化 \- Qiita](https://qiita.com/taka_yayoi/items/36a307e79e9433121c38)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM diamonds

# COMMAND ----------

# MAGIC %md
# MAGIC 上のセルを実行後、グラフボタンを押した後に表示される**Plot Options**をクリックすると以下の画面が表示されます。
# MAGIC ![](https://sajpstorage.blob.core.windows.net/handson20210629/plot_option.png)
# MAGIC 
# MAGIC 以下のように設定を行い棒グラフを表示してみます。設定後、**Apply**をクリックし、確認メッセージ対しては**Confirm**をクリックします。
# MAGIC - **Keys**: All fieldsから**cut**をドラッグ&ドロップ
# MAGIC - **Values**: All fieldsから**price**をドラッグ&ドロップ
# MAGIC - **Aggregation**: SUMを選択
# MAGIC - **Display type**: Bar chartを選択
# MAGIC 
# MAGIC ![](https://sajpstorage.blob.core.windows.net/handson20210629/plot_configuration.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 大量データを高速に処理できます
# MAGIC 
# MAGIC Jupyter Notebookでは処理しきれない大量データも、データブリックスなら簡単に処理・分析できます。
# MAGIC 
# MAGIC **注意**<br>
# MAGIC 本章は、ハンズオンで使用している環境(Community Edition)の都合上、ハンズオンの対象外となります。講師が投影するノートブックを参照ください。説明に用いたノートブックはこちらからダウンロードできます。
# MAGIC - [pandasとSparkの比較ノートブック](https://sajpstorage.blob.core.windows.net/handson20210629/pandasとSparkの比較.html)

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>データレイクに<span style="color='#38a'">信頼性</span>と<span style="color='#38a'">パフォーマンス</span>をもたらすDelta Lake</h3>
# MAGIC <p>さらに、Delta Lakeを活用することで高速なETL処理を実現することができます。</p>
# MAGIC <div style="float:left; padding-right:60px; margin-top:20px; margin-bottom:200px;">
# MAGIC   <img src="https://jixjiadatabricks.blob.core.windows.net/images/delta-lake-square-black.jpg" width="220">
# MAGIC </div>
# MAGIC 
# MAGIC <div style="float:left; margin-top:0px; padding:0;">
# MAGIC   <h3>信頼性</h3>
# MAGIC   <ul style="padding-left: 30px;">
# MAGIC     <li>次世代データフォーマット技術</li>
# MAGIC     <li>トランザクションログによるACIDコンプライアンス</li>
# MAGIC     <li>DMLサポート（INSERTだけではなくUPDATE/DELETE/MERGEをサポート）</li>
# MAGIC     <li>データ品質管理　(スキーマ・エンフォース)</li>
# MAGIC     <li>バッチ処理とストリーム処理の統合</li>
# MAGIC     <li>タイムトラベル (データのバージョン管理)</li>
# MAGIC    </ul>
# MAGIC 
# MAGIC   <h3>パフォーマンス</h3>
# MAGIC   <ul style="padding-left: 30px;">
# MAGIC      <li>スケーラブルなメタデータ</li>
# MAGIC     <li>コンパクション (Bin-Packing)</li>
# MAGIC     <li>データ・パーティショニング</li>
# MAGIC     <li>データ・スキッピング</li>
# MAGIC     <li>ZOrderクラスタリング</li>
# MAGIC     <li>ストリーム処理による低いレイテンシー</li>
# MAGIC   </ul>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC Delta LakeはApache Spark™と連携することで、凄まじいスケーラビリティと性能を提供します。インデックス、パーティショニングなど最適化された機能によって、Delta Lakeの利用者はETLワークロードを48%高速にしています。今日は詳細をご説明できませんが、Delta Lakeに特化したワークショップも開催予定です！
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2021/02/delta-lake-product-2-1.png" width=600>

# COMMAND ----------

# MAGIC %md
# MAGIC ## さいごに
# MAGIC 
# MAGIC 本日はデータブリックスの良さと基本的な機能をご紹介させていただきました。
# MAGIC 
# MAGIC 
# MAGIC データブリックスの利用を開始される方向けに翻訳資料をまとめたクイックスタートガイドを提供しておりますので、データブリックスに興味を持たれた方はご一読いただければと思います。
# MAGIC 
# MAGIC [Databricksクイックスタートガイド \- Qiita](https://qiita.com/taka_yayoi/items/125231c126a602693610)
# MAGIC 
# MAGIC また、以下で提供されているサンプルノートブックなどをお試しいただければと思います。
# MAGIC 
# MAGIC - [DatabricksにおけるCOVID\-19データセットの活用: データコミュニティで何ができるのか \- Qiita](https://qiita.com/taka_yayoi/items/3d62c4dbdc0e39e4772c)
# MAGIC - [RayとMLflow: 分散機械学習アプリケーションの本格運用 \- Qiita](https://qiita.com/taka_yayoi/items/078a5a0a74b18acdb03b)
# MAGIC - [Databricksにおける機械学習による病理画像分析の自動化 \- Qiita](https://qiita.com/taka_yayoi/items/3929677d4e0c9dffaef4)
# MAGIC - [DatabricksとAzure Synapse Analyticsの連携 \(実践編\) \- Qiita](https://qiita.com/taka_yayoi/items/0693337a8fe8f1cbc65a)
# MAGIC - [DatadogによるDatabricksクラスター監視 \- Qiita](https://qiita.com/taka_yayoi/items/722c7ee42d245406c9ca)
# MAGIC - [Koalasのご紹介 \- Qiita](https://qiita.com/taka_yayoi/items/5bbb3280940e73395bf5)
# MAGIC - [Databricks Apache Spark機械学習チュートリアル \- Qiita](https://qiita.com/taka_yayoi/items/f169dcf4d517ac8dd644)
# MAGIC - [機械学習の本格運用：デプロイメントからドリフト検知まで \- Qiita](https://qiita.com/taka_yayoi/items/879506231b9ec19dc6a5)
# MAGIC - [Facebook ProphetとApache Sparkを用いた大規模・高精度時系列データ予測 \- Qiita](https://qiita.com/taka_yayoi/items/82fa2f0c36a355ffa901)
# MAGIC 
# MAGIC データブリックスではユーザーコミュニティによる定例会を開催しています。
# MAGIC - [JEDAI \- connpass](https://jedai.connpass.com/?gmem=1)
# MAGIC 
# MAGIC コミュニティエディションでは使用できない全ての機能を利用するために、フルバージョンのデータブリックスの無料トライアルをご利用いただくことも可能です。
# MAGIC 
# MAGIC - [Databricks 無料トライアル](https://databricks.com/jp/try-databricks)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
