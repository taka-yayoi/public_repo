# Databricks notebook source
# MAGIC %md 
# MAGIC このノートブックの目的は、OSRMとインテグレーションされたDatabricksクラスターのソリューションアクセラレータを紹介することです。

# COMMAND ----------

# MAGIC %md ## イントロダクション
# MAGIC 
# MAGIC 多くの物流シナリオにおける共通のニーズは、2点間以上での移動距離と移動時間を推定することです。Euclidean、Haversine、Manhattanなど類似の距離計算手法はあるシナリオにおいては適切な値を算出しますが、その他のケースではポイント間を移動するために通過すべき経路や道路を考慮しなくてはなりません。
# MAGIC 
# MAGIC [Project OSRM](http://project-osrm.org/)のゴールは、[OpenStreetMap Foundation](https://wiki.osmfoundation.org/wiki/Main_Page)によって提供される地図詳細を用いてルートの計算を行うためのソフトウェアを利用できるようにするというものです。OSRMバックエンドサーバーは、世界中のどこにおいても自動車や徒歩のルーティングに活用できる容易にデプロイできるソリューションを提供します。
# MAGIC 
# MAGIC OSRMバックエンドサーバーは、シンプルかつ高速なREST APIを表現するウェブサービスをデプロイします。多くの企業において、このサーバーはさまざまな内部アプリケーションにアクセスできるコンテナ化されたサービスとしてデプロイされます。大規模な履歴データ、シミュレートされたデータに対してルートを生成する分析チームにとっては、多くの場合専用のデプロイメントが必要となります。このようなニーズを持つ分析を支援するために、OSRMバックエンドサーバーがDatabricksクラスターにどのようにデプロイできるのか、さまざまなデータ処理の取り組みの一部としてアクセスできるのかを説明します。

# COMMAND ----------

# MAGIC %md ## デプロイメントの要件
# MAGIC 
# MAGIC どのようにOSRMソフトウェアがデプロイされるのかをより理解するには、Databricksクラスターがどのように動作するのかに関して知識を持つことが重要です。
# MAGIC 
# MAGIC Databricksクラスターは、共有データ処理のワークロードを実行するために動作するサーバーコンピューターから構成されます。Sparkデータフレームにロードされたデータはクラスターのワーカーノードとして知られる複数台のコンピュータのリソースに分散されます。別のコンピューターであるドライバーノードは、ワーカーノードに割り当てられたデータ処理をコーディネートします。すべてのノードはさまざまなデータセットや他のアセットを読み書きする共有ストレージロケーションにアクセスできます。これは非常に単純化したDatabricksクラスターの説明ですが、我々のアプローチの説明には十分です。</p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/osrm_cluster.png' width=500>
# MAGIC 
# MAGIC 大規模なルーティング情報を生成するために、クラスターのワーカーノードのそれぞれにOSRMバックエンドサーバーをデプロイします。これは、ノードが配備されるとそれぞれのノードで実行されるクラスターinitスクリプトを通じて処理されます。このスクリプトを通じて、OSRMバックエンドサーバーのローカルインスタンスがそれぞれのワーカーにデプロイされます。これらのOSRMソフトウェアのインスタンスによって、Sparkデータフレームでデータを処理する際にローカルにルートを生成できるようになります:
# MAGIC </p>
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/osrm_scaled_deployment2.png' width=500>
# MAGIC 
# MAGIC OSRMバックエンドサーバーのそれぞれのインスタンスは、地図データへのアクセスを必要とします。それぞれのワーカーノードが共有され一貫性のあるロケーションから容易にアクセスできるように、このデータは共有ストレージに保存されます。このデータはOSRMソフトウェアが使用する前にダウンロードされ前処理される必要があります。このデータを使用できるように準備する(そして、OSRMバックエンドサーバーソフトウェア自体をコンパイルする)ために、ワーカーノードを持たない軽量クラスター、すなわちシングルノードクラスターを使用します。Databricksワークスペースの任意のクラスターからアクセスできる共有ストレージロケーションに処理済み地図データ(とコンパイル済みのソフトウェア)を格納します。
# MAGIC </p>
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/osrm_preprocessing_deployment2.png' width=500>
# MAGIC 
# MAGIC クラスターデプロイメントのトポロジーを計画する際、OSRMソフトウェアがメモリーに幾分大きな地図ファイルをロードすることに留意することが重要です。これは、ルート解決を高速にしますが、OSRMソフトウェアインスタンスをホストするそれぞれのコンピューターにある一定量のRAMを持つ必要があります。十分なメモリーがない場合、OSRMソフトウェアは多くの場合、明確なメッセージを出さずにシャットダウンしてしまいます。前処理のステップが成功した際に得られる地図ファイルによって必要なRAMの総量に注意してください。そして、必要に応じてルーティングのクラスターデプロイメントのワーカーノードのサイズを調整してください。

# COMMAND ----------

# MAGIC %md ## ソリューションアクセラレータのノートブック
# MAGIC 
# MAGIC このソリューションアクセラレータは3つのノートブックから構成されます。それぞれがOSRMがインテグレーションされたデプロイメントの固有の用途に取り組んでおり、順番に実行する必要があります。
# MAGIC </p>
# MAGIC 
# MAGIC * RT 00: Introduction - ソリューションアクセラレータのシナリオを紹介します
# MAGIC * RT 01: Setup OSRM Server - OSRMソフトウェアをコンパイルし、OpenStreetMapファイルを前処理します
# MAGIC * RT 02: Generate Routes - データ処理の一環としてルートを生成するためにOSRMソフトウェアを使用します

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | OSRM Backend Server                                  | High performance routing engine written in C++14 designed to run on OpenStreetMap data | BSD 2-Clause "Simplified" License    | https://github.com/Project-OSRM/osrm-backend                   |
# MAGIC | Mosaic | An extension to the Apache Spark framework that allows easy and fast processing of very large geospatial datasets | Databricks License| https://github.com/databrickslabs/mosaic | 
# MAGIC | Tabulate | pretty-print tabular data in Python | MIT License | https://pypi.org/project/tabulate/ |
