# Databricks notebook source
# MAGIC %md 
# MAGIC このノートブックの目的は、OSRMソフトウェアをコンパイルし、Databricksクラスターで地図のルートを生成するのに必要なOpenStreetMapファイルを前処理することです。

# COMMAND ----------

# MAGIC %md ## イントロダクション
# MAGIC 
# MAGIC このノートブックで行うステップは、DatabricksクラスターでOSRMバックエンドサーバーを実行するために必要なアセットをどのように準備するのかを説明します。これらのステップは頻繁には実行せず、OSRMを装備したDatabricksクラスターを起動する前に一度実行する場合が多いです。
# MAGIC 
# MAGIC この作業を行うためには、シングルノードクラスター、すなわち、ドライバーのみでワーカーノードを持たないクラスターを使用することをお勧めします。(シングルノードクラスターをデプロイするには*Create Cluster*ページの*Cluster Mode*ドロップダウンから*single node*を選択します) ドライバーノードには、処理する地図ファイルのサイズよりも遥かに大きいRAMを割り当てる必要があります。ドライバーノードのサイジングに関するガイドラインとしては、[GeoFabrik website](https://download.geofabrik.de/)からダウンロードした11.5GBの北米の *.osm.pbf* 地図ファイルの処理に128GBのRAMのドライバーノードが必要となりました。
# MAGIC </p>
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/osrm_preprocessing_deployment2.png' width=500>

# COMMAND ----------

# MAGIC %md ## Step 1: サーバーソフトウェアの構築
# MAGIC 
# MAGIC スタートするには、[its GitHub repository](https://github.com/Project-OSRM/osrm-backend)から利用できる現在のソースコードからOSRMバックエンドサーバーを構築する必要があります。プロジェクトチームによって提供されている[*Build from Source* instructions](https://github.com/Project-OSRM/osrm-backend#building-from-source)は、このための基本的なステップをカバーしています。以下のように我々の環境にパッケージの依存関係をインストールするところからスタートします。

# COMMAND ----------

# DBTITLE 1,依存関係のインストール
# MAGIC %sh -e
# MAGIC 
# MAGIC sudo apt -qq install -y build-essential git cmake pkg-config \
# MAGIC libbz2-dev libxml2-dev libzip-dev libboost-all-dev \
# MAGIC lua5.2 liblua5.2-dev libtbb-dev

# COMMAND ----------

# MAGIC %md
# MAGIC 次に、我々のローカルシステムにOSRMバックエンドサーバーリポジトリをクローンします。

# COMMAND ----------

# DBTITLE 1,OSRMバックエンドサーバーリポジトリのクローン
# MAGIC %sh -e 
# MAGIC 
# MAGIC # make directory for repo clone
# MAGIC mkdir -p /srv/git
# MAGIC cd /srv/git
# MAGIC 
# MAGIC # clone the osrm backend server repo
# MAGIC rm -rf osrm-backend
# MAGIC git clone --depth 1 -b v5.26.0 https://github.com/Project-OSRM/osrm-backend

# COMMAND ----------

# MAGIC %md そしてサーバーを構築します:
# MAGIC 
# MAGIC **注意** このステップは、お使いのDatabrikcsクラスターのサイズに応じて完了するまでに20分以上かかることがあります。

# COMMAND ----------

# DBTITLE 1,OSRMバックエンドサーバーの構築
# MAGIC %sh -e
# MAGIC 
# MAGIC cd /srv/git/osrm-backend
# MAGIC 
# MAGIC mkdir -p build
# MAGIC cd build
# MAGIC 
# MAGIC cmake ..
# MAGIC cmake --build .
# MAGIC sudo cmake --build . --target install

# COMMAND ----------

# MAGIC %md ## Step 2: 地図ファイルの準備
# MAGIC 
# MAGIC OSRMバックエンドサーバーは地図データを使ってルートを生成します。特定リージョンの地図は *.osm.pbf* ファイルとして[GeoFabrik download site](https://download.geofabrik.de/)から利用することができます。要件に応じて、特定の大陸、国、リージョンレベルの地図ファイルを活用しても構いません。
# MAGIC 
# MAGIC 使用するファイルはOSRMソフトウェアが利用する前にダウンロードして前処理する必要があります。前処理の過程では、2つの前処理のパスから選択し、ルーティングが車によるものなのか、徒歩や他の移動手段によるものなのかを選択する必要があります。これらのオプションの詳細に関しては、[こちら](https://github.com/Project-OSRM/osrm-backend/wiki/Running-OSRM)を参照ください。ここでは、前処理のパスとしては、車移動を選択する際にはOSRMドキュメントで好まれているMulti-Level Dijkstra (MLD)を選択しました。
# MAGIC 
# MAGIC **注意** 地図ファイルのサイズによっては、ダウンロードと前処理ステップが完了するまである程度の時間を要する場合があります。

# COMMAND ----------

# MAGIC %md 最初のステップは使用する地図ファイルのダウンロードです:
# MAGIC 
# MAGIC **注意** このステップは北米の地図ファイルの場合は15分程度必要とします。

# COMMAND ----------

# DBTITLE 1,地図ファイルのダウンロード
# MAGIC %sh -e 
# MAGIC 
# MAGIC # create clean folder to house downloaded map file
# MAGIC rm -rf /srv/git/osrm-backend/maps/north-america
# MAGIC mkdir -p /srv/git/osrm-backend/maps/north-america
# MAGIC 
# MAGIC # download map file to appropriate folder
# MAGIC cd /srv/git/osrm-backend/maps/north-america
# MAGIC wget --quiet https://download.geofabrik.de/north-america-latest.osm.pbf
# MAGIC 
# MAGIC # list folder contents
# MAGIC ls -l .

# COMMAND ----------

# MAGIC %md 
# MAGIC 次に、我々が選択した移動手段を示すプロファイルを指定する地図ファイルアセットを抽出します。*osrm-extract* コマンドの出力は非常に多いことに注意してください。セルの出力を溢れさせないために、ログファイルに標準出力をリダイレクトし、完了したことを検証するためにそのファイルの最後の数行を確認します:
# MAGIC 
# MAGIC **注意** このステップは北米地図ファイルの場合1時間程度必要とします。

# COMMAND ----------

# DBTITLE 1,地図ファイルのコンテンツの抽出
# MAGIC %sh -e
# MAGIC 
# MAGIC # setup location to house log files
# MAGIC mkdir -p /srv/git/osrm-backend/logs
# MAGIC 
# MAGIC # move to folder housing map file
# MAGIC cd /srv/git/osrm-backend/maps/north-america
# MAGIC 
# MAGIC # extract map file contents
# MAGIC /srv/git/osrm-backend/build/osrm-extract north-america-latest.osm.pbf -p /srv/git/osrm-backend/profiles/car.lua > /srv/git/osrm-backend/logs/extract_log.txt
# MAGIC 
# MAGIC # review output from extract command
# MAGIC #echo '----------------------------------------'
# MAGIC #tail /srv/git/osrm-backend/logs/extract_log.txt

# COMMAND ----------

# MAGIC %sh -e
# MAGIC tail /srv/git/osrm-backend/logs/extract_log.txt

# COMMAND ----------

# MAGIC %md 
# MAGIC 続ける前に、*osrm-extract*コマンドの出力の最後の数行をチェックするようにしてください。***\[info\] RAM: peak bytes used***で終わるメッセージを受け取らなかった場合、クラスターのドライバーノードに十分なメモリーがなかったことから抽出プロセスがクラッシュしている可能性が高いです。クラッシュした場合には、コマンド出力でエラーが確認できない場合があります。
# MAGIC 
# MAGIC RAM使用に関するメッセージを受け取ったとしても、 *.osm.pbf* ファイルが保存されているフォルダーに大量のファイルが存在していることを確認することで、抽出が成功したことを検証することは良いアイデアです。

# COMMAND ----------

# DBTITLE 1,地図ファイル抽出の検証
# MAGIC %sh -e ls -l /srv/git/osrm-backend/maps/north-america

# COMMAND ----------

# MAGIC %md 
# MAGIC MLD前処理パスの次のステップは抽出ファイルからコンテンツをパーティショニングすることとなります。
# MAGIC 
# MAGIC **注意** このステップは北米地図ファイルの場合、1時間程度要します。

# COMMAND ----------

# DBTITLE 1,抽出地図ファイルのパーティション
# MAGIC %sh -e 
# MAGIC 
# MAGIC cd /srv/git/osrm-backend/maps/north-america
# MAGIC 
# MAGIC /srv/git/osrm-backend/build/osrm-partition north-america-latest.osrm

# COMMAND ----------

# MAGIC %md 
# MAGIC そして、最後にこの前処理パスに関連づけられた指示に従ってコンテンツをカスタマイズします:
# MAGIC 
# MAGIC **注意** 5分程度かかります。

# COMMAND ----------

# DBTITLE 1,抽出地図ファイルのカスタマイズ
# MAGIC %sh -e 
# MAGIC 
# MAGIC cd /srv/git/osrm-backend/maps/north-america
# MAGIC 
# MAGIC /srv/git/osrm-backend/build/osrm-customize north-america-latest.osrm

# COMMAND ----------

# MAGIC %md ## Step 3: OSRMアセットの永続化
# MAGIC 
# MAGIC OSRMバックエンドサーバーと関連する地図アセットは、クラスターのドライバーノードの */srv/git* フォルダーに作成されました。このフォルダーは一時的なものであり、ドライバーノードからしかアクセスすることができません。これは、クラスターが停止されると、これらすべてのアセットは失われることを意味します。クラスターが再起動しても再利用できる様に、これらのアセットを永続化するには、これらを永続化ロケーションにコピーする必要があります。Databricksでは、[クラウドストレージのマウント](https://docs.databricks.com/data/databricks-file-system.html#mount-object-storage-to-dbfs)を利用することができます。また、シンプルにビルトインの[FileStore](https://docs.databricks.com/data/filestore.html)ストレージロケーションを使用することもできます。
# MAGIC 
# MAGIC **注意** 15分程度かかります。

# COMMAND ----------

# DBTITLE 1,OSRMアセットを永続化ロケーションにコピー
# MAGIC %sh -e
# MAGIC 
# MAGIC rm -rf /dbfs/FileStore/osrm-backend
# MAGIC cp -L -R /srv/git/osrm-backend /dbfs/FileStore/osrm-backend

# COMMAND ----------

# MAGIC %md ## Step 4: initスクリプトの作成
# MAGIC 
# MAGIC この時点で、OSRMバックエンドサーバーを実行するために必要なすべての要素が揃いました。ここでは、Databricksクラスターのそれぞれのワーカーノードにサーバーがデプロイされる様に、[クラスターinitスクリプト](https://docs.databricks.com/clusters/init-scripts.html#cluster-scoped-init-scripts)を定義する必要があります。
# MAGIC 
# MAGIC このスクリプトのロジックは非常にわかりやすいものとなっています。パッケージの依存関係をインストール(すでに大部分は行いました)し、ルーティングサーバーを起動します。
# MAGIC 
# MAGIC ルーティングサーバーがルーティングのリクエストに反応するまでには少々の時間を要します。我々はサーバーをテストし、完全に起動するまで待つロジックを追加しました。このスクリプトを用いるクラスターにおいては、ジョブを完全に実行できるようになるまで数分の遅れが生じることになりますが、ワークフロー(ジョブ)の一部として実行されるロジックの様に、クラスターの起動後即座にロジックが実行され、成功することを保証することができる様になります。
# MAGIC 
# MAGIC initスクリプトはDBFSファイルシステムの中のアクセス可能なロケーションに書き込む様にしてください。この様なロケーションは、クラスターのスタートアップの過程でスクリプトにアクセスするために必要となります。
# MAGIC 
# MAGIC **注意** initスクリプトを設定したクラスターの起動には10分程度要します。

# COMMAND ----------

# DBTITLE 1,initスクリプトの作成
# make folder to house init script
dbutils.fs.mkdirs('dbfs:/databricks/scripts')

# write init script
dbutils.fs.put(
  '/databricks/scripts/osrm-backend.sh',
  '''
#!/bin/bash

if [[ $DB_IS_DRIVER != "TRUE" ]]; then  

  echo "installing osrm backend server dependencies"
  sudo apt -qq install -y build-essential git cmake pkg-config libbz2-dev libxml2-dev libzip-dev libboost-all-dev lua5.2 liblua5.2-dev libtbb-dev
  
  echo "launching osrm backend server"
  /dbfs/FileStore/osrm-backend/build/osrm-routed --algorithm=MLD /dbfs/FileStore/osrm-backend/maps/north-america/north-america-latest.osrm &
  
  echo "wait until osrm backend server becomes responsive"
  res=-1
  i=1

  # while no response
  while [ $res -ne 0 ]
  do

    # test connectivity
    curl --silent "http://127.0.0.1:5000/route/v1/driving/-74.005310,40.708750;-73.978691,40.744850"
    res=$?
    
    # increment the loop counter
    if [ $i -gt 40 ] 
    then 
      break
    fi
    i=$(( $i + 1 ))

    # if no response, sleep
    if [ $res -ne 0 ]
    then
      sleep 30
    fi

  done  
  
fi
''', 
  True
  )

# show script content
print(
  dbutils.fs.head('dbfs:/databricks/scripts/osrm-backend.sh')
  )

# COMMAND ----------

# MAGIC %md 
# MAGIC initスクリプトを定義すると、OSRMバックエンドサーバーが実行されるクラスターを設定することができます。このクラスターのワーカーノードのサイジングに関しては、ノートブック *RT 00* のガイダンスを参照してください。
# MAGIC 
# MAGIC このアクセラレータのジョブとクラスターを作成するために、このフォルダーにあるRUNMEファイルを使用している場合、initスクリプトのセットアップステップが自動化されることに注意してください。手動でクラスターの設定を定義する場合は、上のセルで書き込まれるinitスクリプトのパスを指定してください。クラスターが起動すると、クラスターのそれぞれのノードは設定を反映するためにこのスクリプトを実行します。
# MAGIC 
# MAGIC **注意** このノートブックを実行する際に使用するクラスターにおいても十分なメモリーを搭載したインスタンスタイプ(128GBのRAM以上)を使用する様にしてください。そうしないと、initスクリプトでOSRMサーバーが起動しません。OSRMサーバーに接続できない場合には、[クラスターログ](https://qiita.com/taka_yayoi/items/8d951b660cd87c6c5f18#%E3%82%AF%E3%83%A9%E3%82%B9%E3%82%BF%E3%83%BC%E3%83%AD%E3%82%B0%E3%83%87%E3%83%AA%E3%83%90%E3%83%AA%E3%83%BC)や[Webターミナル](https://qiita.com/taka_yayoi/items/b3be567839a6fcb84136)でデバッグすることをお勧めします。
# MAGIC </p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/osrm_cluster_config.PNG' width=600>

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
