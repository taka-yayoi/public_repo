# Databricks notebook source
# MAGIC %md
# MAGIC # DatabricksとApache Sparkのご紹介

# COMMAND ----------

# MAGIC %md
# MAGIC ## Databricksにようこそ！
# MAGIC 
# MAGIC このノートブックは、DatabricksでApache Sparkを使いこなすための方法を学ぶ最初の一歩として作成されました。ノートブックを通じて、コアのコンセプト、基本的な概要、必要となるツールをご紹介します。このノートブックは、Apache Sparkの開発者から提供されるコアコンセプトとベスプラクティスを提供するものです。
# MAGIC 
# MAGIC > ノートブックを実際に試したいのであれば、無料の[Databricksコミュニティエディション](https://databricks.com/jp/try-databricks)にサインアップしてください。ログイン後、このノートブックをインポートし、指示に従って進めてください。
# MAGIC 
# MAGIC まず初めに、Databricksを説明させてください。DatabricksはApache Sparkを実行するためのマネージドプラットフォームです。つまり、Sparkを利用するために、複雑なクラスター管理の考え方や、面倒なプラットフォームの管理タスクを学ぶ必要がないということです。また、Databricksは、Sparkを利用したワークロードを円滑にするための機能も提供しています。GUIでの操作をこのむデータサイエンティストやデータアナリストの方向けに、マウスのクリックで操作できるプラットフォームとなっています。しかし、UIに加えて、データ処理のワークロードをジョブで自動化したい方向けには、洗練されたAPIも提供しています。エンタープライズでの利用に耐えるために、Databricksにはロールベースのアクセス制御や、使いやすさを改善するためだけではなく、管理者向けにコストや負荷軽減のための最適化が図られています。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 入門シリーズ
# MAGIC 
# MAGIC このノートブックは、Apache Sparkの基礎を学べるように作成された一連の入門編の一部です。このノートブックは、Sparkを触ったことがない方、少しだけ知っているという方向けに作成されています。このシリーズは、Sparkを使ったことがあるが、UDFの作成、機械学習パイプラインなどの機能を知らない方向けのものでもあります。本シリーズのノートブックは以下の通りとなります。
# MAGIC <br><br>
# MAGIC - [Apache Spark on Databricks for Data Scientists](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/346304/2168141618055194/484361/latest.html)
# MAGIC - [Apache Spark on Databricks for Data Engineers](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/346304/2168141618055109/484361/latest.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Databricksにおける用語
# MAGIC 
# MAGIC Databricksには理解すべきキーコンセプトが存在します。これらの多くは画面左側のサイドバーにアイコンとして表示されています。これらは、エンドユーザーであるあなたのために用意された基本的なツールとなります。これらはウェブアプリケーションのUIからも利用できますし、REST APIからも利用できます。<br><br>
# MAGIC 
# MAGIC - **Workspaces(ワークスペース)**
# MAGIC   - Databricksで行う作業はワークスペース上で実施することになります。あなたのコンピューターのフォルダー階層のように、**notebooks(ノートブック)**や**libraries(ライブラリ)**を保存でき、それらを他のユーザーと共有することができます。ワークスペースはデータとは別となっており、データを格納するために用いるものではありません。これらは単に**notebooks(ノートブック)**や**libraries(ライブラリ)**を保存するためのものであり、データを操作するためにこれらを使用することになります。
# MAGIC - **Notebooks(ノートブック)**
# MAGIC   - ノートブックはコマンドを実行するためのセルの集合です。セルには以下の言語を記述することができます： `Scala`、`Python`、`R`、`SQL`、`Markdown`。ノートブックにはでフォルtの言語が存在しますが、セルレベルで言語を指定することも可能です。これは、セルの先頭に`%[言語名]`十することで実現できます。例えば、`%python`です。すぐにこの機能を利用することになります。
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
# MAGIC   - クラスターは、あなたが単一のコンピューターとして取り扱うことのできるコンピューター群です。Databricksにおいては、効果的に20台のコンピューターを1台としてと扱えることを意味します。クラスターを用いることで、あなたのデータに対して**notebooks(ノートブック)**や**libraries(ライブラリ)**のコードを実行することができます。これらのデータはS3に格納されている構造化データかもしれませんし、作業しているクラスターに対して**table(テーブル)**としてアップロードされた日構造化データかもしれません。
# MAGIC   - クラスターにはアクセス制御機能があることに注意してください。
# MAGIC   - クラスターのデモ動画は[こちら](http://www.youtube.com/embed/2-imke2vDs8)となります。
# MAGIC - **Jobs(ジョブ)**
# MAGIC   - ジョブによって、既存の**cluster(クラスター)**あるいはジョブ専用のクラスター上で実行をスケジュールすることができます。実行対象は**notebooks(ノートブック)**、jars、pythonスクリプトとなります。ジョブは手動で作成できますが、REST API経由でも作成できます。
# MAGIC   - ジョブのデモ動画は[こちら](<http://www.youtube.com/embed/srI9yNOAbU0)となります。
# MAGIC - **Apps(アプリケーション)**
# MAGIC   - AppsはDatabricksプラットフォームと連携するサードパーティツールです。Tableauなどが含まれます。

# COMMAND ----------

# MAGIC %md
# MAGIC ## DatabricksとApache Sparkのヘルプリソース
# MAGIC 
# MAGIC Databricksには、Apache SparkとDatabriksを効果的に使うために学習する際に、助けとなる様々なツールが含まれています。Databricksには膨大なApache Sparkのドキュメントが含まれており。Webのどこからでも利用可能です。リソースには大きく二つの種類があります。Apace SparkとDatabricksの使い方を学ぶためのものと、基本を理解した方が参照するためのリソースです。
# MAGIC 
# MAGIC これらのリソースにアクセスするには、画面右上にあるクエスチョンマークをクリックします。サーチメニューでは、以下のドキュメントを検索することができます。
# MAGIC 
# MAGIC ![img](https://sajpstorage.blob.core.windows.net/demo20210421-spark-introduction/help_menu.png)
# MAGIC 
# MAGIC - **Help Center(ヘルプセンター)**
# MAGIC   - [Help Center \- Databricks](https://help.databricks.com/s/)にアクセスして、ドキュメント、ナレッジベース、トレーニングなどのリソースにアクセスできます。
# MAGIC - **Release Notes(リリースノート)**
# MAGIC   - 定期的に実施される機能アップデートの内容を確認できます。
# MAGIC - **Documentation(ドキュメント)**
# MAGIC   - マニュアルにアクセスできます。
# MAGIC - **Knowledge Base(ナレッジベース)**
# MAGIC   - 様々なノウハウが蓄積されているナレッジベースにアクセスできます。
# MAGIC - **Feedback(フィードバック)**
# MAGIC   - 製品に対するフィードバックを投稿できます。
# MAGIC - **Shortcuts(ショートカット)**
# MAGIC   - キーボードショートカットを表示します。
# MAGIC     
# MAGIC また、Databricksではプロフェッショナルサービス、テクニカルサポートを提供しています。こちらの[窓口](https://databricks.com/jp/company/contact)からお問い合わせください。

# COMMAND ----------

# MAGIC %md
# MAGIC ## DatabricksとApache Sparkの概要
# MAGIC 
# MAGIC ここまで、用語と学習のためのリソースを説明してきました。ここからは、Apache SparkとDatabricksの基本を説明していきます。もしかしたら、Sparkのコンセプトはご存知かもしれませんが、みんなが同じ理解をしているのかどうかを確認するための時間をいただければと思います。また、せっかくですので、ここでSparkの歴史を学びましょう。

# COMMAND ----------

# MAGIC %md
# MAGIC ### Apache Sparkプロジェクトの歴史
# MAGIC 
# MAGIC SparkはDatabricksの創始者たちがUC Berkeleyにいるときに誕生しました。Sparkプロジェクトは2009年にスタートし、2010年にオープンソース化され、2013年にApacheにコードが寄贈されApache Sparkになりました。Apache Sparkのコードの75%以上がDatabricksの従業員の手によって書かれており、他の企業に比べて10倍以上の貢献をし続けています。Apache Sparkは、多数のマシンにまたがって並列でコードを実行するための、洗練された分散処理フレームワークです。概要とインタフェースはシンプルですが、クラスターの管理とプロダクションレベルの安定性を確実にすることはそれほどシンプルではありません。Databricksにおいては、Apache Sparkをホストするソリューションとして提供することで、ビッグデータをシンプルなものにします。

# COMMAND ----------

# MAGIC %md
# MAGIC ### コンテキスト、環境
# MAGIC 
# MAGIC DatabricksとSparkの効果的な使い方を理解するために必要なピース全てに慣れ親しめるように、Apache Sparkのコアコンセプトを学んでいきましょう。
# MAGIC 
# MAGIC 歴史的に、Apache Sparkにはユーザーが利用できる2つのコアコンテキストが存在していました。`sc`として利用できる`sparkContext`と`sqlContext`として利用できる`SQLContext`です。これらのコンテキストによって、ユーザーは様々な関数、情報を利用することができます。`sqlContext`では多くのデータフレームに関する機能を利用できる一方、`sparkContext`ではApache Sparkのエンジン自身に焦点を当てていました。
# MAGIC 
# MAGIC しかし、Apache Spark 2.X以降は、一つのコンテキストになります。それが`SparkSession`です。

# COMMAND ----------

# MAGIC %md
# MAGIC ### データインタフェース
# MAGIC 
# MAGIC Sparkを使う際に理解すべき、いくつかのキーインタフェースが存在します。
# MAGIC 
# MAGIC - **Dataset(データセット)**
# MAGIC   - データセットはApache Sparkにおける最新の分散コレクションであり、データフレームとRDDを組み合わせたものと考えることができます。RDDで利用できる型インタフェースを提供しつつも、データフレームの利便性を兼ね備えています。これは、今後コアの概念になるでしょう。
# MAGIC - **Dataframe(データフレーム)**
# MAGIC   - データフレームは、分散された`Row`タイプのコレクションです。これにより、柔軟なインタフェースを保ちながらも、pythonのpandasやR言語で慣れ親しんだデータフレームのコンセプトをほぼそのまま活用することができます。
# MAGIC - **RDD(Resilient Distributed Dataset)**
# MAGIC   - Apache Sparkにおける最初の抽象化はRDDでした。基本的に、これはクラスターにある複数のマシンにまたがって配置される、1種類以上の型から構成されるデータオブジェクトのリストに対するインタフェースとなります。RDDは様々な方法で作成することができ、ユーザーに対して低レベルのAPIを提供します。これは利用可能なオリジナルのデータ構造ではありますが、新たなユーザーは、RDDの機能のスーパーセットとなるデータセットにフォーカスすべきです。

# COMMAND ----------

# MAGIC %md
# MAGIC # コーディングを始めましょう!
# MAGIC 
# MAGIC やれやれ、これまでに多くのことをカバーしてきました！でも、これでようやくApache SparkとDatabricksのパワーを体感できるデモに進むことができます。しかし、その前にいくつかの作業が必要となります。最初にすべきシンプルなことは、このノートブックをあなたの環境に取り込むことです。
# MAGIC 
# MAGIC ノートブックをダウンロードした後の手順は[ノートブックのインポート](https://qiita.com/taka_yayoi/items/c306161906d6d34e8bd5#%E3%83%8E%E3%83%BC%E3%83%88%E3%83%96%E3%83%83%E3%82%AF%E3%81%AE%E3%82%A4%E3%83%B3%E3%83%9D%E3%83%BC%E3%83%88)を参照ください。

# COMMAND ----------

# MAGIC %md ## クラスターの作成
# MAGIC 
# MAGIC 左側のClustersボタンをクリックします。Clusterページで、左上にある![img](http://training.databricks.com/databricks_guide/create_cluster.png)をクリックします。
# MAGIC 
# MAGIC 新たに作成するクラスターに以下の設定を入力します。
# MAGIC 
# MAGIC ![img](https://sajpstorage.blob.core.windows.net/demo20210421-spark-introduction/create_cluster.png)
# MAGIC 
# MAGIC <br>
# MAGIC - 区別がつくようにユニークなクラスター名をつけます。
# MAGIC - Sparkのバージョンを選択します。
# MAGIC   - 必要に応じて、実験段階のバージョンのSparkを試すこともできます。
# MAGIC - 起動するワーカーノードの数を指定します。最低限1台が必要です。
# MAGIC - オンデマンドインスタンスを使うのか、スポットインスタンスを使うのか、あるいは組み合わせにするのかを指定します。
# MAGIC 
# MAGIC ここでは、Cluster Nameに名前のみ(例：My First Cluster)を指定して、他はデフォルトのままにすることをお勧めします。
# MAGIC 
# MAGIC クラスターに指定できる他のオプションに関しては、[Configure clusters \| Databricks on AWS](https://docs.databricks.com/clusters/configure.html#cluster-configurations)を参照ください。

# COMMAND ----------

# MAGIC %md 
# MAGIC 最初に上で説明した`SparkSession`に触れてみましょう。`spark`変数を介してアクセスすることができます。説明したように、SparkセッションはApache Sparkに関する情報が格納される重要な場所となります。
# MAGIC 
# MAGIC セルのコマンドは、セルが選択されている状態で`Shift+Enter`を押すことで実行できます。

# COMMAND ----------

spark

# COMMAND ----------

# MAGIC %md 
# MAGIC 情報にアクセスするためにSparkコンテキストを利用できますが、コレクションを並列化するためにも利用できます。こちらでは、`DataFrame`を返却するpythonの小規模のrangeを並列化します。

# COMMAND ----------

firstDataFrame = spark.range(1000000)
print(firstDataFrame)

# COMMAND ----------

# MAGIC %md ## トランスフォーメーションとアクション
# MAGIC 
# MAGIC 上で並列化した`DataFrame`の値をprintした際に値が表示されるはずだと思ったかもしれません。しかし、Apache Sparkはそのように動作しません。Sparkには、2種類の明確に異なるユーザーのオペレーションが存在します。それが**transformations(トランスフォーメーション)**と**actions(アクション)**です。
# MAGIC 
# MAGIC ### Transformations(トランスフォーメーション)
# MAGIC 
# MAGIC トランスフォーメーションを記述したセルを実行した時点では処理が完了しないオペレーションです。これらは**action(アクション)**が呼ばれたときにのみ実行されます。トランスフォーメーションの例は、integerをfloatへ変換、値のフィルタリングなどです。
# MAGIC 
# MAGIC ### Actions(アクション)
# MAGIC 
# MAGIC アクションは実行された瞬間にSparkによって処理が行われます。アクションは、実際の結果を取得するために、前にあるトランスフォーメーション全てを実行することから構成されます。アクションは一つ以上のジョブから構成され、ジョブはワーカーノードにおいて可能であれば並列で実行される複数のタスクから構成されます。
# MAGIC 
# MAGIC こちらがシンプルなトランスフォーメーションとアクションの例です。これらが**全てのトランスフォーメーション、アクションではない**ことに注意してください。これはほんのサンプルです。なぜ、Apache Sparkがこのように設計されているのかについてはすぐにご説明します！
# MAGIC 
# MAGIC 
# MAGIC ![transformations and actions](http://training.databricks.com/databricks_guide/gentle_introduction/trans_and_actions.png)

# COMMAND ----------

# トランスフォーメーションの例
# IDカラムを選択し、2倍します
secondDataFrame = firstDataFrame.selectExpr("(id * 2) as value")

# COMMAND ----------

# アクションの例
# firstDataFrameの最初の５行を取得します
print(firstDataFrame.take(5))
# secondDataFrameの最初の５行を取得します
print(secondDataFrame.take(5))

# COMMAND ----------

# display()コマンドでsecondDataFrameを表示します
display(secondDataFrame)

# COMMAND ----------

# MAGIC %md ## Apache Sparkのアーキテクチャ
# MAGIC 
# MAGIC ここまでで、Sparkにはアクションとトランスフォーメーションがあることがわかりました。なぜこれが必要なのかを説明します。これは、個々の処理のピースを最適化するのではなく、処理全体のパイプラインを最適化するためのシンプルな方法だからです。適切な処理を一度に実行できるため、特定のタイプの処理において、処理が劇的に高速になります。技術的に言えば、Sparkは、以下の図に示すように処理を`pipelines(パイプライン化)`します。すなわち、逐次的に処理を実行するのではなく、(フィルタリングやマッピングなどを)一括で処理するということです。
# MAGIC 
# MAGIC ![transformations and actions](http://training.databricks.com/databricks_guide/gentle_introduction/pipeline.png)
# MAGIC 
# MAGIC Apache Sparkは、他のフレームワークのようにそれぞれのタスクの都度ディスクに結果を書き込むのではなく、メモリー上に結果を保持します。
# MAGIC 
# MAGIC サンプルを進める前に、Apache Sparkのアーキテクチャを見てみましょう。上で述べたように、Apache Sparkは大量のマシンを一つのマシンとして取り扱えるようにしてくれます。これは、クラスターに`driver(ドライバー)`ノードと、付随する`worker(ワーカー)`ノードから構成されるドライバー・ワーカータイプのアーキテクチャによって実現されています。ドライバーノードがワーカーノードに作業を割り振り、ワーカーノードに対して、メモリーあるいはディスク（あるいはS3やRedshift）からデータを取得するように指示します。
# MAGIC 
# MAGIC 以下の図では、ワーカー(executor)ノードとやりとりをするドライバーノードから構成されるApache Sparkクラスターの例を示しています。それぞれのexecutorノードには、処理を実行するコアに該当するスロットが存在します。
# MAGIC 
# MAGIC ![spark-architecture](http://training.databricks.com/databricks_guide/gentle_introduction/videoss_logo.png)
# MAGIC 
# MAGIC 処理を実行する際、ドライバーノードは空いているワーカーノードのスロットにタスクを割り当てます。
# MAGIC 
# MAGIC ![spark-architecture](http://training.databricks.com/databricks_guide/gentle_introduction/spark_cluster_tasks.png)
# MAGIC 
# MAGIC > 注意： Databricksコミュニティエディションではワーカーノードがないため、図とは異なりますが、単一のノードが全てのコードを実行します。
# MAGIC 
# MAGIC ![spark-architecture](http://training.databricks.com/databricks_guide/gentle_introduction/notebook_microcluster.png)
# MAGIC 
# MAGIC Apache SparkのWeb UIで、あなたのApache Sparkアプリケーションの詳細を参照することができます。Web UIに移動するには、"Clusters"をクリックし、参照したいクラスターの"Spark UI"リンクをクリックします。あるいは、このノートブック画面の左上に表示されているクラスターをクリックして"Spark UI"リンクをクリックします。
# MAGIC 
# MAGIC ハイレベルにおいては、全てのApache Sparkアプリケーションは、クラスターあるいはローカルマシンの同一マシンで動作するワーカーのJava Virtual Macines(JVMs)における並列処理を起動するドライバープログラムから構成されています。Databricksでは、ノートブックのインタフェースがドライバープログラムとなっています。このドライバープログラムはプログラムのmainループから構成されており、クラスター上に分散データセットを作成し、それらのデータセットにオペレーション(トランスフォーメーションとアクション)を適用します。
# MAGIC 
# MAGIC ドライバープログラムは、デプロイされている場所に関係なく`SparkSession`オブジェクトを介してApache Sparkにアクセスします。

# COMMAND ----------

# MAGIC %md
# MAGIC ## トランスフォーメーションとアクションの実例
# MAGIC 
# MAGIC これらのアーキテクチャ、適切な**transformations(トランスフォーメーション)**と**actions(アクション)**を説明するために、より詳細な例を見ていきましょう。今回は、`DataFrames(データフレーム)`とcsvファイルを使用します。
# MAGIC 
# MAGIC データフレームとSparkSQLはここまで説明した通りに動作します。どのようにデータにアクセスするのかの計画を組み立て、最終的にはそのプランをアクションによって実行します。これらのプロセスを以下の図に示します。クエリを分析し、プランを立て、比較を行い、最終的に実行することで、全体のプロセスをなぞります。
# MAGIC 
# MAGIC ![Spark Query Plan](http://training.databricks.com/databricks_guide/gentle_introduction/query-plan-generation.png)
# MAGIC 
# MAGIC このプロセスがどのように動作するのかについて、あまり詳細には踏み込みませんが、詳細な内容は[Databricks blog](https://databricks.com/blog/2015/04/13/deep-dive-into-spark-sqls-catalyst-optimizer.html)で読むことができます。このプロセスを通じてどのようにApache Sparkが動作しているのかを知りたい方には、こちらの記事を読むことをお勧めします。
# MAGIC 
# MAGIC 以降では、Databricksで利用可能なパブリックなデータセットにアクセスすることになります。Databricksデータセットは、Webから集められた、小規模かつ整理されたデータセットとなっています。これらのデータは、[Databricks File System](https://docs.databricks.com/data/databricks-file-system.html)を通じて利用できます。よく使われるダイアモンドのデータセットを、Sparkの`DataFrame`としてロードしてみましょう。まずは、作業することになるデータセットを見てみましょう。

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/Rdatasets/data-001/datasets.csv

# COMMAND ----------

# MAGIC %md 以下のコードでは、前の行から継続していることを示すためにバックスラッシュ `\` を使用していることに注意してください。Sparkのコマンドは、多くのケースで複数のオペレーションのチェーンを構築することになります(例：`.option(...)`)。コマンドが一行に収まらない場合、バックスラッシュを使うことでコードをきれいに保つことができます。

# COMMAND ----------

dataPath = "/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv"
diamonds = sqlContext.read.format("csv")\
  .option("header","true")\
  .option("inferSchema", "true")\
  .load(dataPath)
  
# inferSchemaはデータを読み込んで自動的にカラムの型を識別するオプションです。データを読み込む分のコストがかかります。

# COMMAND ----------

# MAGIC %md 
# MAGIC データをロードしたので、計算に取りかかります。この作業を通じて、基礎的な機能とDatabricks上で動作するSparkの実行をシンプルにする素敵な機能のいくつかを体験することができます。計算を行うためには、データを理解する必要があります。これは`display`関数で可能となります。

# COMMAND ----------

display(diamonds)

# COMMAND ----------

# MAGIC %md 
# MAGIC `display`の素晴らしいところは、グラフアイコンをクリックするだけで以下のような素晴らしいグラフを容易に作成できるところです。以下のプロットによって、価格、色、カットを比較することができます。

# COMMAND ----------

display(diamonds)

# COMMAND ----------

# MAGIC %md 
# MAGIC データを探索しましたので、**transformations**と**actions**の理解に戻りましょう。ここではいくつかのトランスフォーメーションを作成し、アクションを呼び出します。その後で、内部で何が行われたのかを見ていきます。
# MAGIC 
# MAGIC これらのトランスフォーメーションはシンプルなものです。最初に二つの変数、カットとカラーでグルーピングします。そして、平均価格を計算します。そして、`color`カラムでオリジナルのデータセットと`inner join`を行います。そして、新たなデータセットからカラットと平均価格を選択します。

# COMMAND ----------

df1 = diamonds.groupBy("cut", "color").avg("price") # シンプルなグルーピング

df2 = df1\
  .join(diamonds, on='color', how='inner')\
  .select("`avg(price)`", "carat")
# シンプルなjoin及び列の選択

# COMMAND ----------

# MAGIC %md 
# MAGIC これらのトランスフォーメーションはある意味完成していますが何も起きません。上に表示されているように何の結果も表示されていません！
# MAGIC 
# MAGIC これは、ユーザーによって要求される処理の初めから終わりまでの全体のデータフローを構築するために、これらの処理が*lazy(怠惰)*となっているためです。
# MAGIC 
# MAGIC これは、二つの理由からスマートな最適化と言えます。第一に、大元のソースデータから再計算できることで、Apache Sparkが途中で生じるエラーや、処理に手間取るワーカーノードをハンドリングできるように、データの大元から再計算することができます。第二に、上で述べたようにデータと処理がパイプライン化されるように、処理を最適化することができます。このため、それぞれのトランスフォーメーションに対して、Apache Sparkはどのように処理を行うのか計画を立てます。
# MAGIC 
# MAGIC 計画がどのようなものなのかを理解するために、`explain`メソッドを使用します。この時点ではまだ何の処理も実行されていないことに注意してください。このため、explainメソッドが教えてくれることは、このデータセットに対してどのように処理を行うのかの証跡(leneage)となります。

# COMMAND ----------

df2.explain()

# COMMAND ----------

# MAGIC %md 
# MAGIC 上の結果が何を意味しているのかは、この導入のチュートリアルの範疇外となりますが、中身を読むのは自由です。ここで導き出されることは、Sparkは与えられたクエリーを実行する際には実行計画を生成するということです。上のプランを実行に移すためにアクションを実行しましょう。

# COMMAND ----------

df2.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC これにより、Apache Sparkが上で構築したプランが実行されます。実行後に、`(1) Spark Jobs`と言った表示の隣にある小さい矢印をクリックし、`View`リンクをクリックし、さらに`DAG Visualization`をクリックします。これにより、ノートブックの右側にApache Spark Web UIが表示されます。ノートブックの上部にあるクラスターをクリックしてもこの画面にアクセスすることができます。Spark UIでは、以下のような図を見ることができます。
# MAGIC 
# MAGIC <img src="http://training.databricks.com/databricks_guide/gentle_introduction/spark-dag-ui.png" width=600>
# MAGIC 
# MAGIC これが、結果を得るために実行された処理全ての有向非巡回グラフ(DAG)となります。この可視化によって、Sparkがデータを最終的な形に持っていくまでのステップ全てを参照することができます。
# MAGIC 
# MAGIC 繰り返しになりますが、このDAGはトランスフォーメーションが*lazy*であるため生成されます。これら一連のステップを生成する過程で、Sparkは多くの最適化を行い、そのためのコードをも生成します。このことによって、ユーザはレガシーなRDD APIではなくデータフレームやデータセットにフォーカスすることができます。データフレームやデータセットを用いることで、Apache Sparkは内部で全てのステップ、全てのクエリープラン、パイプラインを最適化することができます。プランの中に`WholeStageCodeGen`や`tungsten`を見ることになるかと思います。これらは[improvements in Spark SQL, which you can read more about on the Databricks blog.](https://databricks.com/blog/2015/04/28/project-tungsten-bringing-spark-closer-to-bare-metal.html)の一部です。
# MAGIC 
# MAGIC 上の図では、左側のCSVから始まり、いくつかの処理を経て、別のCSVファイル(これはオリジナルのデータフレームから作成したものです)にマージし、これらをジョインした上で最終的な結果を得るためにいくつかの集計処理を行っています。

# COMMAND ----------

# MAGIC %md ## キャッシュ
# MAGIC 
# MAGIC Apache Sparkの重要な機能の一つとして、計算の過程でメモリーにデータを格納できるということが挙げられます。これは、よく検索されるテーブルやデータに対するアクセスを高速にする際に活用できるテクニックです。また、同じデータに対して繰り返しアクセスするような繰り返しの処理を行うアルゴリズムにも有効です。全ての性能問題に対する万能薬かと思うかもしれませんが、利用できるツールの一つとして捉えるべきです。データのパーティショニングやクラスタリング、バケッティングなどの他の重要なコンセプトの方が、キャッシングよりも高い性能を実現する場合があります。とは言え、これら全てのツールが利用可能であることを覚えておいて下さい。
# MAGIC 
# MAGIC データフレームやRDDをキャッシュするには、単に`cache`メソッドを呼ぶだけです。

# COMMAND ----------

df2.cache()

# COMMAND ----------

# MAGIC %md 
# MAGIC キャッシュはトランスフォーメーションのようにlazyに評価されます。データセットに対するアクションが呼び出されるまでメモリーにデータはキャッシュされません。
# MAGIC 
# MAGIC 簡単な例で説明します。これまでに我々はデータフレーム`df2`を作成しました。これは本質的には、データフレームをどのように計算するのかを示すロジカルプランです。Apahce Sparkに対してこのデータをキャッシュするように初めて伝えました。countによるデータに対するフルスキャンを2回実行しましょう。最初の時には、データフレームを作成しメモリーにキャッシュし、結果を返却します。2回目は、全てのデータフレームを計算するのではなく、メモリーに存在するバージョンを取得します。
# MAGIC 
# MAGIC どのように動作しているのかを見てみましょう。

# COMMAND ----------

df2.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC キャッシュした後では、同じクエリーに要する処理時間が大きく減少していることがわかります。

# COMMAND ----------

df2.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC 上の例では、データを生成するのに必要な時間を劇的に短縮できた様子が見て取れます。少なくとも一桁レベルで削減できています。より大規模かつ複雑なデータ分析においては、キャッシングによる恩恵はより大きなものになります。

# COMMAND ----------

# MAGIC %md ## まとめ
# MAGIC 
# MAGIC このノートブックでは多くの話題をカバーしました！ しかし、まだあなたはSparkとDatabricksを学ぶ道の入り口に立ったところです！ このノートブックを修了することで、SparkとDatabricksのコアの概念に慣れ親しんだに違いありません。
# MAGIC 
# MAGIC 是非、[コミュニティエディションあるいは実環境のフリートライアル](https://databricks.com/jp/try-databricks)を体験してみて下さい!

# COMMAND ----------

# MAGIC %md # END
