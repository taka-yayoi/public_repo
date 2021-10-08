// Databricks notebook source
// MAGIC %md ### Notebook Discovery
// MAGIC 
// MAGIC [Databricksノートブックのカタログ化、高速検索の実現 \- Qiita](https://qiita.com/taka_yayoi/items/00f5a996246d9bc6f386)
// MAGIC 
// MAGIC Elsevierのデータエンジニア、データサイエンティストはDatabrikcsを6年使用しています。この期間、我々は数千のノートブックを作成しました。何年もの間、ワークスペース内のノートブックを検索しようとしましたが、現状の制限のある検索機能では困難であることを知りました。例えば、特定のユーザーのワークスペースのフォルダー、ノートブック名、特定のコマンド言語、特定セルのコマンドテキスト、コマンドを実行した日付で検索を行えたら素晴らしいと思いませんか？元のノートブックからコピーされ修正が加えられたノートブックを発見できたら素晴らしいと思いませんか？数年前に我々は同様のフィードバックをDatabricksフォーラムに投稿しました。以下に代表的な例を示します。
// MAGIC 
// MAGIC - ソーステーブルを更新していますが、そのテーブルを持っている全てのノートブックを探す必要があります。Databricks UIの検索機能を試しましたが、他のユーザーのフォルダーの結果も表示されてしまいます。条件を指定して検索結果を特定のフォルダーに限定することはできませんか？
// MAGIC - 完全フレーズ一致、部分一致(除外)、ワイルドカード、正規表現を用いたノートブックの検索はできませんか？
// MAGIC - 全てのノートブックの文字列を検索する方法はありませんか？Deltaテーブルを使用しているノートブックを探したいのです。
// MAGIC - デバッグのために他のノートブックで使われている類似コマンドを検索できませんか？
// MAGIC 
// MAGIC これらの制限に打ち勝つために、Elsevier LabsはNotebook Discoveryをオープンソース化しました。
// MAGIC 
// MAGIC これは100%ノートブックのソリューションであり、言語がScala/Python/SQL/Rであるかに関係なく、既存のDatabricks環境に容易にインテグレーションすることができます。あるユーザーは、自身のユーザーフォルダーに適用することに有用性を感じるかもしれません。ある人は、全てのユーザーフォルダーやグループメンバーのフォルダーに適用したいと考えるかもしれません。我々は特にLabのメンバーに適用することの有効性を発見しました。このツールはスケーラブルなので、企業全てのユーザーフォルダーに適用することも可能です。どのようなケースにおいても有用であることを望んでいます。
// MAGIC 
// MAGIC Darin McBeath<br/>
// MAGIC Elsevier Labs
// MAGIC 
// MAGIC ##### ソリューション
// MAGIC Notebook Discoveryは3つのノートブックから構成されます。
// MAGIC 
// MAGIC - <i>NotebookIndex</i>
// MAGIC   - NotebookSearchとNotebookSimilarityで使用するNotebook Discoveryのインデックスparquetファイルを作成します。ノートブックのそれぞれのコマンドは(ノートブック全体ではなく)コマンドレベルで検索できるように別々のレコードに格納されます。生成されるインデックスの構造に関しては、<a href="$./NotebookIndex">NotebookIndex</a>ノートブックの説明をご覧ください。 
// MAGIC 
// MAGIC 
// MAGIC - <i>NotebookSimilarity</i>
// MAGIC   - NotebookIndexで生成されたNotebook Discoveryインデックスparquetファイルのノートブックを比較し、全てのノートブックの類似度を生成します。数千のノートブックにスケールするように、Jaccard距離アルゴリズムの[MinHash](https://spark.apache.org/docs/latest/ml-features.html#minhash-for-jaccard-distance)を使用します。詳細は<a href="$./NotebookSimilarity">NotebookSimilarity</a>ノートブックの説明をご覧ください。
// MAGIC   
// MAGIC - <i>NotebookSearch</i>
// MAGIC   - NotebookIndexで生成されるNotebook Discoveryインデックスparquetファイルを検索します。簡単にアクセスできるようにノートブックへのリンク(あるいは特定のコマンド)を表示するように検索結果をHTMLに整形するUDFが含まれています。詳細は<a href="$./NotebookSearch">NotebookSearch</a>ノートブックの説明をご覧ください。
// MAGIC   - ノートブックのCmd 5でNotebook Discoveryインデックスparquetファイルのパスを指定してください。
// MAGIC   
// MAGIC 
// MAGIC ##### ダウンロード
// MAGIC 
// MAGIC Notebook Discoveryは[download](https://github.com/elsevierlabs-os/NotebookDiscovery/tree/main/download/NotebookDiscovery-1.0.0.dbc)からダウンロードできます。DBC (Databricks archive)ファイルです。
// MAGIC 
// MAGIC 
// MAGIC ##### インストール
// MAGIC 
// MAGIC ダウンロードしたDBCファイルをDatabricks UIからフォルダーにインポートします。アーカイブをインポートしたワークスペースフォルダーと、インデックスを作成したいワークスペースフォルダーは無関係です。
// MAGIC 
// MAGIC ##### 実行
// MAGIC 
// MAGIC 上で述べた3つノートブックに加え、Notebook Discoveryの使用をシンプルにするために作られた実行用ノートブックがあります。実行用ノートブックを使うことで、処理の実行に必要なパラメーターを簡単に調整することができます。実行用ノートブックはscalaで実装されていますが、これらはわかりやすいものであり、他のプログラミング言語に慣れている人でも簡単に使えます。
// MAGIC 
// MAGIC - NotebookIndexRun
// MAGIC   - 他のノートブックが必要とするNotebook Discoveryインデックスparquetファイルを作成するために、最初に実行するノートブックが<a href="$./NotebookIndexRun">NotebookIndexRun</a>です。NotebookIndexRunの処理が成功すると、NotebookSimilarityRunやNotebookSearchにあるサンプルクエリーを実行することができます。
// MAGIC   - 引数には以下があります。
// MAGIC     - path: 実行するノートブックのパス(この場合はNotebookIndex)。ここでは、このノートブックは実行用ノートブックと同じにあると仮定しています。
// MAGIC     - timeoutSeconds: ノートブック実行に必要な時間(秒数、初期値は2時間)。
// MAGIC     - folders: Notebook Discoveryインデックスを作成する際に、クローリング対象となるワークスペースフォルダーのカンマ区切りのリスト
// MAGIC     - indexFilename: Notebook Discoveryインデックスparquetファイルを格納するパス。多くの場合、Databricksにマウントされた領域となります。
// MAGIC     - overwrite: Notebook Discoveryインデックスparquetファイルが存在する場合に上書きするかどうか。'False'の場合、上書きしません。'True'は上書きします。
// MAGIC     - parallelization: インデックス作成時にREST APIを通じてDatabricksに送信されるリクエストの同時実行数(初期値は8)。
// MAGIC 
// MAGIC   - 以下がサンプルのリクエストです。
// MAGIC 
// MAGIC ```
// MAGIC dbutils.notebook.run(path="NotebookIndex",
// MAGIC                               timeoutSeconds=7200,
// MAGIC                               arguments=Map("folders" -> """
// MAGIC                                                             /Users/someone1@elsevier.com/,
// MAGIC                                                             /Users/someone2@elsevier.com/
// MAGIC                                                             """,
// MAGIC                                             "indexFilename" -> "/mnt/some-mount-point/DatabricksNotebookSearch/index",
// MAGIC                                             "overwrite" -> "False",
// MAGIC                                             "parallelization" -> "8"))                                            
// MAGIC ```
// MAGIC 
// MAGIC 
// MAGIC - NotebookIndexRun_for_own_folder
// MAGIC   - ご自身のユーザーフォルダーのみをインデックスします。ログインIDからフォルダーを特定し、その中のサブフォルダーのリストを作成するので、処理が並列化されます。
// MAGIC 
// MAGIC 
// MAGIC - NotebookSimilarityRun
// MAGIC   - ノートブックの類似度スコアを生成したいのであれば、Notebook Discovery類似度ファイルを生成するために<a href="$./NotebookSimilarityRun">NotebookSimilarityRun</a>を実行してください。NotebookSimilarityRunの処理が成功した後は、生成されるparquetファイルを分析し、類似のノートブックを探索することができます。
// MAGIC   - 引数は以下の通りです。
// MAGIC     - path: 実行するノートブックのパス(この場合はNotebookSimilarity)。ここでは、このノートブックは実行用ノートブックと同じにあると仮定しています。
// MAGIC     - timeoutSeconds: ノートブック実行に必要な時間(秒数、初期値は2時間)。
// MAGIC     - indexFilename: Notebook Discoveryインデックスparquetファイルを格納するパス。
// MAGIC     - similarityFilename: Notebook Discovery類似度parquetファイルを格納するパス。多くの場合、Databricksにマウントされた領域となります。
// MAGIC     - overwrite: Notebook Discoveryインデックスparquetファイルが存在する場合に上書きするかどうか。'False'の場合、上書きしません。'True'は上書きします。
// MAGIC     - similarDistance: ノートブックを比較する際に使用するJaccard距離の最大値。
// MAGIC     - vocabSize: インデックスされるノートブックに含まれるユニークな単語の数(ngrams)。
// MAGIC     - ngramSize: ノートブックをトークナイズし、比較する際に用いるngramのサイズ。
// MAGIC     - minDocFreq: ボキャブラリーに含まれる際にngramが含まれるべきノートブックの数。
// MAGIC   
// MAGIC   - 以下がサンプルリクエストです。
// MAGIC 
// MAGIC ```
// MAGIC dbutils.notebook.run(path="NotebookSimilarity",
// MAGIC                               timeoutSeconds=7200,
// MAGIC                               arguments=Map("indexFilename" -> "/mnt/some-mount-point/DatabricksNotebookSearch/index", 
// MAGIC                                             "similarityFilename" -> "/mnt/some-mount-point/DatabricksNotebookSearch/ndex-similarity",
// MAGIC                                             "overwrite" -> "False",
// MAGIC                                             "similarDistance" -> ".25",
// MAGIC                                             "vocabSize" -> "5000000",
// MAGIC                                             "ngramSize" -> "5",
// MAGIC                                             "minDocFreq" -> "1"))                                            
// MAGIC ```
// MAGIC   
// MAGIC - NotebookSearch
// MAGIC   - 特定テキストを含むノートブックやコマンドを検索したいのであれば、<a href="$./NotebookSearch">NotebookSearch</a>のサンプルを参照ください。Notebook Discoveryインデックスはparquetなので、検索条件に合致するノートブック、コマンドを特定するために、基本的なfilter、containsオペレーションを行うことができます。
// MAGIC   - 検索結果に表示されるリンクのURLの置換が必要な場合には、Cmd7の`url_replace`をtrueにして、以降の置換処理で置き換えるURLを指定してください。
// MAGIC   - 以下に、言語が'scala'で、'/Users/d.mcbeath@elsevier.com/'で始まるフォルダー、そして、'udf'を含むコマンドテキストを検索し、結果を10件表示するサンプルを示します。
// MAGIC   
// MAGIC ```
// MAGIC   val notebookCommands = spark.read.parquet(indexFilename)
// MAGIC   displaySearchResults(notebookCommands.filter($"cLang" === "scala")
// MAGIC                                        .filter($"cText".contains("udf"))
// MAGIC                                        .filter($"nbFolder".startsWith("/Users/d.mcbeath@elsevier.com/")),num=10)                                       
// MAGIC ```
// MAGIC 
// MAGIC ##### 仮定
// MAGIC - Notebook Discoveryインデックスと類似度ファイルは現状parquetとなっています。Databricksのテーブルにしたいのであれば、実行用ノートブックに、parquetからテーブルを作成するステップを簡単に追加するか、parquetファイルではなくテーブルに書き込むようにコードを変更することができます。
// MAGIC 
// MAGIC ##### 制限
// MAGIC - ノートブック名に"/"が含まれるノートブックは、現状無視されます。
// MAGIC - ノートブックのコンテンツが10MBを超えるノートブックは処理されません。
// MAGIC - Notebook Discoveryインデックスparquetファイルはある時点のスナップショットです。定期的にアップデートしたいのであれば、NotebookIndexRunを定期的に(例：週次)で実行する必要があります。
// MAGIC - Notebook Discoveryでインデックスを作成するノートブックはgithubのような外部リポジトリではなく、Databricksワークスペースに格納されている必要があります。
// MAGIC - Notebook Discoveryでインデックスを作成するノートブックは、NotebookIndexRunを実行するユーザーに対する読み取り権限が設定されている必要があります。
// MAGIC - NotebookSearchでリンクされるノートブック(コマンド)を参照するには、ユーザーはノートブックに対する読み取り権限を持っている必要があります。
// MAGIC 
// MAGIC ##### 今後の計画
// MAGIC 
// MAGIC - それぞれのノートブックに対するuser/groupのアクセス権をNotebook Discoveryインデックスparquetファイルに含める予定です。これにより、ノートブックに関連づけられたアクセス権に基づきフィルタリングを行えるようになります。また、この情報はユーザーがアクセスできる情報のみにNotebookSearchの結果をフィルタリングする際にも使用できます。
// MAGIC - 並列処理の改善。現状は、NotebookIndexRunに一つのフォルダーが指定された場合には並列化されません。
// MAGIC 
// MAGIC ##### ライセンス
// MAGIC 
// MAGIC MIT License
// MAGIC 
// MAGIC Copyright (c) 2021 Elsevier 
// MAGIC 
// MAGIC Permission is hereby granted, free of charge, to any person obtaining a copy
// MAGIC of this software and associated documentation files (the "Software"), to deal
// MAGIC in the Software without restriction, including without limitation the rights
// MAGIC to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// MAGIC copies of the Software, and to permit persons to whom the Software is
// MAGIC furnished to do so, subject to the following conditions:
// MAGIC 
// MAGIC The above copyright notice and this permission notice shall be included in all
// MAGIC copies or substantial portions of the Software.
// MAGIC 
// MAGIC THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// MAGIC IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// MAGIC FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// MAGIC AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// MAGIC LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// MAGIC OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// MAGIC SOFTWARE.`
