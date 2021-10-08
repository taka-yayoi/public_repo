### Notebook Discovery

[Databricksノートブックのカタログ化、高速検索の実現 \- Qiita](https://qiita.com/taka_yayoi/items/00f5a996246d9bc6f386)

Elsevierのデータエンジニア、データサイエンティストはDatabrikcsを6年使用しています。この期間、我々は数千のノートブックを作成しました。何年もの間、ワークスペース内のノートブックを検索しようとしましたが、現状の制限のある検索機能では困難であることを知りました。例えば、特定のユーザーのワークスペースのフォルダー、ノートブック名、特定のコマンド言語、特定セルのコマンドテキスト、コマンドを実行した日付で検索を行えたら素晴らしいと思いませんか？元のノートブックからコピーされ修正が加えられたノートブックを発見できたら素晴らしいと思いませんか？数年前に我々は同様のフィードバックをDatabricksフォーラムに投稿しました。以下に代表的な例を示します。

- ソーステーブルを更新していますが、そのテーブルを持っている全てのノートブックを探す必要があります。Databricks UIの検索機能を試しましたが、他のユーザーのフォルダーの結果も表示されてしまいます。条件を指定して検索結果を特定のフォルダーに限定することはできませんか？
- 完全フレーズ一致、部分一致(除外)、ワイルドカード、正規表現を用いたノートブックの検索はできませんか？
- 全てのノートブックの文字列を検索する方法はありませんか？Deltaテーブルを使用しているノートブックを探したいのです。
- デバッグのために他のノートブックで使われている類似コマンドを検索できませんか？

これらの制限に打ち勝つために、Elsevier LabsはNotebook Discoveryをオープンソース化しました。

これは100%ノートブックのソリューションであり、言語がScala/Python/SQL/Rであるかに関係なく、既存のDatabricks環境に容易にインテグレーションすることができます。あるユーザーは、自身のユーザーフォルダーに適用することに有用性を感じるかもしれません。ある人は、全てのユーザーフォルダーやグループメンバーのフォルダーに適用したいと考えるかもしれません。我々は特にLabのメンバーに適用することの有効性を発見しました。このツールはスケーラブルなので、企業全てのユーザーフォルダーに適用することも可能です。どのようなケースにおいても有用であることを望んでいます。

Darin McBeath<br/>
Elsevier Labs

##### ソリューション
Notebook Discoveryは3つのノートブックから構成されます。

- <i>NotebookIndex</i>
  - NotebookSearchとNotebookSimilarityで使用するNotebook Discoveryのインデックスparquetファイルを作成します。ノートブックのそれぞれのコマンドは(ノートブック全体ではなく)コマンドレベルで検索できるように別々のレコードに格納されます。生成されるインデックスの構造に関しては、<a href="$./NotebookIndex">NotebookIndex</a>ノートブックの説明をご覧ください。 


- <i>NotebookSimilarity</i>
  - NotebookIndexで生成されたNotebook Discoveryインデックスparquetファイルのノートブックを比較し、全てのノートブックの類似度を生成します。数千のノートブックにスケールするように、Jaccard距離アルゴリズムの[MinHash](https://spark.apache.org/docs/latest/ml-features.html#minhash-for-jaccard-distance)を使用します。詳細は<a href="$./NotebookSimilarity">NotebookSimilarity</a>ノートブックの説明をご覧ください。
  
- <i>NotebookSearch</i>
  - NotebookIndexで生成されるNotebook Discoveryインデックスparquetファイルを検索します。簡単にアクセスできるようにノートブックへのリンク(あるいは特定のコマンド)を表示するように検索結果をHTMLに整形するUDFが含まれています。詳細は<a href="$./NotebookSearch">NotebookSearch</a>ノートブックの説明をご覧ください。
  - ノートブックのCmd 5でNotebook Discoveryインデックスparquetファイルのパスを指定してください。
  

##### ダウンロード

Notebook Discoveryは[download](https://github.com/elsevierlabs-os/NotebookDiscovery/tree/main/download/NotebookDiscovery-1.0.0.dbc)からダウンロードできます。DBC (Databricks archive)ファイルです。


##### インストール

ダウンロードしたDBCファイルをDatabricks UIからフォルダーにインポートします。アーカイブをインポートしたワークスペースフォルダーと、インデックスを作成したいワークスペースフォルダーは無関係です。

##### 実行

上で述べた3つノートブックに加え、Notebook Discoveryの使用をシンプルにするために作られた実行用ノートブックがあります。実行用ノートブックを使うことで、処理の実行に必要なパラメーターを簡単に調整することができます。実行用ノートブックはscalaで実装されていますが、これらはわかりやすいものであり、他のプログラミング言語に慣れている人でも簡単に使えます。

- NotebookIndexRun
  - 他のノートブックが必要とするNotebook Discoveryインデックスparquetファイルを作成するために、最初に実行するノートブックが<a href="$./NotebookIndexRun">NotebookIndexRun</a>です。NotebookIndexRunの処理が成功すると、NotebookSimilarityRunやNotebookSearchにあるサンプルクエリーを実行することができます。
  - 引数には以下があります。
    - path: 実行するノートブックのパス(この場合はNotebookIndex)。ここでは、このノートブックは実行用ノートブックと同じにあると仮定しています。
    - timeoutSeconds: ノートブック実行に必要な時間(秒数、初期値は2時間)。
    - folders: Notebook Discoveryインデックスを作成する際に、クローリング対象となるワークスペースフォルダーのカンマ区切りのリスト
    - indexFilename: Notebook Discoveryインデックスparquetファイルを格納するパス。多くの場合、Databricksにマウントされた領域となります。
    - overwrite: Notebook Discoveryインデックスparquetファイルが存在する場合に上書きするかどうか。'False'の場合、上書きしません。'True'は上書きします。
    - parallelization: インデックス作成時にREST APIを通じてDatabricksに送信されるリクエストの同時実行数(初期値は8)。

  - 以下がサンプルのリクエストです。

```
dbutils.notebook.run(path="NotebookIndex",
                              timeoutSeconds=7200,
                              arguments=Map("folders" -> """
                                                            /Users/someone1@elsevier.com/,
                                                            /Users/someone2@elsevier.com/
                                                            """,
                                            "indexFilename" -> "/mnt/some-mount-point/DatabricksNotebookSearch/index",
                                            "overwrite" -> "False",
                                            "parallelization" -> "8"))                                            
```


- NotebookIndexRun_for_own_folder
  - ご自身のユーザーフォルダーのみをインデックスします。ログインIDからフォルダーを特定し、その中のサブフォルダーのリストを作成するので、処理が並列化されます。


- NotebookSimilarityRun
  - ノートブックの類似度スコアを生成したいのであれば、Notebook Discovery類似度ファイルを生成するために<a href="$./NotebookSimilarityRun">NotebookSimilarityRun</a>を実行してください。NotebookSimilarityRunの処理が成功した後は、生成されるparquetファイルを分析し、類似のノートブックを探索することができます。
  - 引数は以下の通りです。
    - path: 実行するノートブックのパス(この場合はNotebookSimilarity)。ここでは、このノートブックは実行用ノートブックと同じにあると仮定しています。
    - timeoutSeconds: ノートブック実行に必要な時間(秒数、初期値は2時間)。
    - indexFilename: Notebook Discoveryインデックスparquetファイルを格納するパス。
    - similarityFilename: Notebook Discovery類似度parquetファイルを格納するパス。多くの場合、Databricksにマウントされた領域となります。
    - overwrite: Notebook Discoveryインデックスparquetファイルが存在する場合に上書きするかどうか。'False'の場合、上書きしません。'True'は上書きします。
    - similarDistance: ノートブックを比較する際に使用するJaccard距離の最大値。
    - vocabSize: インデックスされるノートブックに含まれるユニークな単語の数(ngrams)。
    - ngramSize: ノートブックをトークナイズし、比較する際に用いるngramのサイズ。
    - minDocFreq: ボキャブラリーに含まれる際にngramが含まれるべきノートブックの数。
  
  - 以下がサンプルリクエストです。

```
dbutils.notebook.run(path="NotebookSimilarity",
                              timeoutSeconds=7200,
                              arguments=Map("indexFilename" -> "/mnt/some-mount-point/DatabricksNotebookSearch/index", 
                                            "similarityFilename" -> "/mnt/some-mount-point/DatabricksNotebookSearch/ndex-similarity",
                                            "overwrite" -> "False",
                                            "similarDistance" -> ".25",
                                            "vocabSize" -> "5000000",
                                            "ngramSize" -> "5",
                                            "minDocFreq" -> "1"))                                            
```
  
- NotebookSearch
  - 特定テキストを含むノートブックやコマンドを検索したいのであれば、<a href="$./NotebookSearch">NotebookSearch</a>のサンプルを参照ください。Notebook Discoveryインデックスはparquetなので、検索条件に合致するノートブック、コマンドを特定するために、基本的なfilter、containsオペレーションを行うことができます。
  - 検索結果に表示されるリンクのURLの置換が必要な場合には、Cmd7の`url_replace`をtrueにして、以降の置換処理で置き換えるURLを指定してください。
  - 以下に、言語が'scala'で、'/Users/d.mcbeath@elsevier.com/'で始まるフォルダー、そして、'udf'を含むコマンドテキストを検索し、結果を10件表示するサンプルを示します。
  
```
  val notebookCommands = spark.read.parquet(indexFilename)
  displaySearchResults(notebookCommands.filter($"cLang" === "scala")
                                       .filter($"cText".contains("udf"))
                                       .filter($"nbFolder".startsWith("/Users/d.mcbeath@elsevier.com/")),num=10)                                       
```

##### 仮定
- Notebook Discoveryインデックスと類似度ファイルは現状parquetとなっています。Databricksのテーブルにしたいのであれば、実行用ノートブックに、parquetからテーブルを作成するステップを簡単に追加するか、parquetファイルではなくテーブルに書き込むようにコードを変更することができます。

##### 制限
- ノートブック名に"/"が含まれるノートブックは、現状無視されます。
- ノートブックのコンテンツが10MBを超えるノートブックは処理されません。
- Notebook Discoveryインデックスparquetファイルはある時点のスナップショットです。定期的にアップデートしたいのであれば、NotebookIndexRunを定期的に(例：週次)で実行する必要があります。
- Notebook Discoveryでインデックスを作成するノートブックはgithubのような外部リポジトリではなく、Databricksワークスペースに格納されている必要があります。
- Notebook Discoveryでインデックスを作成するノートブックは、NotebookIndexRunを実行するユーザーに対する読み取り権限が設定されている必要があります。
- NotebookSearchでリンクされるノートブック(コマンド)を参照するには、ユーザーはノートブックに対する読み取り権限を持っている必要があります。

##### 今後の計画

- それぞれのノートブックに対するuser/groupのアクセス権をNotebook Discoveryインデックスparquetファイルに含める予定です。これにより、ノートブックに関連づけられたアクセス権に基づきフィルタリングを行えるようになります。また、この情報はユーザーがアクセスできる情報のみにNotebookSearchの結果をフィルタリングする際にも使用できます。
- 並列処理の改善。現状は、NotebookIndexRunに一つのフォルダーが指定された場合には並列化されません。

##### ライセンス

MIT License

Copyright (c) 2021 Elsevier 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.`
