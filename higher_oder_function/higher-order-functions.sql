-- Databricks notebook source
-- MAGIC %md
-- MAGIC # 高次関数のイントロダクション

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## ネストされたデータを持つテーブルの作成

-- COMMAND ----------

CREATE
OR REPLACE TEMPORARY VIEW nested_data AS
SELECT
  id AS key,
  ARRAY(
    CAST(RAND(1) * 100 AS INT),
    CAST(RAND(2) * 100 AS INT),
    CAST(RAND(3) * 100 AS INT),
    CAST(RAND(4) * 100 AS INT),
    CAST(RAND(5) * 100 AS INT)
  ) AS
values,
  ARRAY(
    ARRAY(
      CAST(RAND(1) * 100 AS INT),
      CAST(RAND(2) * 100 AS INT)
    ),
    ARRAY(
      CAST(RAND(3) * 100 AS INT),
      CAST(RAND(4) * 100 AS INT),
      CAST(RAND(5) * 100 AS INT)
    )
  ) AS nested_values
FROM
  range(5)

-- COMMAND ----------

-- MAGIC %sql SELECT * FROM nested_data

-- COMMAND ----------

-- MAGIC %md ## シンプルな例
-- MAGIC 
-- MAGIC 基本的な変換処理でコンセプトの基礎を学びましょう。このケースでは、高次関数``transform``が配列``values``に対してイテレーションを行い、関連づけられたラムダ関数をそれぞれの要素に適用し、新たな配列を作成します。ラムダ関数``element + 1``は、それぞれの要素をどの様に操作するのかを指定します。このSQLは以下の様になります。

-- COMMAND ----------

SELECT
  key, 
  values,
  TRANSFORM(values, value -> value + 1) AS values_plus_one
FROM
  nested_data

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 変換処理`TRANSFORM(values, value -> value + 1)`は二つのコンポーネントから構成されます:
-- MAGIC </p>
-- MAGIC 
-- MAGIC 1. `TRANSFORM(values..)`は高次関数です。これは、入力として配列と匿名関数を受け取ります。内部的には、新規配列のセットアップ、それぞれの要素への匿名関数の適用、出力配列への結果の割り当てを行います。
-- MAGIC 1. `value -> value + 1`は匿名関数です。この関数は、シンボル`->`で区切られた2つのコンポーネントから構成されます。
-- MAGIC   - 引数のリスト。この場合は引数は1つの`value`です。`(x, y) -> x + y`のように、括弧で囲まれたカンマ区切りの引数リストを作成することで複数の引数もサポートしています。
-- MAGIC   - 本体。新たな値を計算するために引数と外部の変数を使用できるエクスプレッションです。この場合、`argument`の値に`1`を加算しています。

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 変数のキャプチャ
-- MAGIC 
-- MAGIC ラムダ関数では引数だけではなく、他の変数を使用することもできます。これはキャプチャと呼ばれます。トップレベルで定義されている変数や中間のラムダ関数で定義されている変数を使用することができます。例えば、以下の変換処理では`key`(トップレベル)変数を、配列`values`のそれぞれの要素に加算しています。

-- COMMAND ----------

SELECT
  key,
  values,
  TRANSFORM(values, value -> value + key) AS values_plus_key
FROM
  nested_data

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 
-- MAGIC ## ネスト化
-- MAGIC 
-- MAGIC 深くネストされたデータを変換したい場合、ネストされたラムダ関数を使用することができます。以下の例では、integerの配列の配列を変換し、ネストされた配列のそれぞれの要素に、`key`(トップレベル)カラムの値と中間配列のサイズを加算しています。

-- COMMAND ----------

SELECT
  key,
  nested_values,
  TRANSFORM(nested_values, values -> TRANSFORM(values, value -> value + key + SIZE(values))) AS new_nested_values
FROM
  nested_data

-- COMMAND ----------

-- MAGIC %md ## サポートされる関数
-- MAGIC 
-- MAGIC ##### `transform(array<T>, function<T, U>): array<U>`
-- MAGIC 
-- MAGIC 入力である`array<T>`のそれぞれの要素に`function<T, U>`を適用することで`array<U>`を変換します。
-- MAGIC 
-- MAGIC これは機能的には`map`と同じものとなります。(キーバリューエクスプレッションからmapを作成する)mapエクスプレッションとの混乱を避けるために`transform`と名付けられています。
-- MAGIC 
-- MAGIC 以下のクエリーでは、それぞれの要素に`key`の値を加算することで配列`values`を変換しています。

-- COMMAND ----------

SELECT   key,
         values,
         TRANSFORM(values, value -> value + key) transformed_values
FROM     nested_data

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #####`exists(array<T>, function<T, V, Boolean>): Boolean`
-- MAGIC 
-- MAGIC 入力の`array<T>`の要素が述語`function<T, Boolean>`に合致するかどうかをテストします。
-- MAGIC 
-- MAGIC 以下の例では、配列`values`に10で割った余りが1になる要素があるかどうかをチェックしています。

-- COMMAND ----------

SELECT
  key,
  values,
  EXISTS(values, value -> value % 10 == 1) filtered_values
FROM
  nested_data

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #####`filter(array<T>, function<T, Boolean>): array<T>`
-- MAGIC 
-- MAGIC 入力`array<T>`から述語`function<T, boolean>`にマッチするもののみを追加することで出力`array<T>`にフィルタリングします。
-- MAGIC 
-- MAGIC 以下の例では、`value > 50`を満たす要素のみからなる`values`配列にフィルタリングしています。

-- COMMAND ----------

SELECT   key,
         values,
         FILTER(values, value -> value > 50) filtered_values
FROM     nested_data

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #####`aggregate(array<T>, B, function<B, T, B>, function<B, R>): R`
-- MAGIC 
-- MAGIC `function<B, T, B>`を用いて、要素をバッファー`B`にマージし、最終的なバッファーに最後の`function<B, R>`を適用することで、`array<T>`の要素を単一の値`R`にまとめます。`B`の初期値はゼロエクスプレッションによって決定されます。最後の関数はオプションです。最終化の関数を指定しない場合、何も変化させない関数`(id -> id)`が使用されます。
-- MAGIC 
-- MAGIC これは、2つのラムダ関数を使う唯一の高次関数です。
-- MAGIC 
-- MAGIC 以下の例では、`values`配列を単一の(sum)の値に合計(集計)しています。最終化関数(`summed_values`)のバージョンと、最終化関数なしの`summed_values_simple`バージョンを示しています。
-- MAGIC 
-- MAGIC 
-- MAGIC **注意** 以下で使用している`REDUCE`関数は`AGGREGATE`関数と同じものです。

-- COMMAND ----------

SELECT   key,
         values,
         REDUCE(values, 0, (value, acc) -> value + acc, acc -> acc) summed_values,
         REDUCE(values, 0, (value, acc) -> value + acc) summed_values_simple
FROM     nested_data

-- COMMAND ----------

-- MAGIC %md
-- MAGIC さらに複雑な集計処理を行うこともできます。以下のコードでは、配列の要素のジオメトリックな平均値を計算しています。

-- COMMAND ----------

SELECT   key,
         values,
         AGGREGATE(values,
           (1 AS product, 0 AS N),
           (buffer, value) -> (value * buffer.product, buffer.N + 1),
           buffer -> Power(buffer.product, 1.0 / buffer.N)) geomean
FROM     nested_data

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # END
