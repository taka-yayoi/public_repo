// Databricks notebook source
// MAGIC %md 
// MAGIC # Mergeを用いたSCD(Slowly Changing Dimension) Type 2
// MAGIC 
// MAGIC SCD Type 2はディメンジョナルテーブルで指定されたキーに対する複数のレコードを作成することで履歴データを追跡します。このノートブックでは　`MERGE`オペレーションを用いたSCD Type 2オペレーションをどのように実行するのかをデモします。
// MAGIC 
// MAGIC ある企業で顧客と住所を保持するテーブルをメンテナンスしており、それぞれの住所が適切であれば期間とともに全ての住所の履歴を維持したいものとします。
// MAGIC 
// MAGIC Scalaのcaseクラスを用いてスキーマを定義しましょう。

// COMMAND ----------

// MAGIC %md 
// MAGIC ## Customers Deltaテーブル
// MAGIC 
// MAGIC これは我々がアップデートするslowly changingテーブルです。すべての顧客に対して、複数の住所が存在し得ます。しかし、それぞれの住所にはその住所が有効であった`effectiveDate`から`endDate`に至る期間も存在します。さらに、それぞれの顧客において現在適切な住所であることを示す別のフィールド`current`が存在します。すなわち、それぞれの顧客において`current`が`true`である住所は1行のみであり、他の行はfalseになります。

// COMMAND ----------

// MAGIC %sql
// MAGIC USE 20210712_demo_takaakiyayoi;

// COMMAND ----------

import java.sql.Date
import java.text._
import spark.implicits

case class CustomerUpdate(customerId: Int, address: String, effectiveDate: Date)
case class Customer(customerId: Int, address: String, current: Boolean, effectiveDate: Date, endDate: Date)

implicit def date(str: String): Date = Date.valueOf(str)

sql("drop table if exists customers")

Seq(
  Customer(1, "old address for 1", false, null, "2018-02-01"),
  Customer(1, "current address for 1", true, "2018-02-01", null),
  Customer(2, "current address for 2", true, "2018-02-01", null),
  Customer(3, "current address for 3", true, "2018-02-01", null)
).toDF().write.format("delta").mode("overwrite").saveAsTable("customers")

display(table("customers").orderBy("customerId"))

// COMMAND ----------

// MAGIC %md
// MAGIC ## Updatesテーブル
// MAGIC 
// MAGIC これは新規の住所を含むアップデートテーブルです。それぞれの顧客に対して、いつから有効かを示す日付と住所が含まれています。
// MAGIC 
// MAGIC 簡単にするために、同じcaseクラスを用いており、フィールド`current`と`endDate`を無視していることに注意してください。ここではこれらを使用しません。
// MAGIC 
// MAGIC このテーブルには顧客あたり1行が含まれており`effectiveDate`が適切に設定されている必要があります。

// COMMAND ----------

Seq(
  CustomerUpdate(1, "new address for 1", "2018-03-03"),
  CustomerUpdate(3, "current address for 3", "2018-04-04"),    // customer 3においては現在のアドレスと新規のアドレスは同じです。
  CustomerUpdate(4, "new address for 4", "2018-04-04")
).toDF().createOrReplaceTempView("updates")

// 注意: 
// - effectiveDateはSCD Type2のMergeの後にcustomersテーブルにコピーされるので、ソーステーブルにeffectiveDateが適切に設定されるようにしてください
// - 顧客ごとに1行だけが存在するようにしてください

display(table("updates"))

// COMMAND ----------

// MAGIC %md 
// MAGIC ## SCD Type 2を実行するためのMerge文
// MAGIC 
// MAGIC このMerge文はソーステーブルのそれぞれの顧客に対して以下の処理を同時に行います。
// MAGIC 
// MAGIC - `current`がtrueに設定された新規アドレスをinsert
// MAGIC - 以前の現在行の`current`　をfalseにupdateし、`endDate`を`null`からソースの`effectiveDate`にupdate

// COMMAND ----------

// DBTITLE 1,SQL example
// MAGIC %sql 
// MAGIC 
// MAGIC -- ========================================
// MAGIC -- Merge SQL APIはDBR 5.1以降で利用できます
// MAGIC -- ========================================
// MAGIC 
// MAGIC MERGE INTO customers
// MAGIC USING (
// MAGIC    -- これらの行は既存顧客の現在の住所をUPDATEし、新規顧客の新規住所をINSERTします
// MAGIC   SELECT updates.customerId as mergeKey, updates.*
// MAGIC   FROM updates
// MAGIC   
// MAGIC   UNION ALL
// MAGIC   
// MAGIC   -- これらの行は既存顧客の新規住所をINSERTします 
// MAGIC   -- mergeKeyをNULLに設定すると、これらの行はNOT MATCHとなりINSERTが強制されます
// MAGIC   SELECT NULL as mergeKey, updates.*
// MAGIC   FROM updates JOIN customers
// MAGIC   ON updates.customerid = customers.customerid 
// MAGIC   WHERE customers.current = true AND updates.address <> customers.address 
// MAGIC   
// MAGIC ) staged_updates
// MAGIC ON customers.customerId = mergeKey
// MAGIC WHEN MATCHED AND customers.current = true AND customers.address <> staged_updates.address THEN  
// MAGIC   UPDATE SET current = false, endDate = staged_updates.effectiveDate    -- currentをfalse、endDateをソースのeffective dateに設定します
// MAGIC WHEN NOT MATCHED THEN 
// MAGIC   INSERT(customerid, address, current, effectivedate, enddate) 
// MAGIC   VALUES(staged_updates.customerId, staged_updates.address, true, staged_updates.effectiveDate, null) -- currentをtrue、新規の住所とeffective dateに設定します

// COMMAND ----------

// DBTITLE 1,Scala example
// MAGIC %scala
// MAGIC // ==========================================
// MAGIC // Merge Scala API is available since DBR 6.0
// MAGIC // ==========================================
// MAGIC 
// MAGIC import io.delta.tables._
// MAGIC 
// MAGIC val customersTable: DeltaTable =   // table with schema (customerId, address, current, effectiveDate, endDate)
// MAGIC   DeltaTable.forName("customers")
// MAGIC 
// MAGIC val updatesDF = table("updates")          // DataFrame with schema (customerId, address, effectiveDate)
// MAGIC 
// MAGIC // Rows to INSERT new addresses of existing customers
// MAGIC val newAddressesToInsert = updatesDF
// MAGIC   .as("updates")
// MAGIC   .join(customersTable.toDF.as("customers"), "customerid")
// MAGIC   .where("customers.current = true AND updates.address <> customers.address")
// MAGIC 
// MAGIC // Stage the update by unioning two sets of rows
// MAGIC // 1. Rows that will be inserted in the `whenNotMatched` clause
// MAGIC // 2. Rows that will either UPDATE the current addresses of existing customers or INSERT the new addresses of new customers
// MAGIC val stagedUpdates = newAddressesToInsert
// MAGIC   .selectExpr("NULL as mergeKey", "updates.*")   // Rows for 1.
// MAGIC   .union(
// MAGIC     updatesDF.selectExpr("updates.customerId as mergeKey", "*")  // Rows for 2.
// MAGIC   )
// MAGIC 
// MAGIC // Apply SCD Type 2 operation using merge
// MAGIC customersTable
// MAGIC   .as("customers")
// MAGIC   .merge(
// MAGIC     stagedUpdates.as("staged_updates"),
// MAGIC     "customers.customerId = mergeKey")
// MAGIC   .whenMatched("customers.current = true AND customers.address <> staged_updates.address")
// MAGIC   .updateExpr(Map(                                      // Set current to false and endDate to source's effective date.
// MAGIC     "current" -> "false",
// MAGIC     "endDate" -> "staged_updates.effectiveDate"))
// MAGIC   .whenNotMatched()
// MAGIC   .insertExpr(Map(
// MAGIC     "customerid" -> "staged_updates.customerId",
// MAGIC     "address" -> "staged_updates.address",
// MAGIC     "current" -> "true",
// MAGIC     "effectiveDate" -> "staged_updates.effectiveDate",  // Set current to true along with the new address and its effective date.
// MAGIC     "endDate" -> "null"))
// MAGIC   .execute()

// COMMAND ----------

// MAGIC %md
// MAGIC ## 更新されたCustomersテーブル
// MAGIC 
// MAGIC - customer 1では以前の住所は`current = false`となり、新規の住所が`current = true`に設定されます
// MAGIC - customer 2では更新はありません
// MAGIC - customer 3では新規住所は以前の住所と同じであるので更新はされません
// MAGIC - customer 4では新規住所がinsertされます

// COMMAND ----------

display(table("customers").orderBy("customerId", "current", "endDate"))

// COMMAND ----------

// MAGIC %md
// MAGIC # END
