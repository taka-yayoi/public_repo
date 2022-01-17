-- Databricks notebook source
CREATE LIVE TABLE raw_data
AS SELECT id, makeItSquared(id) AS numSquared FROM RANGE(10);

-- COMMAND ----------

CREATE LIVE TABLE squared_even
(CONSTRAINT even_only EXPECT (passOnlyEven(numSquared) = True) ON VIOLATION DROP ROW)
AS SELECT id, makeItSquared(id) AS numSquared FROM live.raw_data
