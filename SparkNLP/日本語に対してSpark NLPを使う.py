# Databricks notebook source
# MAGIC %md
# MAGIC # Spark NLP
# MAGIC 
# MAGIC - [Databricks_John_Snow_NLP_for_Healthcare_Solution_Sheet](https://databricks.com/it/wp-content/uploads/2021/07/Databricks_John_Snow_NLP_for_Healthcare_Solution_Sheet.pdf)
# MAGIC - [Spark-NLP:Getting started in Databricks](https://johnsnowlabs.github.io/spark-nlp-workshop/databricks/index.html)

# COMMAND ----------

# Install PySpark and Spark NLP
%pip install -q pyspark==3.1.2 spark-nlp

# Install Spark NLP Display lib
%pip install --upgrade -q spark-nlp-display

# COMMAND ----------

import sparknlp
from pyspark.ml import Pipeline
from sparknlp.annotator import *
from sparknlp.base import *
from sparknlp.training import *

# COMMAND ----------

# MAGIC %md
# MAGIC ## 日本語の分かち書き
# MAGIC 
# MAGIC [Word Segmenter for Japanese\- Spark NLP Model](https://nlp.johnsnowlabs.com/2021/03/09/wordseg_gsd_ud_ja.html)

# COMMAND ----------

document_assembler = DocumentAssembler() \
  .setInputCol("text") \
  .setOutputCol("document")

sentence_detector = SentenceDetector() \
  .setInputCols(["document"]) \
  .setOutputCol("sentence")

word_segmenter = WordSegmenterModel.pretrained("wordseg_gsd_ud", "ja").setInputCols(["sentence"]).setOutputCol("token")

pipeline = Pipeline(stages=[document_assembler, sentence_detector, word_segmenter])

ws_model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

example = spark.createDataFrame([['データブリックスは、学術界とオープンソースコミュニティをルーツとするデータ＋AIの企業です。']], ["text"])
result = ws_model.transform(example)

# COMMAND ----------

display(result)

# COMMAND ----------

result.createOrReplaceTempView("word_segmentation_result")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT token.result FROM word_segmentation_result;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 日本語品詞の特定
# MAGIC 
# MAGIC [Part of Speech for Japanese\- Spark NLP Model](https://nlp.johnsnowlabs.com/2021/03/09/pos_ud_gsd_ja.html)

# COMMAND ----------

document_assembler = DocumentAssembler() \
  .setInputCol("text") \
  .setOutputCol("document")

sentence_detector = SentenceDetector() \
  .setInputCols(["document"]) \
  .setOutputCol("sentence")

word_segmenter = WordSegmenterModel.pretrained("wordseg_gsd_ud", "ja")\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

pos_tagger = PerceptronModel.pretrained("pos_ud_gsd", "ja") \
  .setInputCols(["document", "token"]) \
  .setOutputCol("pos")

pipeline = Pipeline(stages=[
  document_assembler,
  sentence_detector,
  word_segmenter,
  pos_tagger
])

example = spark.createDataFrame([['データブリックスは、学術界とオープンソースコミュニティをルーツとするデータ＋AIの企業です。']], ["text"])

pos_result = pipeline.fit(example).transform(example)

# COMMAND ----------

display(pos_result.select("pos"))

# COMMAND ----------

pos_result.createOrReplaceTempView("pos_result")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT token.result, pos.result FROM pos_result;

# COMMAND ----------

# MAGIC %md
# MAGIC ## NER(Named Entity Extraction)
# MAGIC 
# MAGIC [Named Entity Recognition for Japanese \(GloVe 840B 300d\)\- Spark NLP Model](https://nlp.johnsnowlabs.com/2021/01/03/ner_ud_gsd_glove_840B_300d_ja.html)

# COMMAND ----------

document_assembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

sentence_detector = SentenceDetector() \
    .setInputCols('document') \
    .setOutputCol('sentence')

word_segmenter = WordSegmenterModel.pretrained("wordseg_gsd_ud", "ja")\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

embeddings = WordEmbeddingsModel.pretrained("glove_840B_300", "xx")\
          .setInputCols("document", "token") \
          .setOutputCol("embeddings")

ner = NerDLModel.pretrained("ner_ud_gsd_glove_840B_300d", "ja") \
        .setInputCols(["document", "token", "embeddings"]) \
        .setOutputCol("ner")

ner_converter = NerConverter() \
    .setInputCols(['sentence', 'token', 'ner']) \
    .setOutputCol('ner_chunk')

# COMMAND ----------

pipeline = Pipeline(stages=[document_assembler, sentence_detector, word_segmenter, embeddings, ner, ner_converter]) 

# COMMAND ----------

from sparknlp_display import NerVisualizer

def display_ner(text):
  example = spark.createDataFrame([[text]], ["text"])
  result = pipeline.fit(example).transform(example)
  #display(result)
  
  ner_vis = NerVisualizer().display(
    result = result.collect()[0],
    label_col = 'ner_chunk',
    document_col = 'document',
    return_html=True
  )

  displayHTML(ner_vis)

# COMMAND ----------

# MAGIC %md
# MAGIC [徳川家康 \- Wikipedia](https://ja.wikipedia.org/wiki/%E5%BE%B3%E5%B7%9D%E5%AE%B6%E5%BA%B7)

# COMMAND ----------

display_ner("徳川 家康（とくがわ いえやす、旧字体：德川 家康、1542年 - 1616年）は、戦国時代から江戸時代初期の日本の武将、戦国大名。松平広忠の長子。江戸幕府初代将軍。安祥松平家第9代当主で徳川家の始祖。幼名は竹千代（たけちよ）、諱は元信（もとのぶ）、元康（もとやす）、家康と改称。")

# COMMAND ----------

# MAGIC %md
# MAGIC # END
