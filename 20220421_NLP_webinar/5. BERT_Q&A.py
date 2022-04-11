# Databricks notebook source
# MAGIC %md
# MAGIC # BERTを用いたチャットボット
# MAGIC 
# MAGIC - [自然言語処理\(BERT\)を用いたチャットボット \- Qiita](https://qiita.com/sugiyama404/items/7691d7ff6a5b8c24eddf)
# MAGIC - [Huggingface Transformers 入門 \(14\) \- 日本語の質問応答の学習｜npaka｜note](https://note.com/npaka/n/na8721fdc3e24)
# MAGIC - [Driving domain QA datasets \- KUROHASHI\-CHU\-MURAWAKI LAB](https://nlp.ist.i.kyoto-u.ac.jp/index.php?Driving%20domain%20QA%20datasets)
# MAGIC 
# MAGIC **推奨** GPUクラスター

# COMMAND ----------

# MAGIC %md
# MAGIC ## ライブラリのインストール

# COMMAND ----------

# MAGIC %sh
# MAGIC git clone https://github.com/huggingface/transformers
# MAGIC cd transformers
# MAGIC pip install .
# MAGIC pip install -r requirements.txt
# MAGIC pip install fugashi[unidic-lite] ipadic datasets

# COMMAND ----------

# MAGIC %md
# MAGIC 事前にDBFSにアップロードしておいたDDQAをコピー

# COMMAND ----------

# MAGIC %fs
# MAGIC cp dbfs:/FileStore/shared_uploads/takaaki.yayoi@databricks.com/DDQA_1_0_tar.gz file:/tmp

# COMMAND ----------

# MAGIC %sh
# MAGIC tar -zxvf /tmp/DDQA_1_0_tar.gz

# COMMAND ----------

# MAGIC %md
# MAGIC ## トレーニング
# MAGIC 
# MAGIC トレーニングします。GPUマシンで3.6時間かかります。

# COMMAND ----------

# MAGIC %sh
# MAGIC cd transformers
# MAGIC python ./examples/legacy/question-answering/run_squad.py \
# MAGIC     --model_type=bert \
# MAGIC     --model_name_or_path=cl-tohoku/bert-base-japanese-whole-word-masking \
# MAGIC     --do_train \
# MAGIC     --do_eval \
# MAGIC     --train_file=/databricks/driver/DDQA-1.0/RC-QA/DDQA-1.0_RC-QA_train.json \
# MAGIC     --predict_file=/databricks/driver/DDQA-1.0/RC-QA/DDQA-1.0_RC-QA_dev.json \
# MAGIC     --per_gpu_train_batch_size 12 \
# MAGIC     --learning_rate 3e-5 \
# MAGIC     --num_train_epochs 10 \
# MAGIC     --max_seq_length 384 \
# MAGIC     --doc_stride 128 \
# MAGIC     --overwrite_output_dir \
# MAGIC     --output_dir output/

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /databricks/driver/transformers/output

# COMMAND ----------

# MAGIC %md
# MAGIC ### バックアップ

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir /tmp/bert_qa_ja_output

# COMMAND ----------

# MAGIC %sh
# MAGIC cp /databricks/driver/transformers/output/* /tmp/bert_qa_ja_output

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /tmp/bert_qa_ja_output

# COMMAND ----------

# MAGIC %sh
# MAGIC cd /tmp
# MAGIC zip -r bert_qa_ja_output.zip /tmp/bert_qa_ja_output

# COMMAND ----------

# MAGIC %fs
# MAGIC cp file:/tmp/bert_qa_ja_output.zip dbfs:/FileStore/shared_uploads/takaaki.yayoi@databricks.com/

# COMMAND ----------

# MAGIC %md
# MAGIC ### レストア

# COMMAND ----------

# MAGIC %sh
# MAGIC #rm /tmp/bert_qa_ja_output.zip
# MAGIC #rm -r /tmp/bert_qa_ja_output

# COMMAND ----------

# MAGIC %fs
# MAGIC cp dbfs:/FileStore/shared_uploads/takaaki.yayoi@databricks.com/bert_qa_ja_output.zip file:/tmp/

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /tmp

# COMMAND ----------

# MAGIC %sh
# MAGIC unzip /tmp/bert_qa_ja_output.zip

# COMMAND ----------

# MAGIC %md
# MAGIC ## 回答

# COMMAND ----------

from transformers import BertJapaneseTokenizer, AutoModelForQuestionAnswering
import torch

model = AutoModelForQuestionAnswering.from_pretrained('tmp/bert_qa_ja_output/')  # 適宜変更してください
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking') 

def reply(context, question):

    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    output = model(**inputs)
    answer_start = torch.argmax(output.start_logits)  
    answer_end = torch.argmax(output.end_logits) + 1 
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    answer = answer.replace(' ', '')
    return answer

# COMMAND ----------

context  = 'データブリックスは、学術界とオープンソースコミュニティをルーツとするデータ＋AIの企業です。Apache Spark™、Delta Lake、MLflowの開発者グループによる2013年の創業以来、最新のレイクハウスアーキテクチャを基盤に、データウェアハウスとデータレイクの優れた機能を取り入れた、データとAIのためのクラウドベースのオープンな統合プラットフォームを提供しています。'

# COMMAND ----------

question = "あなたの名前は？"
answer = reply(context, question)

print("question: " + question)
print("answer: " + answer)

# COMMAND ----------

question = "何を提供している？"
answer = reply(context, question)

print("question: " + question)
print("answer: " + answer)

# COMMAND ----------

question = "いつ創業？"
answer = reply(context, question)

print("question: " + question)
print("answer: " + answer)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
