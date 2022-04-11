# Databricks notebook source
# MAGIC %md
# MAGIC # BERT
# MAGIC 
# MAGIC - [BERTの日本語事前学習済みモデルをGoogle Colaboratoryで手軽に試す方法 \- Qiita](https://qiita.com/karaage0703/items/30485c2ba1c396760982)
# MAGIC - [【PyTorch】BERTの使い方 \- 日本語pre\-trained modelsをfine tuningして分類問題を解く \- Qiita](https://qiita.com/kenta1984/items/7f3a5d859a15b20657f3)
# MAGIC - [BERTについての簡単な解説と使い方 \- Qiita](https://qiita.com/kuroneko2828/items/9310e5e2c7686315f7be)
# MAGIC - [汎用言語表現モデルBERTの内部動作を解明してみる \- Qiita](https://qiita.com/Kosuke-Szk/items/d49e2127bf95a1a8e19f)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ライブラリのインストール

# COMMAND ----------

# MAGIC %pip install fugashi[unidic-lite] ipadic

# COMMAND ----------

import torch
from transformers import BertJapaneseTokenizer, BertForMaskedLM

# COMMAND ----------

# MAGIC %md
# MAGIC ## 文脈から単語を予測(その1)

# COMMAND ----------

tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

text = '父の父は、祖父。'
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)

# COMMAND ----------

# `BertForMaskedLM`を用いて予測するトークンをマスキング
masked_index = 2
tokenized_text[masked_index] = '[MASK]'
print(tokenized_text)

# トークンをボキャブラリーのインデックスに変換
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
print(indexed_tokens)

# 入力をPyTorchのtensorに変換
tokens_tensor = torch.tensor([indexed_tokens])
print(tokens_tensor)

# COMMAND ----------

# 事前学習済みモデルのロード
model_mask = BertForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
model_mask.eval()

# COMMAND ----------

# 予測
with torch.no_grad():
    outputs = model_mask(tokens_tensor)
    predictions = outputs[0][0, masked_index].topk(5) # 予測結果の上位5件を抽出

# 結果の表示
for i, index_t in enumerate(predictions.indices):
    index = index_t.item()
    token = tokenizer.convert_ids_to_tokens([index])[0]
    print(i, token)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 文脈から単語を予測(その2)

# COMMAND ----------

import torch
from transformers import BertJapaneseTokenizer, BertForMaskedLM

# 事前学習済みトークナイザーのロード
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

# 入力をトークン化
text = 'テレビでサッカーの試合を見る。'
tokenized_text = tokenizer.tokenize(text)
# ['テレビ', 'で', 'サッカー', 'の', '試合', 'を', '見る', '。']

# `BertForMaskedLM`を用いて予測するトークンをマスキング
masked_index = 2
tokenized_text[masked_index] = '[MASK]'
# ['テレビ', 'で', '[MASK]', 'の', '試合', 'を', '見る', '。']

# トークンをボキャブラリーのインデックスに変換
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# [571, 12, 4, 5, 608, 11, 2867, 8]

# 入力をPyTorchのtensorに変換
tokens_tensor = torch.tensor([indexed_tokens])
# tensor([[ 571,   12,    4,    5,  608,   11, 2867,    8]])

# 事前学習済みモデルのロード
model = BertForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
model.eval()

# 予測
with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0][0, masked_index].topk(5) # 予測結果の上位5件を抽出

# 結果の表示
for i, index_t in enumerate(predictions.indices):
    index = index_t.item()
    token = tokenizer.convert_ids_to_tokens([index])[0]
    print(i, token)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
