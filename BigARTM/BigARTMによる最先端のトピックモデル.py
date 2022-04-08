# Databricks notebook source
# MAGIC %md
# MAGIC # LDAの先へ：BigARTMによる最先端のトピックモデル
# MAGIC 
# MAGIC [LDAの先へ：BigARTMによる最先端のトピックモデル \- Qiita](https://qiita.com/taka_yayoi/items/622bedee5231ebadb8d8)

# COMMAND ----------

# MAGIC %md
# MAGIC ## BigARTMのインストール

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install bigartm

# COMMAND ----------

import artm

# COMMAND ----------

artm.version()

# COMMAND ----------

# MAGIC %md
# MAGIC ## BigARTMで使用するデータセットのダウンロード

# COMMAND ----------

# MAGIC %sh 
# MAGIC wget https://s3-eu-west-1.amazonaws.com/artm/docword.kos.txt.gz
# MAGIC wget https://s3-eu-west-1.amazonaws.com/artm/vocab.kos.txt
# MAGIC gunzip docword.kos.txt.gz

# COMMAND ----------

# MAGIC %md
# MAGIC ## はじめにLDAをデータセットに適用してみます

# COMMAND ----------

import artm
batch_vectorizer = artm.BatchVectorizer(data_path='.', data_format='bow_uci',collection_name='kos', target_folder='kos_batches')

# COMMAND ----------

lda = artm.LDA(num_topics=15, alpha=0.01, beta=0.001, cache_theta=True, num_document_passes=5, dictionary=batch_vectorizer.dictionary)
lda.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=10)

# COMMAND ----------

lda.perplexity_value

# COMMAND ----------

top_tokens = lda.get_top_tokens(num_tokens=10)
for i, token_list in enumerate(top_tokens):
    print('Topic #{0}: {1}'.format(i, token_list))

# COMMAND ----------

phi = lda.phi_   # サイズはボキャブラリーのワードの数 x トピックの数となります
theta = lda.get_theta() # 行の数はトピックの数に対応します
print(phi)

# COMMAND ----------

print(theta)

# COMMAND ----------

dictionary = batch_vectorizer.dictionary
dictionary

# COMMAND ----------

# MAGIC %md
# MAGIC ## BigARTMとPLSAを使う

# COMMAND ----------

model_plsa = artm.ARTM(num_topics=15, cache_theta=True,
                       scores=[artm.PerplexityScore(name='PerplexityScore',
                                                    dictionary=dictionary)])

model_artm = artm.ARTM(num_topics=15, cache_theta=True,
                       scores=[artm.PerplexityScore(name='PerplexityScore',
                                                    dictionary=dictionary)],
                       regularizers=[artm.SmoothSparseThetaRegularizer(name='SparseTheta',
                                                                       tau=-0.15)])

model_plsa.scores.add(artm.TopTokensScore(name='TopTokensScore', num_tokens=6))
model_artm.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore'))
model_artm.scores.add(artm.TopicKernelScore(name='TopicKernelScore', probability_mass_threshold=0.3))
model_artm.scores.add(artm.TopTokensScore(name='TopTokensScore', num_tokens=6))
model_artm.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SparsePhi', tau=-0.1))
model_artm.regularizers.add(artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi', tau=1.5e+5))
model_artm.num_document_passes = 1
model_artm.initialize(dictionary=dictionary)
model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=15)

# COMMAND ----------

dir(model_artm)

# COMMAND ----------

model_artm.score_tracker['PerplexityScore'].value

# COMMAND ----------

model_artm.scores

# COMMAND ----------

model_artm.phi_

# COMMAND ----------

model_artm.get_theta()

# COMMAND ----------

model_artm.info

# COMMAND ----------

for topic_name in model_artm.topic_names:
    print(topic_name + ': ',model_artm.score_tracker['TopTokensScore'].last_tokens[topic_name])

# COMMAND ----------

# MAGIC %md
# MAGIC # END
