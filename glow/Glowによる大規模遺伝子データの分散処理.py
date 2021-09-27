# Databricks notebook source
# MAGIC %md
# MAGIC # Glowによる大規模遺伝子データの分散処理
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2021/07/13</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>クラスター</td><td>7.3 LTS ML(8.3MLでは動作未確認)</td></tr>
# MAGIC </table>
# MAGIC <img style="margin-top:25px;" src="https://sajpstorage.blob.core.windows.net/workshop20210205/databricks-logo-small-new.png" width="140">
# MAGIC 
# MAGIC **参考資料**
# MAGIC - [Glow V1\.0\.0 \- 次世代ゲノムワイド分析 \- Qiita](https://qiita.com/taka_yayoi/items/d218797152fa480b6673)
# MAGIC - [Get Started with Glow V1\.0\.0 \- For Next Generation Genome Wide Analytics \- The Databricks Blog](https://databricks.com/blog/2021/03/09/glow-v1-0-0-next-generation-genome-wide-analytics.html)
# MAGIC - [Glow — Glow documentation](https://glow.readthedocs.io/en/latest/index.html)
# MAGIC - [Glow \| Databricks on AWS](https://docs.databricks.com/applications/genomics/genomics-libraries/glow.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 事前準備
# MAGIC 
# MAGIC **Glowのインストール** クラスターライブラリとして以下の二つをインストールしてください。
# MAGIC - Maven: io.projectglow:glow-spark3_2.12:1.0.0
# MAGIC - PyPI: glow.py==1.0.0

# COMMAND ----------

# MAGIC %md
# MAGIC ## 人口規模のGWAS(ゲノムワイド関連研究)におけるエンジニアリング

# COMMAND ----------

# MAGIC %md
# MAGIC ゲノムワイド関連研究(GWAS)は遺伝子の変数と、関心のある疾病、特性との相関を導き出します。
# MAGIC 
# MAGIC 対照集団は数百万規模に増加しているため、頑健性をもって大規模GWASエンジニアリングを行うための手段が必要となっています。このため、Glowを用いてSparkネイティブの拡張可能な実装を開発したのです。
# MAGIC 
# MAGIC このノートブックでは[Delta Lake](https://delta.io)による高性能ビッグデータ蓄積機能と、[mlflow](https://mlflow.org/)によるパラメーター、メトリクス、グラフのトラッキング機能を活用しています。

# COMMAND ----------

import re
from pyspark.sql.types import * 

# ログインIDからUsernameを取得
username_raw = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
# Username の英数字以外を除去し、全て小文字化。Username をファイルパスやデータベース名の一部で使用可能にするため。
username = re.sub('[^A-Za-z0-9]+', '', username_raw).lower()

# ファイル格納パス
work_path = f"dbfs:/tmp/{username}/gwas"
work_path_from_local = f"/dbfs/tmp/{username}/gwas"

print("path: " + work_path)
print("path referred from local API: " + work_path_from_local)

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pyspark.sql.functions as fx
from pyspark.sql.types import StringType
from pyspark.ml.linalg import Vector, Vectors, SparseVector, DenseMatrix
from pyspark.ml.stat import Summarizer
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.util import MLUtils

from dataclasses import dataclass

import mlflow
import glow

# needed?
#mlflow.set_experiment(f"/Users/{username_raw}/20210713_glow/gwas")
spark = glow.register(spark)

# COMMAND ----------

# MAGIC %md
# MAGIC ### パラメータ

# COMMAND ----------

allele_freq_cutoff = 0.05
num_pcs = 5 # 主成分コンポーネントの数
mlflow.log_param("minor allele frequency cutoff", allele_freq_cutoff)
mlflow.log_param("principal components", num_pcs)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ファイルパス

# COMMAND ----------

# データソース
vcf_path = "/databricks-datasets/genomics/1kg-vcfs/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"
phenotype_path = "/databricks-datasets/genomics/1000G/phenotypes.normalized"
sample_info_path = "/databricks-datasets/genomics/1000G/samples/populations_1000_genomes_samples.csv"

# 出力先
delta_silver_path = f"{work_path}/gwas_test/snps.delta"
delta_gold_path = f"{work_path}/gwas_test/snps.qced.delta"
principal_components_path = f"{work_path_from_local}/gwas_test/pcs.csv"
gwas_results_path = f"{work_path}/gwas_test/gwas_results.delta"

# COMMAND ----------

# MAGIC %md
# MAGIC ### ヘルパー関数

# COMMAND ----------

def plot_layout(plot_title, plot_style, xlabel):
  plt.style.use(plot_style) #e.g. ggplot, seaborn-colorblind, print(plt.style.available)
  plt.title(plot_title)
  plt.xlabel(r'${0}$'.format(xlabel))
  plt.gca().spines['right'].set_visible(False)
  plt.gca().spines['top'].set_visible(False)
  plt.gca().yaxis.set_ticks_position('left')
  plt.gca().xaxis.set_ticks_position('bottom')
  plt.tight_layout()
  
def plot_histogram(df, col, xlabel, xmin, xmax, nbins, plot_title, plot_style, color, vline, out_path):
  plt.close()
  plt.figure()
  bins = np.linspace(xmin, xmax, nbins)
  df = df.toPandas()
  plt.hist(df[col], bins, alpha=1, color=color)
  if vline:
    plt.axvline(x=vline, linestyle='dashed', linewidth=2.0, color='black')
  plot_layout(plot_title, plot_style, xlabel)
  plt.savefig(out_path)
  plt.show()
  
def calculate_pval_bonferroni_cutoff(df, cutoff=0.05):
  bonferroni_p =  cutoff / df.count()
  return bonferroni_p

def get_sample_info(vcf_df, sample_metadata_df):
  """
  get sample IDs from VCF dataframe, index them, then join to sample metadata dataframe
  """
  sample_id_list = vcf_df.limit(1).select("genotypes.sampleId").collect()[0].__getitem__("sampleId")
  sample_id_indexed = spark.createDataFrame(sample_id_list, StringType()). \
                            coalesce(1). \
                            withColumnRenamed("value", "Sample"). \
                            withColumn("index", fx.monotonically_increasing_id())
  sample_id_annotated = sample_id_indexed.join(sample_metadata_df, "Sample")
  return sample_id_annotated

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 1000のVCF遺伝子データをDelta Lakeに取り込む
# MAGIC 
# MAGIC Variant call format (VCF)ファイルをSparkのデータソースとしてクラウドストレージから直接読み込めるGlowのVCFリーダーを用いることで、遺伝子型データを、ACIDトランザクションを有する高性能ビッグデータストアDelta Lakeに書き込みます。Delta Lakeはデータを構造化、インデックス付け、圧縮することで、増加し続ける遺伝子データに対して高性能、高信頼の処理を実現します。

# COMMAND ----------

vcf_view_unsplit = (spark.read.format("vcf")
   .option("flattenInfoFields", "false")
   .load(vcf_path))

# COMMAND ----------

# MAGIC %md Split multiallelics variants to biallelics

# COMMAND ----------

vcf_view = glow.transform("split_multiallelics", vcf_view_unsplit)

# COMMAND ----------

display(vcf_view.withColumn("genotypes", fx.col("genotypes")[0]))

# COMMAND ----------

# MAGIC %md
# MAGIC **注意** ここでは、Glowに組み込まれている`call_summary_stats`と`hardy_weinberg`を用いてバリアントワイズのサマリー統計情報とハーディー・ワインベルグ平衡のP値を計算します。

# COMMAND ----------

# MAGIC %md
# MAGIC **注意** 以下の処理はi3.xlargeの8台ワーカー構成で約19分かかります。

# COMMAND ----------

(vcf_view
  .select(
    fx.expr("*"),
    glow.expand_struct(glow.call_summary_stats(fx.col("genotypes"))),
    glow.expand_struct(glow.hardy_weinberg(fx.col("genotypes")))
  )
  .write
  .mode("overwrite")
  .format("delta")
  .save(delta_silver_path))

# COMMAND ----------

# MAGIC %md
# MAGIC Delta Lakeに関連づけられているメタデータはトランザクションログに直接保存されているので、Delta Lakeのサイズを容易に計算することができ、これをMLflowに記録することができます。

# COMMAND ----------

num_variants = spark.read.format("delta").load(delta_silver_path).count()
mlflow.log_metric("Number Variants pre-QC", num_variants)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 品質管理の実行
# MAGIC 
# MAGIC ハーディー・ワインベルグ平衡のP値と対立遺伝子頻度に対するバリアントワイズのフィルタリングを行います。

# COMMAND ----------

hwe = (spark.read.format("delta")
                 .load(delta_silver_path)
                 .where((fx.col("alleleFrequencies").getItem(0) >= allele_freq_cutoff) & 
                       (fx.col("alleleFrequencies").getItem(0) <= (1.0 - allele_freq_cutoff)))
                 .withColumn("log10pValueHwe", fx.when(fx.col("pValueHwe") == 0, 26).otherwise(-fx.log10(fx.col("pValueHwe")))))

# COMMAND ----------

hwe_cutoff = calculate_pval_bonferroni_cutoff(hwe)
mlflow.log_param("Hardy-Weinberg P value cutoff", hwe_cutoff)

# COMMAND ----------

display(plot_histogram(df=hwe.select("log10pValueHwe"), 
                       col="log10pValueHwe",
                       xlabel='-log_{10}(P)',
                       xmin=0, 
                       xmax=25, 
                       nbins=50, 
                       plot_title="hardy-weinberg equilibrium", 
                       plot_style="ggplot",
                       color='#e41a1c',
                       vline = -np.log10(hwe_cutoff),
                       out_path = "/databricks/driver/hwe.png"
                      )
       )

# COMMAND ----------

mlflow.log_artifact("/databricks/driver/hwe.png")

# COMMAND ----------

(spark.read.format("delta")
   .load(delta_silver_path)
   .where((fx.col("alleleFrequencies").getItem(0) >= allele_freq_cutoff) & 
         (fx.col("alleleFrequencies").getItem(0) <= (1.0 - allele_freq_cutoff)) &
         (fx.col("pValueHwe") >= hwe_cutoff))
   .write
   .mode("overwrite")
   .format("delta")
   .save(delta_gold_path))

# COMMAND ----------

num_variants = spark.read.format("delta").load(delta_gold_path).count()
mlflow.log_metric("Number Variants post-QC", num_variants)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 主成分分析(PCA)の実行
# MAGIC 
# MAGIC GAWSにおける系統を制御するためのものです。
# MAGIC 
# MAGIC 注意: `array_to_sparse_vector`はGlowに組み込まれている関数です。

# COMMAND ----------

vectorized = (spark.read.format("delta")
                        .load(delta_gold_path)
                        .select(glow.array_to_sparse_vector(glow.genotype_states(fx.col("genotypes"))).alias("features"))
                        .cache())

# COMMAND ----------

# MAGIC %md
# MAGIC ### スパースなベクトルにおいて主成分分析を行うために`pyspark.ml`を使用します

# COMMAND ----------

matrix = RowMatrix(MLUtils.convertVectorColumnsFromML(vectorized, "features").rdd.map(lambda x: x.features))
pcs = matrix.computeSVD(num_pcs)

# COMMAND ----------

pd.DataFrame(pcs.V.toArray()).to_csv(principal_components_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### サンプル情報を読み込み、主成分のプロットを出力
# MAGIC 
# MAGIC ここでは、インデックスで両方のデータフレームを結合し、サンプル情報に対して主成分の注釈を付与します。
# MAGIC 
# MAGIC **注意** インデックスはSpark SQLの`monotonically_increasing_id()`関数を用いて付与します。

# COMMAND ----------

pcs_df = spark.createDataFrame(pcs.V.toArray().tolist(), ["pc" + str(i) for i in range(num_pcs)])

# COMMAND ----------

sample_metadata = spark.read.option("header", True).csv(sample_info_path)
sample_info = get_sample_info(vcf_view, sample_metadata)
sample_count = sample_info.count()
mlflow.log_param("number of samples", sample_count)
pcs_indexed = pcs_df.coalesce(1).withColumn("index", fx.monotonically_increasing_id())
pcs_with_samples = pcs_indexed.join(sample_info, "index")

# COMMAND ----------

# MAGIC %md
# MAGIC ### pc1とpc2の散布図プロットを作成するためにdisplay関数を使用します
# MAGIC 
# MAGIC **注意** ここではchromosome 22しか分析していないため、PCAの散布図は総数と全体のゲノムデータを区別しません。

# COMMAND ----------

display(pcs_with_samples)

# COMMAND ----------

# MAGIC %md
# MAGIC ### GAWSのデータ準備

# COMMAND ----------

phenotype_df = pd.read_parquet('/dbfs/' + phenotype_path).explode('values').rename({'values': 'bmi'}, axis='columns').reset_index(drop=True)
del phenotype_df['phenotype']
phenotype_df

# COMMAND ----------

covariate_df = pd.read_csv(principal_components_path)

# COMMAND ----------

phenotype = phenotype_df.columns[0]
mlflow.log_param("phenotype", phenotype)

# COMMAND ----------

genotypes = spark.read.format("delta").load(delta_gold_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### `linear_regression`の実行
# MAGIC 
# MAGIC **注意** `genotype_states`は遺伝子型の配列を変換するGlowのユーティリティ関数です。例えば、`[0,1]`を変異型を含む数(`1`)に変換します。

# COMMAND ----------

results = glow.gwas.linear_regression(
  genotypes.select('contigName', 'start', 'names', 'genotypes'),
  phenotype_df,
  covariate_df,
  values_column=glow.genotype_states(fx.col('genotypes'))
)

(results.write
  .format("delta")
  .mode("overwrite")
  .save(gwas_results_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 結果の表示

# COMMAND ----------

display(spark.read.format("delta").load(gwas_results_path).limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC ### GWASの結果をRで読み込み、`qqman`ライブラリを用いてプロットします

# COMMAND ----------

# MAGIC %r
# MAGIC library(SparkR)
# MAGIC # Cmd10で指定したgwas_results_pathのパスを設定してください
# MAGIC gwas_df <- read.df("/tmp/takaakiyayoidatabrickscom/gwas/gwas_test/gwas_results.delta", source="delta")
# MAGIC gwas_results <- select(gwas_df, c(cast(alias(gwas_df$contigName, "CHR"), "double"), alias(gwas_df$start, "BP"), alias(gwas_df$pValue, "P"), alias(element_at(gwas_df$names, 1L), "SNP")))
# MAGIC gwas_results_rdf <- as.data.frame(gwas_results)

# COMMAND ----------

# MAGIC %r
# MAGIC install.packages("qqman", repos="http://cran.us.r-project.org")
# MAGIC library(qqman)

# COMMAND ----------

# MAGIC %r
# MAGIC png('/databricks/driver/manhattan.png')
# MAGIC manhattan(gwas_results_rdf, 
# MAGIC           col = c("#228b22", "#6441A5"), 
# MAGIC           chrlabs = NULL,
# MAGIC           suggestiveline = -log10(1e-05), 
# MAGIC           genomewideline = -log10(5e-08),
# MAGIC           highlight = NULL, 
# MAGIC           logp = TRUE, 
# MAGIC           annotatePval = NULL, 
# MAGIC           ylim=c(0,17))
# MAGIC dev.off()

# COMMAND ----------

# MAGIC %r
# MAGIC manhattan(gwas_results_rdf, col = c("#228b22", "#6441A5"), chrlabs = NULL,
# MAGIC suggestiveline = -log10(1e-05), genomewideline = -log10(5e-08),
# MAGIC highlight = NULL, logp = TRUE, annotatePval = NULL, ylim=c(0,17))

# COMMAND ----------

mlflow.log_artifact('/databricks/driver/manhattan.png')

# COMMAND ----------

# MAGIC %r
# MAGIC png('/databricks/driver/qqplot.png')
# MAGIC qq(gwas_results_rdf$P)
# MAGIC dev.off()

# COMMAND ----------

# MAGIC %r
# MAGIC qq(gwas_results_rdf$P)

# COMMAND ----------

mlflow.log_artifact('/databricks/driver/qqplot.png')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean up

# COMMAND ----------

dbutils.fs.rm(f"{work_path}/gwas_test", True)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
