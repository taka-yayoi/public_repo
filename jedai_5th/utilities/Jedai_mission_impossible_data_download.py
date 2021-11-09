# Databricks notebook source
# MAGIC %sh mkdir data
# MAGIC mkdir data/latest
# MAGIC mkdir model
# MAGIC curl -o data/dns_events.json https://raw.githubusercontent.com/zaferbil/dns-notebook-datasets/master/data/dns_events.json 
# MAGIC curl -o data/GeoLite2_City.mmdb https://raw.githubusercontent.com/zaferbil/dns-notebook-datasets/master/data/GeoLite2_City.mmdb
# MAGIC curl -o data/ThreatDataFeed.txt https://raw.githubusercontent.com/zaferbil/dns-notebook-datasets/master/data/ThreatDataFeed.txt
# MAGIC curl -o data/alexa_100k.txt https://raw.githubusercontent.com/zaferbil/dns-notebook-datasets/master/data/alexa_100k.txt
# MAGIC curl -o data/dga_domains_header.txt https://raw.githubusercontent.com/zaferbil/dns-notebook-datasets/master/data/dga_domains_header.txt
# MAGIC curl -o data/domains_dnstwists.csv https://raw.githubusercontent.com/zaferbil/dns-notebook-datasets/master/data/domains_dnstwists.csv
# MAGIC curl -o data/words.txt https://raw.githubusercontent.com/zaferbil/dns-notebook-datasets/master/data/words.txt
# MAGIC curl -o data/latest/dns_test_1.json https://raw.githubusercontent.com/zaferbil/dns-notebook-datasets/master/data/dns_test_1.json
# MAGIC curl -o model/MLmodel https://raw.githubusercontent.com/zaferbil/dns-notebook-datasets/master/model/MLmodel
# MAGIC curl -o model/conda.yaml https://raw.githubusercontent.com/zaferbil/dns-notebook-datasets/master/model/conda.yaml
# MAGIC curl -o model/python_model.pkl https://raw.githubusercontent.com/zaferbil/dns-notebook-datasets/master/model/python_model.pkl

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /databricks/driver/data

# COMMAND ----------

# ダウンロードしたデータをワークスペースのDBFSにコピーします
dbutils.fs.cp("file:///databricks/driver/data", f"dbfs:{work_path}/tables/datasets/", True)
dbutils.fs.cp("file:///databricks/driver/model", f"dbfs:{work_path}/tables/model/", True)
