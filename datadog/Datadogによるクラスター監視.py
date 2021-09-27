# Databricks notebook source
# MAGIC %md ## DatadogエージェントのインストールとSparkとシステムのモニタリング
# MAGIC 
# MAGIC ここで作成するinit scriptをクラスターに設定することで、Datadogエージェントをインストールし、SparkメトリクスとログをDatadogに送信するようになります。<br><br>
# MAGIC 
# MAGIC 1. [Datadog](https://www.datadoghq.com/)でアカウントを作成し、APIキーを取得します。以下の画面の`DD_API_KEY=`以降がAPIキーです。
# MAGIC ![](https://sajpstorage.blob.core.windows.net/demo20210501-datadog/get-API-key.png)
# MAGIC 1. DBFS上にinit scriptを作成するので、以下のセルの4行目の`<init-script-folder>`に格納先を指定してください。
# MAGIC 1. 以下のセルを実行し、`datadog-install-driver-workers.sh`を作成します。
# MAGIC 1. クラスターのinit scriptとして、`datadog-install-driver-workers.sh`を指定します。
# MAGIC ![](https://sajpstorage.blob.core.windows.net/demo20210501-datadog/set-init-script.png)
# MAGIC 1. クラスターの**Advanced Options**の**Spark > Environment Variables**にDatadogのAPIキーを、`DD_API_KEY=<your-api-key>`の形式で指定します。
# MAGIC ![](https://sajpstorage.blob.core.windows.net/demo20210501-datadog/Env.png)
# MAGIC 1. クラスターの起動に合わせてエージェントが起動し、Datadogにメトリクスとログが送信されます。
# MAGIC ![](https://sajpstorage.blob.core.windows.net/demo20210501-datadog/Dashboards.png)
# MAGIC ![](https://sajpstorage.blob.core.windows.net/demo20210501-datadog/Datadog-logs.png)
# MAGIC 
# MAGIC Databricks参考情報
# MAGIC - [Manage clusters \| Databricks on AWS](https://docs.databricks.com/clusters/clusters-manage.html#view-cluster-logs)
# MAGIC - [Apache Spark Cluster Monitoring with Databricks and Datadog \- The Databricks Blog](https://databricks.com/blog/2017/06/01/apache-spark-cluster-monitoring-with-databricks-and-datadog.html)
# MAGIC 
# MAGIC Datadog参考情報
# MAGIC - [Databricks](https://docs.datadoghq.com/integrations/databricks/?tab=driveronly)
# MAGIC - 最新のinit scriptはこちらから取得できます。 [Spark](https://docs.datadoghq.com/integrations/spark/?tab=host)

# COMMAND ----------

# MAGIC %md デフォルトではログが出力されないので、`echo "logs_enabled: true" >> /etc/datadog-agent/datadog.yaml`でログを有効化する必要があります(31行目)。

# COMMAND ----------

# 例
# dbfs:/FileStore/shared_uploads/takaaki.yayoi@databricks.com/datadog-install-driver-workers.sh
  
dbutils.fs.put("dbfs:/<init-script-folder>/datadog-install-driver-workers.sh","""
#!/bin/bash
cat <<EOF >> /tmp/start_datadog.sh

#!/bin/bash
  
  hostip=$(hostname -I | xargs)

if [[ \${DB_IS_DRIVER} = "TRUE" ]]; then

  echo "Installing Datadog agent in the driver (master node) ..."
  # CONFIGURE HOST TAGS FOR DRIVER
  DD_TAGS="environment:\${DD_ENV}","databricks_cluster_id:\${DB_CLUSTER_ID}","databricks_cluster_name:\${DB_CLUSTER_NAME}","spark_host_ip:\${SPARK_LOCAL_IP}","spark_node:driver"

  # INSTALL THE LATEST DATADOG AGENT 7 ON DRIVER AND WORKER NODES
  DD_AGENT_MAJOR_VERSION=7 DD_API_KEY=\$DD_API_KEY DD_HOST_TAGS=\$DD_TAGS bash -c "\$(curl -L https://s3.amazonaws.com/dd-agent/scripts/install_script.sh)"
  
  # WAIT FOR DATADOG AGENT TO BE INSTALLED
  while [ -z \$datadoginstalled ]; do
    if [ -e "/etc/datadog-agent/datadog.yaml" ]; then
      datadoginstalled=TRUE
    fi
    sleep 2
  done
  echo "Datadog Agent is installed"

  # ENABLE LOGS IN datadog.yaml TO COLLECT DRIVER LOGS
  echo "logs_enabled: true" >> /etc/datadog-agent/datadog.yaml

  while [ -z \$gotparams ]; do
    if [ -e "/tmp/driver-env.sh" ]; then
      DB_DRIVER_PORT=\$(grep -i "CONF_UI_PORT" /tmp/driver-env.sh | cut -d'=' -f2)
      gotparams=TRUE
    fi
    sleep 2
  done

  # WRITING CONFIG FILE FOR SPARK INTEGRATION WITH STRUCTURED STREAMING METRICS ENABLED
  # MODIFY TO INCLUDE OTHER OPTIONS IN spark.d/conf.yaml.example
  echo "init_config:
instances:
    - spark_url: http://\${DB_DRIVER_IP}:\${DB_DRIVER_PORT}
      spark_cluster_mode: spark_driver_mode
      cluster_name: \${hostip}
      streaming_metrics: true
logs:
    - type: file
      path: /databricks/driver/logs/*.log
      source: databricks
      service: databricks
      log_processing_rules:
        - type: multi_line
          name: new_log_start_with_date
          pattern: \d{2,4}[\-\/]\d{2,4}[\-\/]\d{2,4}.*" > /etc/datadog-agent/conf.d/spark.yaml
else

  # CONFIGURE HOST TAGS FOR WORKERS
  DD_TAGS="environment:\${DD_ENV}","databricks_cluster_id:\${DB_CLUSTER_ID}","databricks_cluster_name:\${DB_CLUSTER_NAME}","spark_host_ip:\${SPARK_LOCAL_IP}","spark_node:worker"

  # INSTALL THE LATEST DATADOG AGENT 7 ON DRIVER AND WORKER NODES
  DD_AGENT_MAJOR_VERSION=7 DD_API_KEY=\$DD_API_KEY DD_HOST_TAGS=\$DD_TAGS bash -c "\$(curl -L https://s3.amazonaws.com/dd-agent/scripts/install_script.sh)"

fi

  # RESTARTING AGENT
  sudo service datadog-agent restart
EOF

# CLEANING UP
chmod a+x /tmp/start_datadog.sh
/tmp/start_datadog.sh >> /tmp/datadog_start.log 2>&1 & disown
""", True)
