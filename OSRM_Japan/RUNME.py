# Databricks notebook source
# MAGIC %md This notebook sets up the companion cluster(s) to run the solution accelerator. It also creates the Workflow to create a Workflow DAG and illustrate the order of execution. Feel free to interactively run notebooks with the cluster or to run the Workflow to see how this solution accelerator executes. Happy exploring!
# MAGIC 
# MAGIC The pipelines, workflows and clusters created in this script are user-specific, so you can alter the workflow and cluster via UI without affecting other users. Running this script again after modification resets them.
# MAGIC 
# MAGIC **Note**: If the job execution fails, please confirm that you have set up other environment dependencies as specified in the accelerator notebooks. Accelerators sometimes require the user to set up additional cloud infra or data access, for instance. 

# COMMAND ----------

# DBTITLE 0,Install util packages
# MAGIC %pip install git+https://github.com/databricks-industry-solutions/notebook-solution-companion git+https://github.com/databricks-academy/dbacademy-rest git+https://github.com/databricks-academy/dbacademy-gems 

# COMMAND ----------

from solacc.companion import NotebookSolutionCompanion

# COMMAND ----------

job_json = {
        "timeout_seconds": 14400,
        "max_concurrent_runs": 1,
        "tags": {
            "usage": "solacc_testing",
            "group": "RCG"
        },
        "tasks": [
            {
                "job_cluster_key": "routing_cluster",
                "notebook_task": {
                    "notebook_path": f"00_ Introduction",
                    "base_parameters": {}
                },
                "task_key": "routing_00"
            },  
            {
                "job_cluster_key": "routing_cluster",
                "notebook_task": {
                    "notebook_path": f"01_ Setup OSRM Server",
                    "base_parameters": {}
                },
                "task_key": "routing_01",
                "depends_on": [
                    {
                        "task_key": "routing_00"
                    }
                ]
            },
            {
                "job_cluster_key": "routing_cluster_w_init",
                "libraries": [],
                "notebook_task": {
                    "notebook_path": f"02_ Generate Routes"
                },
                "task_key": "routing_02",
                "depends_on": [
                    {
                        "task_key": "routing_01"
                    }
                ]
            }
        ],
        "job_clusters": [
            {
                "job_cluster_key": "routing_cluster",
                "new_cluster": {
                    "spark_version": "10.4.x-cpu-ml-scala2.12",
                    "num_workers": 0,
                    "node_type_id": {"AWS": "i3.4xlarge", "MSA": "Standard_E16_v3", "GCP": "n1-highmem-16"}
                }
            },
            {
                "job_cluster_key": "routing_cluster_w_init",
                "new_cluster": {
                    "spark_version": "10.4.x-cpu-ml-scala2.12",
                    "num_workers": 2,
                    "node_type_id": {"AWS": "i3.4xlarge", "MSA": "Standard_E16_v3", "GCP": "n1-highmem-16"},
                    "init_scripts": [
                        {
                            "dbfs": {
                                "destination": "dbfs:/databricks/scripts/osrm-backend.sh"
                            }
                        }
                    ]
                }
            }
        ]
    }

# COMMAND ----------

dbutils.widgets.dropdown("run_job", "False", ["True", "False"])
run_job = dbutils.widgets.get("run_job") == "True"
NotebookSolutionCompanion().deploy_compute(job_json, run_job=run_job)

# COMMAND ----------



# COMMAND ----------


