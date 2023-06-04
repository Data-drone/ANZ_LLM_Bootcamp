# Databricks notebook source
# MAGIC %pip install fastapi==0.95.2 gradio==3.20.1 uvicorn

# COMMAND ----------
import os

# COMMAND ----------
app_port = 7777
os.environ['DB_APP_PORT'] = f'{app_port}'

cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")

workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
org_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterOwnerOrgId")

# AWS this works at least
proxy_prefix = f'dbc-dp-{org_id}.cloud.databricks.com'
endpoint_url = f"https://{proxy_prefix}/driver-proxy/o/{org_id}/{cluster_id}/{app_port}/"
print(f"Access this API at {endpoint_url}")

# COMMAND ----------

!uvicorn basic_app:app --host 0.0.0.0 --port $DB_APP_PORT

# COMMAND ----------
