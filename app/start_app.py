# Databricks notebook source
# MAGIC %pip install uvicorn gradio==3.20.0


# COMMAND ----------

cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
org_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterOwnerOrgId")
endpoint_url = f"https://{workspace_url}/driver-proxy-api/o/{org_id}/{cluster_id}/7777/"
print(f"Access this API at {endpoint_url}")

# COMMAND ----------

!streamlit run app.py

# COMMAND ----------

# app:app refers to app function inside the app file
# !uvicorn --host 0.0.0.0 --port 7777 app:app --root-path 
