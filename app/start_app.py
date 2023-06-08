# Databricks notebook source
# MAGIC %pip install fastapi==0.95.2 gradio==3.20.1 uvicorn pypdf faiss-cpu

# COMMAND ----------
# MAGIC %md
# MAGIC # Starting a Gradio Application

# MAGIC These apps are basic examples only.
# MAGIC 
# MAGIC For more information on the framework see: https://gradio.app/docs/

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

# DBTITLE 1,Basic Application

# uncomment the latter for a basic app
#!uvicorn basic_app:app --host 0.0.0.0 --port $DB_APP_PORT

# COMMAND ----------

# uncomment the latter to try out the basic chatbot note you will need to supply your own subscription endpoint
# the key will be retrieved through db secrets
os.environ['OPENAI_API_KEY'] = dbutils.secrets.get(scope='brian_dl', key='brian_openai_deployment')
os.environ['OPENAI_API_TYPE']='azure'
os.environ['OPENAI_API_VERSION']="2022-12-01"

# your endpoint address will differ
os.environ['OPENAI_API_BASE']="https://dbdemos-open-ai.openai.azure.com/"

# COMMAND ----------

# DBTITLE 1,Chat Application

# uncomment the latter to try out the basic chatbot note you will need to supply your own subscription endpoint
#!uvicorn chat_app:app --host 0.0.0.0 --port $DB_APP_PORT

# COMMAND ----------

# the advanced app uses certain folder paths
# hardcoded for now
vector_store_path = '/tmp/doc_app/vector_store'
upload_file_path = '/tmp/doc_app/source_documents'

dbutils.fs.mkdirs(vector_store_path)
dbutils.fs.mkdirs(upload_file_path)

os.environ['VECTOR_STORE_PATH'] = f'/dbfs{vector_store_path}'
os.environ['UPLOAD_FILE_PATH'] = f'/dbfs{upload_file_path}'

# COMMAND ----------

# DBTITLE 1,Chat to docs application

# uncomment the latter to try out the basic chatbot note you will need to supply your own subscription endpoint
#!uvicorn advanced_app:app --host 0.0.0.0 --port $DB_APP_PORT
