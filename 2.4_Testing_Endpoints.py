# Databricks notebook source
# MAGIC %md
# MAGIC # Let us deploy and leverage Databricks Model Serving Endpoints
# MAGIC

# COMMAND ----------

%run ./endpoint_utils
# COMMAND ----------

import requests
from mlflow import MlflowClient

mlflow.set_registry_uri('databrick-uc')
client = MlflowClient()

catalog = 'bootcamp_ml'
db = 'rag_chatbot'

# embedding model
endpoint_name = 'brian_embedding_endpoint'
base_model_name = f'{catalog}.{db}.mpnet_base_embedding_model'
workload_sizing = 'Small'

# With GPU Private preview will have: workload_type
# {“CPU”, “GPU_MEDIUM”, “MULTIGPU_MEDIUM”} (AWS) 
# {“CPU”, “GPU_SMALL”, “GPU_LARGE”} (Azure)
workload_type = "CPU"

# COMMAND ----------

%run ./endpoint_utils

# COMMAND ----------

# DBTITLE 1,Deploy Endpoint
# we to deploy the API Endpoint
serving_client = EndpointApiClient()

serving_client.create_endpoint_if_not_exists(endpoint_name, 
                                            model_name = base_model_name, 
                                            model_version = client.get_registered_model(base_model_name).latest_versions[0].version , 
                                            workload_size = workload_sizing,
                                            workload_type = workload_type
                                            )


# COMMAND ----------


# setup config variables
# embedding model
browser_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
db_host = f"https://{browser_host}"

embed_path = f'{db_host}/serving-endpoints/{endpoint_name}/invocations'

# COMMAND ----------

serving_client = EndpointApiClient()

# COMMAND ----------

#Let's try to send some inference to our REST endpoint
dataset =  {"dataframe_split": {'data': ['test sentence']}}
import timeit

#f"{serving_client.base_url}/realtime-inference/dbdemos_embedding_endpoint/invocations"
endpoint_url = embed_path
print(f"Sending requests to {endpoint_url}")
starting_time = timeit.default_timer()
inferences = requests.post(endpoint_url, json=dataset, headers=serving_client.headers).json()
print(f"Embedding inference, end 2 end :{round((timeit.default_timer() - starting_time)*1000)}ms {inferences}")