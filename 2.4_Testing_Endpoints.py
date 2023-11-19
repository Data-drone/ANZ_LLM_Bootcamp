# Databricks notebook source
# MAGIC %md
# MAGIC # Let us deploy and leverage UC Endpoints
# MAGIC

# COMMAND ----------

%run ./endpoint_utils
# COMMAND ----------

import requests

# COMMAND ----------

## TEMP ###
# model serving settings
endpoint_name = 'brian_embedding_endpoint'
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

# Start the enpoint using the REST API (you can do it using the UI directly)

serving_client.create_endpoint_if_not_exists(endpoint_name, 
                                            model_name=model_path, 
                                            model_version = latest_model.version, 
                                            workload_size=workload_sizing,
                                            workload_type=workload_type
                                            )


# COMMAND ----------


# setup config variables
# embedding model
embed_path = 'https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/brian_embedding_endpoint/invocations'

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