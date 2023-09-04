# Databricks notebook source
# MAGIC %md
# MAGIC # Let us use our endpoings

# COMMAND ----------

%run ./endpoint_utils
# COMMAND ----------

import requests

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