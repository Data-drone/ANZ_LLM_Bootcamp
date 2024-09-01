# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-agents mlflow mlflow-skinny==2.15.1 databricks-vectorsearch langchain==0.2.11 langchain_core==0.2.23 langchain_community==0.2.10 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import mlflow
import time
from databricks import agents
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
from databricks.sdk.errors import NotFound, ResourceDoesNotExist

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

with mlflow.start_run(run_name='brian_test_run'):
    # Tag to differentiate from the data pipeline runs
    mlflow.set_tag("type", "chain")

    logged_chain_info = mlflow.langchain.log_model(
        lc_model=os.path.join(
            os.getcwd(), '0.4_QnA_Bot_Dev'
        ),  # Chain code file e.g., /path/to/the/chain.py
        model_config=common_config,  # Chain configuration set in 00_config
        artifact_path="chain",  # Required by MLflow
        input_example={
            "messages": [
            {
                "role": "user",
                "content": "What is RAG?",
            },
            ]
        },  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
        example_no_conversion=True,  # Required by MLflow to use the input_example as the chain's schema
        extra_pip_requirements=["databricks-agents"] # TODO: Remove this
    )

    # Attach the data pipeline's configuration as parameters
    #mlflow.log_params(_flatten_nested_params({"data_pipeline": data_pipeline_config}))

    # Attach the data pipeline configuration 
    #mlflow.log_dict(data_pipeline_config, "data_pipeline_config.json")

# COMMAND ----------

chain_input = {
    "messages": [
        {
            "role": "user",
            "content": "What is RAG?", # Replace with a question relevant to your use case
        }
    ]
}
chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
chain.invoke(chain_input)

# COMMAND ----------

UC_MODEL_NAME = f"{db_catalog}.{db_schema}.rag_chain"

# Use Unity Catalog to log the chain
mlflow.set_registry_uri('databricks-uc')

# Register the chain to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=UC_MODEL_NAME)

# Deploy to enable the Review APP and create an API endpoint
deployment_info = agents.deploy(model_name=UC_MODEL_NAME, model_version=uc_registered_model_info.version)

browser_url = mlflow.utils.databricks_utils.get_browser_hostname()
print(f"\n\nView deployment status: https://{browser_url}/ml/endpoints/{deployment_info.endpoint_name}")
