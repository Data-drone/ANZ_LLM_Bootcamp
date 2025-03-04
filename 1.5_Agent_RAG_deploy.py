# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-agents  mlflow-skinny==2.16.0 databricks-vectorsearch langchain langgraph langchain_core langchain_community langchain-databricks 
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

with mlflow.start_run(run_name='test_agent_run'):
    # Tag to differentiate from the data pipeline runs
    mlflow.set_tag("type", "chain")

    # TODO eval loop

    logged_chain_info = mlflow.langchain.log_model(
        lc_model=os.path.join(
            os.getcwd(), '1.5_Agent_RAG_Bot_Dev'
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

# COMMAND ----------
chain_input = {
    "messages": [
        {
            "role": "user",
            "content": "hi im bob! and i live in sf?", # Replace with a question relevant to your use case
        }
    ]
}
chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
chain.invoke(chain_input)

