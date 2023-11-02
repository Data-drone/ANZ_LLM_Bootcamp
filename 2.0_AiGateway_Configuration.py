# Databricks notebook source
# MAGIC %md
# MAGIC # MLOps and Setting up an environment
# MAGIC
# MAGIC The easiest way to get started is to use a mosaicml-endpoint

# COMMAND ----------

%pip install llama_index==0.8.54

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

%run ./utils

# COMMAND ----------

# MLFlow Gateway
import mlflow
from mlflow import gateway

gateway.set_gateway_uri(gateway_uri="databricks")

mosaic_embeddings_route_name = "mosaicml-instructor-xl-embeddings"

try:
    route = gateway.get_route(mosaic_embeddings_route_name)
except:
    # Create a route for embeddings with MosaicML
    print(f"Creating the route {mosaic_embeddings_route_name}")
    print(gateway.create_route(
        name=mosaic_embeddings_route_name,
        route_type="llm/v1/embeddings",
        model={
            "name": "instructor-xl",
            "provider": "mosaicml",
            "mosaicml_config": {
                "mosaicml_api_key": dbutils.secrets.get(scope="bootcamp_training", key="mosaic_ml_api_key")
            }
        }
    ))

# COMMAND ----------

mosaic_route_name = "mosaicml-llama2-70b-chat"

try:
    route = gateway.get_route(mosaic_route_name)
except:
    # Create a route for embeddings with MosaicML
    print(f"Creating the route {mosaic_route_name}")
    print(gateway.create_route(
        name=mosaic_route_name,
        route_type="llm/v1/chat",
        model={
            "name": "llama2-70b-chat",
            "provider": "mosaicml",
            "mosaicml_config": {
                "mosaicml_api_key": dbutils.secrets.get(scope="bootcamp_training", key="mosaic_ml_api_key")
            }
        }
    ))

# COMMAND ----------

# DBTITLE 1,Test the Langchain and the logging
import mlflow

from langchain.llms import MlflowAIGateway
from langchain.chat_models import ChatMLflowAIGateway
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain

mosaic_route_name = "mosaicml-llama2-70b-chat"

my_gateway = ChatMLflowAIGateway(
    gateway_uri="databricks",
    route=mosaic_route_name,
    params={
        "temperature": 0.0,
        "top_p": 1,
        "max_tokens": 1025
    },
)

llm_chain = LLMChain(
    llm=my_gateway,
    prompt=ChatPromptTemplate.from_messages([
        ("system", "You are an unhelpful bot called Bossy that speaks in Korean"),
        ("human", "{user_input}")
    ])
)
# COMMAND ----------

llm_chain.run(user_input="안녕하세요!")

# COMMAND ----------

experiment_path = f'/Users/{username}/langchain_testing'
mlflow.set_experiment(experiment_path)

with mlflow.start_run(run_name='initial_chain'):
    mlflow.langchain.log_model(llm_chain,
                               artifact_path='langchain_model',
                               extra_pip_requirements=['mlflow==2.8.0',
                                                       'llama_index==0.8.54'])

# COMMAND ----------