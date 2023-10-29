# Databricks notebook source
# MAGIC %md
# MAGIC # MLOps and Setting up an environment
# MAGIC
# MAGIC The easiest way to get started is to use a mosaicml-endpoint

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