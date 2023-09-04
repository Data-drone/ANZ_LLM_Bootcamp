# Databricks notebook source
# MAGIC %md
# MAGIC # Creating Serving Endpoints and Testing

# COMMAND ----------
from sentence_transformers import SentenceTransformer
import mlflow
# COMMAND ----------

username = spark.sql("SELECT current_user()").first()['current_user()']

model_name='sentence-transformers/all-mpnet-base-v2'

# UC Catalog Settings
catalog = 'brian_ml'
db = 'rag_chatbot'
uc_model_name = 'hf_embedding_model'

# mlflow settings
experiment_name = f'/Users/{username}/rag_llm_embedding'
run_name = 'embedding_model'
artifact_path = 'embedding_model'

# model serving settings
endpoint_name = 'brian_embedding_endpoint'
workload_sizing = 'Small'

# With GPU Private preview will have: workload_type
# {“CPU”, “GPU_MEDIUM”, “MULTIGPU_MEDIUM”} (AWS) 
# {“CPU”, “GPU_SMALL”, “GPU_LARGE”} (Azure)
workload_type = "CPU"

# COMMAND ----------

# MAGIC %sql
# MAGIC -- we need to make sure that the schemas exist
# MAGIC CREATE CATALOG IF NOT EXISTS brian_ml;
# MAGIC CREATE SCHEMA IF NOT EXISTS brian_ml.rag_chatbot;

# COMMAND ----------

# DBTITLE 1,Setting Up a Model

embedding_model = SentenceTransformer(model_name)

# Lets create a signature example
example_sentences = ["welcome to sentence transformers", 
                    "This model is for embedding sentences"]

# COMMAND ----------

# DBTITLE 1,Setting Up the mlflow experiment
#Enable Unity Catalog with mlflow registry
mlflow.set_registry_uri('databricks-uc')

try:
  mlflow.create_experiment(experiment_name)
except mlflow.exceptions.RestException:
  print('experiment exists already')

mlflow.set_experiment(experiment_name)

client = mlflow.MlflowClient()

embedding_signature = mlflow.models.infer_signature(
    model_input=example_sentences,
    model_output=embedding_model.encode(example_sentences)
)

with mlflow.start_run(run_name=run_name) as run:
    mlflow.sentence_transformers.log_model(embedding_model,
                                  artifact_path=artifact_path,
                                  signature=embedding_signature,
                                  input_example=example_sentences)
    
# COMMAND ----------

# DBTITLE 1,Register Model

# We need to know the Run id first. When running this straight then we can extract the run_id
latest_model = mlflow.register_model(f'runs:/{run.info.run_id}/{artifact_path}', 
                                     f"{catalog}.{db}.{uc_model_name}")

client.set_registered_model_alias(name=f"{catalog}.{db}.{uc_model_name}", 
                                  alias="prod", 
                                  version=latest_model.version)

# COMMAND ----------

%run ./endpoint_utils

# COMMAND ----------

# DBTITLE 1,Deploy Endpoint

# we to deploy the API Endpoint
serving_client = EndpointApiClient()

# Start the enpoint using the REST API (you can do it using the UI directly)

serving_client.create_endpoint_if_not_exists(endpoint_name, 
                                            model_name=f"{catalog}.{db}.{uc_model_name}", 
                                            model_version = latest_model.version, 
                                            workload_size=workload_sizing,
                                            workload_type=workload_type
                                            )
