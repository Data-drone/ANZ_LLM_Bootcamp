# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow logging for embedding models
# MAGIC Embedding models use sentence transformers which are different

# COMMAND ----------

%pip install -U mlflow==2.8.0 llama_index==0.8.54

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------
from sentence_transformers import SentenceTransformer
import mlflow
# COMMAND ----------

dbutils.library.restartPython()
# COMMAND ----------

model_name='sentence-transformers/all-mpnet-base-v2'

# UC Catalog Settings
use_uc = True
catalog = 'bootcamp_ml'
db = 'rag_chatbot'
base_model_name = 'mpnet_base_embedding_model'

# mlflow settings
experiment_name = f'/Users/{username}/rag_llm_embedding'
run_name = 'embedding_model'
artifact_path = 'embedding_model'

# COMMAND ----------

# DBTITLE 1,Set up UC Settings
if use_uc:
   spark.sql(f'CREATE CATALOG IF NOT EXISTS {catalog}')
   spark.sql(f'CREATE SCHEMA IF NOT EXISTS {catalog}.{db}')

   mlflow.set_registry_uri('databricks-uc')

   model_path = f"{catalog}.{db}.{base_model_name}"
else:
   model_path = base_model_name

# COMMAND ----------

# DBTITLE 1,Setting Up a Model

embedding_model = SentenceTransformer(model_name)

# Lets create a signature example
example_sentences = ["welcome to sentence transformers", 
                    "This model is for embedding sentences"]

# COMMAND ----------

# DBTITLE 1,Setting Up the mlflow experiment
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
                                     model_path)

client.set_registered_model_alias(name=model_path, 
                                  alias="prod", 
                                  version=latest_model.version)

# COMMAND ----------

# MAGIC %md
# MAGIC # 