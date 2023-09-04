# Databricks notebook source
# MAGIC %md
# MAGIC # Creating Serving Endpoints and Testing

# COMMAND ----------

%pip install flash-attn
%pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary

# COMMAND ----------

dbutils.library.restartPython()
# COMMAND ----------
from transformers import (
   AutoModelForCausalLM,
   AutoTokenizer,
   AutoConfig,
   pipeline
)
import mlflow
import torch
# COMMAND ----------

import huggingface_hub
huggingface_key = dbutils.secrets.get(scope='brian_hf', key='hf_hub_token')
huggingface_hub.login(token=huggingface_key)

# COMMAND ----------

username = spark.sql("SELECT current_user()").first()['current_user()']

model_name = 'meta-llama/Llama-2-7b-chat-hf'
revision = '08751db2aca9bf2f7f80d2e516117a53d7450235'

# UC Catalog Settings
catalog = 'brian_ml'
db = 'rag_chatbot'
uc_model_name = 'hf_inference_model'

# mlflow settings
experiment_name = f'/Users/{username}/rag_llm_inference'
run_name = 'inference_model'
artifact_path = 'inference_model'

# model serving settings
endpoint_name = 'brian_inference_endpoint'
workload_sizing = 'Small'

# With GPU Private preview will have: workload_type
# {“CPU”, “GPU_MEDIUM”, “MULTIGPU_MEDIUM”} (AWS) 
# {“CPU”, “GPU_SMALL”, “GPU_LARGE”} (Azure)
workload_type = "GPU_MEDIUM"

# COMMAND ----------

# MAGIC %sql
# MAGIC -- we need to make sure that the schemas exist
# MAGIC CREATE CATALOG IF NOT EXISTS brian_ml;
# MAGIC CREATE SCHEMA IF NOT EXISTS brian_ml.rag_chatbot;

# COMMAND ----------

# DBTITLE 1,Setting Up a Model


tokenizer = AutoTokenizer.from_pretrained(model_name)

model_config = AutoConfig.from_pretrained(model_name,
                                          trust_remote_code=True, # this can be needed if we reload from cache
                                          revision=revision
                                      )

model = AutoModelForCausalLM.from_pretrained(model_name,
                                               revision=revision,
                                               trust_remote_code=True, # this can be needed if we reload from cache
                                               config=model_config,
                                               device_map='auto',
                                               torch_dtype=torch.bfloat16 # This will only work A10G / A100 and newer GPUs
                                              )
  
pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer
        )

inference_config = {
   "do_sample": True,
   "max_new_tokens": 512
}


# Lets create a signature example
example_sentences = ["<s>[INST]<<SYS>>Answer questions succintly<</SYS>> Who are you?[/INST]", 
                    "<s>[INST]<<SYS>>Answer questions succintly<</SYS>> How can you help me?[/INST]"]

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

# LLama 2 special type currently not supported
# embedding_signature = mlflow.models.infer_signature(
#     model_input=example_sentences,
#     model_output=pipe(example_sentences)
# )

with mlflow.start_run(run_name=run_name) as run:
    mlflow.transformers.log_model(pipe,
                                  artifact_path=artifact_path,
                                  #signature=embedding_signature,
                                  input_example=example_sentences,
                                  inference_config=inference_config
                                  )
    
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
                                            workload_type=workload_type,
                                            scale_to_zero_enabled=False
                                            )
