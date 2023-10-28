# Databricks notebook source
# MAGIC
# MAGIC %md
# MAGIC # MLflow Logging
# MAGIC
# MAGIC Most organisations do not want to have workspaces permanently open to huggingface.co \
# MAGIC We can leverage MLflow in order to log and store models
# MAGIC NOTE
# MAGIC - 7b can be logged with single G5
# MAGIC - 13b 

# COMMAND ----------

# DBTITLE 1,Install Note GPU Only
%pip install -U mlflow==2.7.1 transformers==4.33.1 accelerate==0.23.0

# COMMAND ----------

dbutils.library.restartPython()
# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

from transformers import (
   AutoModelForCausalLM,
   AutoTokenizer,
   AutoConfig,
   pipeline
)
import mlflow
from mlflow.models import infer_signature
from mlflow.transformers import generate_signature_output
import torch
import os

# COMMAND ----------

# DBTITLE 1,Setting Up the model
model = 'llama_2_13b'
cached_model_files = f'{bootcamp_dbfs_model_folder}/{model}'

tokenizer = AutoTokenizer.from_pretrained(cached_model_files, cache_dir=dbfs_tmp_cache)

model_config = AutoConfig.from_pretrained(cached_model_files)
model = AutoModelForCausalLM.from_pretrained(cached_model_files,
                                               config=model_config,
                                               #device_map='auto',
                                               torch_dtype=torch.bfloat16, # This will only work A10G / A100 and newer GPUs
                                               cache_dir=dbfs_tmp_cache
                                              )

pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128 
        )
# COMMAND ----------

# DBTITLE 1,Verify Pipeline Working
example_sentences = ["<s>[INST]<<SYS>>Answer questions succintly<</SYS>> Who are you?[/INST]", 
                    "<s>[INST]<<SYS>>Answer questions succintly<</SYS>> How can you help me?[/INST]"]

for sentence in example_sentences:
   print(pipe(sentence, max_new_tokens=256))

# COMMAND ----------

# DBTITLE 1,Configure Experiment
experiment_name = f'/Users/{username}/{model}'
try:
  mlflow.create_experiment(experiment_name)
except mlflow.exceptions.RestException:
  print('experiment exists already')

mlflow.set_experiment(experiment_name)

# setup model experiment details
run_name = model

# UC Catalog Settings
use_uc = True
catalog = 'bootcamp_ml'
db = 'rag_chatbot'
uc_model_name = f'{model}_inference_model'

artifact_path = "transformers_pipeline"

if use_uc:
   spark.sql(f'CREATE CATALOG IF NOT EXISTS {catalog}')
   spark.sql(f'CREATE SCHEMA IF NOT EXISTS {catalog}.{db}')

# set registry and the model naming
if use_uc:
  mlflow.set_registry_uri('databricks-uc')
  register_name = f'{catalog}.{db}.{uc_model_name}'
else:
  register_name = uc_model_name

# COMMAND ----------

if not use_uc:
   # LLM Models are large and it can take a while to update
   # Default is 5min wait. This sets to 15min wait
   ## NOTE ## This problem goes away with UC
   os.environ['AWAIT_REGISTRATION_FOR'] = 900


# Model Examples

# Setting up generation parameters
## These can be updated each call to the model
inference_config = {
    "top_p": 0.85,
    "top_k": 2,
    "num_beams": 5,
    "max_new_tokens": 125,
    "do_sample": True,
    "temperature": 0.62,
    "repetition_penalty": 1.15,
    "use_cache": True
}

data = "MLflow is great!"
output = generate_signature_output(pipe, example_sentences)
signature = infer_signature(data, output)

# See: https://mlflow.org/docs/latest/python_api/mlflow.transformers.html#mlflow.transformers.log_model
# If we set registered_model_name then log will register too. Otherwise we can use the following line
with mlflow.start_run(run_name=run_name) as run:
    mlflow.transformers.log_model(
        transformers_model=pipe,
        artifact_path=artifact_path,
        inference_config=inference_config,
        input_example=example_sentences,
        signature=signature,
        extra_pip_requirements=['transformers==4.33.1',
                                'accelerate==0.23.0'],
        metadata = {"task": "llm/v1/completions"},
        registered_model_name = register_name
    )

