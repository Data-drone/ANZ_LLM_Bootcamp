# Databricks notebook source
# MAGIC
# MAGIC %md
# MAGIC # MLflow Logging
# MAGIC
# MAGIC Most organisations do not want to have workspaces permanently open to huggingface.co \
# MAGIC We can leverage MLflow in order to log and store models
# MAGIC The approach with huggingface model and embedding model is slightly different so we will use different notebooks
# MAGIC NOTE - We don't even need a GPU instance in order to go through and do logging

# COMMAND ----------

# DBTITLE 1,Library Install
# Mistral model is only supported with transformers 4.34.1 and above
# We need mlflow 2,8,0 for latest LLM Updates
## TODO Peft configs?
%pip install -U mlflow==2.8.0 transformers==4.34.1 accelerate==0.23.0 llama_index==0.8.54

# COMMAND ----------

dbutils.library.restartPython()
# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

# MAGIC %md ## Load Model
# MAGIC As long as we don't use the model for inference it is fine to just load with CPU
# MAGIC If we did want to test the model out first we would need to setup GPU properly
# MAGIC We haven't set `device_map` because it glitches with mlflow logging so GPU won't work

# COMMAND ----------

from transformers import (
   AutoModelForCausalLM,
   AutoTokenizer,
   AutoConfig,
   pipeline
)
import mlflow
from mlflow.models import infer_signature
import torch
import os

# COMMAND ----------

# DBTITLE 1,Setting Up the model
# For most bootcamps we cache the models that we need first
# this code sets huggingface to retrieve model from cache
# We can use llama from marketplace so we just log zephyr
model_name = 'zephyr_7b'
cached_model_files = f'{bootcamp_dbfs_model_folder}/{model_name}'

tokenizer = AutoTokenizer.from_pretrained(cached_model_files, cache_dir=dbfs_tmp_cache)

model_config = AutoConfig.from_pretrained(cached_model_files)
model = AutoModelForCausalLM.from_pretrained(cached_model_files,
                                               config=model_config,
                                               #device_map='auto', # disabled for mlflow compatability
                                               torch_dtype=torch.bfloat16, # This will only work A10G / A100 and newer GPUs
                                               cache_dir=dbfs_tmp_cache
                                              )

pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128 
        )
# COMMAND ----------

# DBTITLE 1,Creating Sample Data and defining the schema for the signature
# This is in llama format
#example_sentences = ["<s>[INST]<<SYS>>Answer questions succintly<</SYS>> Who are you?[/INST]", 
#                    "<s>[INST]<<SYS>>Answer questions succintly<</SYS>> How can you help me?[/INST]"]

# This is in zephyr format
example_sentences = ["<|system|>Answer questions succintly <|user|>Who are you?", 
                    "<|system|>Answer questions succintly <|user|>How can you help me?"]


# To use the generate signature we need to do inference on the model which can be slow

# we disable generate_signature as that runs the model and it will be slow on cpu
# The key is the format so even though these strings are fake it is fine
#output = generate_signature_output(pipe, example_sentences)
signature = infer_signature(["MLflow is great!"], ['Yay for ml'])


# COMMAND ----------

# DBTITLE 1,Configure Experiment
experiment_name = f'/Users/{username}/{model_name}'
try:
  mlflow.create_experiment(experiment_name)
except mlflow.exceptions.RestException:
  print('experiment exists already')

mlflow.set_experiment(experiment_name)

# setup model experiment details
run_name = model_name

# UC Catalog Settings
use_uc = True
catalog = 'bootcamp_ml'
db = 'rag_chatbot'
uc_model_name = f'{model_name}_inference_model'

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


# See: https://mlflow.org/docs/latest/python_api/mlflow.transformers.html#mlflow.transformers.log_model
# If we set registered_model_name then log will register too. Otherwise we can use the following line
# The extra metadata is how we switch to optimised serving
with mlflow.start_run(run_name=run_name) as run:
    mlflow.transformers.log_model(
        transformers_model=pipe,
        artifact_path=artifact_path,
        inference_config=inference_config,
        input_example=example_sentences,
        signature=signature,
        extra_pip_requirements=['transformers==4.34.1',
                                'accelerate==0.23.0'],
        metadata = {"task": "llm/v1/completions"},
        registered_model_name = register_name
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # Discussion
# MAGIC Logging Models into UC is the best way to govern them
# MAGIC There can be scenarios where we want to load again from huggingface
# MAGIC
# MAGIC ie getting rid of bfloat16

