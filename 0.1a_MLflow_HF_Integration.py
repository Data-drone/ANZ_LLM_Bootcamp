# Databricks notebook source
# MAGIC
# MAGIC %md
# MAGIC # MLflow Logging
# MAGIC
# MAGIC Most organisations do not want to have workspaces permanently open to huggingface.co \
# MAGIC We can leverage MLflow in order to log and store models

# COMMAND ----------

# DBTITLE 1,Install Note GPU Only
%pip install mlflow==2.7.1

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

# COMMAND ----------

run_mode = 'gpu'
# reload our model pipe
pipe = load_model(run_mode, dbfs_tmp_cache)

# COMMAND ----------

# Logging Model into MLflow

# Model Examples
example_sentences = ["<s>[INST]<<SYS>>Answer questions succintly<</SYS>> Who are you?[/INST]", 
                    "<s>[INST]<<SYS>>Answer questions succintly<</SYS>> How can you help me?[/INST]"]

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
with mlflow.start_run():
    mlflow.transformers.log_model(
        transformers_model=pipe,
        artifact_path="ctransformers_pipeline",
        inference_config=inference_config,
        input_example=example_sentences,
        signature=signature,
        extra_pip_requirements=['transformers==4.31.0']
    )