# Databricks notebook source
# MAGIC %md
# MAGIC # Supplementary Exercises on HuggingFace
# MAGIC Huggingface provides a series of livraries that are critical in the OSS llm space\
# MAGIC To use them properly later on and debug issues we need to have an understand of how they work\
# MAGIC This is not a full tutorial. See HuggingFace docs for that. But it will function as a crash course
# MAGIC
# MAGIC In these exercises we will focus on the _transformers_ library but _datasets_, _evaluate_ and _accelerate_ are commonly used in training models
# MAGIC
# MAGIC *NOTE* The LLM Space is fast moving. Many models are provided by independent companies as well so code version is important
# MAGIC
# MAGIC All code here is tested on MLR 13.2 on a g5 AWS instance (A10G GPU).
# MAGIC We suggest a g5.4xlarge single node cluster to start
# MAGIC The Azure equivalent is XXXX

# MAGIC *Notes*
# MAGIC Falcon requires Torch 2.0 coming soon....


# COMMAND ----------

# DBTITLE 1,Needed prior to 13.2
#%pip install einops

# COMMAND ----------

# Sometimes needed to fix loading
#%pip install xformers

# COMMAND ----------

# DBTITLE 1,Load Libs

# Manual Model building
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig, GenerationConfig
import torch
import os

# COMMAND ----------

# Setup

username = spark.sql("SELECT current_user()").first()['current_user()']
os.environ['USERNAME'] = username

tmp_user_folder = f'/tmp/{username}'
dbutils.fs.mkdirs(tmp_user_folder)
dbfs_tmp_dir = f'/dbfs{tmp_user_folder}'
os.environ['PROJ_TMP_DIR'] = dbfs_tmp_dir

# COMMAND ----------

# MAGIC %md
# MAGIC ## Understanding pipelines
# MAGIC
# MAGIC Language Pipelines contain two key components the tokenizer and the LLM model itself.
# MAGIC The tokenizer component converts integers into 

# MAGIC ### Loading the model
# MAGIC 

# COMMAND ----------

# DBTITLE 1,First we setup the model config

model_id = 'mosaicml/mpt-7b'
model_revision = '72e5f594ce36f9cabfa2a9fd8f58b491eb467ee7'
tokenizer = AutoTokenizer.from_pretrained(model_id)

# lets move to the config version
# using max_length here is deprecating max_length=4096
mpt_config = AutoConfig.from_pretrained(model_id,
                                          trust_remote_code=True, # needed on both sides
                                          revision=model_revision,
                                          init_device='meta'
                                      )

mpt_config.save_pretrained(save_directory=dbfs_tmp_dir,
                           config_file_name='mpt_config.json')

# COMMAND ----------

# MAGIC %sh
# MAGIC cat $PROJ_TMP_DIR/mpt_config.json

# COMMAND ----------

# revision is important to ensure same behaviour
# device_map moves it to gpu
# torch.bfloat16 helps save memory by halving numeric precision
model = AutoModelForCausalLM.from_pretrained(model_id,
                                               revision=model_revision,
                                               trust_remote_code=True, # needed on both sides
                                               config=mpt_config,
                                               device_map='auto',
                                               torch_dtype=torch.bfloat16 # This will only work A10G / A100 and newer GPUs
                                              )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Understanding Generation Config
# MAGIC
# MAGIC The pipeline connects the tokenizer to the model entity itself.
# MAGIC
# MAGIC Key Parameters:
# MAGIC - max_new_tokens
# MAGIC - temperature
# MAGIC - eos_token_id
# MAGIC - pad_token_id

# COMMAND ----------

mpt_generation_config = GenerationConfig(
    max_new_tokens=1024,
    temperature = 0.1,
    top_p = 0.92,
    top_k = 0, 
    use_cache = True,
    do_sample = True,
    eos_token_id = tokenizer.eos_token_id,
    pad_token_id = tokenizer.pad_token_id
  )

pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, config=mpt_generation_config 
        )

# COMMAND ----------

# We seem to need to set the max length here for mpt model
output = pipe("How are you?", max_length=200, repetition_penalty=1.2)
print(output[0]['generated_text'])

# COMMAND ----------

# We seem to need to set the max length here for mpt model
output = pipe("Give me a list in dot points of 10 types of cookies?", max_length=200, repetition_penalty=1.2)
print(output[0]['generated_text'])

# COMMAND ----------


# MAGIC %md 
# MAGIC Experimenting with open-llama

# COMMAND ----------