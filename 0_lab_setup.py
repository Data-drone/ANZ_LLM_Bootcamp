# Databricks notebook source
# MAGIC %md
# MAGIC This notebook will setup the datasets and preload the big models to use for exploring LLM RAGs
# MAGIC This notebook is just if you are setting up locally after the workshop
# MAGIC In a workshop the instructor should run and set this up

# COMMAND ----------

import os
import requests

# COMMAND ----------
# DBTITLE 1,Setup dbfs folder paths
# MAGIC %run ./utils

# COMMAND ----------

# DBTITLE 1,Config Params
# We will setup a folder to store the files
user_agent = "me-me-me"

# If running this on your own in multiuser environment then use this
library_folder = dbfs_source_docs

# When teaching a class
class_lib = '/bootcamp_data/pdf_data'
dbutils.fs.mkdirs(class_lib)
library_folder = f'/dbfs{class_lib}'

# COMMAND ----------

def load_file(file_uri, file_name, library_folder):
    
    # Create the local file path for saving the PDF
    local_file_path = os.path.join(library_folder, file_name)

    # Download the PDF using requests
    try:
        # Set the custom User-Agent header
        headers = {"User-Agent": user_agent}

        response = requests.get(file_uri, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            # Save the PDF to the local file
            with open(local_file_path, "wb") as pdf_file:
                pdf_file.write(response.content)
            print("PDF downloaded successfully.")
        else:
            print(f"Failed to download PDF. Status code: {response.status_code}")
    except requests.RequestException as e:
        print("Error occurred during the request:", e)


# COMMAND ----------

pdfs = {'2203.02155.pdf':'https://arxiv.org/pdf/2203.02155.pdf',
        '2302.09419.pdf': 'https://arxiv.org/pdf/2302.09419.pdf',
        'Brooks_InstructPix2Pix_Learning_To_Follow_Image_Editing_Instructions_CVPR_2023_paper.pdf': 'https://openaccess.thecvf.com/content/CVPR2023/papers/Brooks_InstructPix2Pix_Learning_To_Follow_Image_Editing_Instructions_CVPR_2023_paper.pdf',
        '2303.10130.pdf':'https://arxiv.org/pdf/2303.10130.pdf',
        '2302.06476.pdf':'https://arxiv.org/pdf/2302.06476.pdf',
        '2302.06476.pdf':'https://arxiv.org/pdf/2302.06476.pdf',
        '2303.04671.pdf':'https://arxiv.org/pdf/2303.04671.pdf',
        '2209.07753.pdf':'https://arxiv.org/pdf/2209.07753.pdf',
        '2302.07842.pdf':'https://arxiv.org/pdf/2302.07842.pdf',
        '2302.07842.pdf':'https://arxiv.org/pdf/2302.07842.pdf',
        '2204.01691.pdf':'https://arxiv.org/pdf/2204.01691.pdf'}

for pdf in pdfs.keys():
    load_file(pdfs[pdf], pdf, library_folder)

# COMMAND ----------

dbutils.fs.ls(class_lib)

# COMMAND ----------

# MAGIC %md
# MAGIC Setting up huggingface
# MAGIC Lets try to cache a bunch of stuff and reload it elsewhere
# MAGIC Then we can save students having to wait for downloads or worry about tokens for HF
     
# COMMAND ----------

%pip install ctransformers
# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
hf_home = '/bootcamp_data/hf_cache'
transformers_cache = f'{hf_home}/transformers'
dbutils.fs.mkdirs(hf_home)
dbutils.fs.mkdirs(transformers_cache)

dbfs_hf_home = f'/dbfs{hf_home}'
dbfs_transformers_home = f'/dbfs{transformers_cache}'

os.environ['TRANSFORMERS_CACHE'] = dbfs_transformers_home
os.environ['HF_HOME'] = dbfs_hf_home

# COMMAND ----------

%sh export TRANSFORMERS_CACHE=$dbfs_transformers_home


# COMMAND ----------
# this is needed for llama 2 downloading
# You need to create a huggingface account
# The follow the instructions here: https://huggingface.co/docs/hub/security-tokens#:~:text=To%20create%20an%20access%20token,you're%20ready%20to%20go!
#  

from huggingface_hub import notebook_login
notebook_login()
# COMMAND ----------

# setting up ctransformers:

from ctransformers.langchain import CTransformers
model = 'TheBloke/open-llama-7B-v2-open-instruct-GGML'
model = 'TheBloke/Llama-2-7B-Chat-GGML'
llm_model = CTransformers(model=model, model_type='llama')

# COMMAND ----------

# setting up gpu transformers

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, pipeline
import torch

#model_id = 'VMware/open-llama-7b-v2-open-instruct'
#model_revision = 'b8fbe09571a71603ab517fe897a1281005060b62'

# you need to sign up on huggingface first
model_id = 'meta-llama/Llama-2-7b-chat-hf'
model_revision = '40c5e2b32261834431f89850c8d5359631ffa764'

# note when on gpu then this will auto load to gpu
# this will take approximately an extra 1GB of VRAM
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=dbfs_transformers_home)

model_config = AutoConfig.from_pretrained(model_id,
                                  trust_remote_code=True, # this can be needed if we reload from cache
                                  revision=model_revision
                              )
  
# NOTE only A10G support `bfloat16` - g5 instances
# V100 machines ie g4 need to use `float16`
# device_map = `auto` moves the model to GPU if possible.
# Note not all models support `auto`

model = AutoModelForCausalLM.from_pretrained(model_id,
                                       revision=model_revision,
                                       trust_remote_code=True, # this can be needed if we reload from cache
                                       config=model_config,
                                       device_map='auto',
                                       torch_dtype=torch.bfloat16, # This will only work A10G / A100 and newer GPUs
                                       cache_dir=dbfs_transformers_home
                                      )
  
pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer 
)

