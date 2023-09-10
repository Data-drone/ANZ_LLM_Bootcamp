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
dbutils.fs.rm(class_lib, True)
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
# MAGIC Setting up huggingface \
# MAGIC Lets load the models that we need \
# MAGIC Then we can save students having to wait for downloads or worry about tokens for HF

# COMMAND ----------

import os
hf_home = '/bootcamp_data/hf_cache'
transformers_cache = f'{hf_home}/transformers'
download_dir = f'{hf_home}/downloads'

dbutils.fs.rm(hf_home, True)

dbutils.fs.mkdirs(hf_home)
dbutils.fs.mkdirs(transformers_cache)
dbutils.fs.mkdirs(download_dir)

dbfs_hf_home = f'/dbfs{hf_home}'
dbfs_transformers_home = f'/dbfs{transformers_cache}'
dbfs_downloads_home = f'/dbfs{download_dir}'

os.environ['TRANSFORMERS_CACHE'] = dbfs_transformers_home
os.environ['HF_HOME'] = dbfs_hf_home

# COMMAND ----------

%sh export TRANSFORMERS_CACHE=$dbfs_transformers_home

# COMMAND ----------
# this is needed for llama 2 downloading
# You need to create a huggingface account
# The follow the instructions here: https://huggingface.co/docs/hub/security-tokens#:~:text=To%20create%20an%20access%20token,you're%20ready%20to%20go!
# we could also use notebook login

# we can use a secret to setup the huggingface connection

import huggingface_hub
huggingface_key = dbutils.secrets.get(scope='brian-hf', key='hf-key')
huggingface_hub.login(token=huggingface_key)

# COMMAND ----------

# Lets use snapshot downloads
from huggingface_hub import hf_hub_download, list_repo_files

repo_list = {'llama_2_gpu': 'meta-llama/Llama-2-7b-chat-hf',
             'llama_2_cpu': 'TheBloke/Llama-2-7B-chat-GGUF'}

for lib_name in repo_list.keys():
    for name in list_repo_files(repo_list[lib_name]):
        # skip all the safetensors data as we aren't using it and it's time consuming to download
        if "safetensors" in name:
            continue
        target_path = os.path.join(dbfs_downloads_home, lib_name, name)
        if not os.path.exists(target_path):
            print(f"Downloading {name}")
            hf_hub_download(
                repo_list[lib_name],
                filename=name,
                local_dir=os.path.join(dbfs_downloads_home, lib_name),
                local_dir_use_symlinks=False,
            )