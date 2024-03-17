# Databricks notebook source
# MAGIC %md
# MAGIC This notebook will setup the datasets and preload the big models to use for exploring LLM RAGs
# MAGIC This notebook is just if you are setting up locally after the workshop
# MAGIC In a workshop the instructor should run and set this up

# COMMAND ----------

# MAGIC %pip install --upgrade --force-reinstall databricks-vectorsearch huggingface_hub
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import requests

# COMMAND ----------
# DBTITLE 1,Setup utils to ensure consistency
# MAGIC %run ./utils

# COMMAND ----------

# Setup Catalogs and directories
spark.sql(f"CREATE CATALOG IF NOT EXISTS {db_catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {db_catalog}.{db_schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {db_catalog}.{db_schema}.{db_volume}")
volume_folder = f"/Volumes/{db_catalog}/{db_schema}/{volume_name}/"

# COMMAND ----------

# DBTITLE 1,Download Files
# We will setup a folder to store the files
def load_file(file_uri, file_name, library_folder):
    
    # Create the local file path for saving the PDF
    local_file_path = os.path.join(library_folder, file_name)

    # Download the PDF using requests
    try:
        # Set the custom User-Agent header
        headers = {"User-Agent": "me-me-me"}

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
    load_file(pdfs[pdf], pdf, volume_folder)

# COMMAND ----------

# DBTITLE 1,To Examine Embeddings we need to download from huggingface
spark.sql(f"CREATE VOLUME IF NOT EXISTS {db_catalog}.{db_schema}.{hf_volume}")
hf_volume_path = f'/Volumes/{db_catalog}/{db_schema}/{hf_volume}'

transformers_cache = f'{hf_volume_path}/transformers'
downloads_dir = f'{hf_volume_path}/downloads'
os.mkdirs[transformers_cache]

os.environ['HF_HOME'] = hf_volume_path
os.environ['TRANSFORMERS_CACHE'] = transformers_cache

from huggingface_hub import hf_hub_download, list_repo_files

repo_list = {'mistral_7b_instruct': 'mistralai/Mistral-7B-v0.1'}

for lib_name in repo_list.keys():
    for name in list_repo_files(repo_list[lib_name]):
        target_path = os.path.join(downloads_dir, lib_name, name)
        if not os.path.exists(target_path):
            hf_hub_download(
                repo_list[lib_name],
                filename=name,
                local_dir=os.path.join(downloads_dir, lib_name),
                local_dir_use_symlinks=False
            )

# COMMAND ----------
            
# Setup Vector Search Endpoint
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

vsc.create_endpoint(
    name = vector_search_endpoint,
    endpoint_type="STANDARD"
)