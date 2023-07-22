# Databricks notebook source
# MAGIC %md
# MAGIC This notebook will setup the datasets to use for exploring LLM RAGs

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
