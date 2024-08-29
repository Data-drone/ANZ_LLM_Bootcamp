# Databricks notebook source
# MAGIC %md
# MAGIC Utils notebook\
# MAGIC With Databricks we can create a utils notebook that is then used in other notebooks via the `%run` magic\
# MAGIC We will make some of the code from hugging_face_basics available for general use.

# COMMAND ----------

# setup env
import os
import requests
from pathlib import Path

username = spark.sql("SELECT current_user()").first()['current_user()']
os.environ['USERNAME'] = username

db_catalog = 'gen_ai_workshop'
db_schema = 'datasets'
db_volume = 'raw_data'
raw_table = 'arxiv_data'
hf_volume = 'hf_volume'

#Internal dev
#vector_search_endpoint = 'one-env-shared-endpoint-6' 
vector_search_endpoint = 'gen_ai_workshop'

# # setting up transformers cache
hf_volume_path = f'/Volumes/{db_catalog}/{db_schema}/{hf_volume}'

transformers_cache = f'{hf_volume_path}/transformers'
downloads_dir = f'{hf_volume_path}/downloads'
tf_cache_path = Path(transformers_cache)
dload_path = Path(downloads_dir)
tf_cache_path.mkdir(parents=True, exist_ok=True)
dload_path.mkdir(parents=True, exist_ok=True)
