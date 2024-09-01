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
vector_search_endpoint = 'one-env-shared-endpoint-6' 
#vector_search_endpoint = 'gen_ai_workshop'

# # setting up transformers cache
spark.sql(f"CREATE VOLUME IF NOT EXISTS {db_catalog}.{db_schema}.{hf_volume}")
hf_volume_path = f'/Volumes/{db_catalog}/{db_schema}/{hf_volume}'

transformers_cache = f'{hf_volume_path}/transformers'
downloads_dir = f'{hf_volume_path}/downloads'
tf_cache_path = Path(transformers_cache)
dload_path = Path(downloads_dir)
tf_cache_path.mkdir(parents=True, exist_ok=True)
dload_path.mkdir(parents=True, exist_ok=True)

# COMMAND ----------

# DBTITLE 1,AI Agent Framework config

# The AI Agent Framework relies on yaml files for config
# We cannot use the %run imports that we have been using
import yaml

common_config = {
    "paths_and_locations": {
        "db_catalog": db_catalog,
        "db_schema": db_schema,
        "db_volume": db_volume,
        "raw_table": raw_table,
        "hf_volume": hf_volume,
        "vector_search_endpoint": vector_search_endpoint
    },  
}


with open('common_config.yaml', 'w') as f:
    yaml.dump(common_config, f)