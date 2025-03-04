# Databricks notebook source
# MAGIC %md
# MAGIC # Utils notebook
# MAGIC With Databricks we can create a utils notebook that is then used in other notebooks via the `%run` magic\
# MAGIC We will make some of the code from hugging_face_basics available for general use.
# MAGIC
# MAGIC Edited for AWS Workshop format
# COMMAND ----------

# DBTITLE 1,Aux Functions for AU AWS Workshops
import boto3
from botocore.exceptions import ClientError
import json
import requests

def get_region():
    # Define the URL and headers
    token_url = "http://169.254.169.254/latest/api/token"
    token_headers = {"X-aws-ec2-metadata-token-ttl-seconds": "21600"}

    # Make the PUT request to get the token
    token_response = requests.put(token_url, headers=token_headers)

    # Get the token from the response
    token = token_response.text

    # Define the URL and headers for the second request
    metadata_url = "http://169.254.169.254/latest/meta-data/placement/region"
    metadata_headers = {"X-aws-ec2-metadata-token": token}

    # Make the GET request using the token
    metadata_response = requests.get(metadata_url, headers=metadata_headers)

    # Print the response
    return metadata_response.text

def get_cfn():
    client = boto3.client('cloudformation',region_name=get_region())
    response = client.describe_stacks()
    cfn_outputs = {}
    for stack in response['Stacks']:

        outputs = stack.get('Outputs', [])
        if outputs:
            exists = any('DatabrickWorkshopBucket' in d['OutputKey'] for d in outputs)
            
            if(exists):
                desired_output_keys = ['DatabrickWorkshopBucket', 'DatabricksCatalog']
                

                for output in outputs:
                    output_key = output['OutputKey']
                    if output_key in desired_output_keys:
                        cfn_outputs[output_key] = output['OutputValue']

                workshop_bucket = cfn_outputs['DatabrickWorkshopBucket']
                workshop_catalog = cfn_outputs['DatabricksCatalog']
              
                
                spark.conf.set("da.workshop_bucket",workshop_bucket)
                spark.conf.set("da.workshop_catalog",workshop_catalog)

                print(f"""
                S3 Bucket:                  {workshop_bucket}
                Catalog:                    {workshop_catalog} 
                """)

#get_cfn()


# COMMAND ----------

# setup env
import os
import requests
from pathlib import Path

username = spark.sql("SELECT current_user()").first()['current_user()']
os.environ['USERNAME'] = username

# spark.conf.get("da.workshop_catalog")
db_catalog = 'workshop_ml_dev' 
db_schema = 'genai_workshop'
db_volume = 'raw_data'
raw_table = 'arxiv_data'
hf_volume = 'hf_volume'

#Internal dev
vector_search_endpoint = 'vector-search-endpoint'
#vector_search_endpoint = 'gen_ai_workshop'

# # setting up transformers cache
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {db_catalog}.{db_schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {db_catalog}.{db_schema}.{hf_volume}")
hf_volume_path = f'/Volumes/{db_catalog}/{db_schema}/{hf_volume}'

transformers_cache = f'{hf_volume_path}/transformers'
downloads_dir = f'{hf_volume_path}/downloads'
tf_cache_path = Path(transformers_cache)
dload_path = Path(downloads_dir)
#tf_cache_path.mkdir(parents=True, exist_ok=True)
#dload_path.mkdir(parents=True, exist_ok=True)

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