# Databricks notebook source
# MAGIC %md
# MAGIC Utils notebook\
# MAGIC With Databricks we can create a utils notebook that is then used in other notebooks via the `%run` magic\
# MAGIC We will make some of the code from hugging_face_basics available for general use.

# COMMAND ----------

# setup env
# TODO - adjust and use bootcamp ones later
import os
import requests
import pprint

username = spark.sql("SELECT current_user()").first()['current_user()']
os.environ['USERNAME'] = username

tmp_user_folder = f'/tmp/{username}'
dbutils.fs.mkdirs(tmp_user_folder)
dbfs_tmp_dir = f'/dbfs{tmp_user_folder}'
os.environ['PROJ_TMP_DIR'] = dbfs_tmp_dir

# setting up transformers cache
cache_dir = f'{tmp_user_folder}/.cache'
dbutils.fs.mkdirs(cache_dir)
dbfs_tmp_cache = f'/dbfs{cache_dir}'
os.environ['TRANSFORMERS_CACHE'] = dbfs_tmp_cache

# setup source file_docs
source_doc_folder = f'/home/{username}/pdf_data'
dbfs_source_docs = '/dbfs' + source_doc_folder

# setup vectorstore path
vector_store_path = f'/home/{username}/vectorstore_persistence/db'
linux_vector_store_directory = f'/dbfs{vector_store_path}'

# is that right env var?
os.environ['PERSIST_DIR'] = linux_vector_store_directory

# COMMAND ----------

bootcamp_dbfs_model_folder = '/dbfs/bootcamp_data/hf_cache/downloads'

# COMMAND ----------

# MAGIC %md 
# MAGIC Here we will make the model loaders into functions that receive the run_mode var

# COMMAND ----------

class QueryEndpoint:
    """
    A class designed to be interchangeable with pipe but calls databricks model serving instead
    """
   
    def __init__(self, uri:str, token:str):
      
      self.uri = uri
      self.header = {"Context-Type": "text/json", "Authorization": f"Bearer {token}"}
      print(uri)


    def __call__(self, prompt: list[str], **kwargs):
       
      dataset = {'inputs': {'prompt': prompt},
                  'params': kwargs}

      response = requests.post(headers=self.header, url=self.uri, json=dataset)

      return response.json()
# COMMAND ----------

def load_model(run_mode: str, dbfs_cache_dir: str, serving_uri :str='llama_2_13b'):
    """
    run_mode (str) - can be gpu or cpu
    """

    from transformers import pipeline, AutoConfig
    import torch

    assert run_mode in ['cpu', 'gpu', 'serving'], f'run_mode must be cpu, gpu or serving not {run_mode}'

    if run_mode == 'cpu':

      from ctransformers import AutoModelForCausalLM, AutoTokenizer
      model_id = 'llama_2_cpu/llama-2-7b-chat.Q4_K_M.gguf'
      model = AutoModelForCausalLM.from_pretrained(f'{bootcamp_dbfs_model_folder}/{model_id}',
                                              hf=True, local_files_only=True)
      tokenizer = AutoTokenizer.from_pretrained(model)

      pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer , max_new_tokens=128 
      )

      return pipe

    elif run_mode == 'gpu':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        #model_id = 'VMware/open-llama-7b-v2-open-instruct'
        #model_revision = 'b8fbe09571a71603ab517fe897a1281005060b62'

        cached_model = f'{bootcamp_dbfs_model_folder}/llama_2_gpu'
        tokenizer = AutoTokenizer.from_pretrained(cached_model, cache_dir=dbfs_cache_dir)
        
        model_config = AutoConfig.from_pretrained(cached_model)
        model = AutoModelForCausalLM.from_pretrained(cached_model,
                                               config=model_config,
                                               device_map='auto',
                                               torch_dtype=torch.bfloat16, # This will only work A10G / A100 and newer GPUs
                                               cache_dir=dbfs_cache_dir
                                              )
    
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128 
        )

        return pipe
    
    elif run_mode == 'serving':
        
        browser_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
        db_host = f"https://{browser_host}"
        model_uri = f"{db_host}/serving-endpoints/{serving_uri}/invocations"
        db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

        test_pipe = QueryEndpoint(model_uri, db_token)

        return test_pipe

# COMMAND ----------


def string_printer(out_obj, run_mode):
  """
  Short convenience function because the output formats change between CPU and GPU
  """
  try:
    if run_mode in ['cpu', 'gpu']:

      return pprint.pprint(out_obj[0]['generated_text'], indent=2)
  
    elif run_mode == 'serving':
    
      return  pprint.pprint(out_obj['predictions'], indent=2)
  
  except KeyError:
    pprint.pprint(out_obj, indent=2)
