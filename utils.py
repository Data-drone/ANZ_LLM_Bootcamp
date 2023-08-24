# Databricks notebook source
# MAGIC %md
# MAGIC Utils notebook\
# MAGIC With Databricks we can create a utils notebook that is then used in other notebooks via the `%run` magic\
# MAGIC We will make some of the code from hugging_face_basics available for general use.

# COMMAND ----------

# setup env
import os

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

# MAGIC %md 
# MAGIC Here we will make the model loaders into functions that receive the run_mode var

# COMMAND ----------

def load_model(run_mode: str, dbfs_cache_dir: str):
    """
    run_mode (str) - can be gpu or cpu
    """

    from transformers import AutoTokenizer, pipeline, AutoConfig, GenerationConfig
    import torch

    assert run_mode in ['cpu', 'gpu'], f'run_mode must be cpu or gpu not {run_mode}'

    if run_mode == 'cpu':

        ### Note that caching for TheBloke's models don't follow standard HuggingFace routine
        # You would need to `wget` then weights then use a model_path config instead.
        # See ctransformers docs for more info
        from ctransformers import AutoModelForCausalLM
        #model_id = 'TheBloke/open-llama-7B-v2-open-instruct-GGML'
        model_id = 'TheBloke/Llama-2-13B-chat-GGML'
        pipe = AutoModelForCausalLM.from_pretrained(model_id,
                                           model_type='llama')
        
        return pipe

    elif run_mode == 'gpu':
        from transformers import AutoModelForCausalLM
        #model_id = 'VMware/open-llama-7b-v2-open-instruct'
        #model_revision = 'b8fbe09571a71603ab517fe897a1281005060b62'

        # you need to sign up on huggingface first
        model_id = 'meta-llama/Llama-2-7b-chat-hf'
        model_revision = '40c5e2b32261834431f89850c8d5359631ffa764'
        
        # note when on gpu then this will auto load to gpu
        # this will take approximately an extra 1GB of VRAM
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=dbfs_cache_dir)

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
                                               cache_dir=dbfs_cache_dir
                                              )
  
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512 
        )

        return pipe
    

# COMMAND ----------


def string_printer(out_obj, run_mode):
  """
  Short convenience function because the output formats change between CPU and GPU
  """
  if run_mode == 'cpu':
    return out_obj
  elif run_mode == 'gpu':
    return out_obj[0]['generated_text']
