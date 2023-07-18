# Databricks notebook source
# MAGIC %md
# MAGIC # Supplementary Examples on HuggingFace
# MAGIC Huggingface🤗 provides a series of libraries that are critical in the OSS llm space\
# MAGIC To use them properly later on and debug issues we need to have an understand of how they work\
# MAGIC This is not a full tutorial. See HuggingFace docs for that. But it will function as a crash course.
# MAGIC
# MAGIC In these exercises we will focus on the _transformers_ library but _datasets_, _evaluate_ and _accelerate_ are commonly used in training models. 
# MAGIC
# MAGIC All code here is tested on MLR 13.2 on a g5 AWS instance (A10G GPU).
# MAGIC We suggest a ```g5.4xlarge``` single node cluster to start
# MAGIC The Azure equivalent is ```NC6s_v3``` series. However, for this lab we will be using ```g5.4xlarge``` instances.
# MAGIC ----
# MAGIC **Notes**
# MAGIC - Falcon requires Torch 2.0 coming soon....
# MAGIC - The LLM Space is fast moving. Many models are provided by independent companies as well so model revision and pinning library versions is important.
# MAGIC - If using an MLR prior to 13.2, you will need to run ```%pip install einops```
# MAGIC - It may also be necessary to manually install extra Nvidia libraries via [init_scripts](https://docs.databricks.com/clusters/init-scripts.html)
# MAGIC - Sometimes huggingface complains about xformers you can add that install to the below pip command (```%pip install xformers```)

# COMMAND ----------

# DBTITLE 1,Install ctransformers for CPU inference
%pip install ctransformers==0.2.13

# COMMAND ----------

dbutils.library.restartPython()
# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup 🚀

# COMMAND ----------

# MAGIC %md
# MAGIC ### DBFS Cache
# MAGIC Configure databricks storage locations and caching. By default databricks uses dbfs to store information.\ 
# MAGIC HuggingFace will by default cache to a root path folder. We can change that so that we don't have to redownload if the cluster terminates.
# MAGIC [dbutils](https://docs.databricks.com/dev-tools/databricks-utils.html) is a databricks utility for working with the object store tier.
# COMMAND ----------
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

run_mode = 'cpu'

# COMMAND ----------

# MAGIC %md
# MAGIC ## HuggingFace🤗 Crash Course
# MAGIC
# MAGIC There are a few key components that we need to construct an llm object that we can converse with.\
# MAGIC - [Tokenizers](https://huggingface.co/docs/transformers/main_classes/tokenizer) in HuggingFace🤗, the tokenizers are responsible for preparing text for input to transformer models. More technically, they are responsible for taking text and mapping them to a vector representation in integers (which can be interpretted by our models). We will explore this now...
# MAGIC
# MAGIC - [Models](https://huggingface.co/models) are pretrained versions of various transformer architectures that are used for different natural language processing tasks. They are encapsulations of complex neural networks, each with pre-trained weights that can either be used directly for inference or further fine-tuned on specific tasks. Each 
# MAGIC
# MAGIC Once we have the tokenizer and the model then we can put all into a pipeline\
# MAGIC Note with Huggingface components each object will have it's own configuration parameters. ie
# MAGIC - tokenizer configs
# MAGIC - model configs
# MAGIC - pipeline configs
# MAGIC
# MAGIC One known issue is if you run the code that loads a model twice then it will not overwrite GPU memory.\
# MAGIC It will load the new copy in fresh memory and you can get an `Out of Memory` (OOM) error.\
# MAGIC The easiest fix is to [Detach & Attach](https://docs.databricks.com/notebooks/notebook-ui.html#detach-a-notebook)
# MAGIC When loading models, setting the revision can be important to replicate behaviour. See: [HuggingFace🤗 Repository Docs](https://huggingface.co/docs/transformers/model_sharing#repository-features) 
# MAGIC
# MAGIC When working with standard model objects then it we can use all the normal APIs.\
# MAGIC But To make llms fast enough to run on CPU we need to leverage a couple of other opensource components.\
# MAGIC These are not standard huggingface components so work a bit differently.\
# MAGIC - [ggml](https://github.com/ggerganov/ggml) Which is a specialised tensor library for fast inference.\ 
# MAGIC 
# MAGIC - [ctransformer](https://github.com/marella/ctransformers) A wrapper for ggml to give it a python API 
# MAGIC
# MAGIC The CPU version loads differently and we essentially get the model object straight away without having to define the tokenizer.
# MAGIC
# MAGIC To ensure that we have consistency between CPU and GPU experiences, we will use the model - open-llama-7B-v2-open-instruct\
# MAGIC Since this is available in CPU optimized and GPU formats

# COMMAND ----------

from transformers import AutoTokenizer, pipeline, AutoConfig, GenerationConfig
import torch


if run_mode == 'cpu':

  ### Note that caching for TheBloke's models don't follow standard HuggingFace routine
  # You would need to `wget` then weights then use a model_path config instead.
  # See ctransformers docs for more info
  from ctransformers import AutoModelForCausalLM
  model_id = 'TheBloke/open-llama-7B-v2-open-instruct-GGML'
  pipe = AutoModelForCausalLM.from_pretrained(model_id,
                                           model_type='llama')

elif run_mode == 'gpu':
  from transformers import AutoModelForCausalLM
  model_id = 'VMware/open-llama-7b-v2-open-instruct'
  model_revision = 'b8fbe09571a71603ab517fe897a1281005060b62'

  # note when on gpu then this will auto load to gpu
  # this will take approximately an extra 1GB of VRAM
  tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=dbfs_tmp_cache)

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
                                               cache_dir=dbfs_tmp_cache
                                              )
  
  pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer 
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Understanding Generation Config
# MAGIC
# MAGIC We created a `pipe` entity above. That can be used to generate output from our llm\
# MAGIC The syntax is: `output = pipe(<text input>, **kwargs)`
# MAGIC
# MAGIC If you are using GPU then the output will be a list of dictionaries
# MAGIC If you are on CPU then the output will be a string
# MAGIC
# MAGIC **Key Parameters**
# MAGIC
# MAGIC - **max_new_tokens**: Defines the maximum number of tokens produced during text generation. Useful for controlling the length of the output.
# MAGIC - **temperature**: Adjusts the randomness in the model's output. Lower values yield more deterministic results, higher values introduce more diversity.
# MAGIC - **repetition_penalty**: Some models will repeat themselves unless you set a repetition penalty

# COMMAND ----------

# MAGIC %md <img src="https://files.training.databricks.com/images/icon_note_32.png" alt="Note"> In the code below, we reference the ```repetition_penalty```. Is a parameter to penalise the model for repetition. ```1.0``` implies no penalty. This penalty is applied during the sampling phase by discounting the scores of previously generated tokens. In a greedy sampling scheme, this incentivies model exploration. Please see this paper for further details [https://arxiv.org/pdf/1909.05858.pdf](https://arxiv.org/pdf/1909.05858.pdf).

# COMMAND ----------

def string_printer(out_obj, run_mode):
  """
  Short convenience function because the output formats change between CPU and GPU
  """
  if run_mode == 'cpu':
    print(out_obj)
  elif run_mode == 'gpu':
    print(out_obj[0]['generated_text'])

# COMMAND ----------

# We seem to need to set the max length here for mpt model
output = pipe("Tell me how you have been and any signifcant things that have happened to you?", max_new_tokens=20, repetition_penalty=1.2)
string_printer(output, run_mode)

# COMMAND ----------

# repetition_penalty affects whether we get repeats or not
output = pipe("Tell me how you have been and any signifcant things that have happened to you?", max_new_tokens=200, repetition_penalty=1.2)
string_printer(output, run_mode)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Advanced Generation Config
# MAGIC For a full dive into generation config see the [docs](https://huggingface.co/docs/transformers/generation_strategies)\
# MAGIC **NOTE** ctransformers does not support all the same configs. See [docs](https://github.com/marella/ctransformers#method-llmgenerate)\
# MAGIC The ones that are supported will run the same way

# COMMAND ----------

output = pipe("Tell me about what makes a good burger?", max_new_tokens=500, repetition_penalty=1.2)
string_printer(output, run_mode)

# COMMAND ----------

output = pipe("Tell me about what makes a good burger?", max_new_tokens=200, repetition_penalty=1.2, top_k=3)
string_printer(output, run_mode)


# COMMAND ----------

# COMMAND ----------

# MAGIC %md 
# MAGIC **NOTE** TODO move out
# MAGIC # Managing Prompts w MLFlow
# MAGIC We will now integate MLflow and show you how you can use it to log sample prompts and responses.\
# MAGIC

# COMMAND ----------

import mlflow

mlflow_dir = f'/Users/{username}/huggingface_samples'
mlflow.set_experiment(mlflow_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting up prompts / inputs and logging outputs
# MAGIC Mlflow logging is structured as prompts, inputs and outputs\
# MAGIC For full description see: https://mlflow.org/docs/latest/llm-tracking.html

# COMMAND ----------

# DBTITLE 1,Setup Configs
user_inputs = [
  "How can I make a coffee?",
  "How can I book a restaurant?",
  "How can I make idle chit chat when I don't know a person?"
]
prompts = []
model_outputs = []

# COMMAND ----------

with mlflow.start_run(run_name='mpt model'):
    
  for user_input in user_inputs:
    prompt = f"""
            You are an AI assistant that helps people find information and responds in rhyme. 
            If the user asks you a question you don't know the answer to, say so.

            {user_input}
            """

    raw_output = pipe(prompt, max_length=200, repetition_penalty=1.2)
    text_output = raw_output[0]['generated_text']

    mlflow.llm.log_predictions(inputs=user_input, outputs=text_output, prompts=prompt)

# COMMAND ----------

# MAGIC %md
# MAGIC - Setup Open Llama instead
# MAGIC - See: https://huggingface.co/openlm-research/open_llama_7b_v2_easylm

# COMMAND ----------

from transformers import LlamaTokenizer, LlamaForCausalLM

## v2 models
model_path = 'openlm-research/open_llama_7b_v2'

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map='auto',
)

pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer 
        )

user_input = user_inputs[0]

with mlflow.start_run(run_name='open-llama model'):
    
  for user_input in user_inputs:
    prompt = f"""
        You are an AI assistant that helps people find information and responds in rhyme. 
        If the user asks you a question you don't know the answer to, say so.
        {user_input}
        """
    raw_output = pipe(prompt, max_length=200, repetition_penalty=1.2)
    text_output = raw_output[0]['generated_text']
    mlflow.llm.log_predictions(inputs=[user_input], outputs=[text_output], prompts=[prompt])

# COMMAND ----------

# Testing mlflow evaluate



# COMMAND ----------
