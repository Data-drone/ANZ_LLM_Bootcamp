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
# MAGIC - The LLM Space is fast moving. Many models are provided by independent companies as well so code version is important.
# MAGIC - If using an MLR prior to 13.2, you will need to run ```%pip install einops```
# MAGIC - Sometimes needed to fix loading (```%pip install xformers```)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup 🚀

# COMMAND ----------

# DBTITLE 1,Load Libs
import os

# Manual Model building
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig, GenerationConfig
import torch

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup Env
# MAGIC Configure databricks storage locations and caching. By default databricks uses dbfs to store information. HuggingFace will by default cache to a root path folder. We can change that to make things easier.

# COMMAND ----------

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

# COMMAND ----------

# Manual Model building
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig, GenerationConfig
import torch

# COMMAND ----------

# MAGIC %md
# MAGIC ## Understanding pipelines
# MAGIC
# MAGIC Once you decide which task you want to takle, language Pipelines contain two key components: the tokenizer and the LLM model itself.
# MAGIC
# MAGIC - [Tokenizers](https://huggingface.co/docs/transformers/main_classes/tokenizer) in HuggingFace🤗, the tokenizers are responsible for preparing text for input to transformer models. More technically, they are responsible for taking text and mapping them to a vector representation in integers (which can be interpretted by our models). We will explore this now...
# MAGIC
# MAGIC - [Models](https://huggingface.co/models) are pretrained versions of various transformer architectures that are used for different natural language processing tasks. They are encapsulations of complex neural networks, each with pre-trained weights that can either be used directly for inference or further fine-tuned on specific tasks. Each 
# MAGIC
# MAGIC
# MAGIC ### Loading the model
# MAGIC We will use the MPT-7B model's tokenizer. We encourage checking out the [model card](https://huggingface.co/mosaicml/mpt-7b) for further information. 
# MAGIC
# MAGIC -----
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_32.png" alt="Note"> Different models require different tokenizers due to variations in their pretraining objectives, architectures, and specific token handling needs. These models may be trained on different vocabularies or employ unique tokenization techniques such as Byte Pair Encoding or SentencePiece. Essentially, each tokenizer transforms text into a format that aligns with the training environment of its corresponding model.
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_32.png" alt="Note"> It is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of mpt-7b in https://huggingface.co/mosaicml/mpt-7b/commits/main

# COMMAND ----------

# DBTITLE 1,First we setup the model config

model_id = 'mosaicml/mpt-7b'
model_revision = '72e5f594ce36f9cabfa2a9fd8f58b491eb467ee7'
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=dbfs_tmp_cache)

# lets move to the config version
# using max_length here is deprecating max_length=4096
mpt_config = AutoConfig.from_pretrained(model_id,
                                          trust_remote_code=True, # needed on both sides
                                          revision=model_revision,
                                          init_device='meta'
                                      )

# we can save out and examine the config through this call
## Remember all models from HuggingFace are Open Source 
## sometimes the default configs have issues and we need to override them by writing it out
## and manually changing values in the json
mpt_config.save_pretrained(save_directory=dbfs_tmp_dir)

# COMMAND ----------

# MAGIC %sh
# MAGIC ls $PROJ_TMP_DIR/

# COMMAND ----------

# MAGIC %md
# MAGIC Could of notes: 
# MAGIC - Revision is important to ensure same behaviour
# MAGIC - ```device_map``` moves it to gpu
# MAGIC - ```torch.bfloat16``` helps save memory by halving numeric precision

# COMMAND ----------

model = AutoModelForCausalLM.from_pretrained(model_id,
                                               revision=model_revision,
                                               trust_remote_code=True, # needed on both sides
                                               config=mpt_config,
                                               device_map='auto',
                                               torch_dtype=torch.bfloat16, # This will only work A10G / A100 and newer GPUs
                                               cache_dir=dbfs_tmp_cache
                                              )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Understanding Generation Config
# MAGIC
# MAGIC The pipeline connects the tokenizer to the model entity itself.
# MAGIC
# MAGIC **Key Parameters**
# MAGIC
# MAGIC 1. **max_new_tokens**: Defines the maximum number of tokens produced during text generation. Useful for controlling the length of the output.
# MAGIC 2. **temperature**: Adjusts the randomness in the model's output. Lower values yield more deterministic results, higher values introduce more diversity.
# MAGIC 3. **eos_token_id**: Token ID that signifies the end of a sentence or sequence. The model stops generating tokens once it generates this token.
# MAGIC 4. **pad_token_id**: Used to equalize the length of multiple sequences by adding 'pad' tokens. For models that pay attention to padding (like BERT), these tokens are ignored.
# MAGIC

# COMMAND ----------

mpt_generation_config = GenerationConfig(
    max_new_tokens = 1024,
    temperature = 0.1,
    top_p = 0.92,
    top_k = 0, 
    use_cache = True,
    do_sample = True,
    eos_token_id = tokenizer.eos_token_id,
    pad_token_id = tokenizer.pad_token_id
  )

pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, config=mpt_generation_config 
        )

# COMMAND ----------

# MAGIC %md <img src="https://files.training.databricks.com/images/icon_note_32.png" alt="Note"> In the code below, we reference the ```repetition_penalty```. Is a parameter to penalise the model for repetition. ```1.0``` implies no penalty. This penalty is applied during the sampling phase by discounting the scores of previously generated tokens. In a greedy sampling scheme, this incentivies model exploration. Please see this paper for further details [https://arxiv.org/pdf/1909.05858.pdf](https://arxiv.org/pdf/1909.05858.pdf).

# COMMAND ----------

# We seem to need to set the max length here for mpt model
output = pipe("How are you?", max_new_tokens=200, repetition_penalty=1.2)
print(output[0]['generated_text'])

# COMMAND ----------

# repetition_penalty affects whether we get repeats or not
output = pipe("How are you?", max_new_tokens=20, repetition_penalty=1.2)
print(output[0]['generated_text'])

# COMMAND ----------

# MAGIC %md 
# MAGIC # Logging with MLflow
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
