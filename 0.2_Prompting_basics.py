# Databricks notebook source
# MAGIC %md
# MAGIC # Prompting Basics

# MAGIC Lets explore the basics of prompting\
# MAGIC For more details see: https://www.promptingguide.ai/

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

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig, GenerationConfig
import torch

from transformers import LlamaTokenizer, LlamaForCausalLM


# COMMAND ----------

# MAGIC %md
# MAGIC # Building out prompts
# MAGIC We first need to make up some questions to play around with

# COMMAND ----------

# MAGIC %md
# MAGIC # Loading Model
# MAGIC chat models are base models that have been finetuned on conversations

# COMMAND ----------

model_id = 'mosaicml/mpt-7b-chat'
model_revision = 'c53dee01e05098f81cac11145f9bf45feedc5b2f'
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=dbfs_tmp_cache)

mpt_config = AutoConfig.from_pretrained(model_id,
                                          trust_remote_code=True, # needed on both sides
                                          revision=model_revision,
                                          init_device='meta'
                                      )

model = AutoModelForCausalLM.from_pretrained(model_id,
                                               revision=model_revision,
                                               trust_remote_code=True, # needed on both sides
                                               config=mpt_config,
                                               device_map='auto',
                                               torch_dtype=torch.bfloat16, # This will only work A10G / A100 and newer GPUs
                                               cache_dir=dbfs_tmp_cache
                                              )

pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer 
        )

# COMMAND ----------

# MAGIC %md
# MAGIC # Basic Prompts
# MAGIC Getting started is easy, we can send text in.
# MAGIC It is worth noting that different models will respond differently\
# MAGIC On huggingface it is common to see a model, MPT, have several variants\
# MAGIC - base
# MAGIC - instruct
# MAGIC - chat
# COMMAND ----------



# MAGIC %md
# MAGIC # Basic Prompting
# MAGIC

# COMMAND ----------

prompt = "The sky is"

print(pipe(prompt)[0]['generated_text'])

# COMMAND ----------

prompt = "The sky is"

print(pipe(prompt, max_new_tokens=40)[0]['generated_text'])

# COMMAND ----------

prompt = "The red sky is"

print(pipe(prompt)[0]['generated_text'])

# COMMAND ----------

print(pipe("Knock Kock")[0]['generated_text'])

# COMMAND ----------

prompt = "Knock Kock\
     Who's there"

print(pipe(prompt, max_new_tokens=150, repetition_penalty=1.2)[0]['generated_text'])

# COMMAND ----------

# MAGIC %md
# MAGIC # Zero Shot Prompting
# MAGIC

# COMMAND ----------

prompt = """
    Classify the text into neutral, negative or positive.
    Text: I think the vacation is okay.
    Sentiment:
"""

print(pipe(prompt, max_new_tokens=150, repetition_penalty=1.2)[0]['generated_text'])

# COMMAND ----------

prompt = """
    Classify the text into neutral, negative or positive.
    Text: I think the vacation sucks.
    Sentiment:
"""

print(pipe(prompt, max_new_tokens=150, repetition_penalty=1.2)[0]['generated_text'])

# COMMAND ----------

prompt = """
    What is the interest rate it following paragraph?
    Text: The minutes from the Fed's June 13-14 meeting show that while almost all officials deemed it “appropriate or acceptable” to keep rates unchanged in a 5% to 5.25% target range, some would have supported a quarter-point increase instead.
    Interest Rate:
"""

print(pipe(prompt, max_new_tokens=100, repetition_penalty=1.2)[0]['generated_text'])


# COMMAND ----------

# MAGIC %md
# MAGIC # Few Shot Prompting
# MAGIC

# COMMAND ----------

prompt = """
    A consumer wants a savings account
    A business wants a business account
    A tech unicorn deserves a special VC account
    What account would you recommend a small business?
"""

print(pipe(prompt, max_new_tokens=60, repetition_penalty=1.2)[0]['generated_text'])

# COMMAND ----------

prompt = """
    A consumer wants a savings account
    A business wants a business account
    A tech unicorn deserves a special VC account

    Question: What account would you recommend a consumer?
"""

print(pipe(prompt, max_new_tokens=60, repetition_penalty=1.2)[0]['generated_text'])

# COMMAND ----------


# MAGIC %md
# MAGIC # Chain of Thought Prompting
# MAGIC

# COMMAND ----------

prompt = """
    I went to the market and bought 10 apples. 
    I gave 2 apples to the neighbor and 2 to the repairman. 
    I then went and bought 5 more apples and ate 1. 
    How many apples did I remain with?
"""

print(pipe(prompt, max_new_tokens=60, repetition_penalty=1.2)[0]['generated_text'])


# COMMAND ----------

prompt = """
    I went to the market and bought 10 apples. 
    I gave 2 apples to the neighbor and 2 to the repairman. 
    I then went and bought 5 more apples and ate 1. 
    How many apples did I remain with?
    Think through it step by step:

"""

print(pipe(prompt, max_new_tokens=100, repetition_penalty=1.2)[0]['generated_text'])

# COMMAND ----------

# MAGIC %md
# MAGIC # Constructing a system prompt
# MAGIC

# COMMAND ----------

user_question = """
    I have 20 cars.
    I crashed 1 and sold 3 others.
    I then went and bought 1 back.
    How many cars do I have?
"""

prompt = f"""
    
    Question:
    I went to the market and bought 10 apples. 
    I gave 2 apples to the neighbor and 2 to the repairman. 
    I then went and bought 5 more apples and ate 1. 
    How many apples did I remain with?

    Answer:
    I had 10 apples -2 for neighbor and -2 for repair man
    10 - 2 - 2 = 6
    I bought 5
    6 + 5 = 11
    and ate 1
    11 - 1 = 10
    So the answer is 10
    
    Based on the above answer the following question

    Question:
    {user_question}

    Think through it step by step:
"""

print(pipe(prompt, max_new_tokens=100, repetition_penalty=1.2)[0]['generated_text'])



# COMMAND ----------

# MAGIC %md
# MAGIC # Retrieval Augmented Generation
# MAGIC
# MAGIC Now if I ask the bot about something left of field
# MAGIC It probably cannot answer
# MAGIC Training is expensive
# MAGIC What if we gave it an except?
# MAGIC


# COMMAND ----------

prompt = """
    What happens to GNNs as you add layers?
"""

print(pipe(prompt, max_new_tokens=100, repetition_penalty=1.2)[0]['generated_text'])

# COMMAND ----------

user_question = 'What happens to GNNs as you add layers?'

prompt = f"""
    page-context:

    Graph neural networks (GNNs), a type of neural network that can learn from graphstructured data and learn the representation of nodes through aggregating neighborhood information, have shown superior performance in various downstream
tasks. However, it is known that the performance of GNNs degrades gradually as
the number of layers increases. 

    Based on the page context, answer the following question.
    Question: {user_question}
"""

print(pipe(prompt, max_new_tokens=200, repetition_penalty=1.2)[0]['generated_text'])

# COMMAND ----------


