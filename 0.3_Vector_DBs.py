# Databricks notebook source
# MAGIC %md
# MAGIC # Exploring Vector DBs
# MAGIC In this notebook we will explore the process of converting text to numbers and what that means for our sentences
# MAGIC We will use the faiss library which provides a large variety of different algorithms that you can try out.
# MAGIC The difference between FAISS and a full Vector Database solution is around things like governance, 
# MAGIC convenience features like updates and production grade featuers like failover and backups.

# COMMAND ----------

%pip install faiss-cpu wikipedia llama_index==0.8.54

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %run ./utils

# COMMAND ----------

import faiss
import wikipedia
import os

# COMMAND ----------

# MAGIC %md
# MAGIC # Load some sample data
# MAGIC We will use wikipedia for our initial sample data
# COMMAND ----------

# Load Sample Data
result = wikipedia.search("Neural networks")

print(result)

# COMMAND ----------

# get the first article
page = wikipedia.page(result[0])
len(page.content.split())

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Embedding Model
# MAGIC In this example, we will use the tokeniser from zephyr-7b to start
# MAGIC 
# MAGIC *NOTE* When we build out our full architecture there will be two functions that turn text to tokens.
# MAGIC - Model Tokenizer - This component we are experimenting with here
# MAGIC - Embedding Tokenizer - This will be explored later and is used to populate the VectorDB
# MAGIC
# MAGIC Whilst the _Model Tokenizer_ is set, you have to use the one intended for your model, the _Embedding Tokenizer_ is something 
# MAGIC that we can select to suit our use case

# COMMAND ----------
from transformers import AutoTokenizer

model_id = 'HuggingFaceH4/zephyr-7b-beta'
model_revision = '3bac358730f8806e5c3dc7c7e19eb36e045bf720'
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=dbfs_tmp_cache)


# COMMAND ----------

# MAGIC %md
# MAGIC # Explore tokenization
# MAGIC Lets explore the way that words are encoded for our LLM

# COMMAND ----------

tokenizer.encode('test')

# COMMAND ----------

tokenizer.encode('party')
tokenizer.encode('partying')

# this word exists!!!!
tokenizer.encode('Pneumonoultramicroscopicsilicovolcanoconiosis')

# COMMAND ----------
# lets decode a little and see what the codes mean
medical_encode = tokenizer.encode('Pneumonoultramicroscopicsilicovolcanoconiosis')
tokenizer.decode(medical_encode[0])

# COMMAND ----------

tokenizer.encode('I am happily eating pizza all day long')

# COMMAND ----------

# MAGIC %md
# MAGIC # Sentence Transformers for Embedding tokenization
# MAGIC The Sentence Transformers library provides a series of embedding algorithms that can be used to popuiate our VectorDB.
# MAGIC Unlike the _Model Tokenizer_ which produced a variable length output depending on the input.
# MAGIC An embedding algorithm produces a fixed length vector so that we can run approximate nearest neighbour algorithms.

# COMMAND ----------

from sentence_transformers import SentenceTransformer
# initialize sentence transformer model
model = SentenceTransformer('bert-base-nli-mean-tokens')
# COMMAND ----------

# Split the document into paragraphs
paragraph_form = page.content.split('\n\n')

len(paragraph_form)

# COMMAND ----------

# MAGIC %md
# MAGIC Encode the paragraphs into dence vectors
# MAGIC Different models will produce a different length vector
# MAGIC In theory, a model that produces a longer length can represent the input data better.
# MAGIC But really it depends on the type of data it is trained on.
# MAGIC
# MAGIC ie a Sentence Transformer that produces 512 length vectors BUT is trained on medical data 
# MAGIC will provide a better representation for medical documents than a Sentence Transformer that produces 1024 length vectors 
# MAGIC but is only trained on social media.

# COMMAND ----------

sentence_encode = model.encode(paragraph_form)
sentence_encode.shape
# COMMAND ----------

# MAGIC %md
# MAGIC # Lets build out a FAISS index
# MAGIC FAISS lets us experiment with a wide variety of different search algorithms
# MAGIC Most VectorDBs will offer just one option.

# COMMAND ----------

index = faiss.IndexFlatL2(sentence_encode.shape[1])

# COMMAND ----------

index.add(sentence_encode)
# COMMAND ----------

# now we can search!

num_results = 3

question = 'Were animals used in neural network development'
query_vector = model.encode([question])

score, index_id = index.search(query_vector, num_results)


# COMMAND ----------

# Retrieve Index id

print(f'index ids retrieved are: {index_id}\n')

for x in index_id[0]:
    print(f'Entry: {x}')
    print(f'{paragraph_form[x]}\n')

# COMMAND ----------

# MAGIC %md
# MAGIC # Discussion
# MAGIC The main goal in this exercise is the find the best snippets.\
# MAGIC Specifically for Vector embeddings there are many algorithms\
# MAGIC You can look at some of the varieties here: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes\
# MAGIC Generally you are trading off between speed of indexing / retrieval and accuracy.


