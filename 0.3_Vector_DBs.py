# Databricks notebook source
# MAGIC %md
# MAGIC # Exploring Vector DBs

# COMMAND ----------

%pip install faiss-cpu wikipedia

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
# MAGIC # Get some sample data
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

# COMMAND ----------
from transformers import AutoTokenizer

model_id = 'mosaicml/mpt-7b'
model_revision = '72e5f594ce36f9cabfa2a9fd8f58b491eb467ee7'
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=dbfs_tmp_cache)


# COMMAND ----------

# MAGIC %md
# MAGIC # Explore tokenization

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
# MAGIC # Sentence Encoders for FAISS
# MAGIC Tokenizers from an LLM and for VectorStores are a bit different
# MAGIC SentenceTransformers from Huggingface is focused on the latter.
# COMMAND ----------

from sentence_transformers import SentenceTransformer
# initialize sentence transformer model
model = SentenceTransformer('bert-base-nli-mean-tokens')
# COMMAND ----------

paragraph_form = page.content.split('\n\n')

len(paragraph_form)

# COMMAND ----------

# MAGIC %md
# MAGIC Tokenizations work best when it receives chunks of the same size
# MAGIC 
# COMMAND ----------

sentence_encode = model.encode(paragraph_form)
sentence_encode.shape
# COMMAND ----------

# MAGIC %md
# MAGIC # Lets build out a FAISS index

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


