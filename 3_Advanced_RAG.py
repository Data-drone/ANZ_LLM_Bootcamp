# Databricks notebook source
# MAGIC %md
# MAGIC # Building a document store
# MAGIC We will now build out a larger document store persist and use that

# COMMAND ----------

# DBTITLE 1,Extra Libs to install
# MAGIC %pip install pypdf sentence_transformers pymupdf ctransformers

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import glob
import re
import os

import chromadb
from chromadb.config import Settings

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# COMMAND ----------

# DBTITLE 1,Setup dbfs folder paths
%run ./utils

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Document Store
# MAGIC The document store has to be created first.
# MAGIC We need to have some sort of index and we will need to manage this ourselves.

# COMMAND ----------
# for class
#source_docs = glob.glob('/dbfs/bootcamp_data/pdf_data/*.pdf')
source_docs = glob.glob(dbfs_source_docs+'/*.pdf')

collection_name = 'arxiv_articles'

# We will use default HuggingFaceEmbeddings for now

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                   model_kwargs={'device': 'cpu'})

def embed_fn(text):
  hfe = HuggingFaceEmbeddings()
  return hfe.embed_documents(text)

# setup Chroma client with persistence
client = chromadb.chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
                                    persist_directory=linux_vector_store_directory),
                                    )

rebuild = True

# COMMAND ----------

# MAGIC %md
# MAGIC # Build ChromaDB
# MAGIC See chroma docs for more information

# COMMAND ----------

if rebuild:

  dbutils.fs.rm(f'dbfs:{linux_vector_store_directory}', True)

# COMMAND ----------

# Initiate the ChromaDB
# Create collection. get_collection, get_or_create_collection, delete_collection also available!
## Colection is where we set embeddings? # embedding_function=embed_fn
collection = client.get_or_create_collection(name=collection_name)
print(f"we have {collection.count()} in the collection.")

# COMMAND ----------

# DBTITLE 1,Collection Building Function
# we can look at other splitters later. 
# Probably Paragraph? And / Or Sentence?
def collection_builder(source_docs:list, 
                        collection:chromadb.api.models.Collection.Collection):

  assert collection.count() == 0, "WARNING This function will append to collection regardless of whether it already exists or not"

  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

  # we will process page by page

  for doc in source_docs:
    
    # This regex will only work for arxiv
    match = re.search(r'/([\d.]+)\.pdf$', doc)
    article_number = match.group(1)

    loader = PyMuPDFLoader(doc)
    pages = loader.load_and_split()

    # for page in pages:
    #print(type(page))
    texts = text_splitter.split_documents(pages)
    #print(texts)
    # print(len(texts))    

    doc_list = [x.page_content for x in texts]
    embed_list = embeddings.embed_documents(doc_list)

    collection.add(
      documents=doc_list,
      embeddings=embed_list,
      metadatas=[x.metadata for x in texts],
      ids=[article_number+str(texts.index(x)) for x in texts]
    )

  # See: https://github.com/chroma-core/chroma/issues/275
  client.persist()    

# COMMAND ----------

try:
  collection_builder(source_docs, collection)
  print(f"we now have {collection.count()} in the collection.")
except AssertionError:
  print("Doing nothing, we will not rebuild the collection")

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup LLM to interface with chroma DB
# MAGIC NOTE that reloading with langchain seems glitchy hence why we need to do it manually 

# COMMAND ----------

# Load the collection
# we reuse the previous client and embeddings
docsource = Chroma(collection_name=collection_name,
                  persist_directory=linux_vector_store_directory,
                  embedding_function=embeddings)

# we can verify that our docsearch index has objects in it with this
print('The index includes: {} documents'.format(docsource._collection.count()))



# COMMAND ----------

# MAGIC %md
# MAGIC Note that the llm_model funciton doesn't clean up after itself. so if you call it repeatedly it will fill up the VRAM
# MAGIC
# MAGIC We will add some code to quickly stop reinitiating
# MAGIC In order to understand the HuggingFace Pipeline we need to look at: 
# MAGIC - https://huggingface.co/docs/transformers/main_classes/pipelines
# MAGIC The task set for this pipe is text-generation the def of this is:
# MAGIC - https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TextGenerationPipeline
# MAGIC Device needs to be set in order to utilise GPU
# MAGIC - See: https://huggingface.co/transformers/v3.0.2/main_classes/pipelines.html#transformers.Pipeline


# COMMAND ----------

## One problem with the library at the moment is that GPU ram doesn't get relinquished when the object is overridden
# The only way to clear GPU ram is to detach and reattach
# This snippet will make sure we don't keep reloading the model and running out of GPU ram
try:
  llm_model
except NameError:
  if run_mode == 'cpu':
    # the cTransformers class interfaces with langchain differently
    from ctransformers.langchain import CTransformers
    llm_model = CTransformers(model='TheBloke/Llama-2-7B-Chat-GGML', model_type='llama')
  elif run_mode == 'gpu':
    pipe = load_model(run_mode, dbfs_tmp_cache)
    llm_model = HuggingFacePipeline(pipeline=pipe)

else:
  pass

# COMMAND ----------

# Broken at the moment
qa = RetrievalQA.from_chain_type(llm=llm_model, chain_type="stuff", 
                                 retriever=docsource.as_retriever(search_kwargs={"k": 1}))

# COMMAND ----------

# DBTITLE 1,Verify docsource is valid
# Basic Vector Similarity Search
query = "What is this is a token limit?"
query_embed = embeddings.embed_query(query)

docsource._collection.query(query_embeddings=query_embed, n_results=2)


# COMMAND ----------
# Lets test out querying

# Something is wrong with the similarity search? Are my embeddings not saving?
# Also the docsource has a different embedding structure (vectors don't line up)

query_embed = embeddings.embed_query(query)
query_embed

docs = docsource.similarity_search_by_vector(query_embed)

# Broken at the moment
resp = qa({"query": query}, return_only_outputs=True)
resp

# COMMAND ----------
# MAGIC %md NOTE setting k greater than 2?

# COMMAND ----------

# Broken at the moment
qa.run(query)

# COMMAND ----------
