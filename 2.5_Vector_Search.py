# Databricks notebook source
# MAGIC %pip install --upgrade --force-reinstall databricks-vectorsearch langchain==0.1.10 sqlalchemy==2.0.27 pypdf==4.1.0 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Setting Up Databricks Vector Search

# COMMAND ----------

# We will create the following source Delta table.
source_catalog = "bootcamp_ml"
source_schema = "rag_chatbot"
source_volume = "datasets"
source_table = "arxiv_parse"
vs_endpoint = "bootcamp_vs_endpoint"
embedding_endpoint_name = "databricks-bge-large-en"

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS bootcamp_ml;
# MAGIC CREATE SCHEMA IF NOT EXISTS bootcamp_ml.rag_chatbot;
# MAGIC CREATE VOLUME IF NOT EXISTS bootcamp_ml.rag_chatbot.datasets

# COMMAND ----------

# MAGIC %md
# MAGIC # Loading Data

# COMMAND ----------

import urllib
file_uri = 'https://arxiv.org/pdf/2203.02155.pdf'
volume_path = f'/Volumes/{source_catalog}/{source_schema}/{source_volume}/2203.02155.pdf'
urllib.request.urlretrieve(file_uri, volume_path)

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Ingestion Pipeline

# COMMAND ----------

# 1) Chunk the documents
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import os

def chunk_pdf_from_dir(directory:str='./docs'):

    documents = []
    for file in os.listdir(directory):
        pdf_path = os.path.join(directory, file)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(documents)

    return document_chunks

docs = chunk_pdf_from_dir(directory=f"/Volumes/{source_catalog}/{source_schema}/{source_volume}")

# COMMAND ----------

# 2) setup the dataframe
import pandas as pd

decoded_docs = []
for doc in docs:
  chunk_id = 0
  decoded_docs.append(
    {
      'row_id': f"chunk_{chunk_id}",
      'page_content': doc.page_content,
      'source_doc': doc.metadata['source'],
      'doc_page': doc.metadata['page']
    }
  )
  chunk_id += 1

pandas_frame = pd.DataFrame(decoded_docs)

spk_df = spark.createDataFrame(pandas_frame)

# COMMAND ----------

display(spk_df)

# COMMAND ----------

spk_df.write.mode("overwrite").option("delta.enableChangeDataFeed", "true") \
    .saveAsTable(f'{source_catalog}.{source_schema}.{source_table}')

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup the Vector Search Endpoint

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
# vs_endpoint
vsc.get_endpoint(
  name=vs_endpoint
)

vs_index = f"{source_table}_bge_index"
vs_index_fullname = f"{source_catalog}.{source_schema}.{vs_index}"

# COMMAND ----------

# See all the indices we have
vsc.list_indexes(name=vs_endpoint)

# COMMAND ----------

index = vsc.create_delta_sync_index(
  endpoint_name=vs_endpoint,
  source_table_name=f'{source_catalog}.{source_schema}.{source_table}',
  index_name=vs_index_fullname,
  pipeline_type='TRIGGERED',
  primary_key="row_id",
  embedding_source_column="page_content",
  embedding_model_endpoint_name=embedding_endpoint_name
)
index.describe()['status']['message']

# COMMAND ----------

import time
index = vsc.get_index(endpoint_name=vs_endpoint,index_name=vs_index_fullname)
while not index.describe().get('status')['ready']:
  print("Waiting for index to be ready...")
  time.sleep(30)
print("Index is ready!")
index.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC # Similarity search

# COMMAND ----------

results = index.similarity_search(
  columns=["page_content"],
  # vs_index_fullname,
  query_text="Tell me about tuning LLMs",
  num_results=3
  )

# COMMAND ----------


