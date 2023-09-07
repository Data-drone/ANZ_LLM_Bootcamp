# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluations
# MAGIC Running Evaluations on RAGs is still more art than science \
# MAGIC We will use llama_index to assist in generating evaluation questions \
# MAGIC And the ragas library for generating metrics to assess your RAG \
# MAGIC 
# MAGIC We use an older llama_index to align with MLR 13.3 LTS Langchain version \
# MAGIC as llama_index relies a lot on Langchain
# MAGIC %pip install llama_index==0.6.36 ragas

# COMMAND ----------

%pip install llama_index==0.8.9 ragas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Intro to Llama Index
# MAGIC Much like langchain, llama_index is an orchestration layer for LLM logic \
# MAGIC Where they differ is that llama_index is a lot more focused on RAGs and doing intelligent indexing \
# MAGIC Langchain is more generalist and has been focused on enabling complex workflows
# MAGIC
# MAGIC Llama Index has a few key concepts we will use for this notebook:
# MAGIC - Service Context - wrapper class to hold llm model / embeddings
# MAGIC - An Index

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

# DBTITLE 1,Configurations
run_mode = 'gpu' # or gpu
test_pdf = f'{dbfs_source_docs}/2010.11934.pdf'
test_pdf


# COMMAND ----------

import os
import openai
# For this example we will use azure openai for now
# Setup OpenAI Creds
openai_key = dbutils.secrets.get(scope='brian_dl', key='dbdemos_openai')

openai.api_type = "azure"
#openai.api_base = "https://dbdemos-open-ai.openai.azure.com/"
#openai.api_key = openai_key
#openai.api_version = "2023-07-01-preview"
os.environ['OPENAI_API_BASE'] = 'https://dbdemos-open-ai.openai.azure.com/'
os.environ['OPENAI_API_KEY'] = openai_key
os.environ['OPENAI_API_VERSION'] = "2023-07-01-preview"

deployment_name = 'dbdemo-gpt35'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Service Context
# MAGIC By default, llama_index assumes that OpenAI is the service context \
# MAGIC we will manually override these and create our own HF powered service_context

# COMMAND ----------

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import (
  ServiceContext,
  set_global_service_context,
  LLMPredictor
)
from llama_index.embeddings import LangchainEmbedding
from llama_index.callbacks import CallbackManager, LlamaDebugHandler, CBEventType
from langchain.chat_models import AzureChatOpenAI

# Azure OpenAi doesn't have an embedding set as standard so we will use that
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                   model_kwargs={'device': 'cpu'})

ll_embed = LangchainEmbedding(langchain_embeddings=embeddings)

llm = AzureChatOpenAI(deployment_name=deployment_name)
llm_predictor = LLMPredictor(llm=llm)

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, 
                                               embed_model=ll_embed,
                                               callback_manager = callback_manager 
                                               )

# we can now set this context to be a global default
set_global_service_context(service_context)

# COMMAND ----------

# MAGIC %md
# MAGIC # Load and Chunk Document
# MAGIC we will load a sample doc to test on

# COMMAND ----------

# chunk the output
from llama_index import (
    download_loader, VectorStoreIndex
)
from pathlib import Path

PDFReader = download_loader('PDFReader')
loader = PDFReader()

# This produces a list of llama_index document objects
# ? 1 document per index?
documents = loader.load_data(file=Path(test_pdf))

# COMMAND ----------

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

# Quick test
reply = query_engine.query('what is a neural network?')

print(reply.response)

# COMMAND ----------

# MAGIC %md
# MAGIC # Build out evaluation
# COMMAND ----------

from llama_index.evaluation import DatasetGenerator

data_generator = DatasetGenerator.from_documents(documents=documents)

# this takes 9 mins and gens approx 470
question = data_generator.generate_questions_from_nodes()

# COMMAND ----------