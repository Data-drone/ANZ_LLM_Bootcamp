# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluations
# MAGIC Running Evaluations on RAGs is still more art than science \
# MAGIC We will use llama_index to assist in generating evaluation questions \
# MAGIC And the ragas library for generating metrics to assess your RAG \
# MAGIC 
# MAGIC We use an older llama_index to align with MLR 13.3 LTS Langchain version \
# MAGIC as llama_index relies a lot on Langchain

# COMMAND ----------

%pip install llama_index==0.8.9

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
# MAGIC - An Index - this is the core of llama index. At it's base, an index consists of a complex structure of nodes which contain text and embeddings 

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

import os
import openai

# COMMAND ----------

# DBTITLE 1,Configurations
test_pdf = f'{dbfs_source_docs}/2010.11934.pdf'
test_pdf

# COMMAND ----------

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
# MAGIC We are using AzureOpen AI so the setup is a little different. \
# MAGIC Azure OpenAI notably requires two deployments, an embedder and the model \
# MAGIC We will demonstrate a hybrid setup here where we use a huggingface sentence transformer \
# MAGIC that will do the embeddings for our vector store \
# MAGIC Whilst AzureOpenAI (gpt-3.5-turbo) provides the brains

# COMMAND ----------

from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
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

# Azure OpenAI Embeddings - needed cause ragas uses async
embedding_llm = LangchainEmbedding(
    OpenAIEmbeddings(
        model="text-embedding-ada-002",
        deployment="dbdemos-embedding",
        openai_api_key=openai_key,
        openai_api_base=os.environ['OPENAI_API_BASE'],
        openai_api_type=openai.api_type,
        openai_api_version=os.environ['OPENAI_API_VERSION'],
    ),
    embed_batch_size=1,
)

# See: https://github.com/openai/openai-python/issues/318
llm = AzureChatOpenAI(deployment_name=deployment_name,
                      model_name="gpt-35-turbo")

llm_predictor = LLMPredictor(llm=llm)

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, 
                                               embed_model=embedding_llm,
                                               callback_manager = callback_manager 
                                               )

# we can now set this context to be a global default
set_global_service_context(service_context)

# COMMAND ----------

# MAGIC %md
# MAGIC # Load and Chunk Document
# MAGIC We will load a sample doc to test on, firstly with a naive default chunking strategy
# MAGIC
# COMMAND ----------

# chunk the output
from llama_index import (
    download_loader, VectorStoreIndex
)
from pathlib import Path

PDFReader = download_loader('PDFReader')
loader = PDFReader()

# This produces a list of llama_index document objects
documents = loader.load_data(file=Path(test_pdf))

# COMMAND ----------

# we are just setting up a simple in memory Vectorstore here
index = VectorStoreIndex.from_documents(documents)

# and turning it into a query engine
query_engine = index.as_query_engine()

# Let's validate that it is all working
reply = query_engine.query('what is a neural network?')

print(reply.response)

# COMMAND ----------

# MAGIC %md
# MAGIC # Build out evaluation Questions
# MAGIC In order to run evaluation we need to have feasible questions to feed the model \
# MAGIC It is time consuming to manually construct questions so we will use a LLM to do this \
# MAGIC Note that this will have limitations, namely in the types of questions it will generate
# COMMAND ----------

from llama_index.evaluation import DatasetGenerator, QueryResponseEvaluator

# this is the question generator. Note that it has additional settings to customise prompt etc
data_generator = DatasetGenerator.from_documents(documents=documents)

# this is the call to generate the questions
eval_questions = data_generator.generate_questions_from_nodes()

# COMMAND ----------

# MAGIC %md
# MAGIC # Use Questions to generate evaluations
# MAGIC Now we have our queries we need to run some responses
# MAGIC
# MAGIC This next step can be slow so we will cut it down to 20 questions \
# MAGIC We can then use the `QueryResponseEvaluator`` looks at whether the query is answered by the response

# COMMAND ----------

import pandas as pd

eval_questions = eval_questions[0:20]

# Yes we are using a LLM to evaluate a LLM
evaluator_azure_openai = QueryResponseEvaluator()

# lets create and log the data properly
results = []

for question in eval_questions:
    
    engine_response = query_engine.query(question)
    evaluation = evaluator_azure_openai.evaluate(question, engine_response)
    results.append(
      {
        "query": question,
        "response": str(engine_response.response),
        "source": engine_response.source_nodes[0].node.text,
        "evaluation": evaluation
      }   
    )

# we will load it into a pandas frame: 
response_df = pd.DataFrame(results)

# Lets do a simple YES/NO evaluation
evaluation_counts = response_df.groupby('evaluation').size().reset_index(name='Count')

# COMMAND ----------

# Let see what is in the frame
response_df

# COMMAND ----------

evaluation_counts

# COMMAND ----------
