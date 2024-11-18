# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluations
# MAGIC Running Evaluations on RAGs is still more art than science \
# MAGIC We will use llama_index to assist in generating evaluation questions \
# MAGIC And use the inbuilt assessment prompt in llama_index \

# COMMAND ----------

# MAGIC %pip install databricks-langchain llama_index==0.11.23 langchain==0.3.7 langchain-community==0.3.7 llama-index-llms-langchain llama-index-embeddings-langchain swifter databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import pandas as pd
import nest_asyncio

# Needed for the async calls to work
nest_asyncio.apply()

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

# MAGIC %md
# MAGIC ## Setting up Llama Index default models
# MAGIC If we don't setup new defaults, Llama_index will go to OpenAI by default

# COMMAND ----------

from databricks_langchain import ChatDatabricks
from databricks_langchain import DatabricksEmbeddings
from llama_index.core import Settings
from llama_index.llms.langchain import LangChainLLM
from llama_index.embeddings.langchain import LangchainEmbedding

embedding_model = 'databricks-gte-large-en'
model_name = 'databricks-meta-llama-3-1-405b-instruct'

llm_model = model = ChatDatabricks(
  target_uri='databricks',
  endpoint = model_name,
  temperature = 0.1
)

embeddings = DatabricksEmbeddings(endpoint=embedding_model)

llama_index_chain = LangChainLLM(llm=llm_model)
llama_index_embeddings = LangchainEmbedding(langchain_embeddings=embeddings)
Settings.llm = llama_index_chain 
Settings.embed_model = llama_index_embeddings 

# COMMAND ----------

# MAGIC %md
# MAGIC # Load and Chunk Documents
# MAGIC We will load a sample doc to test on, firstly with a naive default chunking strategy
# MAGIC
# COMMAND ----------

vol_path = f'/Volumes/{db_catalog}/{db_schema}/{db_volume}/'

# validate we have files
os.listdir(vol_path)

# COMMAND ----------

from llama_index.core import (
  SimpleDirectoryReader, VectorStoreIndex, Response   
)

reader = SimpleDirectoryReader(vol_path)
documents = reader.load_data()

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

from llama_index.core.evaluation import DatasetGenerator

data_generator = DatasetGenerator.from_documents(documents)


# this is the call to generate the questions
# if you set the number it will run multithreaded and be gaster
eval_questions = data_generator.generate_questions_from_nodes(num=16)
eval_questions

# Some of these questions might not be too useful. It could be because of the model we are using for generation
# It could also be that the chunk is particularly bad

# COMMAND ----------

# Lets generate the answer to the questions
# When running in lab env we may pregenerate ahead of the class and store it for reloading
#question_frame = spark.sql(f"SELECT * FROM {db_catalog}.{db_schema}.evaluation_questions").toPandas()
question_frame = pd.DataFrame(eval_questions, columns=["request"])

#query_engine.query('Hihihihi').source_nodes[0].text
question_frame['query_result'] = question_frame['request'].apply(lambda x: query_engine.query(x))

question_frame['expected_response'] = question_frame['query_result'].apply(lambda x: x.response)

question_frame['retrieved_context'] = question_frame['query_result'].apply(
  lambda x: [{'doc_uri': f"{x.metadata['file_path']} -  page {x.metadata['page_label']}",
  "content": x.text} for x in question_frame['query_result'][0].source_nodes]
)

response_frame = question_frame.drop(columns=['query_result'], axis=1)

dataframe = spark.createDataFrame(response_frame)

dataframe.write.mode("overwrite").saveAsTable(f"{db_catalog}.{db_schema}.evaluation_questions")
display(dataframe)
# COMMAND ----------

# MAGIC %md
# MAGIC # Run Databricks AI Evaluations
# MAGIC
# MAGIC Lets see how to run these evaluations
# COMMAND ----------

import mlflow


with mlflow.start_run():
    
    mlflow.evaluate(
        data=response_frame, # Evaluation set 
        model_type="databricks-agent", # Use Agent Evaluation
    )