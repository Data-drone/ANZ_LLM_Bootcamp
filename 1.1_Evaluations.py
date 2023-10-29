# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluations
# MAGIC Running Evaluations on RAGs is still more art than science \
# MAGIC We will use llama_index to assist in generating evaluation questions \
# MAGIC And use the inbuilt assessment prompt in llama_index \

# COMMAND ----------

%pip install llama_index==0.8.54 spacy ragas==0.0.18

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

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

import os

# COMMAND ----------

# DBTITLE 1,Configurations
test_pdf = '/dbfs/bootcamp_data/pdf_data/2302.09419.pdf'
#test_pdf = f'{dbfs_source_docs}/2302.09419.pdf'
test_pdf

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Service Context
# MAGIC The service context sets up the LLM and embedding model that we will use for our exercises
# MAGIC In this case, the Embedding Model and the LLM are both setup onto Databricks serving

# COMMAND ----------

from llama_index import (
  ServiceContext,
  set_global_service_context,
  LLMPredictor
)
from llama_index.callbacks import CallbackManager, LlamaDebugHandler, CBEventType

# Using Databricks Model Serving
browser_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
db_host = f"https://{browser_host}"
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

serving_uri = 'vicuna_13b'
serving_model_uri = f"{db_host}/serving-endpoints/{serving_uri}/invocations"

embedding_uri = 'brian_embedding_endpoint'
embedding_model_uri = f"{db_host}/serving-endpoints/{embedding_uri}/invocations"

llm_model = ServingEndpointLLM(endpoint_url=serving_model_uri, token=db_token)

llm_predictor = LLMPredictor(llm=llm_model)

### define embedding model setup
from langchain.embeddings import MosaicMLInstructorEmbeddings
embeddings = ModelServingEndpointEmbeddings(db_api_token=db_token)

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, 
                                               embed_model=embeddings,
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

from llama_index.evaluation import DatasetGenerator, RelevancyEvaluator

# this is the question generator. Note that it has additional settings to customise prompt etc
data_generator = DatasetGenerator.from_documents(documents=documents, service_context=service_context)

# this is the call to generate the questions
eval_questions = data_generator.generate_questions_from_nodes()
eval_questions

# Some of these questions might not be too useful. It could be because of the model we are using for generation
# It could also be that the chunk is particularly bad

# COMMAND ----------

# MAGIC %md
# MAGIC # Use Questions to generate evaluations
# MAGIC Now we have our queries we need to run some responses
# MAGIC
# MAGIC This next step can be slow so we will cut it down to 20 questions \
# MAGIC We can then use the `ResponseEvaluator`` looks at whether the query is answered by the response

# COMMAND ----------

import pandas as pd

eval_questions = eval_questions[0:20]

# Yes we are using a LLM to evaluate a LLM
## When doing this normally you might use a more powerful but more expensive evaluator
## to assess the quality of your input
evaluator = RelevancyEvaluator(service_context=service_context)

# lets create and log the data properly
results = []

for question in eval_questions:
    
    engine_response = query_engine.query(question)
    evaluation = evaluator.evaluate_response(question, engine_response)
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
# COMMAND ----------

# Let see what is in the frame
response_df

# COMMAND ----------

# MAGIC %md # Introducing the Ragas Framework
# MAGIC Ragas provides us with a more comprehensive framework for evaluation
# MAGIC Lets see how the Question Generator for Ragas works

# COMMAND ----------

from ragas.testset import TestsetGenerator

question_generator = TestsetGenerator(generator_llm=llm_model,
                                      critic_llm=llm_model,
                                      embeddings_model=embeddings,
                                      chat_qa=0.0)

# COMMAND ----------

generation_result = question_generator.generate(documents=documents, test_size=10)

pd_generation_result = generation_result.to_pandas()
pd_generation_result

# COMMAND ----------

# Due to our bad chunking, some of the questions are pretty bad
# We can do a basic filter
filtered_result = pd_generation_result[pd_generation_result.str.len() >= 50]
filtered_result

# COMMAND ----------

# We can spot check some of the data
filtered_result.iloc[0].question

# COMMAND ----------

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.metrics.critique import harmfulness

faithfulness.llm.langchain_llm = llm_model
answer_relevancy.llm.langchain_llm = llm_model
context_precision.llm.langchain_llm = llm_model
context_recall.llm.langchain_llm = llm_model
harmfulness.llm.langchain_llm = llm_model

metrics = [
    faithfulness,
    #answer_relevancy,
    #context_precision,
    #context_recall,
    #harmfulness,
]

# COMMAND ----------

from ragas.llama_index import evaluate

generated_questions = filtered_result['question'].astype(str).tolist()
generated_ground_truths = filtered_result['answer'].astype(str).tolist()

result = evaluate(query_engine=query_engine, 
                  metrics=metrics, 
                  questions=generated_questions) #, 
                  #ground_truths=generated_ground_truths)
# COMMAND ----------