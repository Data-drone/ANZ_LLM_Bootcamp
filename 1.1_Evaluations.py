# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluations
# MAGIC Running Evaluations on RAGs is still more art than science \
# MAGIC We will use the AI Agent Evaluations Synthetic Data API to assist in generating evaluation questions \
# MAGIC And use the inbuilt the AI Agent Evaluation Judge API to assess

# COMMAND ----------

# MAGIC %pip install databricks-agents databricks-langchain mlflow==2.21.1 langchain==0.3.21 langchain-community==0.3.20 swifter databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import pandas as pd

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

# MAGIC %md
# MAGIC # Review Existing Chunks
# MAGIC Lets look at the chunking table that we created in an earlier exercise

# COMMAND ----------

chunks = spark.sql(f"SELECT * FROM {db_catalog}.{db_schema}.{raw_table}")
display(chunks)

# COMMAND ----------

# we need to refactor the dataframe to suit the structure expected by the API
chunks_renamed = spark.sql(f"""
SELECT 
  page_content as content,
  source_doc as doc_uri
FROM 
  {db_catalog}.{db_schema}.{raw_table}
""")

# COMMAND ----------

# DBTITLE 1,Data Generation Setup
agent_description = """
The Agent is a RAG chatbot that answers questions about LLM Research. The Agent has access to a corpus of arxiv research papers, 
and its task is to answer the user's questions by retrieving the relevant docs from the corpus and synthesizing a helpful, accurate response. 
The corpus covers a lot of info, but the Agent is specifically designed to interact with research scientists who want to quickly check facts. 
Questions outside of this scope are considered irrelevant.
"""

question_guidelines = """
# User personas
- A phd level research scientist intersted in LLMs
- An experienced, highly technical researcher or engineer

# Example questions
- what can you tell me about Chain Of Thought techniques with LLMs?
- What are some of the key things that I should know about training LLM models?

# Additional Guidelines
- Questions should be succinct, and human-like
"""

# COMMAND ----------

# we would normally set the num evals to be at least as large as the number of chunks
num_evals = 10


from databricks.agents.evals import generate_evals_df

evals = generate_evals_df(
    chunks_renamed,
    # The total number of evals to generate. The method attempts to generate evals that have full coverage over the documents
    # provided. If this number is less than the number of documents, is less than the number of documents,
    # some documents will not have any evaluations generated. See "How num_evals is used" below for more details.
    num_evals=num_evals,
    # A set of guidelines that help guide the synthetic generation. These are free-form strings that will be used to prompt the generation.
    agent_description=agent_description,
    question_guidelines=question_guidelines
)

display(evals)

# COMMAND ----------

# MAGIC %md
# MAGIC # Run Databricks AI Evaluations
# MAGIC
# MAGIC As a baseline lets see how a model with no content performs
# COMMAND ----------

import mlflow

model_name = 'databricks-meta-llama-3-3-70b-instruct'

with mlflow.start_run():
    
    mlflow.evaluate(
        model=f"endpoints:/{model_name}",
        data=evals, # Evaluation set 
        model_type="databricks-agent", # Use Agent Evaluation
    )
    
# COMMAND ----------

# now lets try our previously deployed RAG agent
# OPTIONMAL - depends on having an agent already deployed

MODEL_SERVING_ENDPOINT_NAME = 'agents_brian_ml_dev-genai_workshop-rag_chain'

def agent_fn(input):
  client = mlflow.deployments.get_deploy_client("databricks")
  return client.predict(endpoint=MODEL_SERVING_ENDPOINT_NAME, inputs=input)

with mlflow.start_run():
    
    mlflow.evaluate(
        model=agent_fn,
        data=evals, # Evaluation set 
        model_type="databricks-agent", # Use Agent Evaluation
    )