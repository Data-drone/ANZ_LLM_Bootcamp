# Databricks notebook source
# MAGIC %md # Building an Agent RAG
# MAGIC 
# MAGIC Lets try out some later Agent Abstractions into the Agent Framework

# COMMAND ----------

# MAGIC %pip install -U databricks_langchain langchain==0.3.7 langchain-community==0.3.7 langgraph mlflow-skinny==2.17.2 databricks-vectorsearch 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Dev Configs
# MAGIC %run ./utils

# COMMAND ----------
# DBTITLE 1,Setup Params 
import mlflow
db_table = "arxiv_data"

# this is key to retrieving parameters during mlflow logging process
model_config = mlflow.models.ModelConfig(development_config="common_config.yaml")

paths_and_locations = model_config.get("paths_and_locations")

db_catalog = paths_and_locations.get("db_catalog")
db_schema = paths_and_locations.get("db_schema")
db_volume = paths_and_locations.get("db_volume")
raw_table = paths_and_locations.get("raw_table")
hf_volume = paths_and_locations.get("hf_volume")
vector_search_endpoint = paths_and_locations.get("vector_search_endpoint")

vs_index = f"{db_table}_vs_index"
vs_index_fullname = f"{db_catalog}.{db_schema}.{vs_index}"

# temp need to change later
embedding_model = "databricks-gte-large-en"
chat_model = "databricks-meta-llama-3-1-70b-instruct"

# COMMAND ----------

from operator import itemgetter

# Construct and Test LLM Chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableBranch

from databricks.vector_search.client import VectorSearchClient
from databricks_langchain import DatabricksVectorSearch
from databricks_langchain import ChatDatabricks
from databricks_langchain import DatabricksEmbeddings

mlflow.langchain.autolog()

# COMMAND ----------

# Here we setup the Models that we will be using and check to make sure that they are working
chat = ChatDatabricks(
    target_uri="databricks",
    endpoint=chat_model,
    temperature=0.1,
)

# Test that it is working
chat([HumanMessage(content="hello")])

# COMMAND ----------

# Construct and Test Embedding Model
embeddings = DatabricksEmbeddings(endpoint=embedding_model)
embeddings.embed_query("hello")[:3]

# COMMAND ----------

# MAGIC %md ## Setting Up the Agent
# MAGIC The Retriever is the module that extracts data from the vector-search component 

# COMMAND ----------

import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

os.environ['TAVILY_API_KEY'] = dbutils.secrets.get(scope="brian_hf", key="tavily_api_key")

# COMMAND ----------

# Create the agent
memory = MemorySaver()
#model = ChatAnthropic(model_name="claude-3-sonnet-20240229")
search = TavilySearchResults(max_results=2)
tools = [search]
agent_executor = create_react_agent(chat, tools, checkpointer=memory)

# COMMAND ----------

# Test code
# Use the agent
# config = {"configurable": {"thread_id": "abc123"}}
# for chunk in agent_executor.stream(
#     {"messages": [HumanMessage(content="hi im bob! and i live in sf")]}, config
# ):
#     print(chunk)
#     print("----")

# for chunk in agent_executor.stream(
#     {"messages": [HumanMessage(content="whats the weather where I live?")]}, config
# ):
#     print(chunk)
#     print("----")

# COMMAND ----------

mlflow.models.set_model(model=agent_executor)
