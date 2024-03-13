# Databricks notebook source
# MAGIC %md
# MAGIC # Building a Chat to Docs app
# MAGIC Questioning one document

# COMMAND ----------

# MAGIC %pip install -U langchain==0.1.10 sqlalchemy==2.0.27 pypdf==4.1.0 mlflow==2.11.0 databricks-vectorsearch 

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Config Variables

# COMMAND ----------

source_catalog = "bootcamp_ml"
source_schema = "rag_chatbot"
source_volume = "datasets"
source_table = "arxiv_parse"
vs_endpoint = "bootcamp_vs_endpoint"
embedding_endpoint_name = "databricks-bge-large-en"
llm_model = 'databricks-mixtral-8x7b-instruct'
vs_index = f"{source_table}_bge_index"
vs_index_fullname = f"{source_catalog}.{source_schema}.{vs_index}"

# COMMAND ----------

# Setup Model
from langchain_community.chat_models import ChatDatabricks
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.embeddings import DatabricksEmbeddings

pipe = ChatDatabricks(
    target_uri = 'databricks',
    endpoint = llm_model,
    temperature = 0.1
)

# Construct and Test Embedding Model
embeddings = DatabricksEmbeddings(endpoint=embedding_endpoint_name)

# COMMAND ----------

# Setup Retriever
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

vsc = VectorSearchClient()
index = vsc.get_index(endpoint_name=vs_endpoint,
                      index_name=vs_index_fullname)

retriever = DatabricksVectorSearch(
    index, text_column="page_content", 
    embedding=embeddings, columns=["source_doc"]
).as_retriever()

retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a response to answer this question")
    ])

retriever_chain = create_history_aware_retriever(pipe, retriever, retriever_prompt)

# COMMAND ----------

prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
         You are a research assistance that helps to answer questions based on the below context

         """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

stuff_documents_chain = create_stuff_documents_chain(pipe, prompt_template)

retrieval_chain = create_retrieval_chain(retriever_chain, stuff_documents_chain)

# COMMAND ----------

retrieval_chain.invoke({
  "chat_history": [{"role": "user", "content": "I am interested in LLMs"}],
  "input": "Tell me about how to tune them"})