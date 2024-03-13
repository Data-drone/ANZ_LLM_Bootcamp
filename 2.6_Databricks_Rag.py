# Databricks notebook source
# MAGIC %md
# MAGIC # Building a Chat to Docs app
# MAGIC Questioning one document

# COMMAND ----------

# MAGIC %pip install -U langchain==0.1.10 sqlalchemy==2.0.27 pypdf==4.1.0 mlflow==2.11.1 databricks-vectorsearch langchainhub==0.1.15

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

from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

pipe = ChatDatabricks(
    target_uri = 'databricks',
    endpoint = llm_model,
    temperature = 0.1
)

# Construct and Test Embedding Model
embeddings = DatabricksEmbeddings(endpoint=embedding_endpoint_name)

rag_prompt = hub.pull("rlm/rag-prompt-mistral")

rag_runnable = rag_prompt | pipe

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


rag_with_message_history = RunnableWithMessageHistory(
    rag_runnable,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# COMMAND ----------

# Setup Retriever
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

vsc = VectorSearchClient()
index = vsc.get_index(endpoint_name=vs_endpoint,
                      index_name=vs_index_fullname)

retriever = DatabricksVectorSearch(
    index, text_column="page_content", 
    embedding=embeddings, columns=["source_doc"]
).as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_with_message_history
)

# COMMAND ----------

answer = rag_chain.invoke(input="test",
                 config={"configurable": {"session_id": "abc123"}})