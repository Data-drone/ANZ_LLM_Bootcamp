# Databricks notebook source
# MAGIC %md # Building your first RAG with agent framework
# MAGIC 
# MAGIC Lets try and use the ai-cookbook example

# COMMAND ----------

# MAGIC %pip install -U langchain==0.3.7 langchain-community==0.3.7 mlflow-skinny==2.17.2 databricks-vectorsearch databricks-langchain
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

# MAGIC %md ## Setting Up Retriever
# MAGIC The Retriever is the module that extracts data from the vector-search component 

# COMMAND ----------

vsc = VectorSearchClient()
index = vsc.get_index(endpoint_name=vector_search_endpoint,
                      index_name=vs_index_fullname)

retriever = DatabricksVectorSearch(
    index, text_column="page_content", 
    embedding=embeddings, columns=["row_id", "source_doc"]
).as_retriever()

mlflow.models.set_retriever_schema(
    primary_key = 'row_id',
    text_column = 'page_content',
    doc_uri = 'source_doc'
)

# COMMAND ----------

# DBTITLE 1,Utils
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]


# Return the chat history, which is is everything before the last question
def extract_chat_history(chat_messages_array):
    return chat_messages_array[:-1]

def format_chat_history_for_prompt(chat_messages_array):
    history = extract_chat_history(chat_messages_array)
    formatted_chat_history = []
    if len(history) > 0:
        for chat_message in history:
            if chat_message["role"] == "user":
                formatted_chat_history.append(
                    HumanMessage(content=chat_message["content"])
                )
            elif chat_message["role"] == "assistant":
                formatted_chat_history.append(
                    AIMessage(content=chat_message["content"])
                )
    return formatted_chat_history

# COMMAND ----------

query_rewrite_template = """Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natural language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.

Chat history: {chat_history}

Question: {question}"""

query_rewrite_prompt = PromptTemplate(
    template=query_rewrite_template,
    input_variables=["chat_history", "question"],
)


prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
         You are a research and eduation assistant that takes a conversation between a learner and yourself and answer their questions based on the below context:
         
         <context>
         {context}
         </context> 
         
         <instructions>
         - Focus your answers based on the context but provide additional helpful notes from your background knowledge caveat those notes though.
         - If the context does not seem relevant to the answer say as such as well.
         </instructions>
         """),
        MessagesPlaceholder(variable_name="formatted_chat_history"),
        ("user", "{question}")
    ])

def format_context(docs):
    chunk_template = "Passage: {chunk_text}\n"
    chunk_contents = [
        chunk_template.format(
            chunk_text=d.page_content,
            document_uri=d.metadata["row_id"],
        )
        for d in docs
    ]
    return "".join(chunk_contents)

# COMMAND ----------

# DBTITLE 1,Setup the chain
chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_chat_history),
        "formatted_chat_history": itemgetter("messages")
        | RunnableLambda(format_chat_history_for_prompt),
    }
    | RunnablePassthrough()
    | {
        "context": RunnableBranch(  # Only re-write the question if there is a chat history
            (
                lambda x: len(x["chat_history"]) > 0,
                query_rewrite_prompt | chat | StrOutputParser(),
            ),
            itemgetter("question"),
        )
        | retriever
        | RunnableLambda(format_context),
        "formatted_chat_history": itemgetter("formatted_chat_history"),
        "question": itemgetter("question"),
    }
    | prompt_template
    | chat
    | StrOutputParser()
)

mlflow.models.set_model(model=chain)

# COMMAND ----------

# DBTITLE 1,Input Sample and testing
input_example = {
        "messages": [
            {
                "role": "user",
                "content": "What is RAG?",
            },
        ]
    }

# testing in development - comment out to deploy and log
#chain.invoke(input_example)