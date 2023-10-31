# Databricks notebook source
# MAGIC %md
# MAGIC # Building an Advanced RAG System
# MAGIC We will now build out an advanced RAG system with multiple files and complex structures

# COMMAND ----------

# DBTITLE 1,Extra Libs to install
%pip install pypdf ctransformers==0.2.26 unstructured["local-inference"] sqlalchemy 'git+https://github.com/facebookresearch/detectron2.git' poppler-utils scrapy llama_index==0.8.54 opencv-python chromadb==0.4.15

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup Utils
%run ./utils

# COMMAND ----------

# MAGIC %md
# MAGIC # Building our Vector Store and Index
# MAGIC First step is to build out our VectorStore and Index \
# MAGIC Requirements:
# MAGIC - VectorDB
# MAGIC - Documents
# MAGIC - Embeddings

# COMMAND ----------

# DBTITLE 1,Setup Embedder
# We will go a little fancier and use a local embedder this can help save cost
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding

db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
embeddings = ModelServingEndpointEmbeddings(db_api_token=db_token)
hf_embed = LangchainEmbedding(langchain_embeddings=embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC ChromaDB uses SQLlite which isn't structured to work well on object store \
# MAGIC See: https://github.com/chroma-core/chroma/issues/985 \
# MAGIC We will set up a tmp store in a local_disk folder (will be purged on cluster terminate)

# COMMAND ----------

# DBTITLE 1,Setup Chromadb
import chromadb

from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext

tmp_store = '/local_disk0/vector_store'
print(f'Creating persistent db here: {tmp_store}')
chroma_client = chromadb.PersistentClient(path=tmp_store)
chroma_collection = chroma_client.get_or_create_collection("advanced_rag")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
# COMMAND ----------

# DBTITLE 1,Load Documents
from llama_index import SimpleDirectoryReader, download_loader

UnstructuredReader = download_loader("UnstructuredReader", refresh_cache=True, use_gpt_index_import=True)
unstruct_loader = UnstructuredReader()

dbfs_source_docs = '/dbfs/bootcamp_data/pdf_data'
print(f'loading documents from: {dbfs_source_docs}')
documents = SimpleDirectoryReader(
    input_dir=dbfs_source_docs,
    file_extractor = {'*.pdf': unstruct_loader}
).load_data()

# COMMAND ----------

# MAGIC %md
# MAGIC # Assembling the Components
# MAGIC Now that we have our documents, VectorDB and Embeddings we can assemble it all into an index \
# MAGIC Requirements:
# MAGIC - Service Context
# MAGIC - Choose an Indexing Scheme

# COMMAND ----------

# DBTITLE 1,Creating Llama-index Service Context
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

# DBTITLE 1,Creating the Index
from llama_index import VectorStoreIndex

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# Query Data from the persisted index
query_engine = index.as_query_engine()
response = query_engine.query("Tell me about the mT5 model")

print(response.response)
# COMMAND ----------

# MAGIC %md
# MAGIC ## Adding in Langchain
# MAGIC Whilst llama index provides advanced indexing strategyies, langchain can provide us with useful primatives \
# MAGIC Examples include:
# MAGIC - Memory
# MAGIC - Agents
# COMMAND ----------

from llama_index.langchain_helpers.agents import IndexToolConfig, LlamaIndexTool

from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")

tool_config = IndexToolConfig(
    query_engine=query_engine, 
    name=f"Vector Index",
    description=f"useful for when you want to answer queries about X",
    tool_kwargs={"return_direct": True}
)

tool = LlamaIndexTool.from_tool_config(tool_config)

tools = [tool]

agent_executor = initialize_agent(
    tools, llm_model, agent="conversational-react-description", memory=memory
)

agent_executor.run(input="Tell me about mT5 model")
# COMMAND ----------

agent_executor.run(input="Tell me more!")

# COMMAND ----------

agent_executor.run(input="What is bad about it?")

