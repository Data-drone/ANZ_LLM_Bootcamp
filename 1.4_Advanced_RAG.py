# Databricks notebook source
# MAGIC %md
# MAGIC # Building an Advanced RAG System
# MAGIC We will now build out an advanced RAG system with multiple files and complex structures

# COMMAND ----------

# DBTITLE 1,Extra Libs to install
%pip install pypdf ctransformers unstructured["local-inference"] sqlalchemy 'git+https://github.com/facebookresearch/detectron2.git' poppler-utils scrapy llama_index==0.8.9 opencv-python chromadb==0.4.9

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

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                   model_kwargs={'device': 'cpu'})

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

dbfs_source_docs = '/bootcamp_data/pdf_data'
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
from langchain.chat_models import AzureChatOpenAI
import openai

from llama_index import (
  ServiceContext,
  set_global_service_context,
  LLMPredictor
)
from llama_index.callbacks import CallbackManager, OpenInferenceCallbackHandler


# Setup OpenAI Creds
openai_key = dbutils.secrets.get(scope='bootcamp_training', key='bootcamp_openai')

openai.api_type = "azure"
#openai.api_base = "https://dbdemos-open-ai.openai.azure.com/"
#openai.api_key = openai_key
#openai.api_version = "2023-07-01-preview"
os.environ['OPENAI_API_BASE'] = 'https://anz-bootcamp-daiswt.openai.azure.com/'
os.environ['OPENAI_API_KEY'] = openai_key
os.environ['OPENAI_API_VERSION'] = "2023-07-01-preview"

deployment_name = 'daiwt-demo'

# See: https://github.com/openai/openai-python/issues/318
llm = AzureChatOpenAI(deployment_name=deployment_name,
                      model_name="gpt-35-turbo")

llm_predictor = LLMPredictor(llm=llm)

callback_handler = OpenInferenceCallbackHandler()
callback_manager = CallbackManager([callback_handler])
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, 
                                               embed_model=hf_embed,
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

# TODO Integration with langchain and memory

# COMMAND ----------
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
    tools, llm, agent="conversational-react-description", memory=memory
)

agent_executor.run(input="Tell me about mT5 model")
# COMMAND ----------

agent_executor.run(input="Tell me more!")

# COMMAND ----------

agent_executor.run(input="Tell me more!")

# COMMAND ----------

agent_executor.run(input="What is bad about it?")

