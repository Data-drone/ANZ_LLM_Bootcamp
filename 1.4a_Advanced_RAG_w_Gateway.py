# Databricks notebook source
# MAGIC %md
# MAGIC # Building an Advanced RAG System
# MAGIC We will now build out an advanced RAG system with multiple files and complex structures

# COMMAND ----------

# DBTITLE 1,Extra Libs to install
# ctransformers==0.2.26 
%pip install pypdf unstructured["local-inference"] sqlalchemy 'git+https://github.com/facebookresearch/detectron2.git' poppler-utils scrapy llama_index==0.8.54 opencv-python chromadb==0.4.17

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
from langchain.embeddings.mlflow_gateway import MlflowAIGatewayEmbeddings
from llama_index.embeddings import LangchainEmbedding

embeddings = MlflowAIGatewayEmbeddings(
   gateway_uri="databricks",
   route="mosaicml-instructor-xl-embeddings"
)

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

chroma_local_folder = '/local_disk0/vector_store'
print(f'Creating persistent db here: {chroma_local_folder}')
chroma_client = chromadb.PersistentClient(path=chroma_local_folder)
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

# For: https://github.com/run-llama/llama_index/issues/9111
for document in documents:
  document.embedding  = embeddings.embed_query(document.text)

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
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from langchain.chat_models import ChatMLflowAIGateway

# Using Databricks Model Serving
browser_host = spark.conf.get("spark.databricks.workspaceUrl")
db_host = f"https://{browser_host}"
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

mosaic_chat_route_name = "mosaicml-llama2-70b-chat"

llm_model = ChatMLflowAIGateway(
    gateway_uri="databricks",
    route=mosaic_chat_route_name,
    params={
        "temperature": 0.0,
        "candidate_count": 2,
        "stop": [""], # There is something weird with this param but this works for now
        "max_tokens": 256
    },
)

llm_predictor = LLMPredictor(llm=llm_model)

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
from llama_index import VectorStoreIndex, KnowledgeGraphIndex
from langchain.vectorstores import Chroma

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# index = KnowledgeGraphIndex.from_documents(
#     documents,
#     max_triplets_per_chunk=2, 
#     storage_context=storage_context,
#     service_context=service_context
# )

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

# COMMAND ----------

# MAGIC %md
# MAGIC # Logging and Productionisation
# MAGIC Now that we know that our agent is working, we can save it into MLflow to make it ready for deployment

# COMMAND ----------

import mlflow
import pandas as pd


experiment_name = f'/Users/{username}/llm_orchestrator_agent'
mlflow.set_experiment(experiment_name)


class AdvancedLangchainQABot(mlflow.pyfunc.PythonModel):

    def __init__(self, host, token):
        self.host = host
        self.token = token
        self.chroma_local_dir = '/local_disk0/vector_store'

    def _setup_vector_db(self, context):
        import shutil
        
        from langchain.schema.embeddings import Embeddings

        try:
            shutil.copytree(context.artifacts['chroma_db'], self.chroma_local_dir)
        except FileExistsError:
            shutil.rmtree(self.chroma_local_dir)
            shutil.copytree(context.artifacts['chroma_db'], self.chroma_local_dir)
    
        self.embeddings = MlflowAIGatewayEmbeddings(
                        gateway_uri="databricks",
                        route="mosaicml-instructor-xl-embeddings"
                     )
        
        chroma_client = chromadb.PersistentClient(path=self.chroma_local_dir)
        chroma_collection = chroma_client.get_or_create_collection("advanced_rag")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        #storage_context = StorageContext.from_defaults(vector_store=vector_store)

        return vector_store


    def load_context(self, context):
        from langchain.chat_models import ChatMLflowAIGateway
        import os

        # connecting to gateway requires that this is set
        os.environ['DATABRICKS_HOST'] = self.host
        os.environ['DATABRICKS_TOKEN'] = self.token

        self.vector_store  = self._setup_vector_db(context)

        mosaic_chat_route_name = "mosaicml-llama2-70b-chat"

        llm_model = ChatMLflowAIGateway(
            gateway_uri="databricks",
            route=mosaic_chat_route_name,
            params={
                "temperature": 0.0,
                "candidate_count": 2,
                "stop": [""], # There is something weird with this param but this works for now
                "max_tokens": 256
            },
        )

        self.llm_predictor = LLMPredictor(llm=llm_model)

        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])

        service_context = ServiceContext.from_defaults(llm_predictor=self.llm_predictor, 
                                               embed_model=self.embeddings,
                                               callback_manager = callback_manager 
                                               )
        
        advanced_index = VectorStoreIndex.from_vector_store(
            self.vector_store, service_context=service_context
        )

        self.query_engine = advanced_index.as_query_engine() 

    def process_row(self, row):
        return self.ery_engine.query(row['prompt'])
    

    def predict(self, context, data):

        results = data.apply(self.process_row, axis=1) 
        return results

# COMMAND ----------

catalog = 'bootcamp_ml'
schema = 'rag_chatbot'
model_name = 'adv_retrieval_chain'

model = AdvancedLangchainQABot(db_host, db_token)

# We can setup some example questions for testing the chain as well
test_questions = ['What are the basic components of a Transformer?',
                  'What is a tokenizer?',
                  'How can we handle audio?',
                  'Are there alternatives to transformers?']

testing_questions = pd.DataFrame(
    test_questions, columns = ['prompt']
)

user_input = "What is a tokenizer?"
input_example = {"prompt": user_input}

langchain_signature = mlflow.models.infer_signature(
    model_input=input_example,
    model_output=[agent_executor.run(user_input)]
)

with mlflow.start_run() as run:
  mlflow_result = mlflow.pyfunc.log_model(
      python_model = model,
      extra_pip_requirements = ['llama_index==0.8.54', 'chromadb==0.4.17'],
      artifacts = {
         'chroma_db': chroma_local_folder
      },
      artifact_path = 'langchain_pyfunc',
      signature = langchain_signature,
      input_example = input_example,
      registered_model_name = f'{catalog}.{schema}.{model_name}'
  )

  mlflow.evaluate(mlflow_result.model_uri,
                  testing_questions,
                  model_type="text")


# COMMAND ----------
