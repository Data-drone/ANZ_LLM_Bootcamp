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
chroma_archive_folder = f'/dbfs/Users/{username}/advanced_chromadb'
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
from langchain.vectorstores import Chroma

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

# COMMAND ----------

# MAGIC %md
# MAGIC # Logging and Productionisation
# MAGIC Now that we know that our agent is working, we can save it into MLflow to make it ready for deployment

# COMMAND ----------

import mlflow

import shutil
from typing import Any, Dict, List, Mapping, Optional, Tuple
import requests
import pandas as pd

from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.schema.embeddings import Embeddings
from langchain.utils import get_from_dict_or_env


# archive our existing folder so that we don't need to rebuild
try:
    shutil.copytree(tmp_store, chroma_archive_folder)
except FileExistsError:
    pass

experiment_name = f'/Users/{username}/llm_orchestrator_agent'
mlflow.set_experiment(experiment_name)


class AdvancedLangchainQABot(mlflow.pyfunc.PythonModel):

    def __init__(self, model_uri, token, chroma_archive:str=chroma_archive_folder):
        self.model_uri = model_uri
        self.token = token
        self.chroma_archive_dir = chroma_archive
        self.chroma_local_dir = '/local_disk0/vector_store'

    def _setup_vector_db(self):
        from typing import Any, Dict, List, Mapping, Optional, Tuple
        
        from langchain.pydantic_v1 import BaseModel, Extra, root_validator
        from langchain.schema.embeddings import Embeddings

        try:
            shutil.copytree(self.chroma_archive_dir, self.chroma_local_dir)
        except FileExistsError:
            shutil.rmtree(self.chroma_local_dir)
            shutil.copytree(self.chroma_archive_dir, self.chroma_local_dir)

        
        class ModelServingEndpointEmbeddings(BaseModel, Embeddings):
            """Databricks Model Serving embedding service.

            To use, you should have the
            environment variable ``DB_API_TOKEN`` set with your API token, or pass
            it as a named parameter to the constructor.

            Example:
                .. code-block:: python

                    from langchain.llms import MosaicMLInstructorEmbeddings
                    endpoint_url = (
                        "https://dbc-d0c4038e-c5a9.cloud.databricks.com/serving-endpoints/brian_embedding_endpoint/invocations"
                    )
                    mosaic_llm = MosaicMLInstructorEmbeddings(
                        endpoint_url=endpoint_url,
                        db_api_token="my-api-key"
                    )
            """

            endpoint_url: str = (
                "https://dbc-d0c4038e-c5a9.cloud.databricks.com/serving-endpoints/brian_embedding_endpoint/invocations"
            )
            """Endpoint URL to use."""
            embed_instruction: str = "Represent the document for retrieval: "
            """Instruction used to embed documents."""
            query_instruction: str = (
                "Represent the question for retrieving supporting documents: "
            )
            """Instruction used to embed the query."""
            retry_sleep: float = 1.0
            """How long to try sleeping for if a rate limit is encountered"""

            db_api_token: Optional[str] = None

            class Config:
                """Configuration for this pydantic object."""
                extra = Extra.forbid

            @root_validator()
            def validate_environment(cls, values: Dict) -> Dict:
                """Validate that api key and python package exists in environment."""
                db_api_token = get_from_dict_or_env(
                    values, "db_api_token", "DB_API_TOKEN"
                )
                values["db_api_token"] = db_api_token
                return values

            @property
            def _identifying_params(self) -> Mapping[str, Any]:
                """Get the identifying parameters."""
                return {"endpoint_url": self.endpoint_url}

            def _embed(
                self, input: List[Tuple[str, str]], is_retry: bool = False
            ) -> List[List[float]]:
                #payload = {"input_strings": input}
                payload = {
                    "dataframe_split": {
                        "data": [
                            [
                                input
                            ]
                        ]
                    }
                }

                # HTTP headers for authorization
                headers = {
                    "Authorization": f"Bearer {self.db_api_token}",
                    "Content-Type": "application/json",
                }

                # send request
                try:
                    response = requests.post(self.endpoint_url, headers=headers, json=payload)
                except requests.exceptions.RequestException as e:
                    raise ValueError(f"Error raised by inference endpoint: {e}")

                try:
                    if response.status_code == 429:
                        if not is_retry:
                            import time
                            time.sleep(self.retry_sleep)
                            return self._embed(input, is_retry=True)

                        raise ValueError(
                            f"Error raised by inference API: rate limit exceeded.\nResponse: "
                            f"{response.text}"
                        )

                    parsed_response = response.json()

                except requests.exceptions.JSONDecodeError as e:
                    raise ValueError(
                        f"Error raised by inference API: {e}.\nResponse: {response.text}"
                    )

                return parsed_response

            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                """Embed documents using a MosaicML deployed instructor embedding model.

                Args:
                    texts: The list of texts to embed.

                Returns:
                    List of embeddings, one for each text.
                """
                embeddings = [self._embed(x)['predictions'][0] for x in texts]

                return embeddings

            def embed_query(self, text: str) -> List[float]:
                """Embed a query using a Databricks Model Serving embedding model.

                Args:
                    text: The text to embed.

                Returns:
                    Embeddings for the text.
                """
                embedding = self._embed(text)
                return embedding['predictions'][0]
            
        self.embeddings = ModelServingEndpointEmbeddings(db_api_token=self.token)
        
        chroma_client = chromadb.PersistentClient(path=self.chroma_local_dir)
        chroma_collection = chroma_client.get_or_create_collection("advanced_rag")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        #storage_context = StorageContext.from_defaults(vector_store=vector_store)

        return vector_store


    def load_context(self, context):
        from typing import Any, Dict, List, Mapping, Optional, Tuple
        from langchain.pydantic_v1 import BaseModel, Extra, root_validator
        from langchain.schema.embeddings import Embeddings
        from langchain.utils import get_from_dict_or_env
        from langchain.callbacks.manager import CallbackManagerForLLMRun
        from langchain.llms.base import LLM
        
        self.vector_store  = self._setup_vector_db()

        class ServingEndpointLLM(LLM):
            endpoint_url: str
            token: str

            @property
            def _llm_type(self) -> str:
                return "custom"

            def _call(
                self,
                prompt: str,
                stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any,
            ) -> str:
                if stop is not None:
                    raise ValueError("stop kwargs are not permitted.")
                
                header = {"Context-Type": "text/json", "Authorization": f"Bearer {self.token}"}

                dataset = {'inputs': {'prompt': [prompt]},
                          'params': kwargs}

                response = requests.post(headers=header, url=self.endpoint_url, json=dataset)

                return response.json()['predictions'][0]['candidates'][0]['text']

            @property
            def _identifying_params(self) -> Mapping[str, Any]:
                """Get the identifying parameters."""
                return {"endpoint_url": self.endpoint_url} 

        llm_model = ServingEndpointLLM(endpoint_url=self.model_uri, token=self.token)

        self.llm_predictor = LLMPredictor(llm=llm_model)

        # llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        # callback_manager = CallbackManager([llama_debug])

        # service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, 
                                            #    embed_model=embeddings,
                                            #    callback_manager = callback_manager 
                                            #    )
 
        # index = VectorStoreIndex.from_vector_store(
        #     vector_store, service_context=service_context
        # )

    def predict(self, context, data):
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])

        service_context = ServiceContext.from_defaults(llm_predictor=self.llm_predictor, 
                                               embed_model=self.embeddings,
                                               callback_manager = callback_manager 
                                               )
        
        advanced_index = VectorStoreIndex.from_vector_store(
            self.vector_store, service_context=service_context
        )

        query_engine = advanced_index.as_query_engine() 
 
        questions = data['prompt']
        results = [query_engine.query(x).response for x in questions] 
        return results

# COMMAND ----------

model = AdvancedLangchainQABot(serving_uri, db_token, chroma_archive_folder)

# We can setup some example questions for testing the chain as well
test_questions = ['What are the basic components of a Transformer?',
                  'What is a tokenizer?',
                  'How can we handle audio?',
                  'Are there alternatives to transformers?']

testing_questions = pd.DataFrame(
    test_questions, columns = ['prompt']
)

with mlflow.start_run() as run:
  mlflow_result = mlflow.pyfunc.log_model(
      python_model = model,
      extra_pip_requirements = ['llama_index==0.8.54', 'chromadb==0.4.15'],
      artifact_path = 'langchain_pyfunc'
  )

  mlflow.evaluate(mlflow_result.model_uri,
                  testing_questions,
                  model_type="text")


# COMMAND ----------
