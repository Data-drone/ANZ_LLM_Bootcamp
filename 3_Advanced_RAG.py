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
openai_key = dbutils.secrets.get(scope='brian_dl', key='dbdemos_openai')

openai.api_type = "azure"
#openai.api_base = "https://dbdemos-open-ai.openai.azure.com/"
#openai.api_key = openai_key
#openai.api_version = "2023-07-01-preview"
os.environ['OPENAI_API_BASE'] = 'https://dbdemos-open-ai.openai.azure.com/'
os.environ['OPENAI_API_KEY'] = openai_key
os.environ['OPENAI_API_VERSION'] = "2023-07-01-preview"
deployment_name = 'dbdemo-gpt35'

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

import glob
import re
import os

import chromadb
from chromadb.config import Settings

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# COMMAND ----------

# DBTITLE 1,Setup dbfs folder paths
%run ./utils

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Document Store
# MAGIC The document store has to be created first.
# MAGIC We need to have some sort of index and we will need to manage this ourselves.

# COMMAND ----------
# for class
#source_docs = glob.glob('/dbfs/bootcamp_data/pdf_data/*.pdf')
source_docs = glob.glob(dbfs_source_docs+'/*.pdf')

collection_name = 'arxiv_articles'

# We will use default HuggingFaceEmbeddings for now

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                   model_kwargs={'device': 'cpu'})

def embed_fn(text):
  hfe = HuggingFaceEmbeddings()
  return hfe.embed_documents(text)

# setup Chroma client with persistence
client = chromadb.chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
                                    persist_directory=linux_vector_store_directory),
                                    )

rebuild = True

# COMMAND ----------

# MAGIC %md
# MAGIC # Build ChromaDB
# MAGIC See chroma docs for more information

# COMMAND ----------

if rebuild:

  dbutils.fs.rm(f'dbfs:{linux_vector_store_directory}', True)

# COMMAND ----------

# Initiate the ChromaDB
# Create collection. get_collection, get_or_create_collection, delete_collection also available!
## Colection is where we set embeddings? # embedding_function=embed_fn
collection = client.get_or_create_collection(name=collection_name)
print(f"we have {collection.count()} in the collection.")

# COMMAND ----------

# DBTITLE 1,Collection Building Function
# we can look at other splitters later. 
# Probably Paragraph? And / Or Sentence?
def collection_builder(source_docs:list, 
                        collection:chromadb.api.models.Collection.Collection):

  assert collection.count() == 0, "WARNING This function will append to collection regardless of whether it already exists or not"

  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

  # we will process page by page

  for doc in source_docs:
    
    # This regex will only work for arxiv
    match = re.search(r'/([\d.]+)\.pdf$', doc)
    article_number = match.group(1)

    loader = PyMuPDFLoader(doc)
    pages = loader.load_and_split()

    # for page in pages:
    #print(type(page))
    texts = text_splitter.split_documents(pages)
    #print(texts)
    # print(len(texts))    

    doc_list = [x.page_content for x in texts]
    embed_list = embeddings.embed_documents(doc_list)

    collection.add(
      documents=doc_list,
      embeddings=embed_list,
      metadatas=[x.metadata for x in texts],
      ids=[article_number+str(texts.index(x)) for x in texts]
    )

  # See: https://github.com/chroma-core/chroma/issues/275
  client.persist()    

# COMMAND ----------

try:
  collection_builder(source_docs, collection)
  print(f"we now have {collection.count()} in the collection.")
except AssertionError:
  print("Doing nothing, we will not rebuild the collection")

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup LLM to interface with chroma DB
# MAGIC NOTE that reloading with langchain seems glitchy hence why we need to do it manually 

# COMMAND ----------

# Load the collection
# we reuse the previous client and embeddings
docsource = Chroma(collection_name=collection_name,
                  persist_directory=linux_vector_store_directory,
                  embedding_function=embeddings)

# we can verify that our docsearch index has objects in it with this
print('The index includes: {} documents'.format(docsource._collection.count()))
# COMMAND ----------

# MAGIC %md
# MAGIC Note that the llm_model funciton doesn't clean up after itself. so if you call it repeatedly it will fill up the VRAM
# MAGIC
# MAGIC We will add some code to quickly stop reinitiating
# MAGIC In order to understand the HuggingFace Pipeline we need to look at: 
# MAGIC - https://huggingface.co/docs/transformers/main_classes/pipelines
# MAGIC The task set for this pipe is text-generation the def of this is:
# MAGIC - https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TextGenerationPipeline
# MAGIC Device needs to be set in order to utilise GPU
# MAGIC - See: https://huggingface.co/transformers/v3.0.2/main_classes/pipelines.html#transformers.Pipeline


# COMMAND ----------

## One problem with the library at the moment is that GPU ram doesn't get relinquished when the object is overridden
# The only way to clear GPU ram is to detach and reattach
# This snippet will make sure we don't keep reloading the model and running out of GPU ram
try:
  llm_model
except NameError:
  if run_mode == 'cpu':
    # the cTransformers class interfaces with langchain differently
    from ctransformers.langchain import CTransformers
    llm_model = CTransformers(model='TheBloke/Llama-2-7B-Chat-GGML', model_type='llama')
  elif run_mode == 'gpu':
    pipe = load_model(run_mode, dbfs_tmp_cache)
    llm_model = HuggingFacePipeline(pipeline=pipe)

else:
  pass

# COMMAND ----------

# MAGIC %md
# MAGIC Before we used `RetrievalQA` that doesn't have a concept of memory
# MAGIC We can add in memory and use the `ConversationalRetrievalChain` Chain instead

# COMMAND ----------

# DBTITLE 1,Setting up prompt template
from langchain import PromptTemplate
system_template = """<s>[INST] <<SYS>>
As a helpful assistant, answer questions from users but be polite and concise. If you don't know say I don't know.
<</SYS>>


Based on the following context:

{context}

Answer the following question:
{question}[/INST]
"""

# prompt templates in langchain need the input variables specified it can then be loaded in the string
# Note that the names of the input_variables are particular to the chain type.
friendly_template = PromptTemplate(
    input_variables=["question", "context"], template=system_template
)

# COMMAND ----------

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain

memory = ConversationBufferMemory(memory_key="chat_history", output_key='answer')

# Broken at the moment
memory_chain = ConversationalRetrievalChain.from_llm(llm=llm_model,
                                                  retriever=docsource.as_retriever(search_kwargs={"k": 2}),
                                                  chain_type='stuff',
                                                  return_source_documents=True,
                                                  output_key='answer',
                                                  verbose=True,
                                                  combine_docs_chain_kwargs={"prompt": friendly_template},
                                                  memory=memory,
                                                  get_chat_history=lambda h : h)


# COMMAND ----------

# DBTITLE 1,Verify docsource is valid
# Basic Vector Similarity Search
query = "What is this is a token limit?"
query_embed = embeddings.embed_query(query)

docsource._collection.query(query_embeddings=query_embed, n_results=2)


# COMMAND ----------
# Lets test out querying

# Something is wrong with the similarity search? Are my embeddings not saving?
# Also the docsource has a different embedding structure (vectors don't line up)

query_embed = embeddings.embed_query(query)
query_embed

docs = docsource.similarity_search_by_vector(query_embed)

# COMMAND ----------

memory_chain({"question": query}, return_only_outputs=True)


# COMMAND ----------

query = 'tell me more!'
memory_chain({"question": query}, return_only_outputs=True)

# COMMAND ----------

# MAGIC %md
# MAGIC # Adding Human Feedback
# MAGIC **EXPERIMENTAL**
# MAGIC Now if only we could add in human in the loop reasoning and make the chain more intelligent that way
# MAGIC
# MAGIC We can try agents

# COMMAND ----------

# The conversation and memory doesn't occur at the retreival stage so lets use the old RetrievalQA
from langchain import PromptTemplate
system_template = """<s>[INST] <<SYS>>
As a helpful assistant, answer questions from users but be polite and concise. If you don't know say I don't know.
<</SYS>>


Based on the following context:

{context}

Answer the following question:
{question}[/INST]
"""

# prompt templates in langchain need the input variables specified it can then be loaded in the string
# Note that the names of the input_variables are particular to the chain type.
prompt_template = PromptTemplate(
    input_variables=["question", "context"], template=system_template
)

qa = RetrievalQA.from_chain_type(llm=llm_model, chain_type="stuff", 
                                 retriever=docsource.as_retriever(search_kwargs={"k": 3}),
                                 chain_type_kwargs={"prompt": prompt_template})

# COMMAND ----------

from langchain.agents import Tool, load_tools

# turn our qa chain into a tool
retrieval_tool = Tool(
    name = 'Document Search',
    func = qa,
    description ='this is a chain that has access to a cache of arxiv papers on deep learning and large language models'
)

tools = load_tools(
    ["human"],
    llm=llm_model,
)

tools.append(retrieval_tool)

# COMMAND ----------

# Setup agent
from langchain.agents import initialize_agent, AgentType

agent_chain = initialize_agent(
    tools,
    llm_model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# Mileage may vary!!
agent_chain.run("What should I ask you about llms?")






