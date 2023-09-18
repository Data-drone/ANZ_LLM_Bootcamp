# Databricks notebook source
# MAGIC %md
# MAGIC # Understanding Embeddings
# MAGIC Embeddings are just vectors and we can visualise and analyse them as such \
# MAGIC In this case we will use the Arize Phoenix tool \
# MAGIC for more info see: https://docs.arize.com/phoenix/

# COMMAND ----------

%pip install -U llama_index==0.8.9 "arize-phoenix[experimental]" pandas==1.5.3 faiss-cpu datashader bokeh holoviews scikit-image colorcet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup configs

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

import os
import openai
import numpy as np

# COMMAND ----------

# DBTITLE 1,Configurations
# test_pdf = f'{dbfs_source_docs}/2010.11934.pdf'
test_pdf = '/dbfs/bootcamp_data/pdf_data/2302.09419.pdf'
test_pdf

# COMMAND ----------

# For this example we will use azure openai for now
# Setup OpenAI Creds
openai_key = dbutils.secrets.get(scope='bootcamp_training', key='bootcamp_openai')

openai.api_type = "azure"
#openai.api_base = "https://dbdemos-open-ai.openai.azure.com/"
#openai.api_key = openai_key
#openai.api_version = "2023-07-01-preview"
os.environ['OPENAI_API_BASE'] = 'https://anz-bootcamp-daiswt.openai.azure.com/'
os.environ['OPENAI_API_KEY'] = openai_key
os.environ['OPENAI_API_VERSION'] = "2022-12-01"

deployment_name = 'daiwt-demo'


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Understanding Embeddings
# MAGIC
# MAGIC Lets explore how data embeds a bit more in order to see how we can improve retrieval \
# MAGIC We will use openai to start
# COMMAND ----------

# DBTITLE 1,Setup some embedding algorithms
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI

azure_openai_embedding = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        deployment="dbdemos-embedding",
        openai_api_key=openai_key,
        openai_api_base=os.environ['OPENAI_API_BASE'],
        openai_api_type=openai.api_type,
        openai_api_version=os.environ['OPENAI_API_VERSION'],
    )

# See: https://github.com/openai/openai-python/issues/318
llm = AzureChatOpenAI(deployment_name=deployment_name,
                      model_name="gpt-35-turbo")


#embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2', model_kwargs={'device': 'cpu'})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simple Exploration w ReRank

# COMMAND ----------

# most vector stores use cosine_similarity
import faiss

example_sentences = ["The kangaroo population in Australia is declining due to habitat loss and hunting.",
"Australia has a diverse population of people from all over the world.",
"The kangaroo is a symbol of Australia and appears on its coat of arms.",
"The population of Australia is projected to reach 50 million by 2050.",
"Kangaroos are unique to Australia and can only be found in the wild there.",
"The indigenous population of Australia has been marginalized for centuries.",
"Australia is home to a variety of fascinating animals, including the kangaroo.",
"The population density of Australia is relatively low compared to other countries.",
"Kangaroos play an important role in maintaining the ecosystem balance in Australia.",
"Australia has strict laws regulating the hunting and trade of kangaroos to protect their population."] 

encoded_sentences = [azure_openai_embedding.embed_query(sentence) for sentence in example_sentences]
vector_format_encode = np.array(encoded_sentences, dtype=np.float32)
vector_format_encode /= np.linalg.norm(vector_format_encode, axis=1)[:, np.newaxis]

# we will create a vector index
vector_index = faiss.IndexFlatIP(vector_format_encode.shape[1])
vector_index.add(vector_format_encode)

test_question = "What is affecting the population of kangaroos?"
embedded_query = np.array(azure_openai_embedding.embed_query(test_question))

# COMMAND ----------

# we can look at the retrieved entries and how it has been processed
k = 4
scores, index = vector_index.search(np.array([embedded_query]), k)

# look up the index for sentences
top_sentences = [example_sentences[i] for i in index[0]]

human_readable_result = list(zip(scores.reshape(-1, 1), top_sentences))

for score, sentence in human_readable_result:
    print(f"Score: {score[0]:.4f}, Sentence: {sentence}")

# COMMAND ----------

# we can use a rerank to try to improve the result
format_top = []
for i in range(len(top_sentences)):
  format_top.append(
    f"Document {1}:\n"
    f"{top_sentences[i]}"
  )

context_str = "\n\n".join(format_top)

## Our Reranking prompt
rerank_prompt = ("A list of documents is shown below. Each document has a number next to it along "
    "with a summary of the document. A question is also provided. \n"
    "Respond with the numbers of the documents "
    "you should consult to answer the question, in order of relevance, as well \n"
    "as the relevance score. The relevance score is a number from 1-10 based on "
    "how relevant you think the document is to the question.\n"
    "Do not include any documents that are not relevant to the question. \n"
    "Example format: \n"
    "Document 1:\n<summary of document 1>\n\n"
    "Document 2:\n<summary of document 2>\n\n"
    "...\n\n"
    "Document 10:\n<summary of document 10>\n\n"
    "Question: <question>\n"
    "Answer:\n"
    "Doc: 9, Relevance: 7\n"
    "Doc: 3, Relevance: 4\n"
    "Doc: 7, Relevance: 3\n\n"
    "Let's try this now: \n\n"
    f"{context_str}\n"
    f"Question: {test_question}\n"
    "Answer:\n")

reranked_result = llm.call_as_llm(rerank_prompt)

print(reranked_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualising Embeddings

# COMMAND ----------

# So we can use reranking in order to better craft our results.
# Can we also look at our embeddings to understand the content?
# We will use umap and bokeh for this

import pandas as pd

import umap
from umap import plot

import plotly.express as px

from bokeh.resources import CDN
from bokeh.embed import file_html

umap_2d = umap.UMAP(n_components=2, init='random', random_state=0)
#umap_3d = umap.UMAP(n_components=3, init='random', random_state=0)

proj_2d = umap_2d.fit(vector_format_encode)

hover_data =  pd.DataFrame({'index': np.arange(len(example_sentences)) ,
                          'text': example_sentences})

# COMMAND ----------

plot.output_notebook()
# COMMAND ----------

p = plot.interactive(proj_2d, hover_data=hover_data, point_size=10)
html = file_html(p, CDN, "Sample Sentences")
displayHTML(html)

# COMMAND ----------

# MAGIC %md
# MAGIC # Embeddings with Whole Document
# MAGIC

# COMMAND ----------

# MAGIC %md ## Setup Service Context
# MAGIC By default, llama_index assumes that OpenAI is the service context \
# MAGIC We are using AzureOpen AI so the setup is a little different. \
# MAGIC Azure OpenAI notably requires two deployments, an embedder and the model \
# MAGIC We will demonstrate a hybrid setup here where we use a huggingface sentence transformer \
# MAGIC that will do the embeddings for our vector store \
# MAGIC Whilst AzureOpenAI (gpt-3.5-turbo) provides the brains

# COMMAND ----------


from llama_index import (
  ServiceContext,
  set_global_service_context,
  LLMPredictor
)
from llama_index.embeddings import LangchainEmbedding
from llama_index.callbacks import CallbackManager, OpenInferenceCallbackHandler


# Azure OpenAI Embeddings - needed cause ragas uses async
embedding_llm = LangchainEmbedding(
    azure_openai_embedding,
    embed_batch_size=1,
)
#ll_embed = LangchainEmbedding(langchain_embeddings=embeddings)


llm_predictor = LLMPredictor(llm=llm)

callback_handler = OpenInferenceCallbackHandler()
callback_manager = CallbackManager([callback_handler])

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, 
                                               embed_model=embedding_llm,
                                               callback_manager = callback_manager 
                                               )

# we can now set this context to be a global default
set_global_service_context(service_context)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and Chunk Document
# MAGIC We will load a sample doc to test on, firstly with a naive default chunking strategy
# MAGIC
# COMMAND ----------

# DBTITLE 1,Create Index

# chunk the output
from llama_index import (
    download_loader, VectorStoreIndex
)
from llama_index.evaluation import DatasetGenerator
from pathlib import Path

PDFReader = download_loader('PDFReader')
loader = PDFReader()

# This produces a list of llama_index document objects
documents = loader.load_data(file=Path(test_pdf))

# we are just setting up a simple in memory Vectorstore here
index = VectorStoreIndex.from_documents(documents)

# COMMAND ----------

# Lets have a quick look at the embeddings

text_obj = [document.text for document in documents]
encoded_chunks = [azure_openai_embedding.embed_query(document_text) for document_text in text_obj]
vector_chunks = np.array(encoded_chunks, dtype=np.float32)
vector_chunks /= np.linalg.norm(vector_chunks, axis=1)[:, np.newaxis]

# COMMAND ----------

# DBTITLE 1,Examine Chunk text
pd.set_option('display.max_colwidth', 1000)
hover_data

# COMMAND ----------

# DBTITLE 1,Visualise Chunk Text
umap_2d = umap.UMAP(n_components=2, init='random', random_state=0)
#umap_3d = umap.UMAP(n_components=3, init='random', random_state=0)

proj_2d = umap_2d.fit(vector_chunks)

hover_data =  pd.DataFrame({'index': np.arange(len(text_obj)) ,
                          'text': text_obj})

p = plot.interactive(proj_2d, hover_data=hover_data, point_size=10)
html = file_html(p, CDN, "Research Doc")
displayHTML(html)


# COMMAND ----------

# BIER and compare embeddings?

# COMMAND ----------

# DBTITLE 1,Create Sample Questions

# and turning it into a query engine
query_engine = index.as_query_engine()

# this is the question generator. Note that it has additional settings to customise prompt etc
data_generator = DatasetGenerator.from_documents(documents=documents)

# this is the call to generate the questions
eval_questions = data_generator.generate_questions_from_nodes()




# COMMAND ----------

# MAGIC %md
# MAGIC # (WIP) Create Phoenix Visualisations
# MAGIC The phoenix app still needs updating with support for different root_paths first

# COMMAND ----------

# Extract out nodes
# test parse index data
document_ids = []
document_texts = []
document_embeddings = []

docstore = index.storage_context.docstore
for node_id, node in docstore.docs.items():
  document_ids.append(node.hash)  # use node hash as the document ID
  document_texts.append(node.text)
  document_embeddings.append(np.array(index.storage_context.vector_store.get(node_id)))

dataset_df = pd.DataFrame(
        {
            "document_id": document_ids,
            "text": document_texts,
            "text_vector": document_embeddings,
        }
    )
# COMMAND ----------

# create the query frame

from llama_index.callbacks.open_inference_callback import as_dataframe

query_data_buffer = callback_handler.flush_query_data_buffer()
sample_query_df = as_dataframe(query_data_buffer)
sample_query_df

# COMMAND ----------

import phoenix as px

### Create the schema for the documents
database_schema = px.Schema(
    prediction_id_column_name="document_id",
    prompt_column_names=px.EmbeddingColumnNames(
        vector_column_name="text_vector",
        raw_data_column_name="text",
    ),
)
database_ds = px.Dataset(
    dataframe=dataset_df,
    schema=database_schema,
    name="database",
)

query_ds = px.Dataset.from_open_inference(sample_query_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Start Visualisation App

# COMMAND ----------

session = px.launch_app(primary=query_ds, corpus=database_ds, host='0.0.0.0', port='10101')

# COMMAND ----------