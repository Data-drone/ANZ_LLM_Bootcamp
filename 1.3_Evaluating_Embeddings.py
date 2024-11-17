# Databricks notebook source
# MAGIC %md
# MAGIC # Understanding Embeddings & Retrieval
# MAGIC Enhancing our parsing and chunking algorithms can help to make sure that we collect just the full text \
# MAGIC in our doucments. How we embed them can also have an effect.
# MAGIC
# MAGIC Embedding Models like LLMs have to be trained against s set of documents. \
# MAGIC Words that aren't in the training corpus will likely get tokenised letter by letter \
# MAGIC Foreign languages that aren't part of the training will also suffer in quality.

# COMMAND ----------

# MAGIC %pip install databricks_langchain pymupdf4llm faiss-cpu datashader bokeh holoviews scikit-image colorcet llama_index==0.11.23 langchain==0.3.7 langchain-community==0.3.7 llama-index-llms-langchain poppler-utils unstructured[pdf,txt]==0.13.0 databricks-vectorsearch llama-index-embeddings-langchain
# MAGIC dbutils.library.restartPython()
# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

# DBTITLE 1,Setup Configurations
import os
import numpy as np

# Setup Models & Embeddings
from databricks_langchain import ChatDatabricks
from databricks_langchain import DatabricksEmbeddings
from llama_index.core import Settings
from llama_index.llms.langchain import LangChainLLM
from llama_index.embeddings.langchain import LangchainEmbedding
import nltk

nltk.download('averaged_perceptron_tagger')
model_name = 'databricks-dbrx-instruct'
embedding_model = 'databricks-bge-large-en'

llm_model = ChatDatabricks(
  target_uri='databricks',
  endpoint = model_name,
  temperature = 0.1
)
embeddings = DatabricksEmbeddings(endpoint=embedding_model)

llama_index_chain = LangChainLLM(llm=llm_model)
llama_index_embeddings = LangchainEmbedding(embeddings)
Settings.llm = llama_index_chain
Settings.embed_model = llama_index_embeddings


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Exploring Embeddings with ReRank
# MAGIC
# MAGIC Lets explore how data embeds a bit more in order to see how we can improve retrieval \
# MAGIC At it's heart, embeddings just look at the occurences of words, not semantic meaning.
# MAGIC
# MAGIC
# MAGIC The easiest way to understand this is to look at ReRank \
# MAGIC Here we ask an LLM to look at our retrievals then choose which ones are actually relevant to our question \
# MAGIC (I would always recommend including Rerank as a part of your final Orchestrator Logic)

# COMMAND ----------

# most vector stores use cosine_similarity
# We use faiss for ease of use here.
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

encoded_sentences = [embeddings.embed_query(sentence) for sentence in example_sentences]
vector_format_encode = np.array(encoded_sentences, dtype=np.float32)
vector_format_encode /= np.linalg.norm(vector_format_encode, axis=1)[:, np.newaxis]

# we will create a vector index
vector_index = faiss.IndexFlatIP(vector_format_encode.shape[1])
vector_index.add(vector_format_encode)

test_question = "What is affecting the population of kangaroos?"
embedded_query = np.array(embeddings.embed_query(test_question))

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

# MAGIC %md We can see that we are picking up things that just talk about Kangaroos with no relation to the question

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

reranked_result = llm_model.invoke(rerank_prompt)

print(reranked_result.content)

# COMMAND ----------

# MAGIC %md With rerank, we are finally getting the key chunks that matter.

# COMMAND ----------

# MAGIC %md
# MAGIC # Visualising Embeddings
# MAGIC
# MAGIC Whilst rerank can help us to get the right chunks, we still need to hope that we are able to retrieve the correct chunks as part of our search first. \
# MAGIC We can assess the quality of our chunks and embeddings by visualising them first.
# MAGIC
# MAGIC For this, we will use umap.

# COMMAND ----------

# DBTITLE 1,Setup Vector Index
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from pathlib import Path
from llama_index.readers.file.unstructured import UnstructuredReader

index_persist_dir = f'/Volumes/{db_catalog}/{db_schema}/{db_volume}/folder_index'
Path(index_persist_dir).mkdir(exist_ok=True, parents=True)

# This can take up to 18 mins on 8 core node - suggest running before class
# documents = SimpleDirectoryReader(
#    input_dir=f"/Volumes/{db_catalog}/{db_schema}/{db_volume}",
#    file_extractor={
#       ".pdf": UnstructuredReader()
#    }
# ).load_data(num_workers=20)

# folder_index = VectorStoreIndex.from_documents(documents)
# folder_index.storage_context.persist(persist_dir=index_persist_dir)

# If index has been created can reload
from llama_index.core import StorageContext, load_index_from_storage

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir=index_persist_dir)

# load index
folder_index = load_index_from_storage(storage_context)

# COMMAND ----------

import pandas as pd
import umap
import plotly.express as px
from bokeh.resources import CDN
from bokeh.embed import file_html

text_obj = [document.text for document in list(folder_index.docstore.docs.values())]
encoded_chunks = [embeddings.embed_query(text_chk) for text_chk in text_obj]
vector_chunks = np.array(encoded_chunks, dtype=np.float32)
vector_chunks /= np.linalg.norm(vector_chunks, axis=1)[:, np.newaxis]

# COMMAND ----------

# DBTITLE 1,Visualise Chunk Text
umap_2d = umap.UMAP(n_components=2, init='random', random_state=0)
#umap_3d = umap.UMAP(n_components=3, init='random', random_state=0)

proj_2d = umap_2d.fit(vector_chunks)

hover_data =  pd.DataFrame({'index': np.arange(len(text_obj)) ,
                          'text': text_obj})

# COMMAND ----------

# Seems this requires GPU?
from umap import plot

p = plot.interactive(proj_2d,  point_size=10)
html = file_html(p, CDN, "Research Doc")
displayHTML(html)

# COMMAND ----------

# MAGIC %md A further extension to tune your embeddings is to explore finetuning them. \
# MAGIC You would need to then host and deploy the model though
# MAGIC See: [Finetuning Embeddings](https://docs.llamaindex.ai/en/stable/examples/finetuning/embeddings/finetune_embedding/)