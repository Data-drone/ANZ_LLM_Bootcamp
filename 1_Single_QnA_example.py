# Databricks notebook source
# MAGIC %md
# MAGIC # Building a Q&A Knowledge Base - Part 1
# MAGIC Questioning one document

# COMMAND ----------

# MAGIC %pip install pypdf sentence_transformers chromadb ctransformers

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain import HuggingFacePipeline
from langchain.llms import HuggingFaceHub

# Manual Model building
from transformers import pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC In this example we will load up a single pdf and ask questions and answers of it.
# MAGIC Most examples use OpenAI here we wil try out Dolly v2 and huggingface libraries
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_32.png" alt="Note">  The goal here is to get some sort of response not necessarily a good response. We will address that in later sections.

# COMMAND ----------

# DBTITLE 1,Setup dbfs folder paths
# MAGIC %run ./utils

# COMMAND ----------

# can also set to gpu
run_mode = 'cpu' # 'gpu'

# COMMAND ----------

# As a first step we need to load and parse the document
# for a class 
#file_to_load = '/dbfs/bootcamp_data/pdf_data/2203.02155.pdf'
file_to_load = '/dbfs' + source_doc_folder + '/2303.10130.pdf'
file_path = 'https://arxiv.org/pdf/2303.10130.pdf'

loader = PyPDFLoader(file_to_load)
# This splits it into pages
pages = loader.load_and_split()

# COMMAND ----------

# We will view the page and decide what to do with it
# We can see that we get a list of Langchain document objects
page_0 = pages[0]
type(page_0)

# COMMAND ----------

# MAGIC %md We are going to the ```CharacterTextSplitter``` from LangChain to split this document. LangChain has many text splitters, see [here](https://python.langchain.com/docs/modules/data_connection/document_transformers/#text-splitters) for a complete list. This splits only on one type of character (defaults to ```"\n\n"```).

# COMMAND ----------

# We will feed all pages in
# chunk_size is a key parameter.
# For more advanced use we may want to tune this or use a paragraph splitter or something else
text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=100)
texts = text_splitter.split_documents(pages)

# COMMAND ----------

texts[1]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Setup Chromadb
# MAGIC
# MAGIC We utilise the ```HuggingFaceEmbeddings()``` from LangChain which defaults to ```sentence-transformers/all-mpnet-base-v2``` to generate our text embeddings. However, note that Chroma can handle tokenization, embedding, and indexing automatically for you. If you would like to change the embedding model, read [here on how to do that](https://docs.trychroma.com/embeddings). You will need instantiate the ```collection``` yourself instead of using the LangChain wrapper.
# MAGIC
# MAGIC You can read the documentate [here](https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/chroma) to learn more about how Chroma integrates with LangChain.

# COMMAND ----------

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                   model_kwargs={'device': 'cpu'})
docsearch = Chroma.from_documents(texts, embeddings)

# we can verify that our docsearch index has objects in it with this
print('The index includes: {} documents'.format(docsearch._collection.count()))

# COMMAND ----------

docsearch

# COMMAND ----------

# DBTITLE 1,Verify that the index is working

# We want to quickly verify as with the pace these libraries evolve, things can break often
query = "Does making language models bigger improve intent following?"

docs = docsearch.similarity_search(query)
print(docs[0].page_content)

# COMMAND ----------

# MAGIC %md
# MAGIC Although we can get results from Chroma, it's often useful to metadata as well as ids to our partitions of texts (or embedding vectors). Often we don't want to query the entire vector database. This use-case is addressed below.

# COMMAND ----------

for i, t in enumerate(texts): 
  if i % 2:
    t.metadata = {"source": file_path}
  else:
    t.metadata = {"source": "Uknown"}

print(texts[0].metadata)
print(texts[1].metadata)

# COMMAND ----------

docsearch_metadata = (
  Chroma.from_documents(
    collection_name="single_paper",
    documents=texts,
    ids=[f"id{x}" for x in range(len(texts))],
    embedding=HuggingFaceEmbeddings()
  )
)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can query the vector store and filter on a specific metadata condition

# COMMAND ----------

docs = (
  docsearch_metadata.similarity_search(
    query="What do we call models that use reinforcement learning with human feedback?",
    filter={"source": file_path})
)

print(f'Metadata of the document is: {docs[0].metadata}')
print(f'Some text from the returned page: "{docs[0].page_content[0:50]}"')

# COMMAND ----------

# MAGIC %md 
# MAGIC We can also query our Vector DB and retrieve a tuple of (result, score) so we can have a measure of confidence from the returned results. The returned value is a similarity score between the vector corresponding to the query and the vector for the returned document. Lower scores imply that the vectors are closer together and hence have higher relevance with the query vector.

# COMMAND ----------

docs = docsearch_metadata.similarity_search_with_score("What do we call models that use reinforcement learning with human feedback?")
scores = [d[1] for d in docs]
print(scores)

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
# MAGIC # Chaining together logic - Introducing Langchain
# MAGIC Lets now use Langchain to help us connect everything together\
# MAGIC Before, we would have to manually collect the chromadb outputs,\
# MAGIC construct a prompt and add the content then send it to the llm.\
# MAGIC Langchain has a single function for all this: `RetrievalQA`
# MAGIC
# MAGIC We can see the prompt that it uses here:
# MAGIC - https://github.com/hwchase17/langchain/tree/master/libs/langchain/langchain/chains/retrieval_qa


# COMMAND ----------

# We need to add a search key here
# k affects the number of documents retrieved.
### NOTE a document is not document in the human sense but a chunk from the `CharacterTextSplitter`
qa = RetrievalQA.from_chain_type(llm=llm_model, chain_type="stuff", 
                                 retriever=docsearch.as_retriever(search_kwargs={"k": 3}))

# COMMAND ----------

# Test Query 1
query = "What is this document about?"
qa.run(query)

# COMMAND ----------

# Test Query 2
query = "What are some key facts from this document?"
qa.run(query)

# COMMAND ----------


