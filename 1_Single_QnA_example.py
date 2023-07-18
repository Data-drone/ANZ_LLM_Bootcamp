# Databricks notebook source
# MAGIC %md
# MAGIC # Building a Q&A Knowledge Base - Part 1
# MAGIC Questioning one document

# COMMAND ----------

# MAGIC %pip install langchain pypdf sentence_transformers chromadb

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

# MAGIC %md
# MAGIC ## Generate API tokens
# MAGIC For many of the services that we'll using in the notebook, we'll need some API keys. Follow the instructions below to generate your own. 
# MAGIC
# MAGIC ### Hugging Face Hub
# MAGIC 1. Go to this [Inference API page](https://huggingface.co/inference-api) and click "Sign Up" on the top right.
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/llm/hf_sign_up.png" width=700>
# MAGIC
# MAGIC 2. Once you have signed up and confirmed your email address, click on your user icon on the top right and click the `Settings` button. 
# MAGIC
# MAGIC 3. Navigate to the `Access Token` tab and copy your token. 
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/llm/hf_token_page.png" width=500>
# MAGIC

# COMMAND ----------

# Setup and config variables
## We will store data in a local folder for now
username = spark.sql("SELECT current_user()").first()['current_user()']
username

# See: https://docs.databricks.com/security/secrets/example-secret-workflow.html
# To learn how to set secrets
# We need to set this to pull from huggingface hub - You can get a token here
# https://huggingface.co/docs/hub/security-tokens
os.environ['HUGGINGFACEHUB_API_TOKEN'] =  "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"

data_folder = f'/dbfs/home/{username}/pdf_data'
file_to_load = data_folder + '/2109.07306.pdf'

# can also set to gpu
run_mode = 'cpu' # 'gpu'

# COMMAND ----------

# DBTITLE 1,Let's ensure that we were able to get the PDFs from arXiv
display(dbutils.fs.ls(f'dbfs:/home/{username}/pdf_data'))

# COMMAND ----------

# As a first step we need to load and parse the document

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
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(pages)

# COMMAND ----------

texts[1]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Use Chroma wth LangChain
# MAGIC
# MAGIC We utilise the ```HuggingFaceEmbeddings()``` from LangChain which defaults to ```sentence-transformers/all-mpnet-base-v2``` to generate our text embeddings. However, note that Chroma can handle tokenization, embedding, and indexing automatically for you. If you would like to change the embedding model, read [here on how to do that](https://docs.trychroma.com/embeddings). You will need instantiate the ```collection``` yourself instead of using the LangChain wrapper.
# MAGIC
# MAGIC You can read the documentate [here](https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/chroma) to learn more about how Chroma integrates with LangChain.

# COMMAND ----------

embeddings = HuggingFaceEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

# we can verify that our docsearch index has objects in it with this
print('The index includes: {} documents'.format(docsearch._collection.count()))

# COMMAND ----------

docsearch

# COMMAND ----------

# DBTITLE 1,Verify that the index is working

# We want to quickly verify as with the pace these libraries evolve, things can break often
query = "What is important to have open LLMs?"

docs = docsearch.similarity_search(query)
print(docs[0].page_content)

# COMMAND ----------

# MAGIC %md
# MAGIC Although we can get results from Chroma, it's often useful to metadata as well as ids to our partitions of texts (or embedding vectors). Often we don't want to query the entire vector database. This use-case is addressed below.

# COMMAND ----------

for t in texts: 
  t.metadata = {"source": "https://arxiv.org/abs/2109.07306"}

print(texts[0].metadata)

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
    query="What is important to have open LLMs?",
    filter={"source": "https://arxiv.org/abs/2109.07306"})
)

docs[0].metadata

# COMMAND ----------

# MAGIC %md 
# MAGIC We can also query our Vector DB and retrieve a tuple of (result, score) so we can have a measure of confidence from the returned results. The returned value is a similarity score between the vector corresponding to the query and the vector for the returned document. Lower scores imply that the vectors are closer together and hence have higher relevance with the query vector.

# COMMAND ----------

docs = docsearch_metadata.similarity_search_with_score("What is important to have open LLMs?")
scores = [d[1] for d in docs]
print(scores)

# COMMAND ----------

## One problem with the library at the moment is that GPU ram doesn't get relinquished when the object is overridden
# The only way to clear GPU ram is to detach and reattach

# This snippet will make sure we don't keep reloading the model and running out of GPU ram
try:
  llm_model
except NameError:

  # We can just use the model this way but token limits and fine tuning can be problematic
  #llm_model = HuggingFaceHub(repo_id="google/flan-ul2", 
  #                              model_kwargs={"temperature":0.1, "max_new_tokens":1024})

  # We will create a huggingface pipeline and work with that
  # See: https://huggingface.co/docs/transformers/main_classes/pipelines
  # We need to have "text-generation" as the task

  # For the config we can see: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig

  if run_mode == 'cpu':

    from ctransformers.langchain import CTransformers
    model_id = "/local_disk0/mpt-7b-instruct.ggmlv3.q5_0.bin"
    llm = CTransformers(model=model_id, 
                    model_type='mpt')
    #pipe = AutoModelForCausalLM.from_pretrained(model_id, model_type='gpt2', lib='avx')
    llm_model = HuggingFacePipeline(pipeline=llm)

  elif run_mode == 'gpu':

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "databricks/dolly-v2-3b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_length = 2048
        )

    llm_model = HuggingFacePipeline(pipeline=pipe)

else:
  pass

# COMMAND ----------

# We need to add a search key here
# k affects the number of documents retrieved.
### NOTE a document is not document in the human sense but a chunk from the `CharacterTextSplitter`
qa = RetrievalQA.from_chain_type(llm=llm_model, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 2}))

# COMMAND ----------

# Test Query 1
query = "What is this document about?"
qa.run(query)

# COMMAND ----------

# Test Query 2
query = "What are some key facts from this document?"
qa.run(query)

# COMMAND ----------


