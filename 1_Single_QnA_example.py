# Databricks notebook source
# MAGIC %md
# MAGIC # Building a Q&A Knowledge Base - Part 1
# MAGIC Questioning one document

# COMMAND ----------

# MAGIC %pip install langchain pypdf sentence_transformers 

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
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# COMMAND ----------

# MAGIC %md
# MAGIC In this example we will load up a single pdf and ask questions and answers of it.
# MAGIC
# MAGIC Most examples use OpenAI here we wil try out Dolly v2 and huggingface libraries
# MAGIC NOTE The goal here is to get some sort of response not necessarily a good response

# COMMAND ----------

# Setup and config variables
## We will store data in a local folder for now
username = spark.sql("SELECT current_user()").first()['current_user()']
username

# See: https://docs.databricks.com/security/secrets/example-secret-workflow.html
# To learn how to set secrets
# We need to set this to pull from huggingface hub - You can get a token here
# https://huggingface.co/docs/hub/security-tokens
os.environ['HUGGINGFACEHUB_API_TOKEN'] =  dbutils.secrets.get(scope = "brian-hf", key = "hf_key")

data_folder = f'/dbfs/home/{username}/pdf_data'
file_to_load = data_folder + '/2304.10453.pdf'

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

# We will feed all pages in
# chunk_size is a key parameter.
# For more advanced use we may want to tune this or use a paragraph splitter or something else
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(pages)

# COMMAND ----------

embeddings = HuggingFaceEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

# we can verify that our docsearch index has objects in it with this
print('The index includes: {} documents'.format(docsearch._collection.count()))

# COMMAND ----------

# DBTITLE 1,Verify that the index is working

# We want to quickly verify as with the pace these libraries evolve, things can break often
query = "What is important to havve open LLMs?"
docs = docsearch.similarity_search(query)
print(docs[0].page_content)

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
