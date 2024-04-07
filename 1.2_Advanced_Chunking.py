# Databricks notebook source
# MAGIC %md
# MAGIC # Advanced Parsing & Chunking
# MAGIC
# MAGIC In order to building better query engines we need to improve our extraction and chunking process
# MAGIC - Are we extracting all the information coherently
# MAGIC - Are we splitting the document into sensible chunks?
# MAGIC - What size chunks do we need to make sure that we can fit our model context and provide sufficient extracts?
# MAGIC
# MAGIC There are two steps to the process. Parse and Chunk \
# MAGIC With Parse we need to extract all the text and assocaited metadata that we can \
# MAGIC With Chunk we take the parse and break it down into digestible sections for LLM Promptiog
# MAGIC
# MAGIC The default methods are naive and tend to split just on character limits or words
# MAGIC
# MAGIC We will leverage the library - unstructured but there are many other options out there
# MAGIC

# COMMAND ----------

# MAGIC %sh
# MAGIC # we needed to do this for poppler to work in many cases
# MAGIC apt-get install -y poppler-utils

# COMMAND ----------

# MAGIC %pip install pymupdf llama_index==0.10.25 langchain==0.1.13 llama-index-llms-langchain poppler-utils unstructured[pdf,txt]==0.13.0 databricks-vectorsearch==0.23 llama-index-embeddings-langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
# DBTITLE 1,Setup

%run ./utils

# COMMAND ----------

# DBTITLE 1,Config
import os
from langchain_community.chat_models import ChatDatabricks
from langchain.document_loaders import PyMuPDFLoader
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

sample_file_to_load = f'/Volumes/{db_catalog}/{db_schema}/{db_volume}/2302.06476.pdf'
print(f'We will use {sample_file_to_load} to review chunking open it alongside to see how different algorithms work')
print(f'You can access it here https://arxiv.org/pdf/2302.06476.pdf')
# COMMAND ----------

# MAGIC %md
# MAGIC # Basic File Loading
# MAGIC We will just use the basic pymupdf loader for this stage. \
# MAGIC The load_and_split function will handle all the config.

# COMMAND ----------

loader = PyMuPDFLoader(sample_file_to_load)
docu_split = loader.load_and_split()
docu_split

# COMMAND ----------

# We can see that the first page has been all concatenated into a page
# If you search for "reasonable performance" you will see that the footer has been merged into the paragraph
Intro = docu_split[0].page_content
Intro

# COMMAND ----------

# IT looks like we have the last bit of a paragraph and the footer from page 1 here
Weird_snippet = docu_split[1].page_content
Weird_snippet

# COMMAND ----------

# Our table has kinda been picked up with \n separations
# We probably want the table with descriptor as one chunk and the rest split out
Table = docu_split[36].page_content
Table

# COMMAND ----------

# MAGIC %md
# MAGIC ## Manually loading and parsing pdf
# MAGIC
# MAGIC Exploring PDF Parse Primitives \
# MAGIC We could experiment with using the raw pdf parse primitives but this will be slow


# COMMAND ----------
import fitz 

doc = fitz.open(sample_file_to_load)

for page in doc:
  page_dict = page.get_text("dict")
  blocks = page_dict["blocks"]
  print(blocks)
  break
# COMMAND ----------

# MAGIC %md
# MAGIC We can see that the raw PyMuPDF has a lot more information stored on the text block 
# MAGIC We have information on location of text, 

# COMMAND ----------

# lets see what is in these objects
print(page_dict.keys())

# lets see how many blocks there are:
print(len(page_dict['blocks']))

# lets see what is in a block
print(page_dict['blocks'])

# COMMAND ----------
# Title
page_dict['blocks'][0]

# COMMAND ----------

# First Line authors
page_dict['blocks'][1]

# COMMAND ----------

# 2nd Line authors
page_dict['blocks'][2]

# COMMAND ----------

# The image
page_dict['blocks'][5]

# COMMAND ----------

# MAGIC %md
# MAGIC What will it take to keep the context info and make use of it? 
# MAGIC Depending on our docs, we will have to write custom logic to be able to parse and understand the structure of the document
# MAGIC
# MAGIC See [PyMuPDF Docs](https://pymupdf.readthedocs.io/en/latest/tutorial.html) for extra details on how to parse
# MAGIC
# MAGIC Alternative methods:
# MAGIC - Use a document scanning model ie LayoutLM
# MAGIC - Use a PDF to HTML converter then parse the html fligs
# MAGIC   - ie \<p>, \<h1>, \<h2> etc each pdf to html converter would work a bit different though....
# MAGIC
# MAGIC With our improved parser, we could then:
# MAGIC - write it as a pyspark pandas_udf, parse the pdf docs to a standard Delta table that we could then embed with Datbricks VectorSearch

# COMMAND ----------

# MAGIC %md
# MAGIC # Advanced Parsing of PDFs
# MAGIC We can try newer more advanced parsers instead of manual coding
# MAGIC
# MAGIC Unstructured is one option. The OSS Unstructured Library has two modes of operation \
# MAGIC A basic parse that reads the raw pdf structure, analyses it for headings, paragraphs etc then tries to group them logically \
# MAGIC An OCR mode that applies a computer vision model to help with data extration.
# MAGIC - nltk is required and libs should be pre-installed
# MAGIC - OCR Extraction will involving correctly setting up Computer Vision (Pytorch) based libraries
# MAGIC See [Unstructured Docs](https://unstructured-io.github.io/unstructured/installation/full_installation.html) for more information on installing

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unstructured PDF Reader
# MAGIC
# MAGIC Lets use the reader on its own first to see what it extracts before we integrate with LangChain to parse

# COMMAND ----------

from unstructured.partition.pdf import partition_pdf
from collections import Counter

# COMMAND ----------

elements = partition_pdf(sample_file_to_load)

# You can see that the underlying structures have been classified
display(Counter(type(element) for element in elements))

# COMMAND ----------

# Authors on the front page are appearing as Title sections.  
display(*[(type(element), element.text) for element in elements[0:13]])

# COMMAND ----------

# Sections are being extracted as narrative text
display(*[(type(element), element.text) for element in elements[400:410]])

# COMMAND ----------

# MAGIC %md
# MAGIC # Using Unstructured with Llama_index
# MAGIC
# MAGIC Improving the way we parse by adding more custom logic is one way to better performance \
# MAGIC Compared with 2023, models are increasingly able to digest weird chunks and disjointed paragraphs though. \
# MAGIC
# MAGIC Another route to better performance is to leverage more intelligent structuring of chunks with a library like Llama_index
# MAGIC
# MAGIC Llama Index can structure chunks, `Nodes` in Llama_index jargon so that it has an understanding of their spatial relationships
# MAGIC See: [Llama Index Types](https://docs.llamaindex.ai/en/stable/module_guides/indexing/index_guide/)

# COMMAND ----------

# DBTITLE 1,Setting Up Llama_index default models
from langchain_community.chat_models import ChatDatabricks
from langchain_community.embeddings import DatabricksEmbeddings
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

# DBTITLE 1,Data loaders
# Note that this can take a while to run as it downloads a computer Vision model
# in case it needs to do OCR Analysis
from llama_index.core import VectorStoreIndex
from pathlib import Path
from llama_index.readers.file.unstructured import UnstructuredReader

unstruct_loader = UnstructuredReader()
unstructured_document = unstruct_loader.load_data(sample_file_to_load)

# COMMAND ----------

# DBTITLE 1,Generate Index
unstructured_index = VectorStoreIndex.from_documents(unstructured_document)
unstructured_query = unstructured_index.as_query_engine()

# COMMAND ----------

# DBTITLE 1,Query
question = 'Are there any weak points in ChatGPT for Zero Shot Learning?'
unstructured_result = unstructured_query.query(question)
print(unstructured_result.response)

# COMMAND ----------

# MAGIC %md Try out other types of indices too and as an extension, see how well it performs with multiple documents \
# MAGIC We have just looked at single document for now, identifying the best document to use in a multi document situation is different
# COMMAND ----------

