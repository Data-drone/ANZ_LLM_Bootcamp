# Databricks notebook source
# MAGIC %md
# MAGIC # Advanced Chunking & Parsing
# MAGIC In order to building better query engines we need to experiment more with our embeddings
# MAGIC  
# MAGIC Dolly and other GPT-J models are build with 2048 token length whereas OpenAI has a token length of 4096
# MAGIC New models with even longer token lengths have been coming as well ie the MPT 7B Model

# COMMAND ----------

%pip install pymupdf unstructured["local-inference"] sqlalchemy 'git+https://github.com/facebookresearch/detectron2.git' poppler-utils ctransformers

# COMMAND ----------
# DBTITLE 1,Load Libs

from langchain.document_loaders import PyMuPDFLoader
import os


# COMMAND ----------
# DBTITLE 1,Setup

%run ./utils

# COMMAND ----------

# source_doc_folder = f'/dbfs/bootcamp_data/pdf_data'
sample_file_to_load = source_doc_folder + '/2302.07842.pdf'

# COMMAND ----------

# load a model for testing
run_mode = 'cpu'
pipe = load_model(run_mode, dbfs_tmp_cache)

# COMMAND ----------

# MAGIC %md
# MAGIC # Basic File Loading 

# COMMAND ----------

loader = PyMuPDFLoader(sample_file_to_load)
docu_split = loader.load_and_split()

# COMMAND ----------

docu_split

# COMMAND ----------

context = docu_split[0].page_content
docu_split[0].page_content

# COMMAND ----------

system_prompt = f'<<SYS>>You are a helpful chatbot that is review the page context provided and answering a question based on that<<SYS>>'

question = 'How can we augment Large Language Models?'

prompt_template = f"""{system_prompt}
[INST]
Based on the below paragraph, answer the user question:

{context}

User: {question}
[/INST]

Assistant:"""

# COMMAND ----------

pipe(prompt_template, max_new_tokens=100, repetition_penalty=1.2)

# COMMAND ----------

# MAGIC %md
# MAGIC Lets see what happens if we manually bring in a useful paragraph instead 

# COMMAND ----------

context = f'This survey reviews works in which language models (LMs) are augmented with reasoning
skills and the ability to use tools. The former is defined as decomposing a potentially
complex task into simpler subtasks while the latter consists in calling external modules such
as a code interpreter. LMs can leverage these augmentations separately or in combination
via heuristics, or learn to do so from demonstrations. While adhering to a standard missing
tokens prediction objective, such augmented LMs can use various, possibly non-parametric
external modules to expand their context processing ability, thus departing from the pure
language modeling paradigm. We therefore refer to them as Augmented Language Models
(ALMs). The missing token objective allows ALMs to learn to reason, use tools, and even
act, while still performing standard natural language tasks and even outperforming most
regular LMs on several benchmarks. In this work, after reviewing current advance in ALMs,
we conclude that this new research direction has the potential to address common limitations
of traditional LMs such as interpretability, consistency, and scalability issues.'

prompt_template = f"""{system_prompt}
[INST]
Based on the below paragraph, answer the user question:

{context}

User: {question}
[/INST]

Assistant:"""

pipe(prompt_template, max_new_tokens=100, repetition_penalty=1.2)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Manually loading and parsing pdf

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
# MAGIC See: https://towardsdatascience.com/extracting-headers-and-paragraphs-from-pdf-using-pymupdf-676e8421c467 for a detailed example
# MAGIC
# MAGIC Alternative methods:
# MAGIC - Use a document scanning model ie LayoutLM
# MAGIC - Use a PDF to HTML converter then parse the html fligs
# MAGIC   - ie \<p>, \<h1>, \<h2> etc each pdf to html converter would work a bit different though....
# MAGIC
# MAGIC With our improved parser, we would then either:
# MAGIC - write it as a langchain operator
# MAGIC - write it as a pyspark pandas_udf, parse the pdf docs to a standard Delta table then ingest and embed the Delta table
# MAGIC Here is another tutorial: https://towardsdatascience.com/data-extraction-from-a-pdf-table-with-semi-structured-layout-ef694f3f8ff1
# MAGIC Possible science experiment: Train a visual understanding LLM ie OpenFlamingo
# MAGIC (I suspect someone will release an opensource one soon that can do this)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Parsing of PDFs
# We can try newer more advanced parsers instead of manual coding

# Unstructured seems to show some promise
# - nltk is required and libs should be pre-installed
# - unstructured can also use things like detectron2 etc

# COMMAND ----------

import unstructured
from unstructured.partition.pdf import partition_pdf
from collections import Counter

# COMMAND ----------

elements = partition_pdf(sample_file_to_load)

display(Counter(type(element) for element in elements))

# COMMAND ----------

display(*[(type(element), element.text) for element in elements[0:13]])

# COMMAND ----------

display(*[(type(element), element.text) for element in elements[100:105]])

# COMMAND ----------

# Lets see what is in an element

element_to_examine = elements[0]
element_to_examine

# COMMAND ----------



# COMMAND ----------