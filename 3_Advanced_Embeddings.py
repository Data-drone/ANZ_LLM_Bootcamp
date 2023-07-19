# Databricks notebook source
# MAGIC %md
# MAGIC # Advanced Embeddings
# MAGIC In order to building better query engines we need to experiment more with our embeddings
# MAGIC  
# MAGIC Dolly and other GPT-J models are build with 2048 token length whereas OpenAI has a token length of 4096
# MAGIC New models with even longer token lengths have been coming as well ie the MPT 7B Model

# COMMAND ----------
# DBTITLE 1,Load Libs

from langchain.document_loaders import PyMuPDFLoader
import os


# COMMAND ----------
# DBTITLE 1,Setup

# we will build off of the simple doc analyser from notebook 1
# we select PyMuPDFLoader in this case but others maybe applicable too
# Open https://arxiv.org/pdf/2304.10453.pdf on the side as well to follow along

username = spark.sql("SELECT current_user()").first()['current_user()']

#os.environ['HUGGINGFACEHUB_API_TOKEN'] =  dbutils.secrets.get(scope = "brian-hf", key = "hf_key")

data_folder = f'/dbfs/home/{username}/pdf_data'

# COMMAND ----------

sample_file_to_load = data_folder + '/2212.10264.pdf'

# COMMAND ----------

loader = PyMuPDFLoader(sample_file_to_load)
loader.load_and_split()

# COMMAND ----------

# MAGIC %md
# MAGIC We can see that the data isn't really structured properly. 
# MAGIC Headings and text is mixed up any logic is lost. We can probably do better and get better embeddings.

# MAGIC # Experiment with PyMuPDF

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
# MAGIC TODO - Basic parser to pull out headings, disgrams and sections. 


