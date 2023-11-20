# Databricks notebook source
# MAGIC %md
# MAGIC # Building a Q&A Knowledge Base - Part 1
# MAGIC Questioning one document
# MAGIC
# MAGIC This version uses mlflow gateway endpoints

# COMMAND ----------

# MAGIC %pip install pypdf sentence_transformers chromadb==0.4.17 llama_index==0.8.54 mlflow==2.8.0

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
import chromadb

# Manual Model building
from transformers import pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC In this example we will load up a single pdf and ask questions and answers of it.
# MAGIC We will use an External Model via mlflow gateway as our llm and embedding service
# MAGIC
# MAGIC Ours goal is twofold:
# MAGIC - Find a way to convert our source data into useful snippets that can be inserted into prompts as context
# MAGIC - To use our vector db to provide relevant chunks for use in our prompts
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_32.png" alt="Note">  The goal here is to get some sort of response not necessarily a good response. We will address that in later sections.

# COMMAND ----------

# DBTITLE 1,Setup dbfs folder paths
# MAGIC %run ./utils

# COMMAND ----------

# As a first step we need to load and parse the document
# for a class 

# https://arxiv.org/pdf/2204.01691.pdf
file_to_load = '/dbfs/bootcamp_data/pdf_data/2302.09419.pdf'
#file_to_load = '/dbfs' + source_doc_folder + '/2302.09419.pdf'
file_path = 'https://arxiv.org/pdf/2302.09419.pdf'

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

# Lets see what a chunk is
texts[1]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Setup Chromadb
# MAGIC
# MAGIC We utilise the `HuggingFaceEmbeddings()` from LangChain which defaults to `sentence-transformers/all-mpnet-base-v2` to generate our text embeddings. However, note that Chroma can handle tokenization, embedding, and indexing automatically for you. If you would like to change the embedding model, read [here on how to do that](https://docs.trychroma.com/embeddings). You will need instantiate the ```collection``` yourself instead of using the LangChain wrapper.
# MAGIC
# MAGIC You can read the documentate [here](https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/chroma) to learn more about how Chroma integrates with LangChain.

# COMMAND ----------

from langchain.embeddings.mlflow_gateway import MlflowAIGatewayEmbeddings
from chromadb.config import Settings
import uuid

chroma_local_folder = '/local_disk0/chromadb'
chroma_archive_folder = f'/dbfs/Users/{username}/simple_chromadb'

embeddings = MlflowAIGatewayEmbeddings(
   gateway_uri="databricks",
   route="mosaicml-instructor-xl-embeddings"
)
#embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
#                                   model_kwargs={'device': 'cpu'})

settings = Settings(
   allow_reset=True
)

client = chromadb.PersistentClient(settings=settings, path=chroma_local_folder)
client.reset()
collection = client.create_collection('single_paper')

docsearch = Chroma(
   client=client,
   collection_name='single_paper',
   embedding_function=embeddings
)

for text in texts:
   docsearch.add_texts(texts=[text.page_content])

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
# MAGIC We can query our Vector DB and retrieve a tuple of (result, score) so we can have a measure of confidence from the returned results. The returned value is a similarity score between the vector corresponding to the query and the vector for the returned document. Lower scores imply that the vectors are closer together and hence have higher relevance with the query vector.

# COMMAND ----------

docs = docsearch.similarity_search_with_score("What do we call models that use reinforcement learning with human feedback?")
scores = [d[1] for d in docs]
print(scores)

# COMMAND ----------

#We will use our External LLM to provide "the brains"
from langchain.chat_models import ChatMLflowAIGateway

mosaic_chat_route_name = "mosaicml-llama2-70b-chat"

llm_model = ChatMLflowAIGateway(
    gateway_uri="databricks",
    route=mosaic_chat_route_name,
    params={
        "temperature": 0.0,
        "candidate_count": 2,
        "stop": [""], # There is something weird with this param but this works for now
        "max_tokens": 256
    },
)


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
result = qa.run(query)
print(result)

# COMMAND ----------

# MAGIC %md LLMs have progressed a lot
# MAGIC back with MPT-7b we got total nonsense like the below
# MAGIC
# MAGIC ```
# MAGIC The document.\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\
# MAGIC n\n\n\n\n\n\n\n\n\nfashion design to Q  the following prompts 
# MAGIC and \n\n\n\n\nThe question below 5555530 \n\n\n\n\n\nWhat is 
# MAGIC here:The answer, this document.\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n
# MAGIC \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n
# MAGIC \n\n\n\n\n\n\n\n\n\n\n\n\n\n    question : BC in complete in seconds 
# MAGIC eldou can be used to the following the question is required\n\n\n\n\n
# MAGIC Filling Answer at  Please help wanted ----------\n\n\n\n\n\n\n\n\n\n\n\n
# MAGIC \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n
# MAGIC \n\n\n\nThe question at the following context (eBotry Answer Later 777:\n\n\n\n\n\n
# MAGIC ```
# MAGIC
# MAGIC There are a few different things that we can to fix this.
# MAGIC First, langchain defaults to prompts tuned on OpenAI ChatGPT by default  
# MAGIC But newer models like we are using now run fine
# COMMAND ----------

from langchain import PromptTemplate
system_template = """<s>[INST] <<SYS>>
As a helpful assistant, answer questions from users but be polite and concise. If you don't know say I don't know.
<</SYS>>


Based on the following context:

{context}

Answer the following question:
{question}[/INST]
"""

zephyr_template = """<|system|>
As a helpful assistant, answer questions from users but be polite and concise. If you don't know say I don't know.

Based on the following context:

{context}

<|user|>
{question}

<|assistant|>
"""

# prompt templates in langchain need the input variables specified it can then be loaded in the string
# Note that the names of the input_variables are particular to the chain type.
prompt_template = PromptTemplate(
    input_variables=["question", "context"], template=system_template
)

qa = RetrievalQA.from_chain_type(llm=llm_model, chain_type="stuff", 
                                 retriever=docsearch.as_retriever(search_kwargs={"k": 3}),
                                 chain_type_kwargs={"prompt": prompt_template})

result = qa.run(query)
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC Note it can be hard to figure out how to override defaults in langchain.
# MAGIC For example in this case, `RetrievalQA`` is of class `BaseRetrievalQA``. 
# MAGIC `BaseRetrievalQA` is instantiated with `from_chain_type` in our case.
# MAGIC Inside that method, we can see that `load_qa_chain` is the function that generates the chain.
# MAGIC It is only by looking inside `load_qa_chain` can we work out the correct variable to use to override the prompt.
# MAGIC All this of course can be explored only in the source code.
# MAGIC
# MAGIC We should also review results from our Chroma search. 
# COMMAND ----------

docsearch.similarity_search(query)

# COMMAND ----------

# MAGIC %md
# MAGIC if you got the same result as us you would see the snippets are mostly useless
# MAGIC That was a bad retrieval from our doc store
# MAGIC It means that we need to look into chunking strategy and filtering methods
# MAGIC We know that the document is about large language models
# MAGIC Lets adjust the query so that we can "trigger" those keywords and embeddings.
# COMMAND ----------

query = 'What is text summurisation? How can it be useful?'
docsearch.similarity_search(query)
# COMMAND ----------

# Test Query 2
# Generic queries like this tend to do badly.
query = "What are some key facts from this document?"
qa.run(query)

# COMMAND ----------

# MAGIC %md # Logging and Saving Models
# MAGIC Just like with logging and managing prompts, it is possible to log our chain entity as a mlflow model
# MAGIC We can then use mlflow evaluate to compare different module configurations or splitting and parsing strategies

# COMMAND ----------

import mlflow
import pandas as pd
import shutil

mlflow_dir = f'/Users/{username}/simple_rag_chain'
mlflow.set_registry_uri('databricks-uc')
mlflow.set_experiment(mlflow_dir)

# COMMAND ----------

# We need to save out our Chroma database

# this will be deprecated
try:
  shutil.copytree(chroma_local_folder, chroma_archive_folder)

except FileExistsError:
  shutil.rmtree(chroma_local_folder)
  shutil.copytree(chroma_archive_folder, chroma_local_folder)

# COMMAND ----------

# We can setup some example questions for testing the chain as well
test_questions = ['What are the basic components of a Transformer?',
                  'What is a tokenizer?',
                  'How can we handle audio?',
                  'Are there alternatives to transformers?']

testing_questions = pd.DataFrame(
    test_questions, columns = ['prompt']
)

# COMMAND ----------

# This is a custom wrapper allows us to log langchain to mlflow as a model
# for logging the chroma artifacts we need to follow: 
# https://docs.databricks.com/en/machine-learning/model-serving/model-serving-custom-artifacts.html
class LangchainQABot(mlflow.pyfunc.PythonModel):
    
    def __init__(self, host, token, system_template, chroma_archive_dir:str=chroma_archive_folder):
        self.host = host
        self.token = token
        self.system_template = system_template
        self.chroma_archive_dir = chroma_archive_dir
        self.chroma_local_dir = '/local_disk0/chromadb'

    def setup_retriever(self, context):
        from langchain.embeddings.mlflow_gateway import MlflowAIGatewayEmbeddings
        import chromadb

        try:
          shutil.copytree(context.artifacts['chroma_db'], self.chroma_local_dir)
        except FileExistsError:
          shutil.rmtree(self.chroma_local_dir)
          shutil.copytree(context.artifacts['chroma_db'], self.chroma_local_dir)

        embeddings = MlflowAIGatewayEmbeddings(
                        gateway_uri="databricks",
                        route="mosaicml-instructor-xl-embeddings"
                     )
        
        client = chromadb.PersistentClient(settings=settings, path=chroma_local_folder)
        
        doc_search = Chroma(client=client,
                            collection_name='single_paper',
                            persist_directory=self.chroma_local_dir, 
                            embedding_function=embeddings
                            )
        
        return doc_search.as_retriever(search_kwargs={"k": 3})

    def load_context(self, context):
        from langchain.chat_models import ChatMLflowAIGateway
        import os

        # connecting to gateway requires that this is set
        os.environ['DATABRICKS_HOST'] = self.host
        os.environ['DATABRICKS_TOKEN'] = self.token

        self.retriever = self.setup_retriever(context)

        self.prompt_template = PromptTemplate(
            input_variables=["question", "context"], template=self.system_template
        )
        
        mosaic_chat_route_name = "mosaicml-llama2-70b-chat"

        llm_model = ChatMLflowAIGateway(
            gateway_uri="databricks",
            route=mosaic_chat_route_name,
            params={
                "temperature": 0.0,
                "candidate_count": 2,
                "stop": [""], # There is something weird with this param but this works for now
                "max_tokens": 256
            },
        )

        self.qa = RetrievalQA.from_chain_type(llm=llm_model, chain_type="stuff", 
                                 retriever=self.retriever,
                                 chain_type_kwargs={"prompt": self.prompt_template})
    
    def process_row(self, row):
       return self.qa.run(row['prompt'])
    
    def predict(self, context, data):
        results = data.apply(self.process_row, axis=1) 
        return results

# COMMAND ----------


# **NOTE** This doesn't deploy properly as a model serving model yet
# Due to some bug in the code
db_token = '<redacted>'

catalog = 'bootcamp_ml'
schema = 'rag_chatbot'
model_name = 'retrieval_chain'

model = LangchainQABot('https://adb-984752964297111.11.azuredatabricks.net/', db_token, system_template, chroma_archive_folder)

user_input = "What is a tokenizer?"
input_example = {"prompt": user_input}

langchain_signature = mlflow.models.infer_signature(
    model_input=input_example,
    model_output=[qa.run(user_input)]
)

with mlflow.start_run() as run:
  mlflow_result = mlflow.pyfunc.log_model(
      python_model = model,
      extra_pip_requirements = ['llama_index==0.8.54',
                                'chromadb==0.4.17'],
      artifacts = {
         'chroma_db': chroma_local_folder
      },
      artifact_path = 'langchain_pyfunc',
      signature = langchain_signature,
      input_example = input_example,
      registered_model_name = f'{catalog}.{schema}.{model_name}'
  )

  mlflow.evaluate(mlflow_result.model_uri,
                  testing_questions,
                  model_type="text")

# COMMAND ----------