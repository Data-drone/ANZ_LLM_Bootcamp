# Databricks notebook source
# MAGIC %md
# MAGIC # Building a Q&A Knowledge Base - Part 1
# MAGIC Questioning one document

# COMMAND ----------

# MAGIC %pip install pypdf sentence_transformers chromadb==0.4.15 ctransformers==0.2.26 llama_index==0.8.54

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
# MAGIC Most examples use OpenAI here we wil try out Llama v2.
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

# can also set to gpu
run_mode = 'serving' # 'gpu'

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
    t.metadata = {"source": "Unknown"}

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
  if run_mode == 'serving':

    ## the Langchain Databricks LLM definition is currently not compatible with Optimised Serving
    serving_uri = 'vicuna_13b'
    browser_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
    db_host = f"https://{browser_host}"
    model_uri = f"{db_host}/serving-endpoints/{serving_uri}/invocations"
    db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

    llm_model = ServingEndpointLLM(endpoint_url=model_uri, token=db_token)
  else:
    pipe = load_model(run_mode, dbfs_tmp_cache, 'vicuna_13b')
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
result = qa.run(query)
print(result)

# COMMAND ----------

# MAGIC %md If you got the same result as we did in testing it might be total nonsense!
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
mlflow.set_experiment(mlflow_dir)

# COMMAND ----------

# We need to save out our Chroma
# Load and chunk
file_to_load = '/dbfs/bootcamp_data/pdf_data/2302.09419.pdf'

chroma_local_folder = '/local_disk0/chromadb'
chroma_archive_folder = f'/dbfs/Users/{username}/simple_chromadb'

loader = PyPDFLoader(file_to_load)
pages = loader.load_and_split()
text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=100)
texts = text_splitter.split_documents(pages)

docsearch = Chroma.from_documents(texts, embeddings, persist_directory=chroma_local_folder)

## Whilst we can archive Chroma files onto dbfs, the working files must be on a local SSD like local_disk0
shutil.copytree(chroma_local_folder, chroma_archive_folder)

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
class LangchainQABot(mlflow.pyfunc.PythonModel):
    
    def __init__(self, model_uri, token, system_template, chroma_archive_dir:str=chroma_archive_folder):
        self.model_uri = model_uri
        self.token = token
        self.system_template = system_template
        self.chroma_archive_dir = chroma_archive_dir
        self.chroma_local_dir = '/local_disk0/chromadb'

    def setup_retriever(self):
        try:
          shutil.copytree(self.chroma_archive_dir, self.chroma_local_dir)
        except FileExistsError:
          shutil.rmtree(self.chroma_local_dir)
          shutil.copytree(self.chroma_archive_dir, self.chroma_local_dir)

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                   model_kwargs={'device': 'cpu'})
        doc_search = Chroma(persist_directory=self.chroma_local_dir, embedding_function=embeddings)
        return doc_search.as_retriever(search_kwargs={"k": 3})

    def load_context(self, context):
        from typing import Any, List, Mapping, Optional
        from langchain.callbacks.manager import CallbackManagerForLLMRun
        from langchain.llms.base import LLM
        import requests

        self.retriever = self.setup_retriever()

        self.prompt_template = PromptTemplate(
            input_variables=["question", "context"], template=self.system_template
        )
        
        # Our class must be included as it is not pickleable
        class ServingEndpointLLM(LLM):
            endpoint_url: str
            token: str

            @property
            def _llm_type(self) -> str:
                return "custom"

            def _call(
                self,
                prompt: str,
                stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any,
            ) -> str:
                if stop is not None:
                    raise ValueError("stop kwargs are not permitted.")
                
                header = {"Context-Type": "text/json", "Authorization": f"Bearer {self.token}"}

                dataset = {'inputs': {'prompt': [prompt]},
                          'params': kwargs}

                response = requests.post(headers=header, url=self.endpoint_url, json=dataset)

                return response.json()['predictions'][0]['candidates'][0]['text']

            @property
            def _identifying_params(self) -> Mapping[str, Any]:
                """Get the identifying parameters."""
                return {"endpoint_url": self.endpoint_url} 
    
        llm = ServingEndpointLLM(endpoint_url=self.model_uri, token=self.token)

        self.qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", 
                                 retriever=self.retriever,
                                 chain_type_kwargs={"prompt": self.prompt_template})
    
    def predict(self, context, data):
        questions = data['prompt']
        results = [self.qa.run(x) for x in questions] 
        return results

# COMMAND ----------

model = LangchainQABot(model_uri, db_token, system_template, chroma_archive_folder)

with mlflow.start_run() as run:
  mlflow_result = mlflow.pyfunc.log_model(
      python_model = model,
      extra_pip_requirements = ['langchain==0.0.267'],
      artifact_path = 'langchain_pyfunc'
  )

  mlflow.evaluate(mlflow_result.model_uri,
                  testing_questions,
                  model_type="text")

# COMMAND ----------