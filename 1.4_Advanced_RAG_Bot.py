# Databricks notebook source
# MAGIC %md
# MAGIC # Building an Advanced RAG System
# MAGIC
# MAGIC We will now build out an advanced RAG system with multiple files and some more complex logic
# MAGIC
# MAGIC We skip on `Llama_index`` and `Unstructured` here to expedite installs and run speed

# COMMAND ----------

# DBTITLE 1,Extra Libs to install
# MAGIC %pip install -U pymupdf typing_extensions sqlalchemy>=2.0.25 langchain==0.1.16 databricks-vectorsearch==0.23 flashrank mlflow==2.12.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup Utils
# MAGIC %run ./utils

# COMMAND ----------

# Dev override
vector_search_endpoint = 'one-env-shared-endpoint-5'
db_catalog = 'brian_gen_ai'
db_schema = 'lab_05'

# COMMAND ----------

# MAGIC %md
# MAGIC # Building our Vector Store and Index

# COMMAND ----------

# DBTITLE 1,Setup env and Embeddings
# We will go a little fancier and use a local embedder this can help save cost
from langchain_community.chat_models import ChatDatabricks
from langchain_community.embeddings import DatabricksEmbeddings
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

from langchain.schema import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts.prompt import PromptTemplate

# to do message history we need a history aware retriever 
# basically we receive the question and the history
# - Then we ask an LLM to reformulate
# - Then we send updated llm generated question to retriever
from langchain.chains import create_history_aware_retriever

chat_model = 'databricks-dbrx-instruct'
embedding_model_name = 'databricks-bge-large-en'
index_name = 'arxiv_data_bge_index'

vsc = VectorSearchClient()
vs_index_fullname = f'{db_catalog}.{db_schema}.{index_name}'

llm = ChatDatabricks(
    target_uri="databricks",
    endpoint=chat_model,
    temperature=0.1,
)
embeddings = DatabricksEmbeddings(endpoint=embedding_model_name)

# we should detect and raise error on missing index first

# COMMAND ----------

# Setup the logic

# vector search configuration
index = vsc.get_index(endpoint_name=vector_search_endpoint,
                      index_name=vs_index_fullname)

retriever = DatabricksVectorSearch(
    index, text_column="page_content", 
    embedding=embeddings, columns=["source_doc"]
).as_retriever(search_kwargs={"k": 10})

# Our ReRank Module
compressor = FlashrankRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

# formatting for context
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

## adding history
# We reformulate the question input with the chat_history context (if any) before feeding to retriever
contextualize_q_prompt = PromptTemplate(
    input_variables=["input", "chat_history"],
    template="<s> [INST] Your job is to reformulate a question given a user question and the prior conversational history. DO NOT answer the question. If there is no chat history pass through the question [/INST] </s> \n [INST] Question: {input} \nHistory: {chat_history} \nAnswer: [/INST]"
)

history_aware_retriever = create_history_aware_retriever(
    llm, compression_retriever, contextualize_q_prompt
)

rag_prompt = PromptTemplate(input_variables=['context', 'input', 'chat_history'],
                                      template="<s> [INST] You are a helpful personal assistant who helps users find what they need from documents. Be conversational, polite and use the following pieces of retrieved context and the conversational history to help answer the question. <unbreakable-instruction> ANSWER ONLY FROM THE CONTEXT </unbreakable-instruction> <unbreakable-instruction> If you don't know the answer, just say that you don't know. </unbreakable-instruction> Keep the answer concise. [/INST] </s> \n[INST] Question: {input} \nContext: {context} \nHistory: {chat_history} \nAnswer: [/INST]")


chain = (
    {'context': history_aware_retriever | format_docs, "input": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
    | rag_prompt
    | llm 
    | StrOutputParser()
)

# COMMAND ----------

# Test out endpoint
chain.invoke({'input': 'tell me about llms', 'chat_history': ''})

# To add chat history we need to include a list object with alternating `AiMessage` and `HumanMessage` entries

# COMMAND ----------

# MAGIC %md # Productionisation
# MAGIC
# MAGIC Whilst there is a langchain integration into mlflow, we will likely  want to build our own wrapper for extra flexibility \
# MAGIC `Langchain` and `llama_index` are constantly in flux and it it is common for integrations to break. \
# MAGIC See [MLflow Pyfunc](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html) for more information

# COMMAND ----------

import mlflow

class AdvancedLangchainBot(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        """
        When a model is instantiated in Databricks Model serving,
        This function is run first.

        As not all Langchain components are serialisable we should use this function
        to instantiate our whole chain

        The following is just pasted from above
        """

        from langchain_community.chat_models import ChatDatabricks
        from langchain_community.embeddings import DatabricksEmbeddings
        from databricks.vector_search.client import VectorSearchClient
        from langchain_community.vectorstores import DatabricksVectorSearch
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import FlashrankRerank

        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser

        from langchain_core.prompts.prompt import PromptTemplate

        from langchain.chains import create_history_aware_retriever

        chat_model = 'databricks-dbrx-instruct'
        embedding_model_name = 'databricks-bge-large-en'

        vsc = VectorSearchClient()
        vs_index_fullname = f'{db_catalog}.{db_schema}.{index_name}'

        llm = ChatDatabricks(
            target_uri="databricks",
            endpoint=chat_model,
            temperature=0.1,
        )

        embeddings = DatabricksEmbeddings(endpoint=embedding_model_name)

        index = vsc.get_index(endpoint_name=vector_search_endpoint,
                      index_name=vs_index_fullname)

        retriever = DatabricksVectorSearch(
            index, text_column="page_content", 
            embedding=embeddings, columns=["source_doc"]
        ).as_retriever(search_kwargs={"k": 10})

        # Our ReRank Module
        compressor = FlashrankRerank()
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

        # formatting for context
        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])

        ## adding history
        # We reformulate the question input with the chat_history context (if any) before feeding to retriever
        contextualize_q_prompt = PromptTemplate(
            input_variables=["input", "chat_history"],
            template="<s> [INST] Your job is to reformulate a question given a user question and the prior conversational history. DO NOT answer the question. If there is no chat history pass through the question [/INST] </s> \n [INST] Question: {input} \nHistory: {chat_history} \nAnswer: [/INST]"
        )

        history_aware_retriever = create_history_aware_retriever(
            llm, compression_retriever, contextualize_q_prompt
        )

        rag_prompt = PromptTemplate(input_variables=['context', 'input', 'chat_history'],
                                              template="<s> [INST] You are a helpful personal assistant who helps users find what they need from documents. Be conversational, polite and use the following pieces of retrieved context and the conversational history to help answer the question. <unbreakable-instruction> ANSWER ONLY FROM THE CONTEXT </unbreakable-instruction> <unbreakable-instruction> If you don't know the answer, just say that you don't know. </unbreakable-instruction> Keep the answer concise. [/INST] </s> \n[INST] Question: {input} \nContext: {context} \nHistory: {chat_history} \nAnswer: [/INST]")


        # we use self.chain here so that it is usable in our predict function
        self.chain = (
            {'context': history_aware_retriever | format_docs, "input": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
            | rag_prompt
            | llm 
            | StrOutputParser()
        )

    def process_row(self, row):
       return self.chain.invoke({'input': row['input'],
                                 'chat_history': row['chat_history']})
    
    def predict(self, context, data):
        """
        This is another essential function and it processes our input and sends it to the chain
        """
        results = data.apply(self.process_row, axis=1) 

        # remove .content if it is with Databricks
        results_text = results.apply(lambda x: x)
        return results_text 


# COMMAND ----------

# We can then test our wrapper first to make sure it works
import pandas as pd

sample_input = 'Tell me about how good ChatGPT is across various tasks in a Zero shot Prompting paradigm?'

mlflow_pyfunc_model = AdvancedLangchainBot()
mlflow_pyfunc_model.load_context(context='')

# TODO verify if the pandas gets done by Model Serving when deploy ie we just send json?
response = mlflow_pyfunc_model.predict(
  data=pd.DataFrame.from_records({'input': [sample_input], 'chat_history': [[]]}),
  context='')
response.iloc[0]

# COMMAND ----------

# MAGIC %md
# MAGIC If that works, we can log the model to mlflow along with some samples for evaluations \
# MAGIC NOTE - When deploying as an endpoint, the model will need to have two environment variables set: \
# MAGIC `DATABRICKS_HOST` and `DATABRICKS_TOKEN` so that it can access the Databricks models and authenticate itself

# COMMAND ----------

# Since we are using same dataset as before, we will reuse the questions from Notebook 0.4

eval_questions = [
    "Can you describe the process of Asymmetric transitivity preserving graph embedding as mentioned in reference [350]?",
    "What is the main idea behind Halting in random walk kernels as discussed in reference [351]?",
    "What is the title of the paper authored by Ledig et al. in CVPR, as mentioned in the context information?",
    'Who are the authors of the paper "Invertible conditional gans for image editing"?',
    'In which conference was the paper "Generating videos with scene dynamics" presented?',
    'What is the name of the algorithm developed by Tulyakov et al. for video generation?',
    'What is the main contribution of the paper "Unsupervised learning of visual representations using videos" by Wang and Gupta?',
    'What is the title of the paper authored by Wei et al. in CVPR, as mentioned in the context information?',
    'What is the name of the algorithm developed by Ahsan et al. for video action recognition?',
    'What is the main contribution of the paper "Learning features by watching objects move" by Pathak et al.?'
]

data = {'input': [[x] for x in eval_questions],
        'chat_history': [[[]] for x in eval_questions]}


sample_questions = pd.DataFrame(data)
sample_questions

# COMMAND ----------

def eval_pipe(inputs):
    print(inputs)
    answers = []
    for index, row in inputs.iterrows():
        #answer = {'answer': 'test'}
        #print(row)
        dict_obj = {"chat_history": row['input'], 
                    "input": row['chat_history']}
        answer = chain.invoke(dict_obj)
        
        answers.append(answer) #['answer'])
    
    return answers

# COMMAND ----------

experiment_name = 'workshop_rag_evaluations'

username = spark.sql("SELECT current_user()").first()['current_user()']
mlflow_dir = f'/Users/{username}/{experiment_name}'
mlflow.set_experiment(mlflow_dir)

mlflow.set_registry_uri('databricks-uc')

with mlflow.start_run(run_name='advanced_rag'):
  
    model = AdvancedLangchainBot()

    example_input = 'Tell me about how good ChatGPT is across various tasks in a Zero shot Prompting paradigm?'
    input_json = {'input': [example_input,example_input], 
                  'chat_history': [
                        [{'role':'user', 'content': 'Hello'},
                         {'role':'assistant', 'content': 'Hello'}],
                        None
                    ]}

    langchain_signature = mlflow.models.infer_signature(
        model_input=input_json,
        model_output=[response.iloc[0]]
    )

    mlflow_result = mlflow.pyfunc.log_model(
        python_model = model,
        extra_pip_requirements = ['langchain==0.1.16', 
                                'sqlalchemy==2.0.29', 
                                'mlflow==2.12.2', 
                                'databricks-vectorsearch==0.23', 
                                'flashrank==0.2.0'],
        artifact_path = 'langchain_pyfunc',
        signature = langchain_signature,
        input_example = input_json,
        registered_model_name = f'{db_catalog}.{db_schema}.adv_langchain_model'
    )

    # TODO Fix the evals potentially by just using the chain from above?
    eval_results = mlflow.evaluate(eval_pipe, 
                          data=sample_questions, 
                          model_type='text')

# COMMAND ----------

# MAGIC %md
# MAGIC The proper way to send requests to this endpoint is like: 
# MAGIC ```
# MAGIC {
# MAGIC "input": ["What is the main idea behind Halting in random walk kernels as discussed in reference [351]?"],
# MAGIC "chat_history": [[{"role": "user", "content": "I like beef"}]]
# MAGIC }
# MAGIC ````
# MAGIC
# MAGIC You can use Python Requests against that endpoint in order to query it.
# COMMAND ----------