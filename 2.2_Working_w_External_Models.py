# Databricks notebook source
# MAGIC %md
# MAGIC # Working with External Models on Databricks
# MAGIC
# MAGIC Databricks gives you the flexibility to work with external model providers and APIs
# MAGIC The AI Gateway provides a proxy for access control / user management and throttling
# MAGIC 
# MAGIC **NOTE** APIs will all change late Nov / Early Dec 2023 and these calls will be deprecated

# COMMAND ----------

%pip install llama_index==0.8.54 mlflow==2.8.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

%run ./utils

# COMMAND ----------

# DBTITLE 1,Instantiating Gateway in Databricks
# To use the gateway we will instantiate it then set it to connect with databricks
import mlflow
from mlflow import gateway

gateway.set_gateway_uri(gateway_uri="databricks")

# COMMAND ----------

# DBTITLE 1,Using Embedding Endpoints
mosaic_embeddings_route_name = "mosaicml-instructor-xl-embeddings"

try:
    embedding_route = gateway.get_route(mosaic_embeddings_route_name)
except:
    # Create a route for embeddings with MosaicML
    print(f"Creating the route {mosaic_embeddings_route_name}")
    print(gateway.create_route(
        name=mosaic_embeddings_route_name,
        route_type="llm/v1/embeddings",
        model={
            "name": "instructor-xl",
            "provider": "mosaicml",
            "mosaicml_config": {
                "mosaicml_api_key": dbutils.secrets.get(scope="bootcamp_training", key="mosaic_ml_api_key")
            }
        }
    ))

# COMMAND ----------

# DBTITLE 1,Testing Embedding Endpoint
embedding_route.query(route='mosaicml-instructor-xl-embeddings', data={"text": ["test"]})

# COMMAND ----------

# DBTITLE 1,Using Chat Endpoint
mosaic_chat_route_name = "mosaicml-llama2-70b-chat"

try:
    chat_route = gateway.get_route(mosaic_chat_route_name)
except:
    # Create a route for embeddings with MosaicML
    print(f"Creating the route {mosaic_chat_route_name}")
    print(gateway.create_route(
        name=mosaic_chat_route_name,
        route_type="llm/v1/chat",
        model={
            "name": "llama2-70b-chat",
            "provider": "mosaicml",
            "mosaicml_config": {
                "mosaicml_api_key": dbutils.secrets.get(scope="bootcamp_training", key="mosaic_ml_api_key")
            }
        }
    ))

# COMMAND ----------

# DBTITLE 1,Testing Embedding Endpoint
chat_route.query(route='mosaicml-llama2-70b-chat', data={"messages": [{"role": "user", "content": "I am sad"}]})

# COMMAND ----------

# MAGIC %md
# MAGIC # Adding Orchestration with Langchain
# MAGIC That was enough for us to get started but we probably want to use it with an orchestration module
# MAGIC Langchain is pretty popular so we will use that
# MAGIC **NOTE** Tested with `langchain==0.0.338`
# COMMAND ----------


from langchain.llms import MlflowAIGateway
from langchain.chat_models import ChatMLflowAIGateway
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain

# COMMAND ----------

# DBTITLE 1,Instantiating Langchain LLM
# This creates the LLM Object for querying with langchain
# **NOTE** Different `llm`` and `chat_models` have different params check source code to find out what is needed
chat_gateway = ChatMLflowAIGateway(
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

# We can check the required schema through this undocumented method
chat_gateway.get_input_schema().schema()

# COMMAND ----------

# DBTITLE 1,Testing chat_gateway llm
from langchain.schema import HumanMessage, SystemMessage

messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French."
    ),
    HumanMessage(
        content="Translate this sentence from English to French: I love programming."
    ),
]

print(chat_gateway(messages).content, end='\n')

# COMMAND ----------

# MAGIC %md
# MAGIC Once we have the LLM configured we can add it to a simple chain and query it.
# MAGIC

# COMMAND ----------

llm_chain = LLMChain(
    llm=chat_gateway,
    prompt=ChatPromptTemplate.from_messages([
        ("system", "You are an unhelpful bot called Bossy that speaks in Korean"),
        ("human", "{user_input}")
    ])
)
# COMMAND ----------

# DBTITLE 1,Using just a string input
# The param is `user_input` because that is what we defined in the prompt template
llm_chain.run(user_input="안녕하세요!")

# COMMAND ----------

# MAGIC %md
# MAGIC # Logging chains to mlflow
# MAGIC MLflow now includes a langchain module and we can log them as models to deploy
# MAGIC **NOTE** A current gap is that chat models are not supported with the langchain wrapper
# MAGIC This requires fixing on the langchain side (19/11/2023)

# COMMAND ----------

experiment_path = f'/Users/{username}/langchain_testing'

# uc settings
catalog = 'bootcamp_ml'
schema = 'rag_chatbot'
model = 'simple_gateway_chain'
mlflow.set_registry_uri("databricks-uc")

mlflow.set_experiment(experiment_path)

class LangchainChatBot(mlflow.pyfunc.PythonModel):

    def __init__(self, route_name='mosaicml-llama2-70b-chat'):
        self.route_name = route_name

    def load_context(self, context):
        from langchain.chat_models import ChatMLflowAIGateway
        from langchain.chains import LLMChain

        chat_gateway = ChatMLflowAIGateway(
            gateway_uri="databricks",
            route=self.route_name,
            params={
                "temperature": 0.0,
                "candidate_count": 2,
                "stop": [""], # There is something weird with this param but this works for now
                "max_tokens": 256
            },
        )

        self.llm_chain = LLMChain(
            llm=chat_gateway,
            prompt=ChatPromptTemplate.from_messages([
                ("system", "You are an unhelpful bot called Bossy that speaks in Korean"),
                ("human", "{user_input}")
            ])
        )

    def predict(self, context, data):
        import os

        os.environ['DATABRICKS_HOST'] = data['db_host'][0]
        os.environ['DATABRICKS_TOKEN'] = data['db_token'][0]

        result = data.apply(self.process_row, axis=1)

        return result.tolist()

# setting up signature
user_input = "How are you?"
input_example = {"db_host": 'https://adb-984752964297111.11.azuredatabricks.net/',
                 "db_token": '<redacted>',
                 "prompt": user_input}

langchain_signature = mlflow.models.infer_signature(
    model_input=input_example,
    model_output=[llm_chain.run(user_input=user_input)]
)

llm_chain_wrapper = LangchainChatBot()


with mlflow.start_run(run_name='initial_chain'):
    mlflow.pyfunc.log_model(python_model = llm_chain_wrapper,
                            artifact_path = 'langchain_model',
                            signature = langchain_signature,
                            input_example = input_example,
                            extra_pip_requirements = ['mlflow==2.8.0',
                                                     'llama_index==0.8.54',
                                                     'langchain==0.0.338'],
                            registered_model_name = f'{catalog}.{schema}.{model}'
                            )

# COMMAND ----------