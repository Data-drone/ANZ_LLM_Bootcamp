# Databricks notebook source
# MAGIC %md
# MAGIC Utils notebook\
# MAGIC With Databricks we can create a utils notebook that is then used in other notebooks via the `%run` magic\
# MAGIC We will make some of the code from hugging_face_basics available for general use.

# COMMAND ----------

# setup env
# TODO - adjust and use bootcamp ones later
import os
import requests
import pprint

username = spark.sql("SELECT current_user()").first()['current_user()']
os.environ['USERNAME'] = username

tmp_user_folder = f'/tmp/{username}'
dbutils.fs.mkdirs(tmp_user_folder)
dbfs_tmp_dir = f'/dbfs{tmp_user_folder}'
os.environ['PROJ_TMP_DIR'] = dbfs_tmp_dir

# setting up transformers cache
cache_dir = f'{tmp_user_folder}/.cache'
dbutils.fs.mkdirs(cache_dir)
dbfs_tmp_cache = f'/dbfs{cache_dir}'
os.environ['TRANSFORMERS_CACHE'] = dbfs_tmp_cache

# setup source file_docs
source_doc_folder = f'/home/{username}/pdf_data'
dbfs_source_docs = '/dbfs' + source_doc_folder

# setup vectorstore path
vector_store_path = f'/home/{username}/vectorstore_persistence/db'
linux_vector_store_directory = f'/dbfs{vector_store_path}'

# is that right env var?
os.environ['PERSIST_DIR'] = linux_vector_store_directory

# gateway routes
mosaic_chat_route_name = "mosaicml-llama2-70b-chat"

# COMMAND ----------

bootcamp_dbfs_model_folder = '/dbfs/bootcamp_data/hf_cache/downloads'

# COMMAND ----------

# MAGIC %md 
# MAGIC Here we will make the model loaders into functions that receive the run_mode var

# COMMAND ----------

class QueryEndpoint:
    """
    A class designed to be interchangeable with pipe but calls databricks model serving instead
    """
   
    def __init__(self, uri:str, token:str):
      
      self.uri = uri
      self.header = {"Context-Type": "text/json", "Authorization": f"Bearer {token}"}
      print(uri)


    def __call__(self, prompt: list[str], **kwargs):
       
      dataset = {'inputs': {'prompt': prompt},
                  'params': kwargs}

      response = requests.post(headers=self.header, url=self.uri, json=dataset)

      return response.json()


class QueryGateway:
    """
    A class for querying the gateway connection
    """

    def __init__(self, route:str):
        self.route = route

    def __call__(self, prompt: str):

        format = {'messages': [{"role": "user", "content": prompt}]}

        return gateway.query(route=self.route, 
                                     data=format)

# COMMAND ----------

# This is to setup a connection for our model

from mlflow import gateway

gateway.set_gateway_uri(gateway_uri="databricks")

def load_model(run_mode: str, dbfs_cache_dir: str, serving_uri :str='llama_2_13b'):
    """
    run_mode (str) - can be gpu or cpu
    """

    from transformers import pipeline, AutoConfig
    import torch

    assert run_mode in ['cpu', 'gpu', 'serving', 'gateway'], f'run_mode must be cpu/gpu/serving/gateway not {run_mode}'

    if run_mode == 'cpu':

      from ctransformers import AutoModelForCausalLM, AutoTokenizer
      model_id = 'llama_2_cpu/llama-2-7b-chat.Q4_K_M.gguf'
      model = AutoModelForCausalLM.from_pretrained(f'{bootcamp_dbfs_model_folder}/{model_id}',
                                              hf=True, local_files_only=True)
      tokenizer = AutoTokenizer.from_pretrained(model)

      pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer , max_new_tokens=128 
      )

      return pipe

    elif run_mode == 'gpu':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        #model_id = 'VMware/open-llama-7b-v2-open-instruct'
        #model_revision = 'b8fbe09571a71603ab517fe897a1281005060b62'

        cached_model = f'{bootcamp_dbfs_model_folder}/llama_2_gpu'
        tokenizer = AutoTokenizer.from_pretrained(cached_model, cache_dir=dbfs_cache_dir)
        
        model_config = AutoConfig.from_pretrained(cached_model)
        model = AutoModelForCausalLM.from_pretrained(cached_model,
                                               config=model_config,
                                               device_map='auto',
                                               torch_dtype=torch.bfloat16, # This will only work A10G / A100 and newer GPUs
                                               cache_dir=dbfs_cache_dir
                                              )
    
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128 
        )

        return pipe
    
    elif run_mode == 'serving':
        
        browser_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
        db_host = f"https://{browser_host}"
        model_uri = f"{db_host}/serving-endpoints/{serving_uri}/invocations"
        db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

        test_pipe = QueryEndpoint(model_uri, db_token)

        return test_pipe
    
    elif run_mode == 'gateway':

        gateway_pipe = QueryGateway(mosaic_chat_route_name)

        return gateway_pipe


# COMMAND ----------


def string_printer(out_obj, run_mode):
  """
  Short convenience function because the output formats change between CPU and GPU
  """
  try:
    if run_mode in ['cpu', 'gpu']:

      return pprint.pprint(out_obj[0]['generated_text'], indent=2)
  
    elif run_mode == 'serving':
    
      return  pprint.pprint(out_obj['predictions'], indent=2)
  
  except KeyError:
    pprint.pprint(out_obj, indent=2)

# COMMAND ----------

# currently the langchain integration is broken
from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.prompts.chat import ChatPromptTemplate
from langchain.llms.base import LLM
from langchain.schema.messages import HumanMessage
import requests

# TODO setup generation config properly?
class ServingEndpointLLM(LLM):
    endpoint_url: str
    token: str
    temperature: float = 0.1
    max_length: int = 256

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
            #raise ValueError("stop kwargs are not permitted.")
            pass

        header = {"Context-Type": "text/json", "Authorization": f"Bearer {self.token}"}

        if type(prompt) is str:
            dataset = {'inputs': {'prompt': [prompt]},
                  'params': {**{'max_tokens': self.max_length}, **kwargs}}
        elif type(prompt) is ChatPromptTemplate:
            text_prompt = prompt.format()
            dataset = {'inputs': {'prompt': [text_prompt]},
                  'params': {**{'max_tokens': self.max_length}, **kwargs}} 
        #print(dataset)
        try:
            response = requests.post(headers=header, url=self.endpoint_url, json=dataset)

            try:
        
                return response.json()['predictions'][0]['candidates'][0]['text']
            
            except KeyError:
                print(response)
                return str(response.json())

        
        except TypeError:
          print(dataset)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"endpoint_url": self.endpoint_url}  

# COMMAND ----------

# Embedding wrapper

from typing import Any, Dict, List, Mapping, Optional, Tuple

import requests

from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.schema.embeddings import Embeddings
from langchain.utils import get_from_dict_or_env

browser_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
db_host = f"https://{browser_host}"

embedding_endpoint_name = 'brian_embedding_endpoint'

class ModelServingEndpointEmbeddings(BaseModel, Embeddings):
    """Databricks Model Serving embedding service.

    To use, you should have the
    environment variable ``DB_API_TOKEN`` set with your API token, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain.llms import MosaicMLInstructorEmbeddings
            endpoint_url = (
                "https://dbc-d0c4038e-c5a9.cloud.databricks.com/serving-endpoints/brian_embedding_endpoint/invocations"
            )
            mosaic_llm = MosaicMLInstructorEmbeddings(
                endpoint_url=endpoint_url,
                db_api_token="my-api-key"
            )
    """

    endpoint_url: str = (
        f"{db_host}/serving-endpoints/{embedding_endpoint_name}/invocations"
    )
    """Endpoint URL to use."""
    embed_instruction: str = "Represent the document for retrieval: "
    """Instruction used to embed documents."""
    query_instruction: str = (
        "Represent the question for retrieving supporting documents: "
    )
    """Instruction used to embed the query."""
    retry_sleep: float = 1.0
    """How long to try sleeping for if a rate limit is encountered"""

    db_api_token: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        db_api_token = get_from_dict_or_env(
            values, "db_api_token", "DB_API_TOKEN"
        )
        values["db_api_token"] = db_api_token
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"endpoint_url": self.endpoint_url}

    def _embed(
        self, input: List[Tuple[str, str]], is_retry: bool = False
    ) -> List[List[float]]:
        #payload = {"input_strings": input}
        payload = {
            "dataframe_split": {
                "data": [
                    [
                        input
                    ]
                ]
            }
        }

        # HTTP headers for authorization
        headers = {
            "Authorization": f"Bearer {self.db_api_token}",
            "Content-Type": "application/json",
        }

        # send request
        try:
            response = requests.post(self.endpoint_url, headers=headers, json=payload)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        try:
            if response.status_code == 429:
                if not is_retry:
                    import time

                    time.sleep(self.retry_sleep)

                    return self._embed(input, is_retry=True)

                raise ValueError(
                    f"Error raised by inference API: rate limit exceeded.\nResponse: "
                    f"{response.text}"
                )

            parsed_response = response.json()

            # The inference API has changed a couple of times, so we add some handling
            # to be robust to multiple response formats.
            # if isinstance(parsed_response, dict):
            #     output_keys = ["data", "output", "outputs"]
            #     for key in output_keys:
            #         if key in parsed_response:
            #             output_item = parsed_response[key]
            #             break
            #     else:
            #         raise ValueError(
            #             f"No key data or output in response: {parsed_response}"
            #         )

            #     if isinstance(output_item, list) and isinstance(output_item[0], list):
            #         embeddings = output_item
            #     else:
            #         embeddings = [output_item]
            # else:
            #     raise ValueError(f"Unexpected response type: {parsed_response}")

        except requests.exceptions.JSONDecodeError as e:
            raise ValueError(
                f"Error raised by inference API: {e}.\nResponse: {response.text}"
            )

        return parsed_response

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using a MosaicML deployed instructor embedding model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        #instruction_pairs = [(self.embed_instruction, text) for text in texts]
        #embeddings = self._embed(instruction_pairs)
        try:
            embeddings = [self._embed(x)['predictions'][0] for x in texts]
        except KeyError:
            print([self._embed(x) for x in texts])

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using a Databricks Model Serving embedding model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        #instruction_pair = (self.query_instruction, text)
        #embedding = self._embed([instruction_pair])[0]
        embedding = self._embed(text)
        return embedding['predictions'][0]