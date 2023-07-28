from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import gradio as gr
import random
import time
from langchain.chains import ConversationChain
#from langchain.llms import AzureOpenAI
from langchain import HuggingFacePipeline
import openai
import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig
from transformers import StoppingCriteria, StoppingCriteriaList


class StopOnTokens(StoppingCriteria):
  def __init__(self, stop_token_ids):
     self.stop_token_ids = stop_token_ids

  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
    for stop_id in self.stop_token_ids:
        if input_ids[0][-1] == stop_id:
            return True
    return False

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    # openai.api_type = os.getenv("OPENAI_API_TYPE")
    # openai.api_base = os.getenv("OPENAI_API_BASE")
    # openai.api_version = os.getenv("OPENAI_API_VERSION")
    # openai.api_key = os.getenv("OPENAI_API_KEY")

    # dolly works but isn't the best model
    #model_id = "databricks/dolly-v2-3b"
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    model_revision = '01622a9d125d924bd828ab6c72c995d5eda92b8e'
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model_config = AutoConfig.from_pretrained(model_id,
                                          trust_remote_code=True, # this can be needed if we reload from cache
                                          revision=model_revision
                                      )

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                               revision=model_revision,
                                               trust_remote_code=True, # this can be needed if we reload from cache
                                               config=model_config,
                                               device_map='auto',
                                               torch_dtype=torch.bfloat16 # This will only work A10G / A100 and newer GPUs
                                              )

    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_length = 4000,
        repetition_penalty=1.1, stream=True
        )

    llm = HuggingFacePipeline(pipeline=pipe)

    chain = ConversationChain(llm=llm)

    return chain

fastapi_app = FastAPI()
#gradio_app = gr.Interface(lambda x: "Hello, " + x + "!", "textbox", "textbox")
with gr.Blocks() as gradio_app:
    chain = load_chain()
    
    with gr.Row():
        gr.HTML("""<left><img src="https://www.databricks.com/en-website-assets/static/e6b356d9819308e5133bac62bb1e81ff/db-logo-stacked-white-desktop.svg" style="float: left; margin-right: 10px;" alt="Your Image"></left>
        <h2><center>Chatbot Demo</center></h2>""")

    with gr.Row():
      chatbot = gr.Chatbot()
    
    with gr.Row():
      msg = gr.Textbox()
    
    with gr.Row():
      clear = gr.Button("Clear")
    
    def respond(message, chat_history):
        
        bot_message = chain.run(message)
        #random.choice(["How are you?", "I love you", "I'm very hungry"])
        chat_history.append((message, bot_message))
        time.sleep(2)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

@fastapi_app.get("/")
def redirect_to_gradio():
    return RedirectResponse(url="/gradio")

app = gr.mount_gradio_app(fastapi_app, gradio_app, path='/gradio')