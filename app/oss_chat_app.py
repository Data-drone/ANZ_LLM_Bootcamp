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

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    # openai.api_type = os.getenv("OPENAI_API_TYPE")
    # openai.api_base = os.getenv("OPENAI_API_BASE")
    # openai.api_version = os.getenv("OPENAI_API_VERSION")
    # openai.api_key = os.getenv("OPENAI_API_KEY")

    model_id = "databricks/dolly-v2-3b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto',
                                               torch_dtype=torch.bfloat16)

    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_length = 2048, 
        device=0
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