import random
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import gradio as gr

def random_response(message, history):
    return random.choice(["Yes", "No"])

test = gr.ChatInterface(random_response)

@fastapi_app.get("/")
def redirect_to_gradio():
    return RedirectResponse(url="/gradio")

fastapi_app = FastAPI()
app = gr.mount_gradio_app(fastapi_app, test, path='/gradio')