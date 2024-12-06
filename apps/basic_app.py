from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import gradio as gr

fastapi_app = FastAPI()
gradio_app = gr.Interface(lambda x: "Hello, " + x + "!", "textbox", "textbox")

@fastapi_app.get("/")
def redirect_to_gradio():
    return RedirectResponse(url="/gradio")

app = gr.mount_gradio_app(fastapi_app, gradio_app, path='/gradio')