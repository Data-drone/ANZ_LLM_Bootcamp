# ANZ LLM Bootcamp Repo 🧠

This repo is for the ANZ LLM Bootcamp Series.

## Setup

This series of notebooks have been developed and tested on ML Runtime 14.2

If running with GPUs on compute then for optimal performance, we recommend running on ```g5.4xlarge``` instances on AWS.
Otherwise if using an External Model via Gateway or a Databricks serving model then it is possible to run on smaller 4/8 core cpu instances.

On Databricks, you can select the Single Node instance type and select the runtime and infrastructure as required based on how the LLM and embedding models will be provided. As a last resort, it is possible to use a cpu only model for that we recommend ```i3.4xlarge``` or ```m5.4xlarge``` nodes at a minimum. The prompts for RAG may still be too long for a resonable compute time with a CPU model. 

## Overview of Materials

`0.1_lab_setup(instructor_only)` Notebook is to be run by instructor. This downloads HuggingFace models and some sample documents for us to work with. The workspace will need to have access to `*.huggingface.co` for the models and wikipedia and some other websites for pdf data. Note that it doesn't setup mosaic keys / gateway access or serverless endpoints. The code for this is in the 2.x notebooks

`0.x_` series notebooks go through LLM basics and setup a basic RAG app powered by HuggingFace open source models. 
`1.x_` series notebooks cover the more advanced topics.
`2.x_` series notebooks cover deployment and integration of External Models / Deploying Endpoints etc.

## Extras

It is possible to run applications on the driver node in Databricks. The `app` folder contains examples of how to do this. 

## Recordings

To view these materials presented in a webinar see:

[LLM Basics](https://vimeo.com/857791352) 0.x_ materials
[LLM Advanced](https://vimeo.com/862303088) 1.x_ materials


## Further Reading and Learnings
- We have a great catalog of LLM related talks at the Data and AI Summit [link here](https://www.databricks.com/dataaisummit/llm/)
- For a set of great examples on fine-tuning these LLMs, we recommend looking at [the Databricks ML examples repo](https://github.com/databricks/databricks-ml-examples/tree/master)
