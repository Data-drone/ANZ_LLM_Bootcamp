# ANZ LLM Workshop Repo 🧠

This repo is for the ANZ LLM Workshop Series.

## Setup

This series of notebooks have been developed and tested on Databricks ML Runtime 14.3

They are designed to be run alongside Databricks Provisioned Throughput Foundation Model APIs
See: [Databricks AWS Docu](https://docs.databricks.com/en/machine-learning/foundation-models/deploy-prov-throughput-foundation-model-apis.html)

You can deploy a model endpoint with a chat model like DBRX / Mistral / Llama 2.
See: [Creating Model Endpoints](https://docs.databricks.com/en/machine-learning/model-serving/create-foundation-model-endpoints.html)

## Overview of Materials

`0.1_lab_setup(instructor_only)` Notebook is to be run by instructor. This downloads HuggingFace models and some sample documents for us to work with. The workspace will need to have access to `*.huggingface.co` for the models and wikipedia and some other websites for pdf data.

`0.x_` series notebooks go through LLM basics and setup a basic RAG app powered by HuggingFace open source models. \
`1.x_` series notebooks cover go into more detail about constructing and tuning RAG Architectures.

## Extras

It is possible to run applications on the driver node in Databricks. The `app` folder contains examples of how to do this. 

## Recordings

The 2023 version of these materials were presented in a webinar see:
[LLM Basics](https://vimeo.com/857791352) 0.x_ materials
[LLM Advanced](https://vimeo.com/862303088) 1.x_ materials


## Further Reading and Learnings
- We have a great catalog of LLM related talks at the Data and AI Summit [link here](https://www.databricks.com/dataaisummit/llm/)
- For a set of great examples on fine-tuning these LLMs, we recommend looking at [the Databricks ML examples repo](https://github.com/databricks/databricks-ml-examples/tree/master)
