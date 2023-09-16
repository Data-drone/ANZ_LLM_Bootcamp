# ANZ LLM Bootcamp Repo 🧠

This repo is for the ANZ LLM Bootcamp Series.

## Setup

This series of notebooks have been developed and tested on ML Runtime 13.3 LTS

For optimal performance, we recommend running on ```g5.4xlarge``` instances on AWS.

On Databricks, you can select the Single Node instance type and select the runtime and infrastructure above. If you run into capacity issues, we have supplemented the notebooks with CPU editions. In this case, feel free to use ```i3.4xlarge``` or ```m5.4xlarge``` nodes. Please note that this will not have optimal performance. 

## Overview of Materials

`0_lab_setup` Notebook is to be run by instructor. This downloads HuggingFace models and some sample documents for us to work with. The workspace will need to have access to `*.huggingface.co` for the models and wikipedia and some other websites for pdf data.

`0.x_` series notebooks go through LLM basics and setup a basic RAG app powered by HuggingFace open source models. 
`1.x_` series notebooks cover the more advanced topics. At the moment they have been setup mostly to leverage Azure OpenAI

The other notebooks are works in progress. 

## Extras

It is possible to run applications on the driver node in Databricks. The `app` folder contains examples of how to do this. 

## Recordings

To view these materials presented in a webinar see:

[LLM Basics](https://vimeo.com/857791352) 0.x_ materials
[LLM Advanced](https://vimeo.com/862303088) 1.x_ materials


## Further Reading and Learnings
- We have a great catalog of LLM related talks at the Data and AI Summit [link here](https://www.databricks.com/dataaisummit/llm/)
- For a set of great examples on fine-tuning these LLMs, we recommend looking at [the Databricks ML examples repo](https://github.com/databricks/databricks-ml-examples/tree/master)
