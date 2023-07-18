# Databricks notebook source
# MAGIC %md
# MAGIC # Prompting Basics

# MAGIC Lets explore the basics of prompting\
# MAGIC For more details see: https://www.promptingguide.ai/

# COMMAND ----------

run_mode = 'cpu' # or gpu
# COMMAND ----------

# DBTITLE 1,Setup
%run utils

# COMMAND ----------

pipe = load_model(run_mode, dbfs_tmp_cache)
# COMMAND ----------

# MAGIC %md
# MAGIC # Building out prompts

# COMMAND ----------

# MAGIC %md
# MAGIC # Basic Prompts
# MAGIC Getting started is easy, we can send text in.
# MAGIC Remember that different models will respond differently\
# COMMAND ----------

# MAGIC %md
# MAGIC # Basic Prompting
# MAGIC

# COMMAND ----------

prompt = "The sky is"
output = pipe(prompt, max_new_tokens=100)
str_output = string_printer(output, run_mode)
str_output
# COMMAND ----------

prompt = "The red sky is"
output = pipe(prompt, max_new_tokens=100)
str_output = string_printer(output, run_mode)
str_output

# COMMAND ----------

prompt = "Knock Knock"
output = pipe(prompt, max_new_tokens=100)
str_output = string_printer(output, run_mode)
str_output

# COMMAND ----------

prompt = """
    Knock Knock
    Who's there?
    """

output = pipe(prompt, max_new_tokens=100)
str_output = string_printer(output, run_mode)
str_output

# COMMAND ----------

# MAGIC %md
# MAGIC # Zero Shot Prompting
# MAGIC

# COMMAND ----------

prompt = """
    Classify the text into neutral, negative or positive.
    Text: I think the vacation is okay.
    Sentiment:
"""

output = pipe(prompt, max_new_tokens=100)
str_output = string_printer(output, run_mode)
str_output

# COMMAND ----------

prompt = """
    Classify the text into neutral, negative or positive.
    Text: I think the vacation sucks.
    Sentiment:
"""

output = pipe(prompt, max_new_tokens=100)
str_output = string_printer(output, run_mode)
str_output

# COMMAND ----------

prompt = """
    What is the interest rate it following paragraph?
    Text: The minutes from the Fed's June 13-14 meeting show that while almost all officials deemed it “appropriate or acceptable” to keep rates unchanged in a 5% to 5.25% target range, some would have supported a quarter-point increase instead.
    Interest Rate:
"""

output = pipe(prompt, max_new_tokens=100)
str_output = string_printer(output, run_mode)
str_output

# COMMAND ----------

# MAGIC %md
# MAGIC # Few Shot Prompting
# MAGIC

# COMMAND ----------

prompt = """
    A consumer wants a savings account
    A business wants a business account
    A tech unicorn deserves a special VC account
    What account would you recommend a small business?
"""

output = pipe(prompt, max_new_tokens=100)
str_output = string_printer(output, run_mode)
str_output

# COMMAND ----------

prompt = """
    A consumer wants a savings account
    A business wants a business account
    A tech unicorn deserves a special VC account

    Question: What account would you recommend a consumer?
"""

output = pipe(prompt, max_new_tokens=100)
str_output = string_printer(output, run_mode)
str_output

# COMMAND ----------


# MAGIC %md
# MAGIC # Chain of Thought Prompting
# MAGIC

# COMMAND ----------

prompt = """
    I went to the market and bought 10 apples. 
    I gave 2 apples to the neighbor and 2 to the repairman. 
    I then went and bought 5 more apples and ate 1. 
    How many apples did I remain with?
"""

output = pipe(prompt, max_new_tokens=100)
str_output = string_printer(output, run_mode)
str_output


# COMMAND ----------

prompt = """
    I went to the market and bought 10 apples. 
    I gave 2 apples to the neighbor and 2 to the repairman. 
    I then went and bought 5 more apples and ate 1. 
    How many apples did I remain with?
    Think through it step by step:

"""

output = pipe(prompt, max_new_tokens=100)
str_output = string_printer(output, run_mode)
str_output

# COMMAND ----------

# MAGIC %md
# MAGIC # Constructing a system prompt
# MAGIC

# COMMAND ----------

user_question = """
    I have 20 cars.
    I crashed 1 and sold 3 others.
    I then went and bought 1 back.
    How many cars do I have?
"""

prompt = f"""
    
    Question:
    I went to the market and bought 10 apples. 
    I gave 2 apples to the neighbor and 2 to the repairman. 
    I then went and bought 5 more apples and ate 1. 
    How many apples did I remain with?

    Answer:
    I had 10 apples -2 for neighbor and -2 for repair man
    10 - 2 - 2 = 6
    I bought 5
    6 + 5 = 11
    and ate 1
    11 - 1 = 10
    So the answer is 10
    
    Based on the above answer the following question

    Question:
    {user_question}

    Think through it step by step:
"""

output = pipe(prompt, max_new_tokens=100)
str_output = string_printer(output, run_mode)
str_output


# COMMAND ----------

# MAGIC %md
# MAGIC # Retrieval Augmented Generation
# MAGIC
# MAGIC Now if I ask the bot about something left of field
# MAGIC It probably cannot answer
# MAGIC Training is expensive
# MAGIC What if we gave it an except?
# MAGIC


# COMMAND ----------

prompt = """
    What happens to GNNs as you add layers?
"""

output = pipe(prompt, max_new_tokens=100)
str_output = string_printer(output, run_mode)
str_output

# COMMAND ----------

user_question = 'What happens to GNNs as you add layers?'

prompt = f"""
    page-context:

    Graph neural networks (GNNs), a type of neural network that can learn from graphstructured data and learn the representation of nodes through aggregating neighborhood information, have shown superior performance in various downstream
tasks. However, it is known that the performance of GNNs degrades gradually as
the number of layers increases. 

    Based on the page context, answer the following question.
    Question: {user_question}
"""

output = pipe(prompt, max_new_tokens=100)
str_output = string_printer(output, run_mode)
str_output

# COMMAND ----------

# MAGIC %md 
# MAGIC **NOTE** TODO move out
# MAGIC # Managing Prompts w MLFlow
# MAGIC As we can see logging prompts can be hard!\
# MAGIC You might have already ended up with spreadsheets of prompts and replies!\
# MAGIC Whilst MLflow support for LLMs is still an area of improvement we have made great strides already\
# MAGIC 
# MAGIC See: https://www.databricks.com/blog/2023/04/18/introducing-mlflow-23-enhanced-native-llm-support-and-new-features.html
# MAGIC See: https://www.databricks.com/blog/announcing-mlflow-24-llmops-tools-robust-model-evaluation
# MAGIC
# MAGIC We will quickly review the llm tracking API from the 2.3 addition\
# MAGIC For full descriptions see: https://mlflow.org/docs/latest/llm-tracking.html
# COMMAND ----------

import mlflow

mlflow_dir = f'/Users/{username}/mlflow_log_hf'
mlflow.set_experiment(mlflow_dir)

# COMMAND ----------

# DBTITLE 1,Setup Configs
user_inputs = [
  "How can I make a coffee?",
  "How can I book a restaurant?",
  "How can I make idle chit chat when I don't know a person?"
]
prompts = []
model_outputs = []

# COMMAND ----------



# COMMAND ----------

with mlflow.start_run(run_name='openassist model'):
    
  for user_input in user_inputs:
    prompt = f"""
            You are an AI assistant that helps people find information and responds in rhyme. 
            If the user asks you a question you don't know the answer to, say so.

            {user_input}
            """

    raw_output = pipe(prompt, max_length=200, repetition_penalty=1.2)
    text_output = string_printer(raw_output, run_mode)

    mlflow.llm.log_predictions(inputs=user_input, outputs=text_output, prompts=prompt)

