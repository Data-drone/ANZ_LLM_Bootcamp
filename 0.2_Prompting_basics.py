# Databricks notebook source
# MAGIC %md
# MAGIC # Prompting Basics
# MAGIC
# MAGIC Lets explore the basics of prompting\
# MAGIC For more details see: https://www.promptingguide.ai/ \
# MAGIC
# MAGIC This notebook was last tested with:
# MAGIC - MLR 14.3 LTS
# MAGIC
# MAGIC You will need access to databricks token based pricing models as well \
# MAGIC See: [AWS](https://docs.databricks.com/en/machine-learning/model-serving/model-serving-limits.html#region-availability) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/model-serving-limits#--region-availability)
# MAGIC
# MAGIC Note it is possible to explore prompting through the playground as well (subject to availability).
# MAGIC See: [AWS](https://docs.databricks.com/en/large-language-models/ai-playground.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/large-language-models/ai-playground)


# COMMAND ----------

# DBTITLE 1,Library Setup
# MAGIC %pip install mlflow==2.11.1 langchain==0.1.13
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# We will use the Langchain wrapper though it is just a rest call
from langchain_community.chat_models import ChatDatabricks
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

pipe = ChatDatabricks(
    target_uri = 'databricks',
    endpoint = 'databricks-mixtral-8x7b-instruct',
    temperature = 0.1
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Prompting Techniques

# COMMAND ----------

# MAGIC %md
# MAGIC # Basic Prompting
# MAGIC Getting started is easy, we can send text in.
# MAGIC Remember that different models will respond differently!
# MAGIC The same model can also responds differently when we rerun a prompt
# MAGIC (though you likely only see this with basic one line prompts)

# COMMAND ----------

prompt = "The sky is"
output = pipe([HumanMessage(content=prompt)], max_tokens=100)
str_output = print(output.content)
# COMMAND ----------

prompt = "Knock Knock"
output = pipe([HumanMessage(content=prompt)], max_tokens=100)
str_output = print(output.content)
# COMMAND ----------

prompt = """
    Knock Knock
    Who's there?
    """

output = pipe([HumanMessage(content=prompt)], max_tokens=100)
str_output = print(output.content)
# COMMAND ----------

# MAGIC %md
# MAGIC # Zero Shot Prompting
# MAGIC Zero shot is the most basic way to ask something of the model.
# MAGIC Just define your task and ask!

# COMMAND ----------

prompt = """
    Classify the text into neutral, negative or positive.
    Text: I think the vacation is okay.
    Sentiment:
"""

output = pipe([HumanMessage(content=prompt)], max_tokens=100)
str_output = print(output.content)
# COMMAND ----------

# MAGIC %md
# MAGIC You might have gotten some rubbish, we did the first time. (note models are stochastic) 
# MAGIC And that is because our prompt is problematic.
# MAGIC Different models have different "prompt templates" that they use.
# MAGIC Let's try using the official one for Llama 2

# COMMAND ----------

prompt = """<s>[INST]<<SYS>>Classify the text into neutral, negative or positive.<</SYS>>

Text: I think the vacation is okay.
Sentiment: [/INST]
"""

output = pipe([HumanMessage(content=prompt)], max_tokens=100)
str_output = print(output.content)

# COMMAND ----------

# MAGIC %md llama 2 uses the [INST] tag to highlight the whole instruction
# MAGIC <<SYS>> is the system prompt, the guide for the model on how to respond
# MAGIC In our sample the user question comes after the Text: field
# MAGIC you should get a better response after adopting this format.

# COMMAND ----------

prompt = """<s>[INST]<<SYS>>Provide an answer to the question based on the following:<</SYS>>

The minutes from the Fed's June 13-14 meeting show that while almost all officials deemed it “appropriate or acceptable” to keep rates unchanged in a 5% to 5.25% target range, some would have supported a quarter-point increase instead.

User Question: What is the interest rate in following paragraph?
Answer: [/INST]
"""

output = pipe([HumanMessage(content=prompt)], max_tokens=100)
str_output = print(output.content)
# COMMAND ----------

# MAGIC %md
# MAGIC # Few Shot Prompting
# MAGIC One way to help the a model do logic better is to provide it with samples
# MAGIC

# COMMAND ----------

prompt = """
<s>[INST]<<SYS>>
Be helpful and suggest a type of account for a customer.
<</SYS>>    

Here are some examples:
A consumer wants a savings account
A business wants a business account
A tech unicorn deserves a special VC account

Question:
What account would you recommend a small business?[/INST]
"""

output = pipe([HumanMessage(content=prompt)], max_tokens=100)
str_output = print(output.content)
# COMMAND ----------

prompt = """
<s>[INST]<<SYS>>
Be helpful and suggest a type of account for a customer.
<</SYS>>    

Here are some examples:
A consumer wants a savings account
A business wants a business account
A tech unicorn deserves a special VC account

Question:
What account would you recommend a bob the builder?[/INST]
"""


output = pipe([HumanMessage(content=prompt)], max_tokens=100)
str_output = print(output.content)

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC # Chain of Thought Prompting
# MAGIC In chain of thought prompting, we show the model how to rationalise\
# MAGIC This can help it deduce how to do a task properly

# COMMAND ----------

# DBTITLE 1,Standard Prompt - we ask straight away
user_question = """The cafeteria had 23 apples. If they used 20 to
make lunch and bought 6 more, how many apples
do they have?"""

prompt = f"""
<s>[INST]<<SYS>>
Provide helpful responses and guide the customers. 
<</SYS>>    

The follow example shows how to answer:
Question:
I went to the market and bought 10 apples. 
I gave 2 apples to the neighbor and 2 to the repairman. 
I then went and bought 5 more apples and ate 1. 

Answer:
The answer is 10

Based on the above provide the answer to the following question.
Question:
{user_question}[/INST]
"""


output = pipe([HumanMessage(content=prompt)], max_tokens=100)
str_output = print(output.content)

# COMMAND ----------

# DBTITLE 1,Chain of Thought Prompt
user_question = """The cafeteria had 23 apples. If they used 20 to
make lunch and bought 6 more, how many apples
do they have?"""

prompt = f"""
<s>[INST]<<SYS>>
Provide helpful responses and guide the customers. 
<</SYS>>    

The follow example shows how to answer:
Question:
I went to the market and bought 10 apples. 
I gave 2 apples to the neighbor and 2 to the repairman. 
I then went and bought 5 more apples and ate 1. 

Answer:
We had 10 applies. We gave away 2 each to the neighbour and the repairman.
10 - 2 - 2 = 6. We bought 5 and ate 1. 6+5-1=10 The answer is 10

Based on the above provide the answer to the following question.
Question:
{user_question}[/INST]
"""

output = pipe([HumanMessage(content=prompt)], max_tokens=100)
str_output = print(output.content)

# COMMAND ----------

# MAGIC %md
# MAGIC # System Prompts
# MAGIC Systems prompts can be used to instruct a model and also to tune it's reponse
# MAGIC You have already seen them. It is the bit inside the <<SYS>> tags.
# MAGIC They can have a big effect!
# MAGIC

# COMMAND ----------

system_prompt = 'Be helpful and suggest a type of account for a customer. try to be curteous and explain some of the key things to consider in bank account selection.'

user_question = 'I am a single homeless bloke what account should I get?'

prompt = f"""
<s>[INST]<<SYS>>
{system_prompt}
<</SYS>>    

Here are some examples:
A consumer wants a savings account
A business wants a business account
A tech unicorn deserves a special VC account

Question:
{user_question}[/INST]
"""

output = pipe([HumanMessage(content=prompt)], max_tokens=100)
str_output = print(output.content)

# COMMAND ----------

system_prompt = 'As a learned English Gentleman, be helpful and suggest a type of account for a customer. try to be curteous and explain in flowery language some of the key things to consider in bank account selection.'

user_question = 'I am a single and jobless bloke what account suggest me a type of bank account?'

prompt = f"""
<s>[INST]<<SYS>>
{system_prompt}
<</SYS>>    

Question:
{user_question}[/INST]
"""

output = pipe([HumanMessage(content=prompt)], max_tokens=100)
str_output = print(output.content)

# COMMAND ----------

# MAGIC %md
# MAGIC # Prompt Formatting
# MAGIC Prompt formats help to structure the prompts for different LLMs
# MAGIC Each LLM could have a different standard
# MAGIC
# MAGIC Stanford Alpaca structure
# MAGIC
# MAGIC ```
# MAGIC Below is an instruction that describes a task.
# MAGIC Write a response that appropriately completes the request.
# MAGIC ### Instruction:
# MAGIC {user question}
# MAGIC ### Response:
# MAGIC ```
# MAGIC
# MAGIC llama v2 format
# MAGIC ```
# MAGIC <s>[INST] <<SYS>>
# MAGIC You are a friendly assistant. Be Polite and concise.
# MAGIC <</SYS>>
# MAGIC
# MAGIC Answer the following question:
# MAGIC {user question}
# MAGIC [/INST]
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC # Retrieval Augmented Generation
# MAGIC
# MAGIC Now if I ask the bot about something left of field
# MAGIC It probably cannot answer
# MAGIC Training is expensive
# MAGIC What if we gave it an except?
# MAGIC
# MAGIC

# COMMAND ----------

system_prompt = 'As a helpful long island librarian, answer the questions provided in a succint and eloquent way.'

user_question = 'Explain to me like I am 5 LK-99'

prompt = f"""
<s>[INST]<<SYS>>
{system_prompt}
<</SYS>>    

Question:
{user_question}[/INST]
"""

output = pipe([HumanMessage(content=prompt)], max_tokens=100)
str_output = print(output.content)

# COMMAND ----------

system_prompt = 'As a helpful long island librarian, answer the questions provided in a succint and eloquent way.'

user_question = 'Explain to me like I am 5 LK-99'

prompt = f"""<s>[INST]<<SYS>>{system_prompt}<</SYS>>

Based on the below context:

LK-99 is a potential room-temperature superconductor with a gray‒black appearance.[2]: 8  It has a hexagonal structure slightly modified from lead‒apatite, by introducing small amounts of copper. A room-temperature superconductor is a material that is capable of exhibiting superconductivity at operating temperatures above 0 °C (273 K; 32 °F), that is, temperatures that can be reached and easily maintained in an everyday environment.

Provide an answer to the following:
{user_question}[/INST]
"""

output = pipe([HumanMessage(content=prompt)], max_tokens=100)
str_output = print(output.content)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Managing Prompts w MLFlow
# MAGIC As we can see logging prompts can be hard!\
# MAGIC You might have already ended up with spreadsheets of prompts and replies!\
# MAGIC Whilst MLflow support for LLMs is still an area of improvement we have made great strides already\
# MAGIC
# MAGIC TODO Update links
# MAGIC See: https://www.databricks.com/blog/2023/04/18/introducing-mlflow-23-enhanced-native-llm-support-and-new-features.html
# MAGIC See: https://www.databricks.com/blog/announcing-mlflow-24-llmops-tools-robust-model-evaluation
# MAGIC
# MAGIC We will quickly review the llm tracking API from the 2.3 addition\
# MAGIC For full descriptions see: https://mlflow.org/docs/latest/llm-tracking.html

# COMMAND ----------

import mlflow
import pandas as pd

username = spark.sql("SELECT current_user()").first()['current_user()']
mlflow_dir = f'/Users/{username}/mlflow_log_hf'
mlflow.set_experiment(mlflow_dir)

# COMMAND ----------

# DBTITLE 1,Evaluation Prompts
common_test_prompts = [
    "What is the Perth Australia famous for?",
    "Name the top 10 burgers in Perth",
    "Write me a infomercial script on why iron ore is good?",
    "What best way to make an omlet?",
    "What would you do if you had 1M dollars?"
]

testing_pandas_frame = pd.DataFrame(
    common_test_prompts, columns = ['prompt']
)

# COMMAND ----------

# starting with mlflow 2.8 we don't have to use evaluate with a pyfunc function is okay
def eval_pipe(inputs):
    answers = []
    for index, row in inputs.iterrows():
        # pipe([HumanMessage(content=prompt)], max_tokens=100)
        result = pipe( [HumanMessage(content=row.item())], max_tokens=100)
        answer = result.content
        answers.append(answer)
    
    return answers
# COMMAND ----------

# MAGIC %md
# MAGIC *NOTE* ChatDatabricks is for use with "Chat" models only. \
# MAGIC That is indicated but the word "Chat" appearing in it's description \
# MAGIC MPT-7B and MPT-30B are "Completions" models which have a different format. \
# MAGIC In Langchain you can use: ``https://api.python.langchain.com/en/stable/llms/langchain_community.llms.databricks.Databricks.html

# COMMAND ----------

model = 'databricks-mixtral-8x7b-instruct'
with mlflow.start_run(run_name=model):
    pipe = ChatDatabricks(
            target_uri = 'databricks',
            endpoint = model,
            temperature = 0.1
        )
    
    results = mlflow.evaluate(eval_pipe, 
                          data=testing_pandas_frame, 
                          model_type='text')
    

model = 'databricks-llama-2-70b-chat'
with mlflow.start_run(run_name=model):
    pipe = ChatDatabricks(
            target_uri = 'databricks',
            endpoint = model,
            temperature = 0.1
        )
    
    results = mlflow.evaluate(eval_pipe, 
                          data=testing_pandas_frame, 
                          model_type='text')

model = 'databricks-dbrx-instruct'
with mlflow.start_run(run_name=model):
    pipe = ChatDatabricks(
            target_uri = 'databricks',
            endpoint = model,
            temperature = 0.1
        )
    
    results = mlflow.evaluate(eval_pipe, 
                          data=testing_pandas_frame, 
                          model_type='text')
