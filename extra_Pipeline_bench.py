# Databricks notebook source
# MAGIC %md
# MAGIC Test out llama_index instead

# COMMAND ----------
# #%pip install llama-index

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import timeit
import torch

# COMMAND ----------

# DBTITLE 1,Setting Up HF Pipelines


model_id = "databricks/dolly-v2-3b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# COMMAND ----------

generator = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_length = 200
        )

# COMMAND ----------

generator("What is your favourite colour and why?")

# COMMAND ----------

generator_gpu = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_length = 200, device=0
        )

# COMMAND ----------

# DBTITLE 1,Test the GPU pipeline

generator_gpu("What is your favourite colour and why?")

# COMMAND ----------
repeats = 3

setup = """generator_gpu("What is your favourite colour and why?")"""

time_output = timeit.timeit(stmt=setup, number=repeats)

time_per_iteration = time_output / repeats

print(f'Time per run {time_per_iteration} seconds')

# COMMAND ----------

# MAGIC %md NOTE PyTorch 2.0 needs to be installed via init script
# MAGIC doesn't seem to accelerate much

# COMMAND ----------

# DBTITLE 1,Test PyTorch 2.0 compile

compiled_model = torch.compile(model)

compile_generator_gpu = pipeline(
        "text-generation", model=compiled_model, tokenizer=tokenizer, max_length = 200, device=0
        )


# COMMAND ----------
repeats = 3

run_func = """compile_generator_gpu("What is your favourite colour and why?")"""
setup_func = """from __main__ import compile_generator_gpu"""

time_output = timeit.timeit(stmt=run_func, setup=setup_func, number=repeats)

time_per_iteration = time_output / repeats

print(f'Time per run {time_per_iteration} seconds')

# COMMAND ----------

# MAGIC %md NOTE Hidet needs to installed via init script or the terminal
# MAGIC - See: https://pytorch.org/blog/introducing-hidet/
# MAGIC hidet has stack limit issues see: https://github.com/hidet-org/hidet/issues/190
# MAGIC
# MAGIC tested to 7.8 seconds

# COMMAND ----------

# DBTITLE 1,Testing Hidet

hidet_compiled_model = torch.compile(model, backend='hidet')

hidet_compile_generator_gpu = pipeline(
        "text-generation", model=hidet_compiled_model, tokenizer=tokenizer, max_length = 200, device=0
        )


# COMMAND ----------

repeats = 3

run_func = """hidet_compile_generator_gpu("What is your favourite colour and why?")"""
setup_func = """from __main__ import hidet_compile_generator_gpu"""

time_output = timeit.timeit(stmt=run_func, setup=setup_func, number=repeats)

time_per_iteration = time_output / repeats

print(f'Time per run {time_per_iteration} seconds')

# COMMAND ----------
