# Databricks notebook source
# MAGIC %md
# MAGIC Test out loading large models and model performance

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import timeit
import torch
import os

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup Process

# COMMAND ----------

# DBTITLE 1,Model Selection

# setup a dbfs cache folder
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
path = f'/home/{username}/cache/hf'
dbutils.fs.mkdirs(path)

os.environ['TRANSFORMERS_CACHE'] = f"/dbfs{path}"


# we will use 7B as the baseline
model_id = "databricks/dolly-v2-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# COMMAND ----------

# MAGIC %md
# MAGIC # Experiments

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline

# COMMAND ----------

model = AutoModelForCausalLM.from_pretrained(model_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ### CPU Generation

# COMMAND ----------

generator = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_length = 200
        )

# COMMAND ----------

generator("What is your favourite colour and why?")

# COMMAND ----------

# MAGIC %md
# MAGIC ### GPU Generation
# MAGIC 11551 MiB

# COMMAND ----------

# We can move the model into GPU either via the model object or via the pipeline
# We do it in the pipeline in this case

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

# DBTITLE 1,Check GPU Consumption

# MAGIC %sh
# MAGIC nvidia-smi

# COMMAND ----------

# MAGIC %md 
# MAGIC ## PyTorch 2.0 Compule
# MAGIC NOTE PyTorch 2.0 needs to be installed via init script
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

# MAGIC %md 
# MAGIC ## Hidet Compiler
# MAGIC NOTE Hidet needs to installed via init script or the terminal
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

# MAGIC %md 
# MAGIC ## Quantization

# MAGIC fp16 - 9.4 secs
# MAGIC 8 bit - 23 secs???

# COMMAND ----------

# MAGIC %md 
# MAGIC ### FP16
# MAGIC 6309 GB

# COMMAND ----------

fp16_model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.bfloat16)

fp16_generator = pipeline(
        "text-generation", model=fp16_model, tokenizer=tokenizer, max_length = 200, 
        )

# COMMAND ----------

repeats = 3

run_func = """fp16_generator("What is your favourite colour and why?")"""
setup_func = """from __main__ import fp16_generator"""

time_output = timeit.timeit(stmt=run_func, setup=setup_func, number=repeats)

time_per_iteration = time_output / repeats

print(f'Time per run {time_per_iteration} seconds')

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 8 bit
# MAGIC 4241 GB
# MAGIC NOTE this requires bitsandbytes installed

# COMMAND ----------

quant_model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', load_in_8bit=True)

quantized_generator = pipeline(
        "text-generation", model=quant_model, tokenizer=tokenizer, max_length = 200, device_map='auto', 
        )

# COMMAND ----------

repeats = 3

run_func = """quantized_generator("What is your favourite colour and why?")"""
setup_func = """from __main__ import quantized_generator"""

time_output = timeit.timeit(stmt=run_func, setup=setup_func, number=repeats)

time_per_iteration = time_output / repeats

print(f'Time per run {time_per_iteration} seconds')

# COMMAND ----------

# MAGIC %md # Token Length Experiments
# MAGIC generation time with 2048 is 57 secs

# COMMAND ----------

# we can use auto when GPU is active - note it won't do model parallel just GPU dist

# If we put the model_id in the pipeline rather than an instiated model we default to the 
# we need to set trust_remote_code in order to reuse the cache from last time
# setting set by the authors. For this dollyv2 7b see: https://huggingface.co/databricks/dolly-v2-7b/blob/97611f20f95e1d8c1e914b85da55cc3937c31192/instruct_pipeline.py#L60
long_generator = pipeline(
        "text-generation", model=model_id, tokenizer=tokenizer, 
        device_map='auto', trust_remote_code=True, max_length = 2048*2
        )

# COMMAND ----------

long_generator("What is your favourite colour and why?")

# COMMAND ----------

# MAGIC %md We need to manually create the model to get the full flexibility we want

# COMMAND ----------

# Lets manually create the model instead and override the token settings

# default max_new_tokens is 256
# if we set the max_new_tokens higher it uses more vram 512 exceeds limits for 24GB on default FP
# we set to bfloat16 to try to alleviate this
custom_long_model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.bfloat16)

custom_long_generator = pipeline(
        "text-generation", model=custom_long_model, tokenizer=tokenizer, 
        trust_remote_code=True, max_new_tokens = 256
        )

# COMMAND ----------

custom_long_generator("What is your favourite colour and why?")


# COMMAND ----------

# MAGIC %md When we go beyond the limits of the model we start getting weird results back out.
# MAGIC Hence we need bigger models trained on larger context lengths

# COMMAND ----------
