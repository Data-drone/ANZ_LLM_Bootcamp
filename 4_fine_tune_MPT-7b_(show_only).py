# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Fine tune MPT-7b with deepspeed
# MAGIC
# MAGIC This is to fine-tune [MPT-7b](https://huggingface.co/mosaicml/mpt-7b) models on the [databricks-dolly-15k ](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset. The MPT models are [Apache 2.0 licensed](https://huggingface.co/mosaicml/mpt-7b). sciq is under CC BY-SA 3.0 license.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.2 GPU ML Runtime
# MAGIC - Instance: `g5.24xlarge` on AWS with 4 A10 GPUs.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine tune the model with `deepspeed`
# MAGIC
# MAGIC The fine tune logic is written in `scripts/fine_tune_deepspeed.py`. The dataset used for fine tune is [databricks-dolly-15k ](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset.
# MAGIC
# MAGIC Since MPT model does not support gradient checkpointing, we turn it off.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %pip install ninja==1.11.1
# MAGIC %pip install einops==0.6.1 flash-attn==v1.0.3.post0
# MAGIC %pip install xentropy-cuda-lib@git+https://github.com/HazyResearch/flash-attention.git@v1.0.3#subdirectory=csrc/xentropy
# MAGIC %pip install triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir_sm90#subdirectory=python
# MAGIC %pip install deepspeed==0.9.5 xformers==0.0.20 torch==2.0.1 sentencepiece==0.1.97

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Let's explore the dataset we are going to fine-tune the MPT-7B model with
from datasets import Dataset, load_dataset

display(load_dataset("databricks/databricks-dolly-15k")["train"].to_pandas())

# COMMAND ----------

# MAGIC %sh
# MAGIC deepspeed --num_gpus=4 \
# MAGIC         scripts/fine_tune_deepspeed.py \
# MAGIC         --per-device-train-batch-size=1 \
# MAGIC         --per-device-eval-batch-size=1 \
# MAGIC         --epochs=1 \
# MAGIC         --max-steps=1 \
# MAGIC         --no-gradient-checkpointing \
# MAGIC         --dbfs-output-dir /dbfs/mpt-7b/

# COMMAND ----------

# MAGIC %md
# MAGIC Model checkpoint is saved at `/local_disk0/output`. We save the final model to DBFS location `/dbfs/mpt-7b`.

# COMMAND ----------

dbutils.fs.ls("dbfs:/mpt-7b/")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save the model to mlflow

# COMMAND ----------

import pandas as pd
import numpy as np
import transformers
import mlflow
import torch
import accelerate

class MPT(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
          context.artifacts['repository'], padding_side="left")

        config = transformers.AutoConfig.from_pretrained(
            context.artifacts['repository'], 
            trust_remote_code=True
        )
        
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            context.artifacts['repository'], 
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True)
        self.model.to(device='cuda')
        
        self.model.eval()

    def _build_prompt(self, instruction):
        """
        This method generates the prompt for the model.
        """
        INSTRUCTION_KEY = "### Instruction:"
        RESPONSE_KEY = "### Response:"
        INTRO_BLURB = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
        )

        return f"""{INTRO_BLURB}
        {INSTRUCTION_KEY}
        {instruction}
        {RESPONSE_KEY}
        """

    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """
        generated_text = []
        for index, row in model_input.iterrows():
          prompt = row["prompt"]
          temperature = model_input.get("temperature", [1.0])[0]
          max_new_tokens = model_input.get("max_new_tokens", [100])[0]
          full_prompt = self._build_prompt(prompt)
          encoded_input = self.tokenizer.encode(full_prompt, return_tensors="pt").to('cuda')
          output = self.model.generate(encoded_input, do_sample=True, temperature=temperature, max_new_tokens=max_new_tokens)
          prompt_length = len(encoded_input[0])
          generated_text.append(self.tokenizer.batch_decode(output[:,prompt_length:], skip_special_tokens=True))
        return pd.Series(generated_text)

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

# Define input and output schema
input_schema = Schema([
    ColSpec(DataType.string, "prompt"), 
    ColSpec(DataType.double, "temperature"), 
    ColSpec(DataType.long, "max_tokens")])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example=pd.DataFrame({
            "prompt":["what is ML?"], 
            "temperature": [0.5],
            "max_tokens": [100]})

# Log the model with its details such as artifacts, pip requirements and input example
# This may take about 12 minutes to complete
with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=MPT(),
        artifacts={'repository' : "/dbfs/mpt-7b"},
        pip_requirements=[f"torch=={torch.__version__}", 
                          f"transformers=={transformers.__version__}", 
                          f"accelerate=={accelerate.__version__}", "einops", "sentencepiece"],
        input_example=input_example,
        signature=signature
    )

# COMMAND ----------

import mlflow
import pandas as pd

logged_model = "runs:/"+"97e80f3debd54eefb6274062cac34d2e"+"/model"

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
input_example = pd.DataFrame({"prompt":["what is ML?", "Name 10 colors."], "temperature": [0.5, 0.2],"max_tokens": [100, 200]})
mpt7b_outputs = loaded_model.predict(input_example)

# COMMAND ----------

mpt7b_outputs
