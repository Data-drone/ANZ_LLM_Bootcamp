# Databricks notebook source
# MAGIC %md
# MAGIC This notebook will setup the datasets to use for exploring LLM RAGs

# COMMAND ----------

import os

# We will setup a folder to store the files
username = spark.sql("SELECT current_user()").first()['current_user()']
username

os.environ['DATASTASH_FOLDER'] = dbfs_source_docs

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://arxiv.org/pdf/2304.09151.pdf -U me-me-me -P /dbfs$DATASTASH_FOLDER

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://arxiv.org/pdf/2212.10264.pdf -U me-me-me -P /dbfs$DATASTASH_FOLDER

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://arxiv.org/pdf/2109.07306.pdf -U me-me-me -P /dbfs$DATASTASH_FOLDER

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://arxiv.org/pdf/2105.00572.pdf -U me-me-me -P /dbfs$DATASTASH_FOLDER

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://arxiv.org/pdf/2210.14867.pdf -U me-me-me -P /dbfs$DATASTASH_FOLDER

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://arxiv.org/pdf/2010.11934.pdf -U me-me-me -P /dbfs$DATASTASH_FOLDER

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://arxiv.org/pdf/1706.03762.pdf -U me-me-me -P /dbfs$DATASTASH_FOLDER

# COMMAND ----------


