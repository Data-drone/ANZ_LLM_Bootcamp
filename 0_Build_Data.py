# Databricks notebook source
# MAGIC %md
# MAGIC This is an example for how to use Langchain

# COMMAND ----------

# We will setup a folder to store the files
username = spark.sql("SELECT current_user()").first()['current_user()']
username

# COMMAND ----------

# we will use this dbfs folder
dbutils.fs.mkdirs(f'/home/{username}/pdf_data')

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://arxiv.org/pdf/2304.09151.pdf -U me-me-me -P /dbfs/home/brian.law@databricks.com/pdf_data

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://arxiv.org/pdf/2212.10264.pdf -U me-me-me -P /dbfs/home/brian.law@databricks.com/pdf_data

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://arxiv.org/pdf/2109.07306.pdf -U me-me-me -P /dbfs/home/brian.law@databricks.com/pdf_data

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://arxiv.org/pdf/2105.00572.pdf -U me-me-me -P /dbfs/home/brian.law@databricks.com/pdf_data

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://arxiv.org/pdf/2210.14867.pdf -U me-me-me -P /dbfs/home/brian.law@databricks.com/pdf_data

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://arxiv.org/pdf/2010.11934.pdf -U me-me-me -P /dbfs/home/brian.law@databricks.com/pdf_data