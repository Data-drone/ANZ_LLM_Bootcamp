# Databricks notebook source
# MAGIC %pip install --upgrade --force-reinstall databricks-vectorsearch langchain==0.1.10 sqlalchemy==2.0.27 pypdf==4.1.0 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Setting Up Databricks Vector Search

# COMMAND ----------

# We will create the following source Delta table.
source_catalog = "bootcamp_ml"
source_schema = "rag_chatbot"
source_volume = "datasets"
source_table = "arxiv_parse"
vs_endpoint = "bootcamp_vs_endpoint"
embedding_endpoint_name = "databricks-bge-large-en"

# COMMAND ----------

# TODO
# 1) Pull the table in SQL
# 2) Compare to the parse and source
# 3) Run some searches

# COMMAND ----------

# We have loaded the chunks into a delta table
raw_table = spark.sql(f"select * from {source_catalog}.{source_schema}.{source_table}")

display(raw_table)

# COMMAND ----------

# given these chunks Databricks Vector Search will manage the vector sync and we can focus on making sure that the chunking is working
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
# vs_endpoint
vsc.get_endpoint(
  name=vs_endpoint
)

vs_index = f"{source_table}_bge_index"
vs_index_fullname = f"{source_catalog}.{source_schema}.{vs_index}"
index = vsc.get_index(endpoint_name=vs_endpoint,index_name=vs_index_fullname)

# COMMAND ----------

my_query = "Tell me about tuning LLMs"

results = index.similarity_search(
  columns=["page_content"],
  # vs_index_fullname,
  query_text = my_query,
  num_results = 3
  )

# COMMAND ----------

# Explore the results
results

# COMMAND ----------

# pulling the top result
print(results['result']['data_array'][0][0])