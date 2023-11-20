# Databricks notebook source
# MAGIC %pip install --upgrade --force-reinstall databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

# COMMAND ----------

vsc.list_endpoints()

# COMMAND ----------

vsc.list_indexes(name='vector-search-demo-endpoint-internal')

# COMMAND ----------

# We will create the following source Delta table.
source_catalog = "bootcamp_ml"
source_schema = "rag_chatbot"
source_table = "demo_wiki"
source_table_fullname = f"{source_catalog}.{source_schema}.{source_table}"
print(source_table_fullname)

# COMMAND ----------

source_df = spark.read.parquet("dbfs:/databricks-datasets/wikipedia-datasets/data-001/en_wikipedia/articles-only-parquet").limit(10)
display(source_df)

# COMMAND ----------

source_df.write.format("delta").option("delta.enableChangeDataFeed", "true").saveAsTable(source_table_fullname)

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {source_table_fullname}"))

# COMMAND ----------

display(spark.sql(f"SELECT count(1) FROM {source_table_fullname}"))

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Create Vector Search Endpoint

# COMMAND ----------

vector_search_endpoint_name = "vs-demo-endpoint-bootcamp"
vsc.create_endpoint(
    name=vector_search_endpoint_name,
    endpoint_type="STANDARD"
)

# COMMAND ----------

vector_search_endpoint_name = "vector-search-demo-endpoint-internal"

vsc.get_endpoint(
  name=vector_search_endpoint_name
)

# COMMAND ----------

# Vector index
vs_index = "llm_bootcamp_wiki_index"
vs_index_fullname = f"{source_catalog}.{source_schema}.{vs_index}"

# We utilize the previously created embedding model for our vector search index
embedding_model_endpoint = "ananya_mpnet_v2_embedding_endpoint"

#Generate a vector index on top of our endpoint
index = vsc.create_delta_sync_index(
  endpoint_name=vector_search_endpoint_name,
  source_table_name=source_table_fullname,
  index_name=vs_index_fullname,
  pipeline_type='TRIGGERED',
  primary_key="id",
  embedding_source_column="text",
  embedding_model_endpoint_name=embedding_model_endpoint
)
index.describe()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Similarity search

# COMMAND ----------

results = index.similarity_search(
  columns=["text"],
  # vs_index_fullname,
  query_text="Greek myths",
  num_results=3
  )

# COMMAND ----------


