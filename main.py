# Databricks notebook source
# MAGIC %md
# MAGIC We will use this notebook to call the classes and functions

# COMMAND ----------

import pandas as pd
import logging
import yaml

# COMMAND ----------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# COMMAND ----------

with open('project_config.yml', 'r') as file:
    config = yaml.safe_load(file)

print(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

# MAGIC %run  "/Users/chris.wroe@fabledata.com/marvelous-databricks-course-Chris-W-codes/Pre_processing"

# COMMAND ----------

preprocessor = PreProcess()
logger.info("PreProcessor initialized.")

# COMMAND ----------

df = preprocessor.load_data(config['dataset_location'])
logger.info("Data Loaded")

# COMMAND ----------

df2 = preprocessor.append_headers(df)
df3 = preprocessor.filter_df(df2)
train_df,test_df =  preprocessor.test_train_split(df3,config['train_pc'],config['test_pc'])
logger.info("Data Split with train at "+str(config['train_pc'])+" and test at "+str(config['test_pc']))
test_df_cleaned =  preprocessor.clean_df(test_df)
train_df_cleaned =  preprocessor.clean_df(train_df)
test_df_tokenized =  preprocessor.tokenize_df(test_df_cleaned)
train_df_tokenized =  preprocessor.tokenize_df(train_df_cleaned)
test_df_no_stopwords =  preprocessor.remove_stop_words(test_df_tokenized)
train_df_no_stopwords =  preprocessor.remove_stop_words(train_df_tokenized)
logger.info("Data Cleaned")
train_df_hashed =  preprocessor.hash_df(train_df_no_stopwords)
test_df_hashed = preprocessor.hash_df(test_df_no_stopwords)
train_df_vector,test_df_vector = preprocessor.idf_df(train_df_hashed,test_df_hashed)
train_df_final =  preprocessor.finalise_df(train_df_vector)
test_df_final =  preprocessor.finalise_df(test_df_vector)
train_df_final.limit(10).show()
logger.info("Data Vectorized")
