# Databricks notebook source
# MAGIC %md
# MAGIC We will use this notebook to call the classes and functions

# COMMAND ----------

import pandas as pd
import logging
import sys

# COMMAND ----------

sys.path.append('/Workspace/Users/chris.wroe@fabledata.com/marvelous-databricks-course-Chris-W-codes/sentiment')
from Pre_processing import PreProcess

# COMMAND ----------

df = load_data("/Volumes/dbw-dplabdlz-dev-uks-01/chris_training/sample_data/twitter_sentiment.csv")
logging.info(f"Data has loaded")

# COMMAND ----------

#We need to save some parameters in a config file
#We need to include some unit tests
#We need to do some logging
#Need a main.py to import the functions and run the code
#Every class and function should have a doc string

# COMMAND ----------

df2 = append_columns(df)
df3 = filter_df(df2)
train_df,test_df = test_train_split(df3)
test_df_cleaned = clean_df(test_df)
train_df_cleaned = clean_df(train_df)
test_df_tokenized = tokenize_df(test_df_cleaned)
train_df_tokenized = tokenize_df(train_df_cleaned)
test_df_no_stopwords = remove_stop_words(test_df_tokenized)
train_df_no_stopwords = remove_stop_words(train_df_tokenized)
train_df_hashed = hash_df(train_df_no_stopwords)
test_df_hashed = hash_df(test_df_no_stopwords)
train_df_vector,test_df_vector = idf_df(train_df_hashed,test_df_hashed)
train_df_final = finalise_df(train_df_vector)
test_df_final = finalise_df(test_df_vector)
train_df_final.limit(10).show()
