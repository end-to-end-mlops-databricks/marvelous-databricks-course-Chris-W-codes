# Databricks notebook source
# MAGIC %md
# MAGIC We'll use this notebook to build some pre-processing classes and functions

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import regexp_replace, lower, col, trim
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import HashingTF, IDF


# COMMAND ----------

class PreProcess:
    def load_data(filepath):
         #Low the data from the filepath
        df_out = spark.read.csv(filepath,header = False)
        return df_out

    def append_columns(df):
        #Give the dataframe some column headers
        column_names = ["target", "id", "date", "flag", "user", "text"]
        df_out = df.toDF(*column_names)
        return df_out

    def filter_df(df):
        #Filter out the columns we need
        df_out = df.select("target","text")
        return df_out

    def test_train_split(df):
        #Do a test train split, at 80/20
        train_df,test_df = df.randomSplit([0.8,0.2])
        return(train_df,test_df)

    def clean_df(df):
        #Cleans the text column of the data
        #Remove punctation and numbers, make lower case and the trim
        df_out = train_df.withColumn("text",trim(lower(regexp_replace(col("text"), "[^a-zA-Z\\s]", ""))))
        return(df_out)

    def tokenize_df(df):
        #Build a tokenizer from the text column
        tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
        df_out = tokenizer.transform(train_df_cleaned)
        return(df_out)

    def remove_stop_words(df):
        #Remove stop words from the tokens column
        remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
        df_out = remover.transform(df)
        return(df_out)

    def hash_df(df):
        #Hash the filtered token column
        hashing_tf = HashingTF(inputCol="filtered_tokens", outputCol="raw_features")
        df_out =  hashing_tf.transform(df)
        return (df_out)

    def idf_df(train_df,test_df):
        #Create a vectoroizer using the train set and and apply it to both train and test set 
        idf = IDF(inputCol="raw_features", outputCol="features")
        idf_model = idf.fit(train_df)
        train_df_out = idf_model.transform(train_df)
        test_df_out = idf_model.transform(test_df)
        return(train_df_out,test_df_out)

    def finalise_df(df):
        #Finalise the datasets by removing all the columns we don't need
        df_out = df.select("target","features")
        return(df_out)

# COMMAND ----------



# COMMAND ----------


