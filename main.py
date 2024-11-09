# Databricks notebook source
# MAGIC %md
# MAGIC We will use this notebook to call the classes and functions

# COMMAND ----------

import pandas as pd
import logging
import yaml
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
from pyspark.sql.functions import rand
import numpy as np

# COMMAND ----------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# COMMAND ----------

with open('project_config.yml', 'r') as file:
    config = yaml.safe_load(file)

print(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

# MAGIC %run "/Workspace/Users/chris.wroe8@gmail.com/mlops/Pre_processing"

# COMMAND ----------

preprocessor = PreProcess()
logger.info("PreProcessor initialized.")

# COMMAND ----------

df = preprocessor.load_data(config['dataset_location'])
logger.info("Data Loaded")
df2 = preprocessor.filter_data(df)
train_df,test_df =  preprocessor.test_train_split(df2,config['train_pc'],config['test_pc'])
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
idf_model,train_df_vector,test_df_vector = preprocessor.idf_df(train_df_hashed,test_df_hashed)
train_df_final =  preprocessor.finalise_df(train_df_vector)
test_df_final =  preprocessor.finalise_df(test_df_vector)
logger.info("Data Vectorized")
train_df_final.write.format("delta").mode("overwrite").saveAsTable("mlops_students.chriswroe8.trainset")
test_df_final.write.format("delta").mode("overwrite").saveAsTable("mlops_students.chriswroe8.testset")
logger.info("Data written to catalog")

# COMMAND ----------

#Convert train_df_final and test_df_final to pandas
train_df_final_pd = train_df_final.toPandas()
test_df_final_pd = test_df_final.toPandas()

#Create X_train,X_test,y_train and y_test as pandas dataframes
X_train = train_df_final_pd[["features"]]
y_train = train_df_final_pd[["sentiment"]]
X_test = test_df_final_pd[["features"]]
y_test = test_df_final_pd[["sentiment"]]

# COMMAND ----------

#Convert X_train and x_test into arrays
X_train = X_train["features"].apply(lambda x: x.toArray() if hasattr(x, "toArray") else x).tolist()
X_test = X_test["features"].apply(lambda x: x.toArray() if hasattr(x, "toArray") else x).tolist()

# COMMAND ----------

# MAGIC %run "/Workspace/Users/chris.wroe8@gmail.com/mlops/train_model"

# COMMAND ----------

train_model_instance  = TrainModel()
logger.info("Train Model initialized.")

# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_name="/Users/chris.wroe8@gmail.com/sentiment_analysis")
mlflow.set_experiment_tags({"model_type": "logistic-regression"})

# COMMAND ----------

#Start an MLFlow Run
with mlflow.start_run(
    run_name="test-run-defaults",
    description="returning to model defaults",
) as run:
    run_id = run.info.run_id

    #Log parameters
    mlflow.log_params({"maxIter": config['max_Iter']})

    #Train the model
    trained_model = train_model_instance.train_lr_model(X_train,y_train,config['max_Iter'])
    logger.info("Model trained")

    #Evaluate the model
    predictions,accuracy = train_model_instance.predict_eval(X_test,y_test,trained_model)
    logger.info("Predictions made")
    logger.info(f"Model accuracy: {accuracy:.2f}");
    mlflow.log_metrics({"accuracy": accuracy})

    #Convert the training data to Pandas
    dataset = mlflow.data.from_pandas(train_df_final_pd,source="mlops_students.chriswroe8.trainset", name="training_data")

    #Log the training data
    mlflow.log_input(dataset,context="training")

    #Infer the model signature
    model_input = X_train[:10]
    model_output = predictions[:10]
    signature = infer_signature(model_input,model_output)

    #Log the model
    mlflow.sklearn.log_model(trained_model,artifact_path="sentiment_lr_model", signature = signature)
    logger.info("Model logged")

# COMMAND ----------

#Let's register the model to the model store
mlflow.set_registry_uri('databricks-uc')
model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id}/sentiment_lr_model',
    name=f"mlops_students.chriswroe8.lr_sentiment")

logger.info("Model registered")

# COMMAND ----------

#Let's try and build a custom model wrapper
class sentimentmodelwrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        
    def predict(self, context, model_input):
        predictions = self.model.predict(model_input)
        predictions2 = np.where(predictions == 0, "negative",np.where(predictions == 4, "positive", "neutral"))
        return predictions2
    
wrapped_model = sentimentmodelwrapper(trained_model) 

# COMMAND ----------

predictions = wrapped_model.predict(trained_model,X_train[:4])

# COMMAND ----------

mlflow.set_experiment(experiment_name="/Users/chris.wroe8@gmail.com/sentiment_analysis_wrapped")

with mlflow.start_run(
    run_name="wrapped model",
    description="wrapped model",
) as run:
    run_id_2 = run.info.run_id
    mlflow.log_input(dataset,context="training")
    mlflow.pyfunc.log_model(python_model=wrapped_model,artifact_path="sentiment_lr_model_wrapped", signature=signature)
    logger.info("Model logged")

# COMMAND ----------

#Let's register the model to the model store
model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id_2}/sentiment_lr_model_wrapped',
    name=f"mlops_students.chriswroe8.sentiment_lr_model_wrapped")

logger.info("Model registered")
