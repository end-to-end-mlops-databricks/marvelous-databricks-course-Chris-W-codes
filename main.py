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

# COMMAND ----------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# COMMAND ----------

with open('project_config.yml', 'r') as file:
    config = yaml.safe_load(file)

print(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

# MAGIC %run "/Users/chris.wroe@fabledata.com/marvelous-databricks-course-Chris-W-codes/Pre_processing"

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
logger.info("Data Vectorized")
train_df_final.write.format("delta").mode("overwrite").saveAsTable("`dbw-dplabdlz-dev-uks-01`.chris_training.train_data")
test_df_final.write.format("delta").mode("overwrite").saveAsTable("`dbw-dplabdlz-dev-uks-01`.chris_training.test_data")
logger.info("Data written to catalog")

# COMMAND ----------

# MAGIC %run "/Users/chris.wroe@fabledata.com/marvelous-databricks-course-Chris-W-codes/train_model"

# COMMAND ----------

train_model_instance  = TrainModel()
logger.info("Train Model initialized.")

# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_name="/Users/chris.wroe@fabledata.com/sentiment_analysis_custom")
mlflow.set_experiment_tags({"model_type": "custom-logistic-regression"})
model_name = "dbw-dplabdlz-dev-uks-01.chris_training.sentiment_lr_model"


# COMMAND ----------

#Set some parameters
maxIter = 1
regParam= 0
elasticNetParam = 0

#Start an MLFlow Run
with mlflow.start_run(
    run_name="test-run-defaults",
    description="returning to model defaults",
) as run:
    run_id = run.info.run_id

    #Log parameters
    mlflow.log_params({"maxIter": maxIter,"regParam": regParam,"elasticNetParam": elasticNetParam})

    #Train the model
    trained_model = train_model_instance.train_lr_model(train_df_final,maxIter,regParam,elasticNetParam)
    logger.info("Model trained")

    #Evaluate the model
    predictions,accuracy = train_model_instance.predict_eval(test_df_final,trained_model)
    logger.info("Predictions made")
    logger.info(f"Model accuracy: {accuracy:.2f}");
    mlflow.log_metrics({"accuracy": accuracy})

    #Convert the training data to Pandas
    train_df_final_pd = train_df_final.toPandas()
    dataset = mlflow.data.from_pandas(train_df_final_pd,source="dbw-dplabdlz-dev-uks-01.chris_training.train_data", name="training_data")

    #Log the training data
    mlflow.log_input(dataset,context="training")

    #Infer the model signature
    model_input = train_df_final.select("features").limit(10).toPandas()
    model_output = predictions.select("prediction").limit(10).toPandas()
    signature = infer_signature(model_input=model_input, model_output=model_output)

    #Log the model
    mlflow.spark.log_model(trained_model,artifact_path="sentiment_lr_model", signature = signature)
    logger.info("Model logged")


# COMMAND ----------

#Let's register the model to the model store
mlflow.set_registry_uri('databricks-uc')
model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id}/sentiment_lr_model',
    name=f"dbw-dplabdlz-dev-uks-01.chris_training.sentiment_lr_model")

logger.info("Model registered")



# COMMAND ----------

#Let's try and build a custom model wrapper
class sentimentmodelwrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        
    def predict(self, context, model_input):
        predictions = self.model.transform(model_input)
        return predictions
    
wrapped_model = sentimentmodelwrapper(trained_model) 

# COMMAND ----------

example_prediction = wrapped_model.predict(context=None, model_input=test_df_final)
example_prediction.display()

# COMMAND ----------

mlflow.set_experiment(experiment_name="/Users/chris.wroe@fabledata.com/sentiment_analysis_wrapped")

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
    name=f"dbw-dplabdlz-dev-uks-01.chris_training.sentiment_lr_model_wrapped")

logger.info("Model registered")
