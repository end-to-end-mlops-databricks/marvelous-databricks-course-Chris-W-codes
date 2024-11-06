# Databricks notebook source
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

class TrainModel:
    #This class contains functions needed to train and evaluate our model
    def train_lr_model(self,df,maxIter,regParam,elasticNetParam):
        #Initialize logistic regression and train it
        model_out1= LogisticRegression(featuresCol="features", labelCol="target",maxIter=maxIter,regParam=regParam,elasticNetParam=elasticNetParam)
        model_out2 = model_out1.fit(df)
        return model_out2
    
    def predict_eval(self,df,model):
        #Use a model to make predictions
        predictions = model.transform(df)
        evaluator = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions) 
        return(predictions,accuracy)

