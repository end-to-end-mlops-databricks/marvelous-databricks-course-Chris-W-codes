# Databricks notebook source
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# COMMAND ----------

class TrainModel:
    #This class contains functions needed to train and evaluate our model
    def train_lr_model(self,X,y,maxIter=100):
        # Initialize logistic regression and train it
        model = LogisticRegression(max_iter=maxIter)
        model.fit(X, y)
        return model
    
    def predict_eval(self,X,y,model):
        #Use a model to make predictions
        predictions = model.predict(X)
        accuracy = accuracy_score(y, predictions)
        return predictions, accuracy

