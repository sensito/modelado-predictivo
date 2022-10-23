from pprint import pp
import re
from fastapi import FastAPI
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron





app = FastAPI()
@app.get("/")
async def root():           
    return {"message": "Hello World"}



@app.post("/linear/pla/train")
async def train_pla(data_url: str):
    df = pd.read_csv(data_url, header=None)
    print(df.head())
    #drop label column
    y = df['label'].values
    df = df.drop('label', axis=1).values
    #separate dataframe only in two classes
    ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)
    print (ppn.fit(df, y))
    print(ppn.coef_)
    model_weights = np.concatenate((ppn.intercept_, ppn.coef_[0]), axis=0)
    return model_weights.tolist()


@app.post("/linear/pla/predict")
async def predict_pla(model_weights: List[float], input_data: List[List[float]]):
    #print(model_weights)
    #print(input_data)
    oneVector = np.ones((len(input_data),1))
    input_data = np.concatenate((oneVector, input_data), axis=1)
    #print(input_data)
    #print(model_weights)
    y = np.dot(input_data, model_weights)
    #print(y)
    y = np.where(y > 0, 1, -1)
    return y.tolist()
    #a = np.sign(np.dot(input_data, model_weights))
    #return a.tolist()

@app.post("/linear/pocket/train")
async def train_pocket(data_url: str):
    df = pd.read_csv(data_url, header=None)
    # separate df only in two classes
    Y = df['label'].values
    X_train = df.drop('label', axis=1).values
    #conver y to numeric values 
    oneVector = np.ones((X_train.shape[0], 1))
    X_train = np.concatenate((oneVector, X_train), axis=1)
    learningRate = 0.1
    plotData = []
    weights = np.random.rand(X_train.shape[1], 1)
    misClassifications = 100
    minMisclassifications = 10000
    iteration = 0
    plotData = []
    while (misClassifications != 0 and (iteration<1000)):
        iteration += 1
        misClassifications = 0
        for i in range(0, len(X_train)):
            currentX = X_train[i].reshape(-1, X_train.shape[1])
            currentY = Y[i]
            wTx = np.dot(currentX, weights)[0][0]
            if currentY == 1 and wTx < 0:
                misClassifications += 1
                weights = weights + learningRate * np.transpose(currentX)
            elif currentY == -1 and wTx > 0:
                misClassifications += 1
                weights = weights - learningRate * np.transpose(currentX)
        plotData.append(misClassifications)
        if misClassifications<minMisclassifications:
            minMisclassifications = misClassifications
        # if iteration%1==0:
    print(weights.transpose())
    return weights.transpose().tolist()[0]

@app.post("/linear/pocket/predict")
async def predict_pocket(model_weights: List[float], input_data: List[List[float]]):
    #print(model_weights)
    #print(input_data)
    oneVector = np.ones((len(input_data),1))
    input_data = np.concatenate((oneVector, input_data), axis=1)
    #print(input_data)
    #print(model_weights)
    y = np.dot(input_data, model_weights)
    #print(y)
    y = np.where(y > 0, 1, -1)
    return y.tolist()
