from fastapi import FastAPI
from typing import List
import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from pydantic import BaseModel



class data_Url(BaseModel):
    data_url: str = None

class data_Weight(BaseModel):
    model_weights: List[float] = None
    input_data: List[List[float]] = None


app = FastAPI()
@app.get("/")
async def root():           
    return {"message": "Hello World"}



@app.post("/linear/pla/train")
async def train_pla(data: data_Url):
    df = pd.read_csv(data.data_url, header=0)
    #drop label column
    y = df['label'].values
    df = df.drop('label', axis=1).values
    #separate dataframe only in two classes
    ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)
    model_weights = np.concatenate((ppn.intercept_, ppn.coef_[0]), axis=0)
    return model_weights.tolist()
#a

@app.post("/linear/pla/predict")
async def predict_pla(data: data_Weight):
    weights = data.model_weights
    df = data.input_data
    oneVector = np.ones((len(df),1))
    df = np.concatenate((oneVector, df), axis=1)
    y = np.sign(np.dot(df, weights))
    return y.tolist()

@app.post("/linear/pocket/train")
async def train_pocket(data: data_Url):
    df = pd.read_csv(data.data_url, header=0)
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
async def predict_pocket(data: data_Weight):
    weights = data.model_weights
    df = data.input_data
    print(df)
    print(weights)
    oneVector = np.ones((len(df),1))
    df = np.concatenate((oneVector, df), axis=1)
    y = np.dot(df, weights)
    y = np.where(y > 0, 1, -1)
    return y.tolist()
