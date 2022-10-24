from fastapi import FastAPI
from typing import List
import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from pydantic import BaseModel


def perceptron(X, y, epochs):
    
    # X --> Inputs.
    # y --> labels/target.
    # epochs --> Number of iterations.
    
    # m-> number of training examples
    # n-> number of features 
    m, n = X.shape
    
    # Initializing parapeters(theta) to zeros.
    # +1 in n+1 for the bias term.
    theta = np.zeros((n+1,1))
    
    # Empty list to store how many examples were 
    # misclassified at every iteration.
    n_miss_list = []
    
    
    missclassified = True
    epoch = 0
    while epoch < epochs and missclassified:
        epoch += 1
        missclassified = False
        # variable to store #misclassified.
        n_miss = 0
        
        # looping for every example.
        for idx, x_i in enumerate(X):
            
            # Insering 1 for bias, X0 = 1.
            x_i = np.insert(x_i, 0, 1).reshape(-1,1)
            
            # Calculating prediction/hypothesis.
            y_hat = np.sign(np.dot(x_i.T, theta))
            
            # Updating if the example is misclassified.
            if (np.squeeze(y_hat) - y[idx]) != 0:
                theta += y[idx] *x_i
                
                missclassified = True
        
    return theta, n_miss_list


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
    theta, miss_l = perceptron(df, y, 1000)
    return theta.transpose().tolist()[0]

@app.post("/linear/pla/predict")
async def predict_pla(data: data_Weight):
    weights = data.model_weights
    df = data.input_data
    oneVector = np.ones((len(df),1))
    df = np.concatenate((oneVector, df), axis=1)
    Y_pred = np.dot(df, weights)
    Y_pred = np.where(Y_pred > 0, 1, -1)
    return Y_pred.tolist()

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
