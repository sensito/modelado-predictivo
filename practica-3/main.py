from fastapi import FastAPI
from typing import List
import pandas as pd
import numpy as np
from pydantic import BaseModel



def model(X, y, epochs):
    """
    initialise_weights()
    for i in epochs:    
        for j in batches:        
            #forward propagation
            feed_batch_data()
            compute_ŷ()
            compute_loss()        
            compute_partial_differentials()
            update_weights()
    """
    m, n = X.shape
    theta = np.zeros((n+1,1))
    n_miss_list = []
    missclassified = True
    epoch = 0
    while epoch < epochs and missclassified:
        epoch += 1
        missclassified = False
        n_miss = 0
        for idx, x_i in enumerate(X):
            x_i = np.insert(x_i, 0, 1).reshape(-1,1)
            y_hat = np.sign(np.dot(x_i.T, theta))
            if (np.squeeze(y_hat) - y[idx]) != 0:
                theta += y[idx] *x_i
                missclassified = True
    return theta
    
    
class data_Url(BaseModel):
    data_url: str = None
    learning_rate: float = None
    iterations: int = None
    
class data_Weight(BaseModel):
    model_weights: List[float] = None
    input_data: List[List[float]] = None
    


app = FastAPI()

@app.get("/")
async def root():           
    return {"message": "Hello World"}

#Método POST
#Endpoint /linear/regression/train
@app.post("/linear/regression/train")
async def train(data: data_Url):
    #Cargamos los datos
    df = pd.read_csv(data.data_url, header=0)
    #save "prediction" column
    y = df['prediction'].values
    df = df.drop('prediction', axis=1).values
    #train model
    theta, = model(df, y, data.iterations)
    return theta.transpose().tolist()[0]

#Método POST
#Endpoint /linear/regression/sgd/predict
@app.post("/linear/regression/sgd/predict")
async def predict(data: data_Weight):
    weights = data.model_weights
    input_data = data.input_data
    #add bias
    input_data.insert(0, 1)
    #compute prediction
    prediction = np.dot(input_data, weights)
    return prediction.tolist()     

    
