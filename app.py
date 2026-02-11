from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os

app = FastAPI()

BASE_DIM = (os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = os.path.join(BASE_DIM,"src","KMEANS_model.pkl")

model = joblib.load(model_path)

class irisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

cluster_map = {
    0: "Setosa Type",
    1: "Versicolor Type",
    2: "Virginica Type"
}

@app.post("/predict")
def predict(data:irisInput):
    input_data = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])
    cluster = model.predict(input_data)[0]
    return{
        "cluster_number": int(cluster),
        "category": cluster_map[int(cluster)]
    }