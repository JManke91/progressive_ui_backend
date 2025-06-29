from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib

# Load model
model = joblib.load("user_classifier.pkl")

app = FastAPI()

# Define input format
class InputData(BaseModel):
    data: List[List[float]]

@app.get("/")
def read_root():
    return {"status": "ok"}

@app.post("/predict")
def predict(input: InputData):
    predictions = model.predict(input.data)
    return {"prediction": predictions.tolist()}
