from fastapi import FastAPI
import torch
from models.model_registry import get_model

app=FastAPI()

model=get_model()
model.load_state_dict(torch.load("model.pt"))

@app.post("/predict")

def predict(data:dict):

    x=torch.tensor([[
        data["temperature"],
        data["pressure"],
        data["flow_rate"],
        data["density"],
        data["shear_rate"]
    ]],dtype=torch.float32)

    pred=model(x).item()

    return {
        "predicted_viscosity":pred
    }