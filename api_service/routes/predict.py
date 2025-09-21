# api_service/routes/predict.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
import numpy as np
import yaml
import os

router = APIRouter()

# Config yükle
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

MLFLOW_TRACKING_URI = config["deploy"].get("tracking_uri", os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
MODEL_NAME = config["deploy"].get("model_name", "Classic_Models_Model")
MODEL_STAGE = config["deploy"].get("model_stage", "Production")
FORECAST_STEPS = config["model"]["params"].get("forecast_steps", 1)
MODEL_TYPE = config["model"]["type"]

os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
model = mlflow.pyfunc.load_model(model_uri)

class PredictRequest(BaseModel):
    features: dict

@router.post("/predict", tags=["Prediction"])
def predict(data: PredictRequest):
    try:
        if MODEL_TYPE in ["random_forest","linear_regression"]:
            df = pd.DataFrame([data.features])
            pred = model.predict(df)
            return {"prediction": pred.tolist()}

        elif MODEL_TYPE == "lstm":
            window = data.features.get("window")
            if window is None:
                raise HTTPException(status_code=400, detail="LSTM için 'window' verisi gerekli.")
            arr = np.array(window).reshape(1, len(window), 1)
            forecast = []
            current_input = arr.copy().flatten()

            for _ in range(FORECAST_STEPS):
                input_reshaped = current_input.reshape(1, len(window), 1)
                next_pred = model.predict(input_reshaped)[0][0]
                forecast.append(next_pred)
                current_input = np.append(current_input[1:], next_pred)

            return {"forecast": forecast}

        else:
            raise HTTPException(status_code=400, detail="Desteklenmeyen model tipi")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
