from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np
from typing import List
import os

app = FastAPI()

# Global variable to store the current model
current_model = None
current_model_version = None

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_NAME = "IrisRandomForestModel"

class PredictionInput(BaseModel):
    features: List[List[float]]

class PredictionOutput(BaseModel):
    predictions: List[int]

class ModelUpdate(BaseModel):
    version: int

@app.on_event("startup")
async def load_model():
    """Load the model from MLflow on startup"""
    global current_model, current_model_version
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Load latest version by default
    model_uri = f"models:/{MODEL_NAME}/latest"
    current_model = mlflow.pyfunc.load_model(model_uri)
    current_model_version = "latest"
    print(f"Model loaded: {model_uri}")

@app.get("/")
def read_root():
    return {
        "message": "ML Model Service",
        "current_model_version": current_model_version
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """Make predictions using the current model"""
    if current_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to numpy array
        features = np.array(input_data.features)
        
        # Make predictions
        predictions = current_model.predict(features)
        
        return PredictionOutput(predictions=predictions.tolist())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/update-model")
def update_model(model_update: ModelUpdate):
    """Update the model to a specific version"""
    global current_model, current_model_version
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Load the specified model version
        model_uri = f"models:/{MODEL_NAME}/{model_update.version}"
        current_model = mlflow.pyfunc.load_model(model_uri)
        current_model_version = model_update.version
        
        return {
            "message": "Model updated successfully",
            "new_version": model_update.version
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model update error: {str(e)}")

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": current_model is not None,
        "model_version": current_model_version
    }