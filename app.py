from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np
from typing import List
import os
import random

app = FastAPI()

# Global variables for current and next models (canary deployment)
current_model = None
next_model = None
current_model_version = None
next_model_version = None
canary_probability = 0.1  # Probability to use CANARY (next) model

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_NAME = "IrisRandomForestModel"

class PredictionInput(BaseModel):
    features: List[List[float]]

class PredictionOutput(BaseModel):
    predictions: List[int]
    model_used: str  # Track which model was used

class ModelUpdate(BaseModel):
    version: int

class CanaryConfig(BaseModel):
    probability: float  # Probability to use CANARY (next) model (0.0 to 1.0)

@app.on_event("startup")
async def load_model():
    """Load both current and next models from MLflow on startup"""
    global current_model, next_model, current_model_version, next_model_version
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Load latest version for both current and next
    model_uri = f"models:/{MODEL_NAME}/latest"
    current_model = mlflow.pyfunc.load_model(model_uri)
    next_model = mlflow.pyfunc.load_model(model_uri)
    current_model_version = "latest"
    next_model_version = "latest"
    print(f"Models loaded - Current: {current_model_version}, Next: {next_model_version}")

@app.get("/")
def read_root():
    return {
        "message": "ML Model Service with Canary Deployment",
        "current_model_version": current_model_version,
        "next_model_version": next_model_version,
        "canary_probability": canary_probability
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "current_model_loaded": current_model is not None,
        "next_model_loaded": next_model is not None,
        "current_model_version": current_model_version,
        "next_model_version": next_model_version,
        "canary_probability": canary_probability
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """Make predictions using current or next model based on canary probability"""
    if current_model is None or next_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        features = np.array(input_data.features)
        
        # Choose model based on canary probability
        if random.random() < canary_probability:
            # Use canary (next) model
            predictions = next_model.predict(features)
            model_used = f"canary/next (v{next_model_version})"
        else:
            # Use stable (current) model
            predictions = current_model.predict(features)
            model_used = f"stable/current (v{current_model_version})"
        
        return PredictionOutput(
            predictions=predictions.tolist(),
            model_used=model_used
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/update-model")
def update_model(model_update: ModelUpdate):
    """Update the NEXT model to a specific version"""
    global next_model, next_model_version
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        model_uri = f"models:/{MODEL_NAME}/{model_update.version}"
        next_model = mlflow.pyfunc.load_model(model_uri)
        next_model_version = model_update.version
        
        return {
            "message": "Next model updated successfully",
            "current_version": current_model_version,
            "next_version": next_model_version
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model update error: {str(e)}")

@app.post("/accept-next-model")
def accept_next_model():
    """Accept the next model as current"""
    global current_model, current_model_version
    
    try:
        current_model = next_model
        current_model_version = next_model_version
        
        return {
            "message": "Next model accepted as current",
            "current_version": current_model_version,
            "next_version": next_model_version
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Accept model error: {str(e)}")

@app.post("/set-canary-probability")
def set_canary_probability(config: CanaryConfig):
    """Set the probability of using the CANARY (next) model"""
    global canary_probability
    
    if not 0.0 <= config.probability <= 1.0:
        raise HTTPException(status_code=400, detail="Probability must be between 0.0 and 1.0")
    
    canary_probability = config.probability
    
    return {
        "message": "Canary probability updated",
        "canary_probability": canary_probability,
        "canary_model_usage": f"{canary_probability * 100}%",
        "stable_model_usage": f"{(1 - canary_probability) * 100}%"
    }