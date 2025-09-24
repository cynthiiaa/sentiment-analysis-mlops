from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
from src.models.sentiment_model import SentimentModel
from src.monitoring.metrics import MetricsCollector
from src.monitoring.drift_detection import DriftDetector
import time
from datetime import datetime

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")

# initialzation
model = SentimentModel()
metrics_collector = MetricsCollector()
drift_detectior = DriftDetector()

class PredictionRequest(BaseModel):
    texts: List[str]
    return_probabilities: bool = False

class PredictionResponse(BaseModel):
    predictions: List[Dict]
    model_version: str
    timestamp: str 

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model.model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Main prediction endpoint"""
    try:
        start_time = time.time()

        # make predictions
        results = model.predict(request.texts)

        # track metrics
        latency = time.time() - start_time
        for result in results:
            metrics_collector.track_predicition(result, latency/len(results))
            drift_detectior.add_prediction(
                result["confidence"],
                result["sentiment"]
            )

        # filter probabilities if not requested
        if not request.return_probabilities:
            for result in results:
                result.pop("probabilities", None)

        return PredictionResponse(
            predictions=results,
            model_version=model.config.model_name,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/metrics")
async def get_metrics():
    """Get current metrics"""
    metrics = metrics_collector.get_metrics_summary()
    drift_status = drift_detectior.detect_drift()

    return {
        **metrics,
        "drift_status": drift_status
    }

@app.get("/model/info")
async def model_info():
    """Get model information"""
    return model.get_model_info()

@app.post("model/reload")
async def reload_model():
    """Reload the model"""
    try:
        model.load_model()
        return {"status": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)