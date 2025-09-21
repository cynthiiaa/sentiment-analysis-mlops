from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps
from typing import Dict, List
import numpy as np

# define metrics
prediction_counter = Counter("predicition_total", "Total predicitions", ["model", "sentiment"])
prediction_latency = Histogram("prediction_duration_seconds", "Prediciton latency")
model_condifence = Histogram("model_confidence", "Model confidence distribution")
active_models = Gauge("active_models", "Number of active models")

class MetricsCollector:
    def __init__(self):
        self.predicitions = []
        self.latencies = []
        self.confidence_scores = []

    def track_predicition(self, result: Dict, latency: float):
        """Track individual prediction metrics"""
        self.predicitions.append(result)
        self.latencies.append(latency)
        self.confidence_scores.append(result["confidence"])

        # update Prometheus metrics
        prediction_counter.labels(
            model="sentiment-model",
            sentiment=result["sentiment"]
        ).inc()
        prediction_latency.observe(latency)
        model_condifence.observe(result["confidence"])

    def get_metrics_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.latencies:
            return {}
        
        return {
            "total_predicitions": len(self.predicitions),
            "avg_latency": np.mean(self.latencies),
            "p95_latency": np.percentile(self.latencies, 95),
            "avg_confidence": np.mean(self.confidence_scores),
            "sentiment_distribution": self._get_sentiment_distribution()
        }
    
    def _get_sentiment_distribution(self) -> Dict:
        """Calculate sentiment distribution"""
        sentiments = [prediction["sentiment"] for prediction in self.predicitions]
        return {
            "positive": sentiments.count("positive") / len(sentiments),
            "negative": sentiments.count("negative") / len(sentiments)
        }
    
def monitor_performance(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        latency = time.time() - start
        return result, latency
    return wrapper