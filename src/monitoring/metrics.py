import time
from functools import wraps
from typing import Dict

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

# define metrics
prediction_counter = Counter("prediction_total", "Total predictions", ["model", "sentiment"])
prediction_latency = Histogram("prediction_duration_seconds", "Prediction latency")
model_confidence = Histogram("model_confidence", "Model confidence distribution")
active_models = Gauge("active_models", "Number of active models")


class MetricsCollector:
    def __init__(self):
        self.predictions = []
        self.latencies = []
        self.confidence_scores = []

    def track_prediction(self, result: Dict, latency: float):
        """Track individual prediction metrics"""
        self.predictions.append(result)
        self.latencies.append(latency)
        self.confidence_scores.append(result["confidence"])

        # update Prometheus metrics
        prediction_counter.labels(model="sentiment-model", sentiment=result["sentiment"]).inc()
        prediction_latency.observe(latency)
        model_confidence.observe(result["confidence"])

    def get_metrics_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.latencies:
            return {}

        return {
            "total_predictions": len(self.predictions),
            "avg_latency": np.mean(self.latencies),
            "p95_latency": np.percentile(self.latencies, 95),
            "avg_confidence": np.mean(self.confidence_scores),
            "sentiment_distribution": self._get_sentiment_distribution(),
        }

    def _get_sentiment_distribution(self) -> Dict:
        """Calculate sentiment distribution"""
        sentiments = [prediction["sentiment"] for prediction in self.predictions]
        return {
            "positive": sentiments.count("positive") / len(sentiments),
            "negative": sentiments.count("negative") / len(sentiments),
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
