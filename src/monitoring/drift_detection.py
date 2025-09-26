from collections import deque
from typing import Any, Dict

from scipy import stats


class DriftDetector:
    def __init__(self, window_size: int = 1000, threshold: float = 0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.reference_distribution: deque[Dict[str, Any]] = deque(maxlen=window_size)
        self.current_distribution: deque[Dict[str, Any]] = deque(maxlen=window_size)

    def add_prediction(self, confidence: float, sentiment: str):
        """Add new prediction to current distribution"""
        self.current_distribution.append({"confidence": confidence, "sentiment": sentiment})

    def detect_drift(self) -> Dict:
        """Detect distribution drift using KS test"""
        if len(self.current_distribution) < 100:
            return {"drift_detected": False, "message": "Insufficient data"}

        current_conf = [p["confidence"] for p in self.current_distribution]

        if len(self.reference_distribution) > 0:
            ref_conf = [p["confidence"] for p in self.reference_distribution]
            ks_statistic, p_value = stats.ks_2samp(current_conf, ref_conf)

            drift_detected = p_value < self.threshold  # statistically significant

            return {
                "drift_detected": drift_detected,
                "ks_statistic": ks_statistic,
                "p_value": p_value,
                "threshold": self.threshold,
            }

        return {"drift_detected": False, "message": "No reference distribution"}

    def update_reference(self):
        """Update reference distribution with current data"""
        self.reference_distribution = deque(self.current_distribution, maxlen=self.window_size)
