import os
import sys

from fastapi_testclient import TestClient

from src.api.inference import app

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

client = TestClient(app)


class TestAPI:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_predict_endpoint(self):
        payload = {"texts": ["Great product!", "Terrible service"], "return_probabilities": True}
        response = client.post("/predict", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 2
        assert "model_version" in data

    def test_metrics_endpoint(self):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "total_predictions" in response.json()

    def test_model_info_endpoint(self):
        response = client.get("model/info")
        assert response.status_code == 200
        assert "model_name" in response.json()
