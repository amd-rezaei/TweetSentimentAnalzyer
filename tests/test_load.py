import os
import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)
# DEPLOYMENT_TYPE = os.getenv("DEPLOYMENT_TYPE", "docker").lower()

sample_requests = [
    {"text": "I love this product!", "sentiment": "positive"},
    {"text": "This is the worst experience I've ever had.", "sentiment": "negative"},
    {"text": "It's just okay, nothing special.", "sentiment": "neutral"},
    {"text": "Absolutely fantastic! Highly recommend.", "sentiment": "positive"},
    {"text": "Not worth the money.", "sentiment": "negative"},
    {"text": "I absolutely adore how this product makes my life easier!" * 10, "sentiment": "positive"},
]

@pytest.mark.parametrize("request_payload", sample_requests)
@pytest.mark.benchmark(group="load_test")
def test_predict_load(request_payload, benchmark):
    @benchmark
    def send_request():
        response = client.post("/predict", json=request_payload)
        assert response.status_code == 200
        assert "selected_text" in response.json()
