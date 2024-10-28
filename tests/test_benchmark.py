import os
from fastapi.testclient import TestClient
from src.app import app
import pytest

client = TestClient(app)
# DEPLOYMENT_TYPE = os.getenv("DEPLOYMENT_TYPE", "docker").lower()

@pytest.mark.benchmark(group="prediction_latency")
@pytest.mark.parametrize("payload", [
    {"text": "This product is outstanding!", "sentiment": "positive"}
])
def test_predict_latency(benchmark, payload):
    benchmark(client.post, "/predict", json=payload)
