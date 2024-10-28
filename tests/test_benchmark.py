# test_benchmark.py

import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

@pytest.mark.benchmark(group="predict_latency")
def test_predict_latency(benchmark):
    payload = {"text": "Testing prediction latency!", "sentiment": "positive"}
    benchmark(client.post, "/predict", json=payload)

@pytest.mark.parametrize("payload", [
    {"text": "Testing load!", "sentiment": "positive"},
    {"text": "Worst experience.", "sentiment": "negative"}
])
@pytest.mark.benchmark(group="load_test")
def test_predict_load(payload, benchmark):
    @benchmark
    def send_request():
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert "selected_text" in response.json()
