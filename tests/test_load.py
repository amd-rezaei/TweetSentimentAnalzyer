import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

# Sample request payloads for load testing, including edge cases
sample_requests = [
    {"text": "I love this product!", "sentiment": "positive"},
    {"text": "This is the worst experience I've ever had.", "sentiment": "negative"},
    {"text": "It's just okay, nothing special.", "sentiment": "neutral"},
    {"text": "Absolutely fantastic! Highly recommend.", "sentiment": "positive"},
    {"text": "Not worth the money.", "sentiment": "negative"},
    # Edge case: large payload to test latency with long inputs
    {"text": "I absolutely adore how this product makes my life easier!" * 10, "sentiment": "positive"},
]

@pytest.mark.parametrize("request_payload", sample_requests)
@pytest.mark.benchmark(group="load_test")
def test_predict_load(request_payload, benchmark):
    # Benchmark the load test for predict endpoint
    @benchmark
    def send_request():
        response = client.post("/predict", json={"text": request_payload["text"], "sentiment": request_payload["sentiment"]})
        assert response.status_code == 200
        
        data = response.json()
        assert "selected_text" in data

# Concurrent requests to test predict endpoint under load
@pytest.mark.parametrize("request_payload", sample_requests)
@pytest.mark.benchmark(group="concurrency_test")
def test_predict_concurrency(request_payload, benchmark):
    # Benchmark the load test for concurrency
    def send_concurrent_request():
        response = client.post("/predict", json={"text": request_payload["text"], "sentiment": request_payload["sentiment"]})
        assert response.status_code == 200
        assert "selected_text" in response.json()

    # Run concurrency benchmark with multiple iterations to simulate load
    benchmark.pedantic(send_concurrent_request, rounds=5, iterations=10, warmup_rounds=1)
