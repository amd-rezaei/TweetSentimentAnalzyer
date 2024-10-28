# test_benchmark.py

import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

@pytest.mark.benchmark(group="predict_latency")
def test_predict_latency(benchmark):
    payload = {"text": "Testing prediction latency!", "sentiment": "positive"}
    response = benchmark(client.post, "/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "selected_text" in data
    assert data["selected_text"], "Expected 'selected_text' to be non-empty"

@pytest.mark.parametrize("payload", [
    pytest.param({"text": "Short text", "sentiment": "positive"}, id="short_text_positive"),
    pytest.param({"text": "Worst experience.", "sentiment": "negative"}, id="short_text_negative"),
    pytest.param({"text": "This product is wonderful!" * 20, "sentiment": "positive"}, id="long_text_positive"),
    pytest.param({"text": "Just okay.", "sentiment": "neutral"}, id="short_text_neutral"),
])
@pytest.mark.benchmark(group="load_test")
def test_predict_load(payload, benchmark):
    def send_request():
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "selected_text" in data
        assert data["selected_text"], "Expected 'selected_text' to be non-empty"

    benchmark.pedantic(send_request, rounds=10, iterations=1)
