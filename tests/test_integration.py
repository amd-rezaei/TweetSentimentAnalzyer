# test_integration.py

import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "<html" in response.text.lower()

def test_predict_empty_text():
    payload = {"text": "", "sentiment": "positive"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["selected_text"] == ""

@pytest.mark.parametrize("text, sentiment, expected_substr", [
    ("I love this!", "positive", "love"),
    ("This is terrible.", "negative", "terrible"),
    ("It's okay, nothing special.", "neutral", "okay"),
    ("", "positive", ""),
    ("Great quality, amazing product.", "invalid_sentiment", None)
])
def test_predict_endpoint(text, sentiment, expected_substr):
    payload = {"text": text, "sentiment": sentiment}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    if sentiment == "invalid_sentiment":
        assert data["selected_text"] is None
    else:
        assert expected_substr.lower() in data["selected_text"].lower()
