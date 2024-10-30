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
    ("Great quality, amazing product.", "invalid_sentiment", "Great quality, amazing product.")
])
def test_predict_endpoint(text, sentiment, expected_substr):
    payload = {"text": text, "sentiment": sentiment}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200 
    data = response.json()
    
    if sentiment == "invalid_sentiment":
        # For an invalid sentiment, expect the full text in 'selected_text' (case-insensitive)
        assert data["selected_text"].lower() == text.lower(), \
            f"Expected 'selected_text' to be the full text '{text}' for invalid sentiment, got '{data['selected_text']}'"
    else:
        # For valid sentiments, check that the expected substring is in the selected text
        assert expected_substr.lower() in data["selected_text"].lower(), \
            f"Expected '{expected_substr}' in '{data['selected_text']}'"
