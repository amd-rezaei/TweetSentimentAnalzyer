# tests/test_integration.py

from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "<html" in response.text.lower()  # Ensure HTML content is served

def test_predict_endpoint():
    payload = {"text": " just got home from a nice party, just not tired yet", "sentiment": "positive"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "selected_text" in response.json()
    assert "not tired" in response.json()["selected_text"].lower()

def test_predict_empty_text():
    payload = {"text": "", "sentiment": "positive"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["selected_text"] == ""
