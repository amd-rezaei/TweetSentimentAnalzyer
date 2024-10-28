import os
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)
# DEPLOYMENT_TYPE = os.getenv("DEPLOYMENT_TYPE", "docker").lower()

def test_predict_positive():
    response = client.post("/predict", json={"text": "I want to see David cook!!", "sentiment": "positive"})
    assert response.status_code == 200
    data = response.json()
    assert "selected_text" in data
    assert data["selected_text"], f"Expected non-empty selected_text, got: '{data['selected_text']}'"

def test_predict_negative():
    response = client.post("/predict", json={"text": "Back from town and my Mac crashed on me but it's better now", "sentiment": "negative"})
    assert response.status_code == 200
    data = response.json()
    assert "selected_text" in data
    assert data["selected_text"], f"Expected non-empty selected_text, got: '{data['selected_text']}'"
