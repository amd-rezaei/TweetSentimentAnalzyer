# tests/test_app.py
from fastapi.testclient import TestClient
from src.app import app  

client = TestClient(app)

def test_predict_positive():
    response = client.post("/predict", json={"text": "Hello how are you?", "sentiment": "positive"})
    assert response.status_code == 200
    data = response.json()
    assert "selected_text" in data
    assert data["selected_text"] is not None

def test_predict_negative():
    response = client.post("/predict", json={"text": "I don't like it", "sentiment": "negative"})
    assert response.status_code == 200
    data = response.json()
    assert "selected_text" in data
    assert data["selected_text"] is not None
    
