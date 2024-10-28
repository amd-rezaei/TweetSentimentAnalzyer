# test_triton.py

import os
import pytest
import numpy as np
from src.utils import preprocess_text
from fastapi.testclient import TestClient
from src.triton_app import app 

client = TestClient(app)

@pytest.mark.skipif(os.getenv("DEPLOYMENT_TYPE") != "triton", reason="Only for Triton deployment")
def test_triton_preprocessing():
    sample_text = "Triton preprocessing test"
    ids, att, tok = preprocess_text(sample_text, "neutral")
    assert ids.shape == (1, 96) and att.shape == (1, 96) and tok.shape == (1, 96)
    assert ids.dtype == np.int32 and att.dtype == np.int32 and tok.dtype == np.int32

@pytest.mark.skipif(os.getenv("DEPLOYMENT_TYPE") != "triton", reason="Only for Triton deployment")
def test_triton_predict_endpoint():
    payload = {"text": "I am thrilled with the product!", "sentiment": "positive"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "selected_text" in response.json()
