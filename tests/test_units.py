import os
import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)
DEPLOYMENT_TYPE = os.getenv("DEPLOYMENT_TYPE", "encapsulated").lower()

@pytest.mark.parametrize("text,sentiment,expected", [
    ("I love this!", "positive", "love"),
    ("This is terrible.", "negative", "terrible"),
    ("It's okay, nothing special.", "neutral", "okay"),
    ("", "positive", ""),  # Edge case: empty text
    ("This product is amazing!", "invalid_sentiment", None)  # Edge case: invalid sentiment
])
def test_predict(text, sentiment, expected):
    # Construct payload
    payload = {"text": text, "sentiment": sentiment}
    
    # Send request to /predict endpoint using TestClient
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    selected_text = data["selected_text"]

    if sentiment == "invalid_sentiment":
        assert selected_text is None
    else:
        assert expected.lower() in selected_text.lower()


@pytest.mark.skipif(DEPLOYMENT_TYPE != "triton", reason="Only for Triton deployment")
def test_preprocess_triton():
    from src.triton_app import TritonSentimentService
    import numpy as np

    # Initialize Triton service for direct method testing
    model_service = TritonSentimentService()
    
    # Test preprocessing specific to Triton
    sample_text = "Test the Triton preprocessing"
    sample_sentiment = "neutral"
    ids, att, tok = model_service.preprocess_text(sample_text, sample_sentiment)
    
    # Validate shapes and data types for Triton input
    assert ids.shape == (1, 96)
    assert att.shape == (1, 96)
    assert tok.shape == (1, 96)
    assert ids.dtype == np.int32
    assert att.dtype == np.int32
    assert tok.dtype == np.int32

@pytest.mark.skipif(DEPLOYMENT_TYPE == "triton", reason="Only for Docker/TensorFlow deployment")
def test_model_loading():
    from src.model_inference import TweetSentimentModel

    # Set up paths to model files
    MODEL_PATH = os.getenv("MODEL_PATH", "config/pretrained-roberta-base.h5")
    CONFIG_PATH = os.getenv("CONFIG_PATH", "config/config-roberta-base.json")
    TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "config/vocab-roberta-base.json")
    MERGES_PATH = os.getenv("MERGES_PATH", "config/merges-roberta-base.txt")
    WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "models/weights_final.h5")

    # Initialize the model to check loading
    model = TweetSentimentModel(
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        tokenizer_path=TOKENIZER_PATH,
        merges_path=MERGES_PATH,
        weights_path=WEIGHTS_PATH
    )
    
    assert model.model is not None, "Expected the TensorFlow model to be loaded"
