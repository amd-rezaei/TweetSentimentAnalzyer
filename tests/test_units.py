# tests/test_unit.py

from src.model_inference import TweetSentimentModel
import os
import pytest

# Set up paths to model files (ensure these paths are correct for your setup)
MODEL_PATH = os.getenv("MODEL_PATH", "config/pretrained-roberta-base.h5")
CONFIG_PATH = os.getenv("CONFIG_PATH", "config/config-roberta-base.json")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "config/vocab-roberta-base.json")
MERGES_PATH = os.getenv("MERGES_PATH", "config/merges-roberta-base.txt")
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "models/weights_final.h5")

# Initialize the model for testing
model = TweetSentimentModel(
    model_path=MODEL_PATH,
    config_path=CONFIG_PATH,
    tokenizer_path=TOKENIZER_PATH,
    merges_path=MERGES_PATH,
    weights_path=WEIGHTS_PATH
)

@pytest.mark.parametrize("text,sentiment,expected", [
    ("I love this!", "positive", "love"),
    ("This is terrible.", "negative", "terrible"),
    ("It's okay, nothing special.", "neutral", "okay"),
    ("", "positive", ""),  # Edge case: empty text
    ("This product is amazing!", "invalid_sentiment", None)  # Edge case: invalid sentiment
])
def test_predict(text, sentiment, expected):
    try:
        result = model.predict(text, sentiment)
        if sentiment == "invalid_sentiment":
            assert result is None
        else:
            assert expected.lower() in result.lower()
    except Exception as e:
        if sentiment == "invalid_sentiment":
            assert isinstance(e, KeyError)
        else:
            raise
