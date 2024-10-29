# test_tensorflow.py

import os
import pytest


@pytest.mark.skipif(os.getenv("DEPLOYMENT_TYPE") == "triton", reason="Only for TensorFlow deployment")
def test_model_loading():
    from src.model_inference import TweetSentimentModel
    
    model = TweetSentimentModel(
        model_path=os.getenv("MODEL_PATH", "config/pretrained-roberta-base.h5"),
        config_path=os.getenv("CONFIG_PATH", "config/config-roberta-base.json"),
        tokenizer_path=os.getenv("TOKENIZER_PATH", "config/vocab-roberta-base.json"),
        merges_path=os.getenv("MERGES_PATH", "config/merges-roberta-base.txt"),
        weights_path=os.getenv("WEIGHTS_PATH", "models/weights_final.h5")
    )
    assert model.model is not None

# @pytest.mark.skipif(os.getenv("DEPLOYMENT_TYPE") == "triton", reason="Only for TensorFlow deployment")
# def test_tensorflow_predict():
#     model = TweetSentimentModel(
#         model_path=os.getenv("MODEL_PATH"),
#         config_path=os.getenv("CONFIG_PATH"),
#         tokenizer_path=os.getenv("TOKENIZER_PATH"),
#         merges_path=os.getenv("MERGES_PATH"),
#         weights_path=os.getenv("WEIGHTS_PATH")
#     )
#     sample_text = "TensorFlow test"
#     selected_text = model.predict(sample_text, "positive")
#     assert isinstance(selected_text, str)
