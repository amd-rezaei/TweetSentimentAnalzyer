import tf2onnx
import tensorflow as tf
import os

from src.model_inference import TweetSentimentModel

STATIC_DIR = os.getenv('STATIC_DIR', 'static')
MODEL_PATH = os.getenv('MODEL_PATH', 'config/pretrained-roberta-base.h5')
CONFIG_PATH = os.getenv('CONFIG_PATH', 'config/config-roberta-base.json')
TOKENIZER_PATH = os.getenv('TOKENIZER_PATH', 'config/vocab-roberta-base.json')
MERGES_PATH = os.getenv('MERGES_PATH', 'config/merges-roberta-base.txt')
WEIGHTS_PATH = os.getenv('WEIGHTS_PATH', 'models/weights_final.h5')

# Initialize the model
modelobject = TweetSentimentModel(
    model_path=MODEL_PATH,
    config_path=CONFIG_PATH,
    tokenizer_path=TOKENIZER_PATH,
    merges_path=MERGES_PATH,
    weights_path=WEIGHTS_PATH
)
# Load your TensorFlow model
model = modelobject.get_model()

# Convert and save as ONNX format
onnx_model, _ = tf2onnx.convert.from_keras(model)
with open("models/final_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
