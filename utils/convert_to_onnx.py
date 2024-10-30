import tf2onnx
import tensorflow as tf
import os

from src.model_inference import TweetSentimentModel

# Load environment variables for model configuration paths
STATIC_DIR = os.getenv('STATIC_DIR', 'static')
MODEL_PATH = os.getenv('MODEL_PATH', 'config/pretrained-roberta-base.h5')
CONFIG_PATH = os.getenv('CONFIG_PATH', 'config/config-roberta-base.json')
TOKENIZER_PATH = os.getenv('TOKENIZER_PATH', 'config/vocab-roberta-base.json')
MERGES_PATH = os.getenv('MERGES_PATH', 'config/merges-roberta-base.txt')
WEIGHTS_PATH = os.getenv('WEIGHTS_PATH', 'models/weights_final.h5')

# Initialize and load the sentiment model
model_object = TweetSentimentModel(
    model_path=MODEL_PATH,
    config_path=CONFIG_PATH,
    tokenizer_path=TOKENIZER_PATH,
    merges_path=MERGES_PATH,
    weights_path=WEIGHTS_PATH
)
model = model_object.get_model()

# Convert the TensorFlow model to ONNX format
onnx_model, _ = tf2onnx.convert.from_keras(model)

# Ensure output directory exists
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

# Save the ONNX model
output_path = os.path.join(output_dir, "final_model.onnx")
with open(output_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"ONNX model saved at {output_path}")
