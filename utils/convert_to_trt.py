from src.model_inference import TweetSentimentModel
import os 

# Load model configuration paths from environment variables, with fallback defaults
STATIC_DIR = os.getenv('STATIC_DIR', 'static')
MODEL_PATH = os.getenv('MODEL_PATH', 'config/pretrained-roberta-base.h5')
CONFIG_PATH = os.getenv('CONFIG_PATH', 'config/config-roberta-base.json')
TOKENIZER_PATH = os.getenv('TOKENIZER_PATH', 'config/vocab-roberta-base.json')
MERGES_PATH = os.getenv('MERGES_PATH', 'config/merges-roberta-base.txt')
WEIGHTS_PATH = os.getenv('WEIGHTS_PATH', 'models/weights_final.h5')

# Initialize the TweetSentimentModel
model = TweetSentimentModel(
    model_path=MODEL_PATH,
    config_path=CONFIG_PATH,
    tokenizer_path=TOKENIZER_PATH,
    merges_path=MERGES_PATH,
    weights_path=WEIGHTS_PATH
)

# Define the output path for the converted TensorRT model
converted_model_path = "models/weights_final.trt"
os.makedirs(os.path.dirname(converted_model_path), exist_ok=True)

# Convert the model to TensorRT format
model.convert_to_tensorrt(converted_model_path)

print(f"TensorRT model saved at {converted_model_path}")
