# app.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .model_inference import TweetSentimentModel
import os

# Load paths from environment variables with fallback to default paths
STATIC_DIR = os.getenv('STATIC_DIR', 'static') 


MODEL_PATH = os.getenv('MODEL_PATH', 'config/pretrained-roberta-base.h5')
CONFIG_PATH = os.getenv('CONFIG_PATH', 'config/config-roberta-base.json')
TOKENIZER_PATH = os.getenv('TOKENIZER_PATH', 'config/vocab-roberta-base.json')
MERGES_PATH = os.getenv('MERGES_PATH', 'config/merges-roberta-base.txt')
WEIGHTS_PATH = os.getenv('WEIGHTS_PATH', 'models/weights_final.h5')

class PredictionRequest(BaseModel):
    text: str
    sentiment: str

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from the "static" directory
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(STATIC_DIR, 'index.html')) as f:
        return HTMLResponse(content=f.read(), status_code=200)



# Load the model
model = TweetSentimentModel(
    model_path=MODEL_PATH,
    config_path=CONFIG_PATH,
    tokenizer_path=TOKENIZER_PATH,
    merges_path=MERGES_PATH,
    weights_path=WEIGHTS_PATH
)

# Define the /predict endpoint
@app.post("/predict")
async def predict(request: PredictionRequest):
    text = request.text
    sentiment = request.sentiment
    selected_text = model.predict(text, sentiment)
    return {
        'text': text,
        'selected_text': selected_text
    }
