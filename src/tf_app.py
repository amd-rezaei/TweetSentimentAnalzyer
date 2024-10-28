# app_docker.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .model_inference import TweetSentimentModel
import os
import time
from contextlib import asynccontextmanager
import logging

# Load paths from environment variables
STATIC_DIR = os.getenv('STATIC_DIR', 'static')
MODEL_PATH = os.getenv('MODEL_PATH', 'config/pretrained-roberta-base.h5')
CONFIG_PATH = os.getenv('CONFIG_PATH', 'config/config-roberta-base.json')
TOKENIZER_PATH = os.getenv('TOKENIZER_PATH', 'config/vocab-roberta-base.json')
MERGES_PATH = os.getenv('MERGES_PATH', 'config/merges-roberta-base.txt')
WEIGHTS_PATH = os.getenv('WEIGHTS_PATH', 'models/weights_final.h5')

# Initialize logger
logger = logging.getLogger("uvicorn")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    text: str
    sentiment: str

# Initialize the model
model = TweetSentimentModel(
    model_path=MODEL_PATH,
    config_path=CONFIG_PATH,
    tokenizer_path=TOKENIZER_PATH,
    merges_path=MERGES_PATH,
    weights_path=WEIGHTS_PATH
)

# Warm-up function
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up... warming up the model")
    model.predict("Warm-up text", "neutral")
    yield  # Start serving requests
    logger.info("Shutting down...")

# Serve static files
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(STATIC_DIR, 'index.html')) as f:
        return HTMLResponse(content=f.read(), status_code=200)

# Prediction endpoint
@app.post("/predict")
async def predict(request: PredictionRequest):
    start_time = time.perf_counter()
    selected_text = model.predict(request.text, request.sentiment)
    latency = time.perf_counter() - start_time
    logger.info(f"Prediction took {latency:.4f} seconds")
    return {"text": request.text, "selected_text": selected_text}
