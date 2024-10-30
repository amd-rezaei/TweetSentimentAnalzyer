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

# Load configuration paths from environment variables, with defaults
STATIC_DIR = os.getenv('STATIC_DIR', 'static')
MODEL_PATH = os.getenv('MODEL_PATH', 'config/pretrained-roberta-base.h5')
CONFIG_PATH = os.getenv('CONFIG_PATH', 'config/config-roberta-base.json')
TOKENIZER_PATH = os.getenv('TOKENIZER_PATH', 'config/vocab-roberta-base.json')
MERGES_PATH = os.getenv('MERGES_PATH', 'config/merges-roberta-base.txt')
WEIGHTS_PATH = os.getenv('WEIGHTS_PATH', 'models/weights_final.h5')

# Initialize logger
logger = logging.getLogger("uvicorn")

class PredictionRequest(BaseModel):
    text: str
    sentiment: str

# Initialize sentiment model with provided paths
model = TweetSentimentModel(
    model_path=MODEL_PATH,
    config_path=CONFIG_PATH,
    tokenizer_path=TOKENIZER_PATH,
    merges_path=MERGES_PATH,
    weights_path=WEIGHTS_PATH
)

# Warm-up function for initializing the model
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up: warming up the model")
    try:
        # Perform a warm-up prediction to initialize the model
        result = model.predict("Warm-up text", "neutral")
        logger.info(f"Warm-up prediction completed successfully, result: {result}")
    except Exception as e:
        logger.error(f"Error during warm-up: {e}")

    yield  # Continue to serve requests
    logger.info("Application shutting down...")

# Initialize FastAPI app with lifespan management
app = FastAPI(lifespan=lifespan)

# Enable CORS for all origins (consider restricting in production if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the main static HTML file at the root endpoint
@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open(os.path.join(STATIC_DIR, 'index.html')) as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        logger.error("index.html not found in the specified STATIC_DIR")
        return HTMLResponse(content="index.html not found", status_code=404)

# Prediction endpoint for sentiment analysis
@app.post("/predict")
async def predict(request: PredictionRequest):
    start_time = time.perf_counter()
    try:
        selected_text = model.predict(request.text, request.sentiment)
        latency = time.perf_counter() - start_time
        logger.info(f"Prediction completed in {latency:.4f} seconds")
        return {"text": request.text, "selected_text": selected_text}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": "Prediction failed"}
