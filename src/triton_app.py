from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
import time
import logging
import tritonclient.http as httpclient
import numpy as np

from .utils import get_selected_text_from_logits, preprocess_text

# Triton server configuration from environment variables
STATIC_DIR = os.getenv('STATIC_DIR', 'static')
TRITON_URL = os.getenv("TRITON_URL", "localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "nlp_model")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")

# Initialize logger
logger = logging.getLogger("uvicorn")

# Lifespan function to warm up the Triton server client
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Connecting to Triton server and warming up the model...")
    try:
        triton_client = httpclient.InferenceServerClient(url=TRITON_URL)

        # Run a warm-up inference request to initialize the model
        ids, att, tok = preprocess_text("Warm-up text", "neutral")
        inputs = [
            httpclient.InferInput("input_1", ids.shape, "INT32"),
            httpclient.InferInput("input_2", att.shape, "INT32"),
            httpclient.InferInput("input_3", tok.shape, "INT32")
        ]
        outputs = [
            httpclient.InferRequestedOutput("activation"),
            httpclient.InferRequestedOutput("activation_1")
        ]
        inputs[0].set_data_from_numpy(ids)
        inputs[1].set_data_from_numpy(att)
        inputs[2].set_data_from_numpy(tok)

        # Perform inference to warm up the model
        response = triton_client.infer(
            model_name=MODEL_NAME,
            inputs=inputs,
            outputs=outputs,
            model_version=MODEL_VERSION
        )
        logger.info("Warm-up successful, model response received.")
    except Exception as e:
        logger.error(f"Error during warm-up: {e}")

    yield  # Start serving requests

# Initialize FastAPI app with lifespan context
app = FastAPI(lifespan=lifespan)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model for prediction input
class PredictionRequest(BaseModel):
    text: str
    sentiment: str

# Serve main HTML file at the root endpoint
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
        # Initialize Triton client for inference
        triton_client = httpclient.InferenceServerClient(url=TRITON_URL)
        ids, att, tok = preprocess_text(request.text, request.sentiment)

        # Set up inputs and outputs for Triton inference
        inputs = [
            httpclient.InferInput("input_1", ids.shape, "INT32"),
            httpclient.InferInput("input_2", att.shape, "INT32"),
            httpclient.InferInput("input_3", tok.shape, "INT32")
        ]
        outputs = [
            httpclient.InferRequestedOutput("activation"),
            httpclient.InferRequestedOutput("activation_1")
        ]
        inputs[0].set_data_from_numpy(ids)
        inputs[1].set_data_from_numpy(att)
        inputs[2].set_data_from_numpy(tok)

        # Perform inference and extract logits
        response = triton_client.infer(
            model_name=MODEL_NAME,
            inputs=inputs,
            outputs=outputs,
            model_version=MODEL_VERSION
        )
        start_logits = response.as_numpy("activation").flatten()
        end_logits = response.as_numpy("activation_1").flatten()

        # Post-process logits to get selected text
        selected_text = get_selected_text_from_logits(
            request.text, start_logits, end_logits, max_len=96
        )

        latency = time.perf_counter() - start_time
        logger.info(f"Prediction completed in {latency:.4f} seconds")
        return {"text": request.text, "selected_text": selected_text}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction")
