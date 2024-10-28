# app_triton.py
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

# Triton server configuration
STATIC_DIR = os.getenv('STATIC_DIR', 'static')
TRITON_URL = os.getenv("TRITON_URL", "localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "nlp_model")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")


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


# Warm-up function
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Connecting to Triton server and warming up...")
    try:
        triton_client = httpclient.InferenceServerClient(url=TRITON_URL)
        ids, att, tok = preprocess_text("Warm-up text", "neutral")
        inputs = [
            httpclient.InferInput("input_1", ids.shape, "INT32"),
            httpclient.InferInput("input_2", att.shape, "INT32"),
            httpclient.InferInput("input_3", tok.shape, "INT32")
        ]
        outputs = [httpclient.InferRequestedOutput("activation"), httpclient.InferRequestedOutput("activation_1")]
        inputs[0].set_data_from_numpy(ids)
        inputs[1].set_data_from_numpy(att)
        inputs[2].set_data_from_numpy(tok)
        triton_client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs, model_version=MODEL_VERSION)
        logger.info("Triton model warm-up complete")
    except Exception as e:
        logger.error(f"Error during warm-up: {e}")

    yield  # Start serving requests

# Serve static files
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(STATIC_DIR, 'index.html')) as f:
        return HTMLResponse(content=f.read(), status_code=200)

# Prediction endpoint
@app.post("/predict")
async def predict(request: PredictionRequest):
    start_time = time.perf_counter()
    try:
        triton_client = httpclient.InferenceServerClient(url=TRITON_URL)
        ids, att, tok = preprocess_text(request.text, request.sentiment)
        inputs = [
            httpclient.InferInput("input_1", ids.shape, "INT32"),
            httpclient.InferInput("input_2", att.shape, "INT32"),
            httpclient.InferInput("input_3", tok.shape, "INT32")
        ]
        outputs = [httpclient.InferRequestedOutput("activation"), httpclient.InferRequestedOutput("activation_1")]
        inputs[0].set_data_from_numpy(ids)
        inputs[1].set_data_from_numpy(att)
        inputs[2].set_data_from_numpy(tok)
        response = triton_client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs, model_version=MODEL_VERSION)

        # Extract logits and compute selected text
        start_logits = response.as_numpy("activation").flatten()
        end_logits = response.as_numpy("activation_1").flatten()
        selected_text = get_selected_text_from_logits(request.text, start_logits, end_logits, max_len=96)

        latency = time.perf_counter() - start_time
        logger.info(f"Prediction took {latency:.4f} seconds")
        return {"text": request.text, "selected_text": selected_text}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
