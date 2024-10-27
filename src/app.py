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

# Triton server URL and model configuration
STATIC_DIR = os.getenv('STATIC_DIR', 'static')
TRITON_URL = os.getenv("TRITON_URL", "localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "nlp_model")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")

# Initialize logger
logger = logging.getLogger("uvicorn")

# Define request and response schemas
class PredictionRequest(BaseModel):
    text: str
    sentiment: str

# Preprocess text to suitable input tensors for Triton
def preprocess_text(text: str, sentiment: str, max_len=96):
    ids = np.random.randint(0, 100, size=(1, max_len), dtype=np.int32)
    att = np.ones((1, max_len), dtype=np.int32)
    tok = np.zeros((1, max_len), dtype=np.int32)
    return ids, att, tok

# Define the lifespan context manager for startup and shutdown events
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
    logger.info("Shutting down FastAPI...")

# Create the FastAPI app with the lifespan context
app = FastAPI(lifespan=lifespan)

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

# Define the /predict endpoint
@app.post("/predict")
async def predict(request: PredictionRequest):
    start_time = time.perf_counter()

    try:
        # Instantiate a Triton client for each request
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

        # Perform inference
        response = triton_client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs, model_version=MODEL_VERSION)

        # Extract logits as numpy arrays and find start/end positions
        start_logits = response.as_numpy("activation").flatten()
        end_logits = response.as_numpy("activation_1").flatten()

        # Use np.argmax to safely find scalar indices
        start_idx = int(np.argmax(start_logits))
        end_idx = int(np.argmax(end_logits))

        # Ensure start index is before or equal to end index
        if start_idx > end_idx:
            selected_text = request.text.strip()  # Default to the full text
        else:
            selected_text = request.text[start_idx:end_idx].strip()

        end_time = time.perf_counter()
        latency = end_time - start_time
        logger.info(f"Prediction took {latency:.4f} seconds")
        return {"text": request.text, "selected_text": selected_text}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
