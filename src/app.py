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
import tokenizers

# Triton server URL and model configuration
STATIC_DIR = os.getenv('STATIC_DIR', 'static')
TRITON_URL = os.getenv("TRITON_URL", "localhost:8000")  # Triton on 8000
MODEL_NAME = os.getenv("MODEL_NAME", "nlp_model")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
MAX_LEN = 96

# Initialize Triton client and tokenizer
triton_client = httpclient.InferenceServerClient(url=TRITON_URL)
tokenizer = tokenizers.ByteLevelBPETokenizer("config/vocab-roberta-base.json", "config/merges-roberta-base.txt")
sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}

# Initialize logger
logger = logging.getLogger("uvicorn")

class PredictionRequest(BaseModel):
    text: str
    sentiment: str

# Preprocess text to suitable input tensors for Triton
def preprocess(text: str, sentiment: str):
    input_ids = np.ones((1, MAX_LEN), dtype='int32')
    attention_mask = np.zeros((1, MAX_LEN), dtype='int32')
    token_type_ids = np.zeros((1, MAX_LEN), dtype='int32')

    # Clean and encode the input text
    text = " " + " ".join(text.split())
    enc = tokenizer.encode(text)
    sentiment_token = sentiment_id[sentiment]
    available_length = MAX_LEN - 5
    token_ids = enc.ids[:available_length]

    # Construct input arrays with special tokens and sentiment ID
    input_ids[0, :len(token_ids) + 5] = [0] + token_ids + [2, 2] + [sentiment_token] + [2]
    attention_mask[0, :len(token_ids) + 5] = 1

    return input_ids, attention_mask, token_type_ids

# Define the lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Connecting to Triton server and warming up...")
    try:
        ids, att, tok = preprocess("Warm-up text", "neutral")
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
    yield
    logger.info("Shutting down FastAPI...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(STATIC_DIR, 'index.html')) as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/predict")
async def predict(request: PredictionRequest):
    start_time = time.perf_counter()

    try:
        # Preprocess and prepare data for Triton
        input_ids, attention_mask, token_type_ids = preprocess(request.text, request.sentiment)
        inputs = [
            httpclient.InferInput("input_1", input_ids.shape, "INT32"),
            httpclient.InferInput("input_2", attention_mask.shape, "INT32"),
            httpclient.InferInput("input_3", token_type_ids.shape, "INT32")
        ]
        outputs = [httpclient.InferRequestedOutput("activation"), httpclient.InferRequestedOutput("activation_1")]
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)
        inputs[2].set_data_from_numpy(token_type_ids)

        # Send request to Triton and receive response
        response = triton_client.infer(
            model_name=MODEL_NAME,
            inputs=inputs,
            outputs=outputs,
            model_version=MODEL_VERSION
        )

        # Post-process to get selected text
        start_logits = response.as_numpy("activation").flatten()
        end_logits = response.as_numpy("activation_1").flatten()
        a = np.argmax(start_logits)
        b = np.argmax(end_logits)
        
        if a > b:
            selected_text = request.text.strip()
        else:
            enc = tokenizer.encode(" " + " ".join(request.text.split()))
            b = min(b + 1, len(enc.ids))
            selected_text = tokenizer.decode(enc.ids[max(a - 1, 0):b]).strip()

        end_time = time.perf_counter()
        latency = end_time - start_time
        logger.info(f"Prediction took {latency:.4f} seconds")
        return {"text": request.text, "selected_text": selected_text}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
