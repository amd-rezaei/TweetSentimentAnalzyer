# app.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .model_inference import TweetSentimentModel
import os

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
    with open(os.path.join('/app/static', 'index.html')) as f:
        return HTMLResponse(content=f.read(), status_code=200)

# Load the model
model = TweetSentimentModel(
    model_path='/app/config/pretrained-roberta-base.h5',
    config_path='/app/config/config-roberta-base.json',
    tokenizer_path='/app/config/vocab-roberta-base.json',
    merges_path='/app/config/merges-roberta-base.txt',
    weights_path='/app/models/weights_final.h5'
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
