# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from model_inference import TweetSentimentModel

# Define the request body structure using Pydantic
class PredictionRequest(BaseModel):
    text: str

# Initialize FastAPI app
app = FastAPI()

# Load model
model = TweetSentimentModel(
    model_path='../config/pretrained-roberta-base.h5',
    config_path='../config/config-roberta-base.json',
    tokenizer_path='../config/vocab-roberta-base.json',
    merges_path='../config/merges-roberta-base.txt',
    weights_path='../models/weights_final.h5'
)

# Define the prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    text = request.text
    
    # Make prediction (both selected text and sentiment)
    selected_text, predicted_sentiment = model.predict(text)

    return {
        'text': text,
        'selected_text': selected_text,
        'predicted_sentiment': predicted_sentiment
    }
