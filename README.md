
# Tweet Sentiment Extraction Service

A FastAPI-based web service that deploys a pre-trained sentiment extraction model from the Kaggle "Tweet Sentiment Extraction" competition.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
  - [Local Setup](#local-setup)
  - [Docker Setup](#docker-setup)
- [Running the Service](#running-the-service)
  - [Local Run](#local-run)
  - [Docker Run](#docker-run)
- [API Usage](#api-usage)
  - [Endpoints](#endpoints)
  - [Example Request](#example-request)
- [Testing](#testing)
- [Optimization and Performance](#optimization-and-performance)
- [License](#license)

## Overview
This service provides a **Tweet Sentiment Extraction** API using a pretrained RoBERTa model to extract relevant phrases based on sentiment (positive, negative, neutral). It is built with FastAPI, TensorFlow, and tokenizers, with GPU support enabled through Docker for faster processing.

## Project Structure
```
├── config                   # Model configuration files
├── models                   # Pre-trained model weights
├── src                      # Source code for the FastAPI app and model inference
│   ├── app.py               # Main FastAPI app
│   └── model_inference.py   # Model inference code
├── static                   # Static files (e.g., HTML for UI)
├── tests                    # Unit and integration tests
├── environment.yml          # Conda environment configuration
├── Dockerfile               # Docker configuration for GPU support
└── README.md                # Project documentation
```

## Requirements
- **Python 3.9**
- **CUDA-compatible GPU** (for Dockerized GPU acceleration)
- **CUDA Toolkit** version compatible with TensorFlow
- **Docker** and **NVIDIA Docker** for GPU support

## Setup

### Local Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/amd-rezaei/TweetSentimentExtractor.git
   cd TweetSentimentExtractor
   ```

2. **Setup Conda Environment**:
   ```bash
   conda env create -f environment.yml
   conda activate senta
   ```

3. **Set Environment Variables** (optional):
   Adjust paths in `.env` to customize file locations if necessary.

### Docker Setup
1. **Build the Docker Image**:
   ```bash
   docker build -t tweet-sentiment-service:optimized .
   ```

2. **Run with GPU Support**:
   ```bash
   docker run --gpus all -p 8000:8000 tweet-sentiment-service:optimized
   ```

## Running the Service

### Local Run
Run the FastAPI app with:
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Run
Start the container with:
```bash
docker run --gpus all -p 8000:8000 tweet-sentiment-service:optimized
```

Access the service at `http://localhost:8000`.

## API Usage

### Endpoints
- **POST /predict**: Extracts sentiment-based text from a tweet.
- **GET /**: Returns the main HTML page.

### Example Request
#### Request
```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{"text": "I love the sunny weather!", "sentiment": "positive"}'
```

#### Response
```json
{
  "text": "I love the sunny weather!",
  "selected_text": "love the sunny weather"
}
```

## Testing
1. **Run Tests**:
   ```bash
   PYTHONPATH=. pytest tests/
   ```

2. **Test Docker Image**:
   Ensure the Docker container is running and test with an HTTP request as shown in the example above.

## Optimization and Performance
- **Latency Measurement**: The `/predict` endpoint logs latency by measuring the time taken for prediction.
- **Docker Image Optimization**: Uses multi-stage builds to minimize image size.
- **Model Warm-Up**: The model can be warmed up at startup to reduce initial request latency.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
