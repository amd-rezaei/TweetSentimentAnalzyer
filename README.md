
# Tweet Sentiment Extraction Service

A FastAPI-based web service that deploys a pre-trained sentiment extraction model from the Kaggle "Tweet Sentiment Extraction" competition. This service can deploy the model in two ways:
- As an encapsulated FastAPI app for direct model inference.
- Using NVIDIA Triton Inference Server with a FastAPI client proxy.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
  - [Local Setup](#local-setup)
  - [Docker Setup](#docker-setup)
- [Running the Service](#running-the-service)
  - [Encapsulated FastAPI](#encapsulated-fastapi)
  - [Triton Deployment](#triton-deployment)
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
.
├── config                   # Model config files
├── data                     # Dataset files
├── Dockerfile               # Docker configuration
├── environment.yml          # Conda environment file
├── model_inference          # Model training/evaluation scripts
├── models                   # Pre-trained model weights
├── src                      # FastAPI app and inference code
│   ├── app.py               # Main app entry point
│   ├── tf_app.py            # TensorFlow deployment app
│   └── triton_app.py        # Triton deployment app
├── start_services.sh        # Dynamic supervisord setup
├── static                   # HTML for UI
├── tests                    # Test suite
├── triton_models            # Triton model repository
└── utils                    # Utility scripts (e.g., model conversion)
```

## Requirements
- **Python 3.9**
- **CUDA-compatible GPU** (for Dockerized GPU acceleration)
- **CUDA Toolkit** version compatible with TensorFlow and Triton
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

## Running the Service

### Encapsulated FastAPI
This runs the model directly within FastAPI, without Triton.
```bash
docker run --gpus all -e DEPLOYMENT_TYPE=encapsulated -p 9001:9001 tweet-sentiment-service:optimized
```
Access the service at [http://localhost:9001](http://localhost:9001).

### Triton Deployment
This deploys the model using NVIDIA Triton Inference Server, with a FastAPI client on port 9000.
```bash
docker run --gpus all -e DEPLOYMENT_TYPE=triton -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 9000:9000 tweet-sentiment-service:optimized
```
- **Triton Server**:
  - Health: [http://localhost:8000/v2/health/ready](http://localhost:8000/v2/health/ready)
  - Models: [http://localhost:8000/v2/models](http://localhost:8000/v2/models)
- **FastAPI Client Proxy**: [http://localhost:9000](http://localhost:9000)

## API Usage

### Endpoints
- **POST /predict**: Extracts sentiment-based text from a tweet.
- **GET /**: Returns the main HTML page.

### Example Request
#### Request
```bash
curl -X POST "http://localhost:9001/predict" -H "Content-Type: application/json" -d '{"text": "I love the sunny weather!", "sentiment": "positive"}'
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
