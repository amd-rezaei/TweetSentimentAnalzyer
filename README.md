
# Tweet Sentiment Extraction Service

A FastAPI-based web service that deploys a pre-trained sentiment extraction model from the Kaggle "Tweet Sentiment Extraction" competition. This service offers two deployment options:
- **Encapsulated FastAPI**: Deploys the model directly within a FastAPI application.
- **NVIDIA Triton Inference Server**: Uses Triton for optimized inference, with a FastAPI client as a proxy.

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
  - [Docker Compose Commands](#docker-compose-commands)
- [API Usage](#api-usage)
  - [Endpoints](#endpoints)
  - [Example Request](#example-request)
- [Testing](#testing)
  - [Testing with Docker Compose](#testing-with-docker-compose)
  - [Manual Testing Commands](#manual-testing-commands)
- [Performance Measurement and Optimization](#performance-measurement-and-optimization)
- [Reports](#reports)
- [License](#license)

## Overview

The **Tweet Sentiment Extraction Service** provides an API for extracting sentiment-based text from tweets. It uses a **pre-trained RoBERTa model** fine-tuned for sentiment extraction, inspired by **Chris Deotte's** approach from the [Kaggle competition](https://www.kaggle.com/c/tweet-sentiment-extraction). The service is built using FastAPI, TensorFlow, and tokenizers, and it supports GPU acceleration through Docker and Triton Inference Server.

## Project Structure

```plaintext
.
├── config                   # Model configuration files
├── data                     # Dataset files
├── docker                   # Docker configurations for deployment
│   ├── docker-compose.yml   # Docker Compose file for multi-container setup
│   ├── Dockerfile.encapsulated # Dockerfile for encapsulated FastAPI deployment
│   └── Dockerfile.triton    # Dockerfile for Triton-based deployment
├── environment.yml          # Conda environment file for encapsulated setup
├── models                   # Pre-trained model weights
├── report                   # Test and benchmark reports
├── requirements.txt         # Python requirements for Triton deployment
├── src                      # Source code for FastAPI application and utilities
├── static                   # HTML UI files
├── tests                    # Test suite for functionality and performance
├── triton_models            # Triton model repository
└── utils                    # Utility scripts (e.g., for model conversion)
```

## Requirements

- **Python 3.9**
- **CUDA-compatible GPU** for Dockerized GPU acceleration
- **CUDA Toolkit** compatible with TensorFlow and Triton
- **Docker** and **NVIDIA Docker** for GPU support

## Setup

### Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/amd-rezaei/TweetSentimentExtractor.git
   cd TweetSentimentExtractor
   ```

2. **Setup Conda Environment** (for encapsulated setup):
   ```bash
   conda env create -f environment.yml
   conda activate senta
   ```

3. **Set Environment Variables** (optional):
   Adjust any paths in `.env` to customize file locations if needed.

### Docker Setup

The project uses two Dockerfiles: `Dockerfile.encapsulated` for a direct FastAPI-based deployment and `Dockerfile.triton` for a Triton-based deployment.

- **Encapsulated Docker Image**:
  - Based on **NVIDIA CUDA 11.8** with **CUDNN** for TensorFlow support.
  - Installs essential tools, Miniconda, and Python 3.10.
  - Sets up the **Conda environment** specified in `environment.yml`.
  - Entrypoint: `start_encapsulated.sh`, which initializes the FastAPI app.
  
- **Triton Docker Image**:
  - Based on **NVIDIA Triton Inference Server** with Python support.
  - Installs `supervisor` for service management and creates a Python virtual environment for dependencies.
  - Entrypoint: `start_triton.sh`, which starts the Triton server and the FastAPI proxy.

To build the images using Docker Compose:
```bash
docker-compose -f docker/docker-compose.yml up --build
```

## Running the Service

### Encapsulated FastAPI

This option deploys the model directly within FastAPI, providing straightforward inference without the additional layer of Triton Inference Server. This setup is best suited for direct model access and lower complexity.

#### Run Command

To deploy the encapsulated FastAPI service, use the following command:

```bash
docker run --gpus all -e DEPLOYMENT_TYPE=encapsulated -p 9001:9001 tweet-sentiment-service:optimized
```

This will start the service, making it accessible at [http://localhost:9001](http://localhost:9001).

### Triton Deployment

This option deploys the model using **NVIDIA Triton Inference Server**, optimized for high-performance model inference. A FastAPI client proxy is also set up to interact with Triton, separating the inference server and client layers.

#### Run Command

To deploy the service with Triton, use the following command:

```bash
docker run --gpus all -e DEPLOYMENT_TYPE=triton -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 9000:9000 tweet-sentiment-service:optimized
```

In this configuration:

- **Triton Server Endpoints**:
  - **Health Check**: [http://localhost:8000/v2/health/ready](http://localhost:8000/v2/health/ready)
  - **Model Repository**: [http://localhost:8000/v2/models](http://localhost:8000/v2/models)
- **FastAPI Proxy**: Accessible at [http://localhost:9000](http://localhost:9000)

These endpoints allow health monitoring and model management, along with the FastAPI proxy for making inference requests.

### Docker Compose Commands

For streamlined deployment and management of both the encapsulated FastAPI and Triton-based services, Docker Compose can be used. The following commands help build, run, and tear down the services efficiently:

1. **Build the Images without Cache**:

   ```bash
   docker-compose -f docker/docker-compose.yml build --no-cache
   ```

2. **Run the Services in Detached Mode**:

   ```bash
   docker-compose -f docker/docker-compose.yml up -d
   ```

3. **Stop and Remove Containers**:

   ```bash
   docker-compose -f docker/docker-compose.yml down
   ```

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

### Testing with Docker Compose

To automate testing on container startup, set `RUN_TESTS_ON_START=true` in your `docker-compose.yml` file. When enabled, this will trigger the entrypoint to automatically run `pytest` on startup.

```yaml
services:
  encapsulated:
    environment:
      - RUN_TESTS_ON_START=true
```

### Manual Testing Commands

You can run tests manually within each container. Below are the commands for both the encapsulated and Triton containers.

#### 1. Encapsulated FastAPI Testing

In the encapsulated FastAPI container, **activate the Conda environment** first, as `pytest` is installed within it. Here’s how:

```bash
docker exec -it <encapsulated_container_name> /bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate senta
pytest /app/tests
```

#### 2. Triton Deployment Testing

For the Triton container, you can directly use `pytest` if it’s installed globally or within a virtual environment. Access the container and run:

```bash
docker exec -it <triton_container_name> pytest /app/tests
```

This verifies the functionality of the service in both deployment environments.

## Performance Measurement and Optimization

### Key Optimizations
- **Latency Measurement**: Logs response time for the `/predict` endpoint, providing data for identifying bottlenecks.
- **Docker Image Optimization**: Multi-stage builds reduce Docker image size, improving deployment times.
- **Model Warm-Up**: Initial inference runs at startup to minimize first-request latency.

#### Additional Conceptual Optimizations
1. **Batch Processing**: Handles multiple predictions in a single batch to reduce redundant computations.
2. **TensorRT Conversion**: Converts the model to TensorRT to improve inference speed and reduce memory consumption.
3. **Cache Frequent Requests**: Implement caching for commonly repeated requests to reduce inference time.

## Reports

Performance and test reports are saved in the `report` directory:
- **report_test_encapsulated.txt**: Results from tests in the encapsulated FastAPI deployment.
- **report_test_triton.txt**: Results from tests in the Triton deployment.

These reports provide detailed insights into latency, resource usage, and API accuracy.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
