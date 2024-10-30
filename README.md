
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
├── report                   # Test and benchmark reports, including Report.md
├── requirements.txt         # Python requirements for Triton deployment
├── src                      # Source code for FastAPI application and utilities
├── static                   # HTML UI files
├── tests                    # Test suite for functionality and performance, includes Tests.md
├── triton_models            # Triton model repository
└── utils                    # Utility scripts (e.g., for model conversion)
```

## Requirements

- **Python 3.10**
- **CUDA-compatible GPU** for Dockerized GPU acceleration 
- **CUDA Toolkit** compatible with TensorFlow and Triton, tested on Cuda11.8
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
docker-compose -f docker/docker-compose.yml up -d encapsulated
```

This will start the service, making it accessible at [http://localhost:9001](http://localhost:9001).

### Triton Deployment

This option deploys the model using **NVIDIA Triton Inference Server**, optimized for high-performance model inference. A FastAPI client proxy is also set up to interact with Triton, separating the inference server and client layers.

#### Run Command

To deploy the service with Triton, use the following command:

```bash
docker-compose -f docker/docker-compose.yml up -d triton
```

This will start the service, making it accessible at [http://localhost:9000](http://localhost:9000)

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
Replace to "http://localhost:9000/predict" for Encapsulated version.

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
pytest 
```

#### 2. Triton Deployment Testing

For the Triton container, you can directly use `pytest` if it’s installed globally or within a virtual environment. Access the container and run:

```bash
docker exec -it <triton_container_name> pytest
```

This verifies the functionality of the service in both deployment environments. More details can be found in `tests/Tests.md`.


## Performance Measurement and Optimization

### Key Optimizations
- **Latency Measurement**: Tracks response time for `/predict` to identify bottlenecks.
- **Docker Image Optimization**: Multi-stage builds reduce image size and improve deployment time.
- **Model Warm-Up**: Initial inference at startup minimizes first-request latency.

### Additional Conceptual Optimizations
1. **Batch Processing**: Batching reduces redundant computations for high-throughput scenarios.
2. **TensorRT Conversion**: Improves inference speed and reduces memory usage with TensorRT.
3. **Cache Frequent Requests**: Caches common queries to reduce repeated computation.

### Future Optimizations
1. **Enhanced Concurrency and Dynamic Batching**: For Triton, enabling dynamic batching optimizes handling of high volumes of concurrent requests. FastAPI's asynchronous design already supports concurrency, but additional tuning can maximize connection limits.
2. **Mixed Precision**: Using FP16 precision reduces memory usage and improves processing speed.
3. **Distributed Model Serving**: Load balancing across instances or GPUs for high traffic.
4. **Model Distillation**: Creates lighter model versions for faster inference on limited resources.


## Reports

Performance and benchmark comparisons between TensorFlow and TensorRT can be found in `report/Report.md`, with additional test insights in `report_test_encapsulated.txt` and `report_test_triton.txt`.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
