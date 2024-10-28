# Base image with Triton server and Python 3
FROM nvcr.io/nvidia/tritonserver:23.03-py3

# Install system dependencies and Miniconda
RUN apt-get update && apt-get install -y wget supervisor && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Add Conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Copy the environment.yml file to create Conda environment
COPY environment.yml /app/environment.yml

# Create the Conda environment with TensorFlow and other dependencies
RUN conda env create -f /app/environment.yml && conda clean -afy

# Set working directory
WORKDIR /app

# Copy application code, models, and static files
COPY config /app/config
COPY models /app/models
COPY src /app/src
COPY static /app/static
COPY triton_models /app/triton_models
COPY start_services.sh /app/start_services.sh

# Set environment variables
ENV STATIC_DIR=/app/static \
    MODEL_PATH=/app/config/pretrained-roberta-base.h5 \
    CONFIG_PATH=/app/config/config-roberta-base.json \
    TOKENIZER_PATH=/app/config/vocab-roberta-base.json \
    MERGES_PATH=/app/config/merges-roberta-base.txt \
    WEIGHTS_PATH=/app/models/weights_final.h5

# Expose ports for both FastAPI and Triton
EXPOSE 8000 8001 8002 9000 9001

# Start services using the custom startup script
CMD ["/bin/bash", "/app/start_services.sh"]
