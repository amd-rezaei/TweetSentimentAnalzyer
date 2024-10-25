# Stage 1: Base image with CUDA and Miniconda installation
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04 AS base

# Install Miniconda
RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda init && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Add Conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Set working directory
WORKDIR /app

# Stage 2: Environment creation and dependency installation
FROM base AS builder

# Copy the environment.yml file to create Conda environment
COPY environment.yml /app/environment.yml

# Create the Conda environment and clean up package cache
RUN conda env create -f /app/environment.yml && conda clean -afy

# Copy only the Conda environment without extra files to the final stage
RUN echo "source activate senta" > ~/.bashrc

# Stage 3: Final optimized image
FROM base AS final

# Copy Conda environment from builder stage to reduce layers and image size
COPY --from=builder /opt/conda /opt/conda

# Set working directory
WORKDIR /app

# Copy application code and configuration files
COPY config /app/config
COPY models /app/models
COPY src /app/src
COPY static /app/static

# Set environment variables
ENV STATIC_DIR=/app/static \
    MODEL_PATH=/app/config/pretrained-roberta-base.h5 \
    CONFIG_PATH=/app/config/config-roberta-base.json \
    TOKENIZER_PATH=/app/config/vocab-roberta-base.json \
    MERGES_PATH=/app/config/merges-roberta-base.txt \
    WEIGHTS_PATH=/app/models/weights_final.h5

# Activate Conda environment
SHELL ["conda", "run", "-n", "senta", "/bin/bash", "-c"]

# Expose the FastAPI port
EXPOSE 8000

# Command to run the FastAPI app with uvicorn in the production environment
CMD ["bash", "-c", "source activate senta && uvicorn src.app:app --host 0.0.0.0 --port 8000 --log-level info"]
