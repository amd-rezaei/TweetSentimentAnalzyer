# Use a CUDA-compatible base image for TensorFlow with GPU support
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Install Miniconda
RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda init

# Add Conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Set working directory
WORKDIR /app

# Copy the environment file
COPY environment.yml /app/environment.yml

# Copy required files for the app
COPY config /app/config
COPY models /app/models
COPY src /app/src
COPY static /app/static

# Create the Conda environment
RUN conda env create -f /app/environment.yml

# Ensure the shell uses Conda environment
SHELL ["conda", "run", "-n", "senta", "/bin/bash", "-c"]

# Expose the FastAPI port
EXPOSE 8000

# Run FastAPI app with uvicorn in the Conda environment
CMD ["conda", "run", "--no-capture-output", "-n", "senta", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
