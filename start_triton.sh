#!/bin/bash

# Function to wait for a specific port to be ready
wait_for_port() {
  local PORT=$1
  echo "Waiting for service on port $PORT..."
  for i in {1..15}; do
    nc -z localhost $PORT && echo "Service on port $PORT is ready!" && return 0
    echo "Port $PORT not ready, retrying..."
    sleep 2
  done
  echo "Service on port $PORT failed to start."
  exit 1
}

# Start Triton Inference Server in the background
echo "Starting Triton Inference Server..."
tritonserver --model-repository=/models --http-port=8000 --grpc-port=8001 --metrics-port=8002 &

# Wait for Triton server to be ready
wait_for_port 8000

# Start FastAPI for Triton client dynamically using src.app:app
echo "Starting FastAPI server for Triton client..."
source /opt/conda/bin/activate senta
export PYTHONPATH="/app:${PYTHONPATH}"
uvicorn src.app:app --host 0.0.0.0 --port 9000 --log-level info &

# Wait for FastAPI to be ready
wait_for_port 9000

# Run tests if requested
if [ "$RUN_TESTS_ON_START" = "true" ]; then
    echo "Running tests with pytest.ini configuration..."
    pytest --maxfail=5 --disable-warnings -c /app/pytest.ini
fi

# Keep the container running by waiting indefinitely
wait
