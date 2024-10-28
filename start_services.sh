#!/bin/bash

# Function to check if the server is ready
function wait_for_server() {
  local PORT=$1
  echo "Waiting for FastAPI server to be ready on port $PORT..."
  
  # Check if the server is available on the specified port using nc
  for i in {1..15}; do
    nc -z localhost $PORT && echo "FastAPI server is ready!" && return 0
    echo "FastAPI server not ready yet...retrying in 2 seconds"
    sleep 2
  done

  echo "Error: FastAPI server did not become ready on port $PORT within the timeout period."
  exit 1
}

# Generate supervisord.conf based on DEPLOYMENT_TYPE
if [ "$DEPLOYMENT_TYPE" = "triton" ]; then
    cat <<EOF > /etc/supervisor/conf.d/supervisord.conf
[supervisord]
nodaemon=true

[program:tritonserver]
command=tritonserver --model-repository=/app/triton_models --http-port=8000 --grpc-port=8001 --metrics-port=8002
autostart=true
autorestart=true
stderr_logfile=/var/log/tritonserver.err.log
stdout_logfile=/var/log/tritonserver.out.log

[program:triton_fastapi_client]
command=bash -c "source /opt/conda/bin/activate senta && uvicorn src.app:app --host 0.0.0.0 --port 9000 --log-level info"
autostart=true
autorestart=true
stderr_logfile=/var/log/triton_client.err.log
stdout_logfile=/var/log/triton_client.out.log
EOF

elif [ "$DEPLOYMENT_TYPE" = "encapsulated" ]; then
    cat <<EOF > /etc/supervisor/conf.d/supervisord.conf
[supervisord]
nodaemon=true

[program:encapsulated_fastapi]
command=bash -c "source /opt/conda/bin/activate senta && uvicorn src.app:app --host 0.0.0.0 --port 9001 --log-level info"
autostart=true
autorestart=true
stderr_logfile=/var/log/encapsulated_fastapi.err.log
stdout_logfile=/var/log/encapsulated_fastapi.out.log
EOF

else
    echo "Error: DEPLOYMENT_TYPE must be either 'triton' or 'encapsulated'"
    exit 1
fi

# Start supervisord
echo "Starting supervisord..."
/usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf

# Wait for the appropriate server based on DEPLOYMENT_TYPE
if [ "$DEPLOYMENT_TYPE" = "encapsulated" ]; then
    wait_for_server 9001
elif [ "$DEPLOYMENT_TYPE" = "triton" ]; then
    wait_for_server 9000
fi

# Run tests if requested
if [ "$RUN_TESTS_ON_START" = "true" ]; then
    echo "Running tests..."
    source /opt/conda/bin/activate senta && pytest --maxfail=5 --disable-warnings
fi
