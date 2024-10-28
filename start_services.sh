#!/bin/bash

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
/usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
