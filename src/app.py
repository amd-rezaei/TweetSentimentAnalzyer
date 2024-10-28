import os

# Determine the deployment type based on the DEPLOYMENT_TYPE environment variable
DEPLOYMENT_TYPE = os.getenv("DEPLOYMENT_TYPE", "docker").lower()

# Import the correct app module based on DEPLOYMENT_TYPE
if DEPLOYMENT_TYPE == "triton":
    from .triton_app import app  # Triton-based FastAPI app
else:
    from .tf_app import app  # TensorFlow-based FastAPI app
