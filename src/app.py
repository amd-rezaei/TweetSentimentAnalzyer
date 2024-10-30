import os

# Determine the deployment type based on the DEPLOYMENT_TYPE environment variable, defaulting to "encapsulated"
DEPLOYMENT_TYPE = os.getenv("DEPLOYMENT_TYPE", "encapsulated").strip().lower()

# Import the appropriate FastAPI app module based on the deployment type
if DEPLOYMENT_TYPE == "triton":
    from .triton_app import app  # Triton-based FastAPI app
else:
    from .tf_app import app  # TensorFlow-based FastAPI app
