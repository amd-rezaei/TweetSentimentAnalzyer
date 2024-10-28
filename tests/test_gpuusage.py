import os
import time
import subprocess
import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)
DEPLOYMENT_TYPE = os.getenv("DEPLOYMENT_TYPE", "encapsulated").lower()

@pytest.mark.skipif(DEPLOYMENT_TYPE != "encapsulated", reason="Only for GPU encapsulated deployment")
def test_predict_and_check_gpu_usage():
    # Step 1: Define the prediction payload
    payload = {"text": "This is a test for GPU usage.", "sentiment": "positive"}
    
    # Step 2: Run nvidia-smi before prediction to get baseline GPU usage
    print("Running nvidia-smi before prediction...")
    baseline_gpu_utilization = get_gpu_utilization()
    print(f"Baseline GPU Utilization: {baseline_gpu_utilization}%")
    
    # Step 3: Trigger a prediction request and check GPU usage during the prediction
    print("Sending prediction request...")
    start_time = time.time()
    response = client.post("/predict", json=payload)
    prediction_duration = time.time() - start_time
    
    assert response.status_code == 200, "Prediction request failed!"
    
    # Step 4: Run nvidia-smi after prediction to capture GPU usage
    print("Running nvidia-smi after prediction...")
    post_prediction_gpu_utilization = get_gpu_utilization()
    print(f"Post-Prediction GPU Utilization: {post_prediction_gpu_utilization}%")
    
    # Step 5: Validate that GPU usage increased after prediction started
    assert post_prediction_gpu_utilization > baseline_gpu_utilization, \
        "Expected GPU utilization to increase during prediction"

    print(f"Prediction took {prediction_duration:.2f} seconds, and GPU utilization increased as expected.")


def get_gpu_utilization():
    """Helper function to retrieve GPU utilization percentage using nvidia-smi."""
    try:
        # Run nvidia-smi command and capture output
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        # Log the full output for debugging
        full_output = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, text=True)
        print("Full nvidia-smi output:\n", full_output.stdout)
        
        # Parse the GPU utilization percentage (output is a list of values per GPU)
        utilization_values = [int(line) for line in result.stdout.strip().splitlines()]
        return max(utilization_values)  # Return the highest utilization value (handles multi-GPU systems)

    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        return 0
