## Testing Overview

### `test_benchmark.py`
This file covers performance and load testing, measuring the prediction latency under different conditions.

- **`test_predict_latency`**: Benchmarks the latency for a single prediction request using a positive sentiment. It ensures the prediction response is within acceptable latency and includes a non-empty `selected_text`.
- **`test_predict_load`**: Conducts load testing by sending multiple payloads with varying text lengths and sentiments to simulate real-world usage. It repeats each test 10 times to observe performance under load.

### `test_gpuusage.py`
This file focuses on monitoring GPU utilization during predictions to confirm that the model utilizes the GPU effectively under the encapsulated deployment.

- **`test_predict_and_check_gpu_usage`**: Sends multiple prediction requests while monitoring GPU usage via `nvidia-smi`. It compares GPU usage before and during prediction to confirm a performance increase.

### `test_integration.py`
This file contains integration tests for key application endpoints and covers various prediction cases, including edge cases.

- **`test_root_endpoint`**: Verifies that the root endpoint `/` successfully returns the main HTML page.
- **`test_predict_empty_text`**: Ensures that an empty text payload returns an empty `selected_text`.
- **`test_predict_endpoint`**: Uses parameterized testing for different sentiment cases, including valid sentiments and an invalid sentiment. It verifies that each sentiment case returns expected `selected_text`.

### `test_tensorflow.py`
This file is focused on validating model loading specifically within the TensorFlow (encapsulated) deployment.

- **`test_model_loading`**: Confirms that the model loads correctly, with no issues, in a TensorFlow environment by checking the model object.

### `test_triton.py`
This file includes tests specific to the Triton deployment, focusing on preprocessing and prediction consistency.

- **`test_triton_preprocessing`**: Validates that the `preprocess_text` function prepares input correctly for Triton, with expected shapes and data types.
- **`test_triton_predict_endpoint`**: Confirms the `/predict` endpoint in Triton returns a successful response containing `selected_text`.

---

Each test file targets a specific aspect of functionality, performance, or environment, ensuring a comprehensive evaluation of the service under varied conditions. This setup helps catch issues early and verifies that the application runs optimally in both encapsulated and Triton deployments.