# Triton Inference Server

Triton is...

### Setup docker images

Pull nividia Triton server image

```bash!
docker pull nvcr.io/nvidia/tritonserver:24.06-py3
```

Pull nvidia Triton client for inference

```bash!
docker pull nvcr.io/nvidia/tritonserver:24.06-py3-sdk
```

### Triton server

Create and run container for Triton server

```bash!
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v d/Documents/GitHub/MY-REPO/triton/model_repository:/models nvcr.io/nvidia/tritonserver:24.06-py3 tritonserver --model-repository=/models
```

Recommend using docker-compose file

```dockerfile!
services:
  triton-server:
    image: nvcr.io/nvidia/tritonserver:24.06-py3
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: tritonserver --model-repository=/models --model-control-mode=explicit --load-model=densenet_onnx
    ports:
      - "8000:8000"
      - "8001:8001"
      - "8002:8002"
    volumes:
      - ../model_repository:/models
    environment:
      - NVIDIA_VISIBLE_DEVICES=1
```

Then enter bash or attach shell in vscode for tracking logging

```bash!
docker-compose ps
docker-compose exec triton-server bash
```

### Triton client

Create and run container for Triton client:

```bash!
docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:24.06-py3-sdk
```

Then in the bash, run the premade file image_client

```bash!
/workspace/install/bin/image_client -m densenet_onnx -c 3 -s INCEPTION /workspace/images/mug.jpg
```

However, we can also create our own client with own `http` or `grpc` protocols:

```python!
import numpy as np
import requests
import json

# Define the server URL
url = "http://localhost:8000/v2/models/densenet_onnx/infer"

# Create input data (example: an array of zeros)
input_data = np.zeros((3, 224, 224), dtype=np.float32)

# Prepare the data in JSON format
inputs = [
    {
        "name": "data_0",
        "shape": input_data.shape,
        "datatype": "FP32",
        "data": input_data.tolist()
    }
]

outputs = [
    {
        "name": "fc6_1"
    }
]

request_payload = {
    "inputs": inputs,
    "outputs": outputs
}

# Send the request to the Triton server
response = requests.post(url, json=request_payload)

# Check the response status
if response.status_code == 200:
    response_json = response.json()
    print(response_json.keys())
    output_data = np.array(response_json["outputs"][0]["data"]).reshape(response_json["outputs"][0]["shape"])
    print("Output Data: ", output_data)

else:
    print("Request failed with status code: ", response.status_code)
    print("Response: ", response.text)
```

Then run this docker-compose.yml in client directory:

```dockerfile!
services:
  triton-client:
    image: nvcr.io/nvidia/tritonserver:24.06-py3-sdk
    network_mode: host
    tty: true
    stdin_open: true
    restart: unless-stopped
    volumes:
      - ../:/workspace/inference/
```

### Model analyzer

Create the output dir first to avoid error

```bash
mkdir output_model/output
```

Run the triton server with above docker compose file. And now run the container for it to automatically connect to triton server

```bash!
docker run -it --gpus all -v /var/run/docker.sock:/var/run/docker.sock -v d/Documents/GitHub/MY-REPO/triton/model_analyzer:/workspace/model_analyzer --net=host nvcr.io/nvidia/tritonserver:24.06-py3-sdk
```

```bash!
model-analyzer profile \
    --model-repository /workspace/model_analyzer/ \
    --profile-models densenet_onnx --triton-launch-mode=remote \
    --output-model-repository-path /workspace/model_analyzer/model_output/output \
    --export-path /workspace/model_analyzer/profile_results \
    --override-output-model-repository
```

If you just want to test with limit experiments, use this:

```bash!
--run-config-search-max-concurrency 2
    --run-config-search-max-model-batch-size 2
    --run-config-search-max-instance-count 2
```
