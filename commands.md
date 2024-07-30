## Model Anlayzer

### Thorough analyzer

model-analyzer profile --model-repository /workspace/inference/model_analyzer/ --profile-models text_reg_batch --triton-launch-mode=remote --output-model-repository-path /workspace/inference/model_analyzer/text_reg_batch/output --export-path /workspace/inference/model_analyzer/text_reg_batch/profile_results --override-output-model-repository --triton-grpc-endpoint --triton-http-endpoint

### Quick analyzer

#### Mode online default

model-analyzer profile --model-repository /workspace/inference/model_analyzer/ --profile-models text_reg_batch --triton-launch-mode=remote --output-model-repository-path /workspace/inference/model_analyzer/text_recognition/output --export-path /workspace/inference/model_analyzer/text_recognition/profile_results --override-output-model-repository --run-config-search-max-concurrency 2 --run-config-search-max-model-batch-size 2 --run-config-search-max-instance-count 2

model-analyzer profile --model-repository /workspace/inference/model_analyzer/ --profile-models text_recognition --triton-launch-mode=remote --output-model-repository-path /workspace/inference/model_analyzer/text_recognition/output --export-path /workspace/inference/model_analyzer/text_recognition/profile_results --run-config-search-mode quick --checkpoint-directory=/workspace/inference/model_analyzer/text_recognition/checkpoints --override-output-model-repository --run-config-search-max-concurrency 2 --run-config-search-max-model-batch-size 2 --run-config-search-max-instance-count 2

## Perf Analyzer

perf-analyzer -m text_reg_batch -b 2 --shape input.1:1,32,100 --concurrency-range 2:16:2 --percentile=95

## Docker server quick run

docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v d/Documents/GitHub/MY-REPO/triton:/workspace/ -v d/Documents/GitHub/MY-REPO/triton/model_repository:/models nvcr.io/nvidia/tritonserver:24.06-py3

## Triton server run

tritonserver --model-repository=/models --model-control-mode=explicit --load-model=\*

### Load and unload model for debugging

curl -X POST localhost:8000/v2/repository/models/text_reg_batch/load
curl -X POST localhost:8000/v2/repository/models/text_reg_batch/unload

### Temp container for stable diffusion

docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /var/run/docker.sock:/var/run/docker.sock d/Documents/GitHub/MY-REPO/triton:/workspace/ -v d/Documents/GitHub/MY-REPO/triton/model_repository:/models nvcr.io/nvidia/tritonserver:24.06-py3 bash

### Temp container for model_anlyzer pip installing docker mode

docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /var/run/docker.sock:/var/run/docker.sock -v d/Documents/GitHub/MY-REPO/triton/model_repository:/models nvcr.io/nvidia/tritonserver:24.06-py3
