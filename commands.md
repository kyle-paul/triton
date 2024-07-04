Pull nividia Triton server image

```bash
docker pull nvcr.io/nvidia/tritonserver:24.06-py3
```

Pull nvidia Triton client for inference

```bash
docker pull nvcr.io/nvidia/tritonserver:24.06-py3-sdk
```

Create and run container for Triton server

```bash
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v d/Documents/GitHub/MY-REPO/triton/model_repository:/models nvcr.io/nvidia/tritonserver:24.06-py3 tritonserver --model-repository=/models
```

Recommend using docker-compose file and enter bash without VScode attachment:

```bash
docker-compose ps
docker-compose exec triton-server bash
```

Create and run container for Triton client:

```bash
docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:24.06-py3-sdk
```

Then in the bash

```bash
/workspace/install/bin/image_client -m densenet_onnx -c 3 -s INCEPTION /workspace/images/mug.jpg
```
