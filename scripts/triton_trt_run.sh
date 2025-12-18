docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/../dev/model_400ms_fixed:/models \
  nvcr.io/nvidia/tritonserver:25.06-py3 \
  tritonserver --model-repository=/models
