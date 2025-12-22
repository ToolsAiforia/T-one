BATCH_SIZE=16

docker run --gpus all --rm -it \
  -v $(pwd)/../dev/model_400ms_fixed/streaming_acoustic/1:/models \
  nvcr.io/nvidia/tritonserver:25.06-py3 \
  /usr/src/tensorrt/bin/trtexec \
    --onnx=/models/model.onnx \
    --minShapes=signal:1x3200x1,state:1x219729 \
    --optShapes=signal:"$BATCH_SIZE"x3200x1,state:"$BATCH_SIZE"x219729 \
    --maxShapes=signal:"$BATCH_SIZE"x3200x1,state:"$BATCH_SIZE"x219729 \
    --builderOptimizationLevel=5 \
    --stronglyTyped \
    --useSpinWait \
    --noDataTransfers \
    --saveEngine=/models/model.plan
