#!/usr/bin/env bash
set -euo pipefail

BATCH_SIZE=16
SKIP_PREPROCESSOR=1  # 0 = raw audio (int32 Bx3200x1), 1 = features (fp16 Bx40x64)

MODEL_DIR="$(pwd)/../dev/model_400ms_fixed/streaming_acoustic/1"

if [[ "${SKIP_PREPROCESSOR}" -eq 1 ]]; then
  # log-mel features: float16 [B, 40, 64]
  MIN_SHAPES="signal:1x40x64,state:1x219729"
  OPT_SHAPES="signal:${BATCH_SIZE}x40x64,state:${BATCH_SIZE}x219729"
  MAX_SHAPES="signal:${BATCH_SIZE}x40x64,state:${BATCH_SIZE}x219729"
else
  # raw audio: int32 [B, 3200, 1]
  MIN_SHAPES="signal:1x3200x1,state:1x219729"
  OPT_SHAPES="signal:${BATCH_SIZE}x3200x1,state:${BATCH_SIZE}x219729"
  MAX_SHAPES="signal:${BATCH_SIZE}x3200x1,state:${BATCH_SIZE}x219729"
fi

docker run --gpus all --rm -it \
  -v "${MODEL_DIR}:/models" \
  nvcr.io/nvidia/tritonserver:25.06-py3 \
  /usr/src/tensorrt/bin/trtexec \
    --onnx=/models/model.onnx \
    --minShapes="${MIN_SHAPES}" \
    --optShapes="${OPT_SHAPES}" \
    --maxShapes="${MAX_SHAPES}" \
    --builderOptimizationLevel=5 \
    --stronglyTyped \
    --useSpinWait \
    --noDataTransfers \
    --saveEngine=/models/model.plan
