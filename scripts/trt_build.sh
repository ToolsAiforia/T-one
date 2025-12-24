#!/usr/bin/env bash
set -euo pipefail

BATCH_SIZE=16
SKIP_PREPROCESSOR=1  # 0 = raw audio (int32 Bx3200x1), 1 = features (fp32 Bx64x40)

MODEL_DIR="$(pwd)/../triton/model/1"

if [[ "${SKIP_PREPROCESSOR}" -eq 1 ]]; then
  # log-mel features: float32 [B, 64, 40] (C, T)
  MIN_SHAPES="audio_signal:1x64x40,length:1,state:1x219729"
  OPT_SHAPES="audio_signal:${BATCH_SIZE}x64x40,length:${BATCH_SIZE},state:${BATCH_SIZE}x219729"
  MAX_SHAPES="audio_signal:${BATCH_SIZE}x64x40,length:${BATCH_SIZE},state:${BATCH_SIZE}x219729"
else
  # raw audio: int32 [B, 3200, 1]
  MIN_SHAPES="audio_signal:1x3200x1,length:1,state:1x219729"
  OPT_SHAPES="audio_signal:${BATCH_SIZE}x3200x1,length:${BATCH_SIZE},state:${BATCH_SIZE}x219729"
  MAX_SHAPES="audio_signal:${BATCH_SIZE}x3200x1,length:${BATCH_SIZE},state:${BATCH_SIZE}x219729"
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
