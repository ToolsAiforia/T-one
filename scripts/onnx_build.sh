#!/bin/bash

python3 -m tone.scripts.export \
    --path-to-pretrained t-tech/T-one \
    --chunk-duration-ms 400 \
    --skip-preprocessor \
    --output_path $(pwd)/../dev/model_400ms_fixed/streaming_acoustic/1/model.onnx
