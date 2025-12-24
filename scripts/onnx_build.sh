#!/bin/bash

python3 -m tone.scripts.export \
    --path-to-pretrained t-tech/T-one \
    --chunk-duration-ms 400 \
    --skip-preprocessor \
    --output_path $(pwd)/../triton/model/1/model.onnx
