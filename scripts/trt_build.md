# TensorRT build notes (targeted FP32)

This documents the TensorRT engine build settings that eliminate the WER drift
between TRT and ONNX for the T-one streaming model.

## Root cause summary
- TRT accuracy drift was large when the engine was built with generic FP16
  tactics (and especially with `--stronglyTyped`).
- ONNX backend in Triton matched local ORT closely, so the feature pipeline and
  state handling were correct.
- The fix is mixed precision: keep LayerNorm/Softmax/LogSoftmax in FP32 and
  allow BF16 (best so far) or FP16 everywhere else.

## BF16 mixed-precision (best WER so far)
- Precision: BF16 engine with LayerNormalization/Softmax/LogSoftmax forced FP32.
- Flags: `--bf16 --precisionConstraints=obey --layerPrecisions=<LN/Softmax/LogSoftmax>:fp32
  --layerOutputTypes=<LN/Softmax/LogSoftmax>:fp32 --noTF32 --builderOptimizationLevel=0`
  (no `--stronglyTyped`).
- WER (test_rec_support, beam + KenLM, `client_wer.py`): 0.128937.
- Same precision with `--builderOptimizationLevel=5`: 0.129429.

## Default build in `trt_build.sh`
The script now defaults to the "good" build:
- `TRT_FP32_LAYER_TYPES=LayerNormalization,Softmax,LogSoftmax`
- `TRT_FORCE_FP16_ALL=1` (adds `*:fp16` for all other layers)
- `TRT_FP16=1` (enables FP16 tactics)
- `TRT_STRONGLY_TYPED=0`
- `TRT_NO_TF32=0`
- `TRT_INPUT_IO_FORMATS` and `TRT_OUTPUT_IO_FORMATS` default to linear (`:chw`)
  formats derived from the ONNX input/output dtypes (fixes state tensor format drift
  under sequence batching).
- `BATCH_SIZE=64` (OPT/MAX shapes)
- `--builderOptimizationLevel=0`

The script auto-discovers the ONNX node names for those op types and pins their
layer precision and output type to FP32 via:
- `--precisionConstraints=obey`
- `--layerPrecisions=*:fp16,<layer>:fp32,...`
- `--layerOutputTypes=<layer>:fp32,...`

## Build command
Use a Python that has `onnx` installed so the script can read node names.

```
cd /home/mle/T-one/scripts
PYTHON_BIN=/home/mle/T-one/.venv/bin/python ./trt_build.sh
```

The engine is written to:
`/home/mle/T-one/triton/model/1/model.plan`

## Verified WER (beam + KenLM, explicit states, DALI preproc)
These were measured with `dev/triton/client_wer.py` and the targeted engine:
- test_rec_support: 0.127461
- test_collection: 0.151869
- test_support: 0.194081

## Overrides / troubleshooting
- Disable targeted FP32 layers:
  `TRT_FP32_LAYER_TYPES="" TRT_FORCE_FP16_ALL=0 ./trt_build.sh`
- Add or override a specific layer list:
  `TRT_FP32_LAYERS="/decoder/LogSoftmax,/encoder/layers.0/self_attn/Softmax"`
- If the build fails, set `TRT_FORCE_FP16_ALL=0` to remove the `*:fp16` hard
  constraint and let TRT choose per-layer precision (still pins the listed
  layers to FP32).
- Override IO formats manually (use binding order from the ONNX graph):
  `TRT_INPUT_IO_FORMATS="fp32:chw,int64:chw,fp16:chw,fp16:chw,int64:chw" TRT_OUTPUT_IO_FORMATS="fp32:chw,fp16:chw,fp16:chw,int64:chw" ./trt_build.sh`
