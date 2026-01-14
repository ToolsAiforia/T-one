#!/usr/bin/env bash
set -euo pipefail

BATCH_SIZE="${BATCH_SIZE:-64}"
MODEL_DIR="$(pwd)/../triton/model/1"
ONNX_PATH="${MODEL_DIR}/model.onnx"
TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:25.04-py3}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

TRT_STRONGLY_TYPED="${TRT_STRONGLY_TYPED:-0}"
TRT_FP32_LAYER_TYPES="${TRT_FP32_LAYER_TYPES:-LayerNormalization,Softmax,LogSoftmax}"
TRT_FP32_LAYERS="${TRT_FP32_LAYERS:-}"
TRT_FORCE_FP16_ALL="${TRT_FORCE_FP16_ALL:-0}"
TRT_FORCE_BF16_ALL="${TRT_FORCE_BF16_ALL:-1}"
TRT_FP16="${TRT_FP16:-0}"
TRT_BF16="${TRT_BF16:-1}"
TRT_NO_TF32="${TRT_NO_TF32:-1}"
TRT_OPT_LEVEL="${TRT_OPT_LEVEL:-5}"
TRT_INPUT_IO_FORMATS="${TRT_INPUT_IO_FORMATS:-}"
TRT_OUTPUT_IO_FORMATS="${TRT_OUTPUT_IO_FORMATS:-}"

if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "ERROR: ONNX model not found at: ${ONNX_PATH}" >&2
  exit 1
fi

export ONNX_PATH=$ONNX_PATH
# Extract fixed input dims (excluding dynamic batch dim) directly from ONNX
eval "$(
FP32_LAYER_TYPES="${TRT_FP32_LAYER_TYPES}" "${PYTHON_BIN}" - <<'PY'
import os
import onnx
import shlex

onnx_path = os.environ["ONNX_PATH"]
layer_types = [t for t in os.environ.get("FP32_LAYER_TYPES", "").split(",") if t]
m = onnx.load(onnx_path)
initializer_names = {init.name for init in m.graph.initializer}

def emit(name: str, value: str) -> None:
    print(f"{name}={shlex.quote(value)}")

def get_dims(name: str):
    for inp in m.graph.input:
        if inp.name == name:
            t = inp.type.tensor_type
            dims = []
            for d in t.shape.dim:
                if d.dim_value > 0:
                    dims.append(int(d.dim_value))
                elif d.dim_param:
                    dims.append(d.dim_param)
                else:
                    dims.append(None)
            return dims
    raise SystemExit(f"Missing input in ONNX graph: {name}")

def dims_ex_batch(dims):
    # Expect batch as first dim; remove it for trtexec shape string construction
    if len(dims) == 0:
        return []
    return dims[1:]

required = [
    "audio_signal",
    "length",
    "cache_last_time",
    "cache_last_channel",
    "cache_last_chan_len",
]

shapes = {}
for name in required:
    dims = get_dims(name)
    rest = dims_ex_batch(dims)
    # All non-batch dims should be concrete ints for this export style.
    out = []
    for x in rest:
        if isinstance(x, int):
            out.append(str(x))
        else:
            raise SystemExit(f"Non-batch dim for {name} is not a fixed int: {dims}")
    shapes[name] = "x".join(out)

def trt_dtype(onnx_dtype: int) -> str:
    mapping = {
        onnx.TensorProto.FLOAT: "fp32",
        onnx.TensorProto.FLOAT16: "fp16",
        onnx.TensorProto.BFLOAT16: "bf16",
        onnx.TensorProto.INT64: "int64",
        onnx.TensorProto.INT32: "int32",
        onnx.TensorProto.INT8: "int8",
    }
    if onnx_dtype in mapping:
        return mapping[onnx_dtype]
    raise SystemExit(f"Unsupported ONNX dtype: {onnx_dtype}")

def io_formats(values):
    out = []
    for v in values:
        if v.name in initializer_names:
            continue
        t = v.type.tensor_type
        out.append(f"{trt_dtype(t.elem_type)}:chw")
    return out

fp32_layers = []
seen = set()
for node in m.graph.node:
    if node.op_type in layer_types:
        name = node.name or (node.output[0] if node.output else "")
        if name and name not in seen:
            seen.add(name)
            fp32_layers.append(name)

emit("AUDIO_DIMS", shapes["audio_signal"])
emit("LENGTH_DIMS", shapes["length"])
emit("TIME_DIMS", shapes["cache_last_time"])
emit("CHANNEL_DIMS", shapes["cache_last_channel"])
emit("LENSTATE_DIMS", shapes["cache_last_chan_len"])
emit("FP32_LAYERS_AUTO", ",".join(fp32_layers))
emit("INPUT_IO_FORMATS_AUTO", ",".join(io_formats(m.graph.input)))
emit("OUTPUT_IO_FORMATS_AUTO", ",".join(io_formats(m.graph.output)))
PY
)"

if [[ -z "${TRT_FP32_LAYERS}" ]]; then
  TRT_FP32_LAYERS="${FP32_LAYERS_AUTO}"
fi

if [[ -z "${TRT_INPUT_IO_FORMATS}" ]]; then
  TRT_INPUT_IO_FORMATS="${INPUT_IO_FORMATS_AUTO}"
fi

if [[ -z "${TRT_OUTPUT_IO_FORMATS}" ]]; then
  TRT_OUTPUT_IO_FORMATS="${OUTPUT_IO_FORMATS_AUTO}"
fi

make_shape() {
  local b="$1"
  local rest="$2"
  if [[ -z "${rest}" ]]; then
    echo "${b}"
  else
    echo "${b}x${rest}"
  fi
}

MIN_SHAPES="audio_signal:$(make_shape 1 "${AUDIO_DIMS}"),length:$(make_shape 1 "${LENGTH_DIMS}"),cache_last_time:$(make_shape 1 "${TIME_DIMS}"),cache_last_channel:$(make_shape 1 "${CHANNEL_DIMS}"),cache_last_chan_len:$(make_shape 1 "${LENSTATE_DIMS}")"
OPT_SHAPES="audio_signal:$(make_shape "${BATCH_SIZE}" "${AUDIO_DIMS}"),length:$(make_shape "${BATCH_SIZE}" "${LENGTH_DIMS}"),cache_last_time:$(make_shape "${BATCH_SIZE}" "${TIME_DIMS}"),cache_last_channel:$(make_shape "${BATCH_SIZE}" "${CHANNEL_DIMS}"),cache_last_chan_len:$(make_shape "${BATCH_SIZE}" "${LENSTATE_DIMS}")"
MAX_SHAPES="${OPT_SHAPES}"

TRT_LAYER_PRECISIONS=""
TRT_LAYER_OUTPUT_TYPES=""
if [[ "${TRT_FORCE_FP16_ALL}" == "1" && "${TRT_FORCE_BF16_ALL}" == "1" ]]; then
  echo "ERROR: TRT_FORCE_FP16_ALL and TRT_FORCE_BF16_ALL are mutually exclusive" >&2
  exit 1
fi

if [[ "${TRT_FORCE_BF16_ALL}" == "1" ]]; then
  TRT_LAYER_PRECISIONS="*:bf16"
  TRT_BF16="1"
elif [[ "${TRT_FORCE_FP16_ALL}" == "1" ]]; then
  TRT_LAYER_PRECISIONS="*:fp16"
  TRT_FP16="1"
fi

if [[ -n "${TRT_FP32_LAYERS}" ]]; then
  IFS=',' read -r -a FP32_LAYER_LIST <<< "${TRT_FP32_LAYERS}"
  for layer in "${FP32_LAYER_LIST[@]}"; do
    if [[ -z "${layer}" ]]; then
      continue
    fi
    if [[ -n "${TRT_LAYER_PRECISIONS}" ]]; then
      TRT_LAYER_PRECISIONS+=","
    fi
    TRT_LAYER_PRECISIONS+="${layer}:fp32"
    if [[ -n "${TRT_LAYER_OUTPUT_TYPES}" ]]; then
      TRT_LAYER_OUTPUT_TYPES+=","
    fi
    TRT_LAYER_OUTPUT_TYPES+="${layer}:fp32"
  done
fi

TRT_EXTRA_FLAGS=()
if [[ "${TRT_FP16}" == "1" ]]; then
  TRT_EXTRA_FLAGS+=(--fp16)
fi
if [[ "${TRT_BF16}" == "1" ]]; then
  TRT_EXTRA_FLAGS+=(--bf16)
fi
if [[ "${TRT_NO_TF32}" == "1" ]]; then
  TRT_EXTRA_FLAGS+=(--noTF32)
fi
if [[ "${TRT_STRONGLY_TYPED}" == "1" ]]; then
  TRT_EXTRA_FLAGS+=(--stronglyTyped)
fi
if [[ -n "${TRT_INPUT_IO_FORMATS}" ]]; then
  TRT_EXTRA_FLAGS+=(--inputIOFormats="${TRT_INPUT_IO_FORMATS}")
fi
if [[ -n "${TRT_OUTPUT_IO_FORMATS}" ]]; then
  TRT_EXTRA_FLAGS+=(--outputIOFormats="${TRT_OUTPUT_IO_FORMATS}")
fi
if [[ -n "${TRT_LAYER_PRECISIONS}" || -n "${TRT_LAYER_OUTPUT_TYPES}" ]]; then
  TRT_EXTRA_FLAGS+=(--precisionConstraints=obey)
  if [[ -n "${TRT_LAYER_PRECISIONS}" ]]; then
    TRT_EXTRA_FLAGS+=(--layerPrecisions="${TRT_LAYER_PRECISIONS}")
  fi
  if [[ -n "${TRT_LAYER_OUTPUT_TYPES}" ]]; then
    TRT_EXTRA_FLAGS+=(--layerOutputTypes="${TRT_LAYER_OUTPUT_TYPES}")
  fi
fi

docker run --gpus all --rm -i \
  -v "${MODEL_DIR}:/models" \
  -e MIN_SHAPES="${MIN_SHAPES}" \
  -e OPT_SHAPES="${OPT_SHAPES}" \
  -e MAX_SHAPES="${MAX_SHAPES}" \
  "${TRITON_IMAGE}" \
  /usr/src/tensorrt/bin/trtexec \
    --onnx=/models/model.onnx \
    --minShapes="${MIN_SHAPES}" \
    --optShapes="${OPT_SHAPES}" \
    --maxShapes="${MAX_SHAPES}" \
    --builderOptimizationLevel="${TRT_OPT_LEVEL}" \
    "${TRT_EXTRA_FLAGS[@]}" \
    --useSpinWait \
    --noDataTransfers \
    --saveEngine=/models/model.plan
