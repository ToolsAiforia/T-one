#!/usr/bin/env bash
set -euo pipefail

BATCH_SIZE=64
MODEL_DIR="$(pwd)/../triton/model/1"
ONNX_PATH="${MODEL_DIR}/model.onnx"
TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:25.04-py3}"
TRT_STRONGLY_TYPED="${TRT_STRONGLY_TYPED:-1}"
TRT_STRONGLY_TYPED_FLAG=""
if [[ "${TRT_STRONGLY_TYPED}" == "1" ]]; then
  TRT_STRONGLY_TYPED_FLAG="--stronglyTyped"
fi

if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "ERROR: ONNX model not found at: ${ONNX_PATH}" >&2
  exit 1
fi

export ONNX_PATH=$ONNX_PATH
# Extract fixed input dims (excluding dynamic batch dim) directly from ONNX
eval "$(
python3 - <<'PY'
import os
import onnx

onnx_path = os.environ["ONNX_PATH"]
m = onnx.load(onnx_path)

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

print(f'AUDIO_DIMS="{shapes["audio_signal"]}"')
print(f'LENGTH_DIMS="{shapes["length"]}"')
print(f'TIME_DIMS="{shapes["cache_last_time"]}"')
print(f'CHANNEL_DIMS="{shapes["cache_last_channel"]}"')
print(f'LENSTATE_DIMS="{shapes["cache_last_chan_len"]}"')
PY
)"

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
    --builderOptimizationLevel=5 \
    ${TRT_STRONGLY_TYPED_FLAG} \
    --useSpinWait \
    --noDataTransfers \
    --saveEngine=/models/model.plan
