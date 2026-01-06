# T-one Triton/ONNX WER Notes

This file summarizes the key pieces needed to debug WER differences between the
T-one Triton deployment and the ONNX pipeline.

## Exporting ONNX
- `T-one/scripts/onnx_build.sh` runs:
  `python3 -m tone.scripts.export --path-to-pretrained t-tech/T-one --chunk-duration-ms 400 --skip-preprocessor`
  and writes the model to `T-one/triton/model/1/model.onnx`.
- `T-one/tone/scripts/export.py` exports a fixed-shape streaming model via
  `Tone.forward_for_export`.
  - `skip_preprocessor=False` => input is raw audio int32 `(B, T_samples, 1)`.
  - `skip_preprocessor=True` => input is log-mel float32 `(B, C=64, T_frames)`.
  - Inputs always include: `audio_signal`, `length`, `cache_last_time`,
    `cache_last_channel`, `cache_last_chan_len` (see `ModelToExport.input_sample`).
  - Streaming state packing: `cache_last_time` packs MHSA + conv states; the
    preproc/subsampling/reduction states are packed into `cache_last_channel`
    tail; `cache_last_chan_len` is int64.

## TensorRT engine build
- `T-one/scripts/trt_build.sh` builds `model.plan` from the exported ONNX using
  `trtexec` inside `nvcr.io/nvidia/tritonserver:25.04-py3`.
- It extracts fixed non-batch dims from the ONNX for:
  `audio_signal`, `length`, `cache_last_time`, `cache_last_channel`,
  `cache_last_chan_len`, then builds:
  - MIN shapes for batch 1
  - OPT/MAX shapes for `BATCH_SIZE=64`
- `TRT_STRONGLY_TYPED=1` by default (can be overridden).

## TRT precision results (Jan 2026)
- Best TRT build so far: BF16 with LayerNormalization/Softmax/LogSoftmax forced
  FP32 (no `--stronglyTyped`, `--precisionConstraints=obey`,
  `--layerPrecisions/--layerOutputTypes`, `--noTF32`,
  `--builderOptimizationLevel=0`).
- WER on `test_rec_support` (beam + KenLM, `client_wer.py`): 0.128937.
- Same precision with `--builderOptimizationLevel=5`: WER 0.129429.

## Triton repository layout
- `T-one/triton/ensemble/config.pbtxt` is an ensemble:
  `AUDIO_CHUNK (FP32, 1D)` -> `preprocessing` (DALI) -> `model` (ONNX/TRT).
- DALI preprocessor is at `T-one/triton/preprocessing/1/features_8k_tone.py`.
- Acoustic model artifacts live in `T-one/triton/model/1/` (`model.onnx` or
  `model.plan`).
- `T-one/scripts/docker-compose.yml` mounts `/home/mle/T-one/triton` to `/models`
  and exposes ports 8000/8001/8002.

## WER measurement (Triton)
- Script: `T-one/dev/triton/client_wer.py`.
- Data layout:
  `base-data-path/<dataset-name>/tarred_audio_manifest.json` with `audio_filepath`
  and `text`. The script resolves audio by basename under the dataset folder.
- It forces 400 ms chunks:
  `StreamingCTCModel.AUDIO_CHUNK_SAMPLES = int(0.4 * 8000)`.
- Uses `StreamingCTCPipeline` + `StreamingLogprobSplitter` and decodes with:
  - Greedy or Beam search (`BeamSearchCTCDecoder` uses KenLM).
  - Beam params are in `tone/decoder.py` (beam_width=200, alpha=0.4, beta=0.9).
- Input scaling: raw int16 audio is scaled by `INT16_MAX` before Triton.
- Text normalization: `\\u0451` -> `\\u0435` (`SUB` map in `client_wer.py`).
- Example:
  `poetry run python dev/triton/client_wer.py --base-data-path /home/mle/aiphoria-asr-training/data/rus_finetune --dataset-name test_rec_support --manifest tarred_audio_manifest.json --triton-url localhost:8001 --model-name ensemble --decoder beam --kenlm-path /home/mle/T-one/dev/model/kenlm.bin`

## Model context handling
- ONNX export expects explicit streaming states:
  - `cache_last_time` (MHSA + conv),
  - `cache_last_channel` (sub2 + tail of preproc/sub1/reduction),
  - `cache_last_chan_len` (int64).
- ONNX preprocessor (`Tone.forward_for_export`) expects raw audio int32 and scales
  by `torch.iinfo(torch.int16).max` before `FilterbankFeatures.forward_streaming`.
- The DALI preprocessor is stateless, so Triton streaming emulates context by
  prepending a small left-context window in the client:
  `preproc_left_context=80` samples (10 ms) by default in `client_wer.py`.

## Other nuances / gotchas
- `--skip-preprocessor` export is required when Triton uses DALI preproc;
  monolithic ONNX expects raw int32 audio and performs its own preprocessing.
- DALI feature extractor details: 8 kHz, 20 ms window, 10 ms hop, 64 mel bins,
  preemphasis 0.97, Slaney mel scale.
- DALI preproc casts to fp16 and back to fp32 to match ONNX preproc output
  (see `features_8k_tone.py`).
- WER mismatches often come from inconsistent:
  chunk duration, left-context padding, input scaling, or fp16 quantization.
