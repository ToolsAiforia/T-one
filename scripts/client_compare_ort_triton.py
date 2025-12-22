#!/usr/bin/env python3
"""
client_compare_ort_triton.py

Runs streaming inference side-by-side:
  - Local ONNXRuntime (original ONNX model, explicit state)
  - Triton (TensorRT plan) with sequence batching (implicit state)

It:
  - uses the SAME dataset manifest and chunking
  - runs BOTH pipelines chunk-by-chunk on the same audio
  - compares logprobs and state_next tensors (max/mean abs diff, argmax location)
  - computes WER for each path (ORT vs Triton)

Prereqs:
  - Triton model exposes outputs: logprobs + state_next (you already do)
  - Local ONNX model file path is provided

Notes:
  - Triton is driven with mocked counter state:
        0  => sequence_start=True
       -1  => sequence_end=True (used on finalize call)
       >0  => continuation
"""

from __future__ import annotations

import argparse
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from nemo.collections.asr.metrics.wer import word_error_rate
from tqdm.auto import tqdm

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from tone import read_stream_audio, StreamingCTCPipeline, StreamingCTCModel
from tone.decoder import BeamSearchCTCDecoder, GreedyCTCDecoder
from tone.logprob_splitter import StreamingLogprobSplitter


SUB = str.maketrans("ё", "е")


def clean_text(s: str) -> str:
    return s.lower().translate(SUB)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def resolve_audio_path(base_data_path: Path, dataset_name: str, meta: Dict[str, Any]) -> Path:
    return base_data_path / dataset_name / Path(meta["audio_filepath"]).name


def _crc32(a: np.ndarray) -> int:
    return zlib.crc32(a.view(np.uint8))


def _absdiff_stats(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, Tuple[int, ...], float, float]:
    """
    Returns: (max_abs, mean_abs, argmax_index, a_at_argmax, b_at_argmax)
    """
    af = a.astype(np.float32, copy=False)
    bf = b.astype(np.float32, copy=False)
    d = np.abs(af - bf)
    max_abs = float(d.max())
    mean_abs = float(d.mean())
    arg = int(d.reshape(-1).argmax())
    arg_idx = np.unravel_index(arg, d.shape)
    a_val = float(af[arg_idx])
    b_val = float(bf[arg_idx])
    return max_abs, mean_abs, tuple(int(x) for x in arg_idx), a_val, b_val


class RecordingORTModel:
    """
    Wraps tone.onnx_wrapper.StreamingCTCModel and records last outputs.
    """

    def __init__(self, ort_model: StreamingCTCModel):
        self.model = ort_model
        self.last_logprobs: Optional[np.ndarray] = None
        self.last_state_next: Optional[np.ndarray] = None

    def forward(self, signal: np.ndarray, state: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        logprobs, state_next = self.model.forward(signal, state)
        self.last_logprobs = logprobs
        self.last_state_next = state_next
        return logprobs, state_next


class TritonImplicitStateModel:
    """
    Triton implicit-state sequence batching wrapper for StreamingCTCPipeline.

    forward(signal, counter) returns (logprobs, next_counter)
    but also records last_state_next (fused state) for comparison.
    """

    def __init__(
        self,
        *,
        triton_url: str,
        model_name: str,
        input_signal_name: str = "signal",
        output_logprobs_name: str = "logprobs",
        output_state_name: str = "state_next",
        ssl: bool = False,
        verbose: bool = False,
    ) -> None:
        self.client = grpcclient.InferenceServerClient(url=triton_url, verbose=verbose, ssl=ssl)
        self.model_name = model_name
        self.input_signal_name = input_signal_name
        self.output_logprobs_name = output_logprobs_name
        self.output_state_name = output_state_name

        self._next_sequence_id = 1
        self._active_sequence_id: Optional[int] = None

        self.last_logprobs: Optional[np.ndarray] = None
        self.last_state_next: Optional[np.ndarray] = None

    def forward(self, signal: np.ndarray, counter: Optional[int]) -> Tuple[np.ndarray, int]:
        if counter is None:
            counter = 0
        if not isinstance(counter, (int, np.integer)):
            raise TypeError(f"Expected counter int, got {type(counter)}: {counter!r}")

        is_start = (int(counter) == 0)
        is_end = (int(counter) == -1)

        if is_start or self._active_sequence_id is None:
            self._active_sequence_id = self._next_sequence_id
            self._next_sequence_id += 1

        sid = int(self._active_sequence_id)

        if signal.dtype != np.int32:
            signal = signal.astype(np.int32, copy=False)
        if signal.ndim != 3 or signal.shape[2] != 1:
            raise ValueError(f"Expected (B, chunk, 1) int32, got {signal.shape} {signal.dtype}")

        inp_signal = grpcclient.InferInput(self.input_signal_name, list(signal.shape), "INT32")
        inp_signal.set_data_from_numpy(signal)

        outputs = [
            grpcclient.InferRequestedOutput(self.output_logprobs_name),
            grpcclient.InferRequestedOutput(self.output_state_name),
        ]

        try:
            res = self.client.infer(
                model_name=self.model_name,
                inputs=[inp_signal],
                outputs=outputs,
                sequence_id=sid,
                sequence_start=is_start,
                sequence_end=is_end,
            )
        except InferenceServerException as e:
            raise RuntimeError(f"Triton infer failed: {e}") from e

        logprobs = res.as_numpy(self.output_logprobs_name)
        state_next = res.as_numpy(self.output_state_name)

        if logprobs is None:
            raise RuntimeError(f"No output {self.output_logprobs_name!r}")
        if state_next is None:
            raise RuntimeError(f"No output {self.output_state_name!r} (is it listed in output[] in config.pbtxt?)")

        self.last_logprobs = logprobs
        self.last_state_next = state_next

        next_counter = 0 if is_end else (int(counter) + 1)
        if is_end:
            self._active_sequence_id = None

        return logprobs, next_counter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--base-data-path", type=str, required=True)
    p.add_argument("--dataset-name", type=str, default="test_rec_support")
    p.add_argument("--manifest", type=str, default="tarred_audio_manifest.json")
    p.add_argument("--max-items", type=int, default=0, help="0=all")

    # Local ORT model
    p.add_argument("--onnx-model", type=str, required=True, help="Path to local model.onnx")
    p.add_argument(
        "--onnx-providers",
        type=str,
        default="CUDAExecutionProvider",
        help='Comma-separated ORT providers, e.g. "CUDAExecutionProvider" or "CPUExecutionProvider"',
    )

    # Triton
    p.add_argument("--triton-url", type=str, default="localhost:8001")
    p.add_argument("--triton-model-name", type=str, default="streaming_acoustic")
    p.add_argument("--input-signal-name", type=str, default="signal")
    p.add_argument("--output-logprobs-name", type=str, default="logprobs")
    p.add_argument("--output-state-name", type=str, default="state_next")
    p.add_argument("--ssl", action="store_true")
    p.add_argument("--verbose", action="store_true")

    # Decoder
    p.add_argument("--decoder", choices=["greedy", "beam"], default="beam")
    p.add_argument("--kenlm-path", type=str, default="", help="Path to kenlm.bin for beam decoder")

    # Compare controls
    p.add_argument("--compare-utts", type=int, default=2, help="How many utterances to print per-chunk diffs for")
    p.add_argument("--compare-chunks", type=int, default=10, help="How many chunks per compared utterance")
    p.add_argument("--lp-atol", type=float, default=1e-3, help="Flag logprobs mismatch if max_abs > lp-atol")
    p.add_argument("--state-atol", type=float, default=1e-3, help="Flag state mismatch if max_abs > state-atol")
    p.add_argument("--dump-dir", type=str, default="", help="If set, dump tensors when mismatch triggers")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    dump_dir = Path(args.dump_dir) if args.dump_dir else None
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)

    # Match your 0.4s chunking
    StreamingCTCModel.AUDIO_CHUNK_SAMPLES = int(0.4 * StreamingCTCModel.SAMPLE_RATE)
    StreamingCTCPipeline.CHUNK_SIZE = StreamingCTCModel.AUDIO_CHUNK_SAMPLES

    # ORT model + pipeline
    ort_providers = [p.strip() for p in args.onnx_providers.split(",") if p.strip()]
    ort_model = StreamingCTCModel.from_local(Path(args.onnx_model), providers=ort_providers)
    ort_model_rec = RecordingORTModel(ort_model)

    # Triton model + pipeline
    triton_model = TritonImplicitStateModel(
        triton_url=args.triton_url,
        model_name=args.triton_model_name,
        input_signal_name=args.input_signal_name,
        output_logprobs_name=args.output_logprobs_name,
        output_state_name=args.output_state_name,
        ssl=args.ssl,
        verbose=args.verbose,
    )

    # Decoder (use SAME artifact path to avoid confounds)
    if args.decoder == "greedy":
        decoder = GreedyCTCDecoder()
    else:
        if not args.kenlm_path:
            raise SystemExit("For a fair comparison, pass --kenlm-path to the same kenlm.bin you used in baseline.")
        decoder = BeamSearchCTCDecoder.from_local(Path(args.kenlm_path))

    # Two independent splitters (must NOT share state)
    splitter_ort = StreamingLogprobSplitter()
    splitter_triton = StreamingLogprobSplitter()

    pipeline_ort = StreamingCTCPipeline(ort_model_rec, splitter_ort, decoder)
    pipeline_triton = StreamingCTCPipeline(triton_model, splitter_triton, decoder)

    base_data_path = Path(args.base_data_path)
    manifest_path = base_data_path / args.dataset_name / args.manifest
    jdata = load_jsonl(manifest_path)
    if args.max_items and args.max_items > 0:
        jdata = jdata[: args.max_items]

    hypos_ort: List[str] = []
    hypos_triton: List[str] = []
    gts: List[str] = []

    for utt_i, meta in enumerate(tqdm(jdata, desc="Utterances", unit="utt")):
        audio_path = resolve_audio_path(base_data_path, args.dataset_name, meta)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        # ORT pipeline state: None initially (explicit state tensor is created by ORT wrapper)
        state_ort = None
        phrases_ort: List[str] = []

        # Triton pipeline state: mocked counter + splitter state
        state_triton = (0, None)
        phrases_triton: List[str] = []

        for chunk_i, audio_chunk in enumerate(read_stream_audio(path_to_file=audio_path, chunk_size=pipeline_triton.CHUNK_SIZE)):
            out_triton, state_triton = pipeline_triton.forward(audio_chunk, state_triton)
            out_ort, state_ort = pipeline_ort.forward(audio_chunk, state_ort)

            if out_triton:
                phrases_triton.extend([p.text for p in out_triton])
            if out_ort:
                phrases_ort.extend([p.text for p in out_ort])

            # Compare only for first N utterances / chunks
            if utt_i < args.compare_utts and chunk_i < args.compare_chunks:
                lp_t = triton_model.last_logprobs
                st_t = triton_model.last_state_next
                lp_o = ort_model_rec.last_logprobs
                st_o = ort_model_rec.last_state_next

                if lp_t is None or lp_o is None or st_t is None or st_o is None:
                    print(f"[DBG] utt={utt_i} chunk={chunk_i}: missing tensors (lp/st).")
                    continue

                lp_max, lp_mean, lp_idx, lp_oval, lp_tval = _absdiff_stats(lp_o, lp_t)
                st_max, st_mean, st_idx, st_oval, st_tval = _absdiff_stats(st_o, st_t)

                print(
                    f"[CMP] utt={utt_i} chunk={chunk_i} "
                    f"LP: max={lp_max:.6e} mean={lp_mean:.6e} idx={lp_idx} "
                    f"ORT={lp_oval:.6f} TRT={lp_tval:.6f} "
                    f"crc32(ORT/TRT)={_crc32(lp_o)}/{_crc32(lp_t)} | "
                    f"STATE: max={st_max:.6e} mean={st_mean:.6e} idx={st_idx} "
                    f"ORT={st_oval:.6f} TRT={st_tval:.6f} "
                    f"crc32(ORT/TRT)={_crc32(st_o)}/{_crc32(st_t)}"
                )

                mismatch = (lp_max > args.lp_atol) or (st_max > args.state_atol)
                if mismatch and dump_dir:
                    tag = f"utt{utt_i:04d}_chunk{chunk_i:04d}"
                    np.save(dump_dir / f"{tag}.lp_ort.npy", lp_o)
                    np.save(dump_dir / f"{tag}.lp_trt.npy", lp_t)
                    np.save(dump_dir / f"{tag}.st_ort.npy", st_o)
                    np.save(dump_dir / f"{tag}.st_trt.npy", st_t)

        # Finalize
        # ORT: normal finalize
        out_ort_final, _ = pipeline_ort.finalize(state_ort)
        if out_ort_final:
            phrases_ort.extend([p.text for p in out_ort_final])

        # Triton: set end sentinel for finalize call (preserve splitter state)
        state_triton = (-1, state_triton[1])
        out_triton_final, _ = pipeline_triton.finalize(state_triton)
        if out_triton_final:
            phrases_triton.extend([p.text for p in out_triton_final])

        pred_ort = clean_text(" ".join(phrases_ort))
        pred_triton = clean_text(" ".join(phrases_triton))
        gt = clean_text(str(meta.get("text", "")))

        hypos_ort.append(pred_ort)
        hypos_triton.append(pred_triton)
        gts.append(gt)

    wer_ort = word_error_rate(hypos_ort, gts)
    wer_triton = word_error_rate(hypos_triton, gts)

    print(f"WER (ORT local ONNX):   {wer_ort:.6f}")
    print(f"WER (Triton TRT plan): {wer_triton:.6f}")


if __name__ == "__main__":
    main()
