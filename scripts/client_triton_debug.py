#!/usr/bin/env python3
"""
client_triton_debug.py — sequential streaming eval against Triton sequence batching + WER,
with optional state debugging.

Key behavior (matches your intent):
  - Uses Triton sequence batching (implicit server-side state).
  - Pipeline "model state" is mocked as an integer counter:
        0  => start of sequence  (sequence_start=True)
       -1  => end of sequence    (sequence_end=True)
       >0  => continuation
  - Logprob-splitter state is kept normally by StreamingCTCPipeline.

Debug features:
  - Optional self-test to validate:
        (a) outputs/state evolve across steps in the same sequence
        (b) outputs/state reset on a new sequence
  - Optional per-chunk debug prints (CRC32 fingerprints, norms, dtypes, shapes)
  - Optional dumping of logprobs/state_next to .npy for offline diffing

Important:
  - To debug state tensors, Triton must expose "state_next" as an output
    in config.pbtxt (see section 2 / 3 below).
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


def _crc32_of_ndarray(a: np.ndarray) -> int:
    # Stable fingerprint over raw bytes
    view = a.view(np.uint8)
    return zlib.crc32(view)


@dataclass
class InferDebug:
    sequence_id: int
    sequence_start: bool
    sequence_end: bool
    logprobs_shape: Tuple[int, ...]
    logprobs_dtype: str
    logprobs_crc32: int
    logprobs_min: float
    logprobs_max: float
    logprobs_mean: float
    state_next_present: bool
    state_next_shape: Optional[Tuple[int, ...]] = None
    state_next_dtype: Optional[str] = None
    state_next_crc32: Optional[int] = None
    state_next_nonzero: Optional[int] = None
    state_next_mean_abs: Optional[float] = None
    state_next_max_abs: Optional[float] = None


class TritonStreamingCTCModel:
    """
    Interface-compatible wrapper for StreamingCTCPipeline:

      logprobs, next_counter = model.forward(signal, counter)

    Where:
      signal: np.ndarray (B, chunk, 1) int32
      counter: int
          0  => start sequence
         -1  => end sequence (typically used on finalize call)
         >0  => continuation

    Triton holds the real state internally via sequence batching.
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
        debug_state: bool = False,
        debug_dump_dir: str = "",
    ) -> None:
        self.model_name = model_name
        self.input_signal_name = input_signal_name
        self.output_logprobs_name = output_logprobs_name
        self.output_state_name = output_state_name

        self.client = grpcclient.InferenceServerClient(url=triton_url, verbose=verbose, ssl=ssl)

        self._next_sequence_id = 1
        self._active_sequence_id: Optional[int] = None

        self.debug_state = bool(debug_state)
        self.debug_dump_dir = Path(debug_dump_dir) if debug_dump_dir else None
        if self.debug_dump_dir:
            self.debug_dump_dir.mkdir(parents=True, exist_ok=True)

        # Populated after each infer
        self.last_debug: Optional[InferDebug] = None

        # Detect whether server will let us request state_next
        self._state_next_requestable = self._detect_state_next_requestable()

    def _detect_state_next_requestable(self) -> bool:
        # We try metadata first; if it fails, we fall back to "try and see" during infer.
        try:
            md = self.client.get_model_metadata(self.model_name)
            output_names = []
            try:
                output_names = [o.name for o in md.outputs]
            except Exception:
                # Some environments return dict-like
                output_names = [o["name"] for o in md.get("outputs", [])]  # type: ignore[attr-defined]

            return self.output_state_name in output_names
        except Exception:
            return False

    def print_server_and_model_info(self) -> None:
        # Best-effort; does not raise if unavailable.
        try:
            smd = self.client.get_server_metadata()
            print("=== Triton Server Metadata ===")
            print(smd)
        except Exception as e:
            print(f"[WARN] Could not fetch server metadata: {e}")

        try:
            md = self.client.get_model_metadata(self.model_name)
            print("=== Model Metadata ===")
            print(md)
        except Exception as e:
            print(f"[WARN] Could not fetch model metadata for {self.model_name!r}: {e}")

        try:
            cfg = self.client.get_model_config(self.model_name)
            print("=== Model Config ===")
            print(cfg)
        except Exception as e:
            print(f"[WARN] Could not fetch model config for {self.model_name!r}: {e}")

        if self.debug_state and not self._state_next_requestable:
            print(
                f"[WARN] debug_state enabled but output {self.output_state_name!r} not visible in model metadata.\n"
                f"       If you want state fingerprints, add {self.output_state_name!r} to output[] in config.pbtxt."
            )

    def _infer(
        self,
        signal: np.ndarray,
        *,
        sequence_id: int,
        sequence_start: bool,
        sequence_end: bool,
        dump_tag: str = "",
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not isinstance(signal, np.ndarray):
            raise TypeError(f"signal must be np.ndarray, got {type(signal)}")
        if signal.dtype != np.int32:
            signal = signal.astype(np.int32, copy=False)
        if signal.ndim != 3 or signal.shape[2] != 1:
            raise ValueError(f"Expected signal shape (B, chunk, 1), got {signal.shape}")

        inp_signal = grpcclient.InferInput(self.input_signal_name, list(signal.shape), "INT32")
        inp_signal.set_data_from_numpy(signal)

        requested_outputs = [grpcclient.InferRequestedOutput(self.output_logprobs_name)]

        want_state = bool(self.debug_state)
        if want_state and self._state_next_requestable:
            requested_outputs.append(grpcclient.InferRequestedOutput(self.output_state_name))

        try:
            res = self.client.infer(
                model_name=self.model_name,
                inputs=[inp_signal],
                outputs=requested_outputs,
                sequence_id=sequence_id,
                sequence_start=sequence_start,
                sequence_end=sequence_end,
            )
        except InferenceServerException as e:
            # If requesting state_next fails, disable and retry once (common during partial configs)
            if want_state and self._state_next_requestable:
                msg = str(e)
                if self.output_state_name in msg and ("unknown output" in msg.lower() or "not found" in msg.lower()):
                    print(
                        f"[WARN] Server rejected requested output {self.output_state_name!r}. "
                        "Disabling state output requests for the rest of the run."
                    )
                    self._state_next_requestable = False
                    return self._infer(
                        signal,
                        sequence_id=sequence_id,
                        sequence_start=sequence_start,
                        sequence_end=sequence_end,
                        dump_tag=dump_tag,
                    )
            raise

        logprobs = res.as_numpy(self.output_logprobs_name)
        if logprobs is None:
            raise RuntimeError(f"Triton returned no output '{self.output_logprobs_name}'")

        state_next = None
        if want_state and self._state_next_requestable:
            state_next = res.as_numpy(self.output_state_name)
            # If output is defined but not returned for some reason
            if state_next is None:
                print(
                    f"[WARN] Requested {self.output_state_name!r} but got None. "
                    "Disabling state output requests for the rest of the run."
                )
                self._state_next_requestable = False

        # Optional dumping for offline comparisons
        if self.debug_dump_dir and dump_tag:
            np.save(self.debug_dump_dir / f"{dump_tag}.logprobs.npy", logprobs)
            if state_next is not None:
                np.save(self.debug_dump_dir / f"{dump_tag}.state_next.npy", state_next)

        return logprobs, state_next

    def forward(self, signal: np.ndarray, state_counter: Optional[int]) -> Tuple[np.ndarray, int]:
        """
        Called by StreamingCTCPipeline. Returns:
          logprobs, next_counter
        """
        if state_counter is None:
            # Allow calling with None, treat as sequence start
            state_counter = 0

        if not isinstance(state_counter, (int, np.integer)):
            raise TypeError(
                f"Expected mocked state_counter as int (0.. or -1), got {type(state_counter)}: {state_counter!r}"
            )

        is_start = (int(state_counter) == 0)
        is_end = (int(state_counter) == -1)

        if is_start or (self._active_sequence_id is None):
            self._active_sequence_id = self._next_sequence_id
            self._next_sequence_id += 1

        sid = int(self._active_sequence_id)

        logprobs, state_next = self._infer(
            signal,
            sequence_id=sid,
            sequence_start=is_start,
            sequence_end=is_end,
        )

        # Populate last_debug
        dbg = InferDebug(
            sequence_id=sid,
            sequence_start=is_start,
            sequence_end=is_end,
            logprobs_shape=tuple(logprobs.shape),
            logprobs_dtype=str(logprobs.dtype),
            logprobs_crc32=_crc32_of_ndarray(logprobs),
            logprobs_min=float(np.min(logprobs)),
            logprobs_max=float(np.max(logprobs)),
            logprobs_mean=float(np.mean(logprobs)),
            state_next_present=state_next is not None,
        )
        if state_next is not None:
            # Light stats; avoid heavy conversions
            abs_state = np.abs(state_next)
            dbg.state_next_shape = tuple(state_next.shape)
            dbg.state_next_dtype = str(state_next.dtype)
            dbg.state_next_crc32 = _crc32_of_ndarray(state_next)
            dbg.state_next_nonzero = int(np.count_nonzero(state_next))
            dbg.state_next_mean_abs = float(np.mean(abs_state))
            dbg.state_next_max_abs = float(np.max(abs_state))

        self.last_debug = dbg

        # advance mocked counter unless ending
        next_counter = 0 if is_end else (int(state_counter) + 1)
        if is_end:
            self._active_sequence_id = None

        return logprobs, next_counter

    def debug_self_test(self, chunk_size: int, *, atol_logprob_reset: float = 0.0) -> None:
        """
        Quick sanity test:
          - same dummy input, step1 vs step2 in SAME sequence: should differ if state is active
          - step1 in NEW sequence should match step1 in old sequence (reset)
        """
        dummy = np.zeros((1, chunk_size, 1), dtype=np.int32)

        sid1 = self._next_sequence_id
        self._next_sequence_id += 1

        lp1, st1 = self._infer(dummy, sequence_id=sid1, sequence_start=True, sequence_end=False, dump_tag="selftest_s1_step1")
        lp2, st2 = self._infer(dummy, sequence_id=sid1, sequence_start=False, sequence_end=True, dump_tag="selftest_s1_step2")

        # New sequence
        sid2 = self._next_sequence_id
        self._next_sequence_id += 1
        lp1b, st1b = self._infer(dummy, sequence_id=sid2, sequence_start=True, sequence_end=True, dump_tag="selftest_s2_step1")

        # Compare
        maxdiff_12 = float(np.max(np.abs(lp2 - lp1)))
        maxdiff_reset = float(np.max(np.abs(lp1b - lp1)))

        print("=== Self-test (sequence state sanity) ===")
        print(f"step1 (seq {sid1}) logprobs: shape={lp1.shape} dtype={lp1.dtype} crc32={_crc32_of_ndarray(lp1)}")
        print(f"step2 (seq {sid1}) logprobs: shape={lp2.shape} dtype={lp2.dtype} crc32={_crc32_of_ndarray(lp2)}")
        print(f"step1 (seq {sid2}) logprobs: shape={lp1b.shape} dtype={lp1b.dtype} crc32={_crc32_of_ndarray(lp1b)}")
        print(f"max|step2-step1| in same seq: {maxdiff_12:.6e}")
        print(f"max|step1_newseq-step1| reset: {maxdiff_reset:.6e}")

        if maxdiff_12 == 0.0:
            print("[WARN] step2 == step1 for identical input in same sequence. This often indicates state is NOT applied.")
        if atol_logprob_reset == 0.0:
            if maxdiff_reset != 0.0:
                print("[WARN] step1 in new sequence differs from step1 in old sequence for identical input. Reset may be broken.")
        else:
            if maxdiff_reset > atol_logprob_reset:
                print(
                    f"[WARN] reset diff {maxdiff_reset:.6e} exceeds atol {atol_logprob_reset:.6e}. "
                    "This can happen with non-deterministic kernels; use state_next fingerprints if possible."
                )

        if st1 is not None and st2 is not None and st1b is not None:
            st1_crc = _crc32_of_ndarray(st1)
            st2_crc = _crc32_of_ndarray(st2)
            st1b_crc = _crc32_of_ndarray(st1b)
            print(f"state_next step1 crc32={st1_crc}, step2 crc32={st2_crc}, step1_newseq crc32={st1b_crc}")
            if st2_crc == st1_crc:
                print("[WARN] state_next did not change between step1 and step2 in same sequence.")
            if st1b_crc != st1_crc:
                print("[WARN] state_next for first step differs across fresh sequences for identical input (reset may be broken).")
        else:
            print("[INFO] state_next not available (not exposed as output). Only logprobs-based checks were run.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--base-data-path", type=str, required=True)
    p.add_argument("--dataset-name", type=str, default="test_rec_support")
    p.add_argument("--manifest", type=str, default="tarred_audio_manifest.json")
    p.add_argument("--max-items", type=int, default=0, help="0 = all")

    # Triton
    p.add_argument("--triton-url", type=str, default="localhost:8001")
    p.add_argument("--model-name", type=str, default="streaming_acoustic")
    p.add_argument("--input-signal-name", type=str, default="signal")
    p.add_argument("--output-logprobs-name", type=str, default="logprobs")
    p.add_argument("--output-state-name", type=str, default="state_next")
    p.add_argument("--ssl", action="store_true")
    p.add_argument("--verbose", action="store_true")

    # Decoder
    p.add_argument("--decoder", choices=["greedy", "beam"], default="beam")
    p.add_argument("--kenlm-path", type=str, default="", help="Path to kenlm.bin for beam decoder")

    # Debug
    p.add_argument("--debug-print-model-info", action="store_true", help="Print server/model metadata + config")
    p.add_argument("--debug-state", action="store_true", help="Request state_next (if exposed) and print fingerprints")
    p.add_argument("--debug-self-test", action="store_true", help="Run a short sequence reset/advance self-test")
    p.add_argument("--debug-utts", type=int, default=2, help="How many utterances to print per-chunk debug for (0=none)")
    p.add_argument("--debug-chunks", type=int, default=6, help="How many chunks per utterance to print debug for")
    p.add_argument("--debug-dump-dir", type=str, default="", help="If set, dump .npy tensors into this directory")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    base_data_path = Path(args.base_data_path)
    manifest_path = base_data_path / args.dataset_name / args.manifest
    jdata = load_jsonl(manifest_path)
    if args.max_items and args.max_items > 0:
        jdata = jdata[: args.max_items]

    # Chunk config (must match your model export; you set 0.4s)
    StreamingCTCModel.AUDIO_CHUNK_SAMPLES = int(0.4 * StreamingCTCModel.SAMPLE_RATE)
    StreamingCTCPipeline.CHUNK_SIZE = StreamingCTCModel.AUDIO_CHUNK_SAMPLES

    # Triton-backed model
    triton_model = TritonStreamingCTCModel(
        triton_url=args.triton_url,
        model_name=args.model_name,
        input_signal_name=args.input_signal_name,
        output_logprobs_name=args.output_logprobs_name,
        output_state_name=args.output_state_name,
        ssl=args.ssl,
        verbose=args.verbose,
        debug_state=args.debug_state,
        debug_dump_dir=args.debug_dump_dir,
    )

    if args.debug_print_model_info:
        triton_model.print_server_and_model_info()

    if args.debug_self_test:
        # If reset check via logprobs is too strict for your GPU kernels, set atol_logprob_reset ~1e-4.
        triton_model.debug_self_test(chunk_size=StreamingCTCPipeline.CHUNK_SIZE, atol_logprob_reset=0.0)

    # Decoder
    logprob_splitter = StreamingLogprobSplitter()
    if args.decoder == "greedy":
        decoder = GreedyCTCDecoder()
    else:
        # Strongly recommend forcing local kenlm for apples-to-apples.
        if args.kenlm_path:
            decoder = BeamSearchCTCDecoder.from_local(Path(args.kenlm_path))
        else:
            decoder = BeamSearchCTCDecoder.from_hugging_face()

    pipeline = StreamingCTCPipeline(triton_model, logprob_splitter, decoder)

    hypos: List[str] = []
    gts: List[str] = []

    for utt_i, meta in enumerate(tqdm(jdata, desc="Utterances", unit="utt")):
        audio_path = resolve_audio_path(base_data_path, args.dataset_name, meta)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        # mocked model counter + splitter state
        state = (0, None)
        chunk_phrases = []

        for chunk_i, audio_chunk in enumerate(read_stream_audio(path_to_file=audio_path, chunk_size=pipeline.CHUNK_SIZE)):
            new_phrases, state = pipeline.forward(audio_chunk, state)
            if new_phrases:
                chunk_phrases += new_phrases

            if args.debug_utts > 0 and utt_i < args.debug_utts and chunk_i < args.debug_chunks:
                dbg = triton_model.last_debug
                if dbg:
                    print(
                        f"[DBG] utt={utt_i} chunk={chunk_i} "
                        f"sid={dbg.sequence_id} start={dbg.sequence_start} end={dbg.sequence_end} "
                        f"logprobs{dbg.logprobs_shape} {dbg.logprobs_dtype} "
                        f"crc32={dbg.logprobs_crc32} min={dbg.logprobs_min:.4f} "
                        f"max={dbg.logprobs_max:.4f} mean={dbg.logprobs_mean:.4f}"
                    )
                    if args.debug_state and dbg.state_next_present:
                        print(
                            f"      state_next{dbg.state_next_shape} {dbg.state_next_dtype} "
                            f"crc32={dbg.state_next_crc32} nonzero={dbg.state_next_nonzero} "
                            f"mean|x|={dbg.state_next_mean_abs:.6f} max|x|={dbg.state_next_max_abs:.6f}"
                        )
                    elif args.debug_state and not dbg.state_next_present:
                        print("      state_next: (not available from server)")

        # Tell model wrapper to end sequence on finalize request
        state = (-1, state[1])
        new_phrases, _ = pipeline.finalize(state)

        output = chunk_phrases + new_phrases
        pred_text = " ".join([phrase.text for phrase in output])

        hypos.append(clean_text(pred_text))
        gts.append(clean_text(str(meta.get("text", ""))))

    wer = word_error_rate(hypos, gts)
    print(f"WER: {wer:.6f}")


if __name__ == "__main__":
    main()
