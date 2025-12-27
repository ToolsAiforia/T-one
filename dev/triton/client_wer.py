#!/usr/bin/env python3
"""
client_wer.py - streaming WER eval against Triton sequence batching.

Queues audio chunks into Triton, decodes with StreamingCTCPipeline, computes WER.
"""

from __future__ import annotations

import argparse
import json
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


SUB = str.maketrans("\u0451", "\u0435")
INT16_MAX = float(np.iinfo(np.int16).max)


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


class TritonStreamingCTCModel:
    """
    Triton streaming wrapper for StreamingCTCPipeline.

    forward(signal, counter) returns (logprobs, next_counter)
    where counter is mocked state:
      0  => start sequence
     -1  => end sequence
     >0  => continuation

    The Triton preprocessor is stateless, so we prepend a small left-context
    window (state) to each chunk and scale int16 audio to float32 [-1, 1].
    """

    def __init__(
        self,
        *,
        triton_url: str,
        model_name: str,
        input_signal_name: str = "AUDIO_CHUNK",
        output_logprobs_name: str = "logprobs",
        state_len_name: str = "cache_last_chan_len",
        state_len_next_name: str = "cache_last_chan_len_next",
        state_len_shape: Tuple[int, ...] = (1, 1),
        use_sequence_batching: bool = True,
        explicit_cache_len: bool = True,
        ssl: bool = False,
        verbose: bool = False,
        preproc_left_context: int = 80,
        input_scale: float = INT16_MAX,
    ) -> None:
        self.client = grpcclient.InferenceServerClient(url=triton_url, verbose=verbose, ssl=ssl)
        self.model_name = model_name
        self.input_signal_name = input_signal_name
        self.output_logprobs_name = output_logprobs_name
        self.state_len_name = state_len_name
        self.state_len_next_name = state_len_next_name
        self.state_len_shape = tuple(int(x) for x in state_len_shape)
        self.use_sequence_batching = bool(use_sequence_batching)
        self.explicit_cache_len = bool(explicit_cache_len)

        if preproc_left_context < 0:
            raise ValueError("preproc_left_context must be >= 0")
        if input_scale <= 0:
            raise ValueError("input_scale must be > 0")
        self.preproc_left_context = int(preproc_left_context)
        self.input_scale = float(input_scale)
        self._preproc_state: Optional[np.ndarray] = None
        self._cache_last_channel_len: Optional[np.ndarray] = None
        self._next_sequence_id = 1
        self._active_sequence_id: Optional[int] = None

    def _reset_len_state(self, batch_size: int) -> None:
        self._cache_last_channel_len = np.zeros((batch_size, *self.state_len_shape), dtype=np.int32)

    def _reset_preproc_state(self, batch_size: int) -> None:
        if self.preproc_left_context <= 0:
            self._preproc_state = None
            return
        self._preproc_state = np.zeros((batch_size, self.preproc_left_context), dtype=np.float32)

    def _prepare_audio(self, signal: np.ndarray, *, reset_state: bool) -> np.ndarray:
        if signal.ndim == 3 and signal.shape[2] == 1:
            waveform = signal[:, :, 0]
        elif signal.ndim == 2:
            waveform = signal
        elif signal.ndim == 1:
            waveform = signal[None, :]
        else:
            raise ValueError(f"Expected signal shape (B, chunk, 1) or (B, chunk), got {signal.shape}")

        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32, copy=False)
        if self.input_scale != 1.0:
            waveform = waveform / self.input_scale

        batch_size = waveform.shape[0]
        if reset_state or self._preproc_state is None or self._preproc_state.shape[0] != batch_size:
            self._reset_preproc_state(batch_size)

        if self.preproc_left_context > 0:
            if self._preproc_state is None:
                self._reset_preproc_state(batch_size)
            waveform = np.concatenate([self._preproc_state, waveform], axis=1)
            self._preproc_state = waveform[:, -self.preproc_left_context :]

        return waveform

    def forward(self, signal: np.ndarray, counter: Optional[int]) -> Tuple[np.ndarray, int]:
        if counter is None:
            counter = 0
        if not isinstance(counter, (int, np.integer)):
            raise TypeError(f"Expected counter int, got {type(counter)}: {counter!r}")

        is_start = (int(counter) == 0)
        is_end = (int(counter) == -1)

        sequence_id = None
        if self.use_sequence_batching:
            if is_start or self._active_sequence_id is None:
                self._active_sequence_id = self._next_sequence_id
                self._next_sequence_id += 1
            sequence_id = int(self._active_sequence_id)

        audio_data = self._prepare_audio(signal, reset_state=is_start)
        batch_size = audio_data.shape[0]
        if self.explicit_cache_len:
            if is_start or self._cache_last_channel_len is None or self._cache_last_channel_len.shape[0] != batch_size:
                self._reset_len_state(batch_size)

        inp_signal = grpcclient.InferInput(self.input_signal_name, list(audio_data.shape), "FP32")
        inp_signal.set_data_from_numpy(audio_data)

        inputs = [inp_signal]
        if self.explicit_cache_len:
            if self._cache_last_channel_len is None:
                raise RuntimeError("cache_last_chan_len state is missing")
            inp_cache_len = grpcclient.InferInput(
                self.state_len_name,
                list(self._cache_last_channel_len.shape),
                "INT32",
            )
            inp_cache_len.set_data_from_numpy(self._cache_last_channel_len)
            inputs.append(inp_cache_len)

        outputs = [grpcclient.InferRequestedOutput(self.output_logprobs_name)]
        if self.explicit_cache_len:
            outputs.append(grpcclient.InferRequestedOutput(self.state_len_next_name))

        try:
            res = self.client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs,
                sequence_id=sequence_id,
                sequence_start=is_start if self.use_sequence_batching else False,
                sequence_end=is_end if self.use_sequence_batching else False,
            )
        except InferenceServerException as e:
            raise RuntimeError(f"Triton infer failed: {e}") from e

        logprobs = res.as_numpy(self.output_logprobs_name)
        if logprobs is None:
            raise RuntimeError(f"No output {self.output_logprobs_name!r}")

        if self.explicit_cache_len:
            cache_len_next = res.as_numpy(self.state_len_next_name)
            if cache_len_next is None:
                raise RuntimeError("Missing cache_last_chan_len_next output from Triton response")
            self._cache_last_channel_len = cache_len_next

        next_counter = 0 if is_end else (int(counter) + 1)
        if is_end:
            self._active_sequence_id = None
            self._preproc_state = None
            self._cache_last_channel_len = None

        return logprobs, next_counter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--base-data-path", type=str, required=True)
    p.add_argument("--dataset-name", type=str, default="test_rec_support")
    p.add_argument("--manifest", type=str, default="tarred_audio_manifest.json")
    p.add_argument("--max-items", type=int, default=0, help="0 = all")

    # Triton
    p.add_argument("--triton-url", type=str, default="localhost:8001")
    p.add_argument("--model-name", type=str, default="ensemble")
    p.add_argument("--input-signal-name", type=str, default="AUDIO_CHUNK")
    p.add_argument("--output-logprobs-name", type=str, default="logprobs")
    p.add_argument("--ssl", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--no-sequence-batching",
        dest="sequence_batching",
        action="store_false",
        help="Disable Triton sequence batching (use explicit states only).",
    )
    p.add_argument(
        "--explicit-cache-len",
        dest="explicit_cache_len",
        action="store_true",
        help="Enable explicit cache_last_chan_len input/output.",
    )
    p.add_argument(
        "--no-explicit-cache-len",
        dest="explicit_cache_len",
        action="store_false",
        help="Disable explicit cache_last_chan_len input/output.",
    )
    p.set_defaults(sequence_batching=True, explicit_cache_len=False)

    # Preprocessor streaming state
    p.add_argument(
        "--preproc-left-context",
        type=int,
        default=80,
        help="Number of left-context samples to prepend for streaming preprocessor (0 = disable).",
    )
    p.add_argument(
        "--input-scale",
        type=float,
        default=INT16_MAX,
        help="Divisor to scale int16 waveform into float32 (default: int16 max).",
    )

    # Decoder
    p.add_argument("--decoder", choices=["greedy", "beam"], default="beam")
    p.add_argument("--kenlm-path", type=str, default="", help="Path to kenlm.bin for beam decoder")

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

    triton_model = TritonStreamingCTCModel(
        triton_url=args.triton_url,
        model_name=args.model_name,
        input_signal_name=args.input_signal_name,
        output_logprobs_name=args.output_logprobs_name,
        use_sequence_batching=args.sequence_batching,
        explicit_cache_len=args.explicit_cache_len,
        ssl=args.ssl,
        verbose=args.verbose,
        preproc_left_context=args.preproc_left_context,
        input_scale=args.input_scale,
    )

    logprob_splitter = StreamingLogprobSplitter()
    if args.decoder == "greedy":
        decoder = GreedyCTCDecoder()
    else:
        if args.kenlm_path:
            decoder = BeamSearchCTCDecoder.from_local(Path(args.kenlm_path))
        else:
            decoder = BeamSearchCTCDecoder.from_hugging_face()

    pipeline = StreamingCTCPipeline(triton_model, logprob_splitter, decoder)

    hypos: List[str] = []
    gts: List[str] = []

    for meta in tqdm(jdata, desc="Utterances", unit="utt"):
        audio_path = resolve_audio_path(base_data_path, args.dataset_name, meta)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        state = (0, None)
        chunk_phrases = []

        for audio_chunk in read_stream_audio(path_to_file=audio_path, chunk_size=pipeline.CHUNK_SIZE):
            new_phrases, state = pipeline.forward(audio_chunk, state)
            if new_phrases:
                chunk_phrases += new_phrases

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
