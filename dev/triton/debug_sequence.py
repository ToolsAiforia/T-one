#!/usr/bin/env python3
"""
Minimal sequence batching tester for Triton model state.
"""

from __future__ import annotations

import argparse
from typing import List

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--triton-url", type=str, default="localhost:8001")
    p.add_argument("--model-name", type=str, default="model")
    p.add_argument("--input-name", type=str, default="audio_signal")
    p.add_argument("--length-name", type=str, default="length")
    p.add_argument("--output-name", type=str, default="decoder_logprobs")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--feat-dim", type=int, default=64)
    p.add_argument("--frames", type=int, default=40)
    p.add_argument("--length", type=int, default=0, help="0 = use --frames")
    p.add_argument("--sequence-id", type=int, default=1)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--include-state-outputs", action="store_true")
    p.add_argument("--ssl", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _build_inputs(
    input_name: str,
    length_name: str,
    batch_size: int,
    feat_dim: int,
    frames: int,
    length_value: int,
) -> List[grpcclient.InferInput]:
    signal = np.random.randn(batch_size, feat_dim, frames).astype(np.float32)
    length = np.full((batch_size, 1), length_value, dtype=np.int64)

    inp_signal = grpcclient.InferInput(input_name, list(signal.shape), "FP32")
    inp_signal.set_data_from_numpy(signal)

    inp_length = grpcclient.InferInput(length_name, list(length.shape), "INT64")
    inp_length.set_data_from_numpy(length)

    return [inp_signal, inp_length]


def _build_outputs(output_name: str, include_state: bool) -> List[grpcclient.InferRequestedOutput]:
    outputs = [grpcclient.InferRequestedOutput(output_name)]
    if include_state:
        outputs.extend(
            [
                grpcclient.InferRequestedOutput("cache_last_time_next"),
                grpcclient.InferRequestedOutput("cache_last_channel_next"),
                grpcclient.InferRequestedOutput("cache_last_chan_len_next"),
            ]
        )
    return outputs


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if args.feat_dim <= 0 or args.frames <= 0:
        raise ValueError("feat_dim and frames must be > 0")

    length_value = args.length if args.length > 0 else args.frames

    client = grpcclient.InferenceServerClient(url=args.triton_url, verbose=args.verbose, ssl=args.ssl)
    outputs = _build_outputs(args.output_name, args.include_state_outputs)

    for step in range(args.steps):
        inputs = _build_inputs(
            args.input_name,
            args.length_name,
            args.batch_size,
            args.feat_dim,
            args.frames,
            length_value,
        )

        try:
            res = client.infer(
                model_name=args.model_name,
                inputs=inputs,
                outputs=outputs,
                sequence_id=args.sequence_id,
                sequence_start=(step == 0),
                sequence_end=(step == args.steps - 1),
            )
        except InferenceServerException as e:
            raise RuntimeError(f"Triton infer failed on step {step}: {e}") from e

        print(f"step {step}:")
        for out_name in [o.name() for o in outputs]:
            out = res.as_numpy(out_name)
            if out is None:
                print(f"  {out_name}: <missing>")
            else:
                print(f"  {out_name}: shape={out.shape} dtype={out.dtype}")


if __name__ == "__main__":
    main()
