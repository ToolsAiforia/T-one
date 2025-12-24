"""Module that exports a T-one model to ONNX format."""

from __future__ import annotations

import argparse
import io
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from tone.nn.model import Tone

import onnx
import torch
from cloudpathlib import AnyPath

from tone.training.model_wrapper import ToneForCTC

_old_layer_norm = torch.nn.functional.layer_norm
DUMMY_BATCH_SIZE = 5
DUMMY_AUDIO_RANGE_MIN = -32767
DUMMY_AUDIO_RANGE_MAX = 32767


def layer_norm(
    inputs: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:  # pylint: disable=redefined-builtin
    r"""LayerNorm workaround for ONNX export: compute in float32."""
    return _old_layer_norm(inputs.float(), *args, **kwargs)


class ModelToExport(torch.nn.Module):
    """A wrapper class for exporting a T-one model to ONNX format.

    Export input:
      - default (skip_preprocessor=False): raw audio int32, (B, T_samples, 1)
      - if skip_preprocessor=True: log-mel features float32, (B, C=n_mels, T_frames)
    Length input:
      - int64, (B,) for both modes; used for compatibility with Triton configs
    """

    _model: Tone
    _state_shape: list[tuple[int, ...]]
    _state_place: list[tuple[int, int]]
    _signal_len: int
    _feat_dim: int
    _skip_preprocessor: bool

    @property
    def input_sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dummy input tuple for tracing/testing/export."""
        if self._skip_preprocessor:
            # features: (B, C, T_frames)
            signal = torch.randn(
                (DUMMY_BATCH_SIZE, self._feat_dim, self._signal_len),
                dtype=torch.float32,
            )
        else:
            # raw audio: (B, T_samples, 1)
            signal = torch.randint(
                DUMMY_AUDIO_RANGE_MIN,
                DUMMY_AUDIO_RANGE_MAX,
                (DUMMY_BATCH_SIZE, self._signal_len, 1),
                dtype=torch.int32,
            )
        length = torch.full(
            (DUMMY_BATCH_SIZE,),
            self._signal_len,
            dtype=torch.int64,
        )
        return signal, length, self.get_initial_state(DUMMY_BATCH_SIZE)

    def __init__(
        self,
        path_to_pretrained: Path | str,
        chunk_duration_ms: int,
        skip_preprocessor: bool = False,
    ) -> None:
        super().__init__()
        tone_ctc = ToneForCTC.from_pretrained(path_to_pretrained)
        self._model = tone_ctc.tone
        self._model.eval()

        # Export-mode switch is an explicit CLI flag (not inferred from checkpoint).
        self._skip_preprocessor = bool(skip_preprocessor)
        self._model.skip_preprocessor = self._skip_preprocessor

        # Determine exported input length.
        if self._skip_preprocessor:
            # Use streaming preprocessor ONCE to infer exact (C, T_frames) for the given duration.
            self._feat_dim, self._signal_len = self._infer_streaming_feat_shape(chunk_duration_ms)
        else:
            self._feat_dim = 1
            self._signal_len = chunk_duration_ms * self._model.preprocessor.sample_rate // 1000

        # Build fused-state layout (same as original script).
        init_state: tuple[torch.Tensor, ...] = self._model.get_initial_state(
            batch_size=1,
            len_dtype=torch.int32,
            target="export",
        )

        # Fused state order:
        #   [mhsa_len, preproc, mhsa, conv, sub1, sub2, reduction]
        # We keep mhsa_len as a dedicated first slice (initialized to zeros).
        device = init_state[0].device
        mhsa_len_placeholder = torch.zeros((1, 1), device=device, dtype=torch.float16)
        state_for_layout = (mhsa_len_placeholder,) + init_state[:3] + init_state[4:]

        self._state_shape = [tuple(t.shape[1:]) for t in state_for_layout]
        state_size = [t.flatten(1).size(-1) for t in state_for_layout]
        self._state_place = [(sum(state_size[:i]), sum(state_size[: i + 1])) for i in range(len(state_size))]

    def _infer_streaming_feat_shape(self, chunk_duration_ms: int) -> tuple[int, int]:
        """Infer (C, T_frames) produced by preprocessor.forward_streaming for a raw-audio chunk."""
        sample_rate = self._model.preprocessor.sample_rate
        wav_len = chunk_duration_ms * sample_rate // 1000

        device = next(self._model.parameters()).device
        wav = torch.zeros((1, wav_len), device=device, dtype=torch.float16)
        pre_state = torch.zeros(
            (1, self._model.preprocessor.state_size),
            device=device,
            dtype=torch.float16,
        )

        with torch.no_grad():
            feats, _ = self._model.preprocessor.forward_streaming(
                waveform=wav,
                state=pre_state,
            )

        feats_bct = self._model._as_features_bct(feats)  # (B, C, T)
        c = int(feats_bct.size(1))
        t = int(feats_bct.size(2))

        enc_feat_in = getattr(self._model.encoder, "_feat_in", None)
        if enc_feat_in is not None and int(enc_feat_in) != c:
            raise ValueError(
                f"Feature dim mismatch: streaming preprocessor produced C={c}, "
                f"but encoder expects feat_in={int(enc_feat_in)}."
            )
        if t <= 0:
            raise ValueError(f"Inferred non-positive feature length T={t} for chunk_duration_ms={chunk_duration_ms}.")

        return c, t

    def get_initial_state(self, batch_size: int) -> torch.Tensor:
        """Generates the initial fused state tensor for a given batch size."""
        return torch.zeros(batch_size, self._state_place[-1][1], dtype=torch.float16)

    def _checkpoint_to_bytes(self, checkpoint: dict[str, Any]) -> IO:
        """Serializes a PyTorch checkpoint dictionary into an in-memory byte stream."""
        checkpoint_bytes = io.BytesIO()
        torch.save(checkpoint, checkpoint_bytes)
        checkpoint_bytes.seek(0)
        return checkpoint_bytes

    def forward(
        self,
        audio_signal: torch.Tensor,
        length: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass using a single fused state tensor."""
        state_mhsa_len, *state_parts = [
            state[:, place[0] : place[1]].reshape(-1, *shape)
            for place, shape in zip(self._state_place, self._state_shape)
        ]

        with torch.amp.autocast(audio_signal.device.type, dtype=torch.float16):
            res, *state_next = self._model.forward_for_export(
                audio_signal,
                length,
                *state_parts[:3],
                state_mhsa_len.int(),
                *state_parts[3:],
            )

        # Keep the fused-state order identical to the original script:
        # state_next = (preproc, mhsa, conv, mhsa_len, sub1, sub2, reduction)
        # fused order: [mhsa_len, preproc, mhsa, conv, sub1, sub2, reduction]
        fused_next = torch.cat(
            [t.flatten(1) for t in state_next[3:4] + state_next[:3] + state_next[4:]],
            dim=-1,
        ).half()

        return res, fused_next


def _export_onnx(model: ModelToExport) -> bytes:
    output_sample = model(*model.input_sample)

    # Patch LayerNorm for ONNX export stability.
    torch.nn.functional.layer_norm = layer_norm

    onnx_model_bytes = io.BytesIO()
    torch.onnx.export(
        model,
        model.input_sample,
        onnx_model_bytes,
        input_names=["audio_signal", "length", "state"],
        output_names=["decoder_logprobs", "state_next"],
        opset_version=17,
        dynamic_axes={
            "audio_signal": {0: "batch_size"},
            "length": {0: "batch_size"},
            "state": {0: "batch_size"},
            "decoder_logprobs": {0: "batch_size"},
            "state_next": {0: "batch_size"},
        },
    )

    onnx_model_bytes.seek(0)
    onnx_model = onnx.load(onnx_model_bytes)

    # Freeze non-batch output dims.
    onnx_model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = output_sample[0].size(1)
    onnx_model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = output_sample[0].size(2)
    onnx_model.graph.output[1].type.tensor_type.shape.dim[1].dim_value = output_sample[1].size(1)

    onnx_model_bytes = io.BytesIO()
    onnx.save(onnx_model, onnx_model_bytes)
    return onnx_model_bytes.getvalue()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-to-pretrained",
        type=str,
        default="t-tech/T-one",
        help="Path to huggingface pretrained checkpoint",
    )
    parser.add_argument(
        "--chunk-duration-ms",
        type=int,
        default=300,
        help="Input chunk duration in ms (raw-audio export; also used to infer T_frames when --skip-preprocessor is set)",
    )
    parser.add_argument(
        "--skip-preprocessor",
        action="store_true",
        help="Export model expecting log-mel features instead of raw audio.",
    )
    parser.add_argument(
        "--output_path",
        type=AnyPath,
        required=True,
        help="Path to output model (on s3 or locally)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = ModelToExport(
        args.path_to_pretrained,
        args.chunk_duration_ms,
        skip_preprocessor=args.skip_preprocessor,
    )
    model_bytes = _export_onnx(model)
    args.output_path.write_bytes(model_bytes)
