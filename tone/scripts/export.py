"""Module that exports a T-one model to ONNX or NeMo format."""

from __future__ import annotations

import argparse
import io
import math
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


def _prod(shape: tuple[int, ...]) -> int:
    out = 1
    for v in shape:
        out *= int(v)
    return int(out)


class ModelToExport(torch.nn.Module):
    """A wrapper class for exporting a T-one model to ONNX format.

    Export input:
      - default (skip_preprocessor=False): raw audio int32, (B, T_samples, 1)
      - if skip_preprocessor=True: log-mel features float32, (B, C=n_mels, T_frames)

    Length input:
      - int64, (B,) for both modes; used for compatibility with Triton configs
    """

    _model: Tone
    _signal_len: int
    _feat_dim: int
    _skip_preprocessor: bool

    # Derived shapes (excluding batch)
    _cache_last_time_shape: tuple[int, int, int]      # (L_time, H, Tcache)
    _cache_last_channel_shape: tuple[int, int, int]   # (C1, C2, Tbase+Tpad)

    # Internal state shapes (excluding batch)
    _shape_preproc: tuple[int, ...]
    _shape_mhsa: tuple[int, ...]
    _shape_conv: tuple[int, ...]
    _shape_sub1: tuple[int, ...]
    _shape_sub2: tuple[int, ...]
    _shape_reduction: tuple[int, ...]

    # Packing parameters
    _n_mhsa: int
    _n_conv: int
    _time_h: int
    _time_t: int

    _sub2_c1: int
    _sub2_c2: int
    _sub2_tbase: int
    _sub2_tpad: int
    _tail_capacity: int
    _tail_required: int

    _preproc_elems: int
    _sub1_elems: int
    _reduction_elems: int

    @property
    def input_sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dummy input tuple for tracing/testing/export."""
        device = next(self._model.parameters()).device

        if self._skip_preprocessor:
            # features: (B, C, T_frames)
            signal = torch.randn(
                (DUMMY_BATCH_SIZE, self._feat_dim, self._signal_len),
                dtype=torch.float32,
                device=device,
            )
        else:
            # raw audio: (B, T_samples, 1)
            signal = torch.randint(
                DUMMY_AUDIO_RANGE_MIN,
                DUMMY_AUDIO_RANGE_MAX,
                (DUMMY_BATCH_SIZE, self._signal_len, 1),
                dtype=torch.int32,
                device=device,
            )

        length = torch.full(
            (DUMMY_BATCH_SIZE,),
            self._signal_len,
            dtype=torch.int64,
            device=device,
        )

        cache_last_time = torch.zeros(
            (DUMMY_BATCH_SIZE, *self._cache_last_time_shape),
            dtype=torch.float16,
            device=device,
        )
        cache_last_channel = torch.zeros(
            (DUMMY_BATCH_SIZE, *self._cache_last_channel_shape),
            dtype=torch.float16,
            device=device,
        )
        cache_last_channel_len = torch.zeros(
            (DUMMY_BATCH_SIZE, 1),
            dtype=torch.int32,
            device=device,
        )

        return signal, length, cache_last_time, cache_last_channel, cache_last_channel_len

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

        # Pull the true internal state shapes from the model.
        init_state = self._model.get_initial_state(
            batch_size=1,
            dtype=torch.float16,
            len_dtype=torch.int64,
            target="export",
        )
        # init_state order (per Tone.get_initial_state):
        # (preproc, mhsa, conv, mhsa_len, sub1, sub2, reduction)
        state_preproc, state_mhsa, state_conv, _state_mhsa_len, state_sub1, state_sub2, state_reduction = init_state

        self._shape_preproc = tuple(int(x) for x in state_preproc.shape[1:])
        self._shape_mhsa = tuple(int(x) for x in state_mhsa.shape[1:])       # (N_mhsa, Tcache, H)
        self._shape_conv = tuple(int(x) for x in state_conv.shape[1:])       # (N_conv, H, Tcache)
        self._shape_sub1 = tuple(int(x) for x in state_sub1.shape[1:])
        self._shape_sub2 = tuple(int(x) for x in state_sub2.shape[1:])
        self._shape_reduction = tuple(int(x) for x in state_reduction.shape[1:])

        # Build cache_last_time = pack mhsa + conv along layer axis into (L_time, H, Tcache).
        if len(self._shape_mhsa) != 3 or len(self._shape_conv) != 3:
            raise ValueError(
                "Unexpected mhsa/conv state rank. "
                f"mhsa={self._shape_mhsa}, conv={self._shape_conv}"
            )

        self._n_mhsa = self._shape_mhsa[0]
        self._n_conv = self._shape_conv[0]

        mhsa_t = self._shape_mhsa[1]
        mhsa_h = self._shape_mhsa[2]
        conv_h = self._shape_conv[1]
        conv_t = self._shape_conv[2]

        if mhsa_t != conv_t or mhsa_h != conv_h:
            raise ValueError(
                "Cannot pack mhsa and conv into a single cache_last_time: "
                f"mhsa (N,T,H)={self._shape_mhsa}, conv (N,H,T)={self._shape_conv} "
                f"(requires mhsa.T == conv.T and mhsa.H == conv.H)."
            )

        self._time_h = conv_h
        self._time_t = conv_t
        self._cache_last_time_shape = (self._n_mhsa + self._n_conv, self._time_h, self._time_t)

        # Build cache_last_channel:
        # Head region stores sub2 exactly: (C1, C2, Tbase)
        # Tail region stores flattened {preproc, sub1, reduction} with padding:
        # tail has shape (C1, C2, Tpad), capacity = C1*C2*Tpad.
        if len(self._shape_sub2) != 3:
            raise ValueError(f"Unexpected sub2 state rank (expected 3 dims excl batch). Got {self._shape_sub2}.")

        self._sub2_c1, self._sub2_c2, self._sub2_tbase = self._shape_sub2
        elems_per_last = self._sub2_c1 * self._sub2_c2

        self._preproc_elems = _prod(self._shape_preproc)
        self._sub1_elems = _prod(self._shape_sub1)
        self._reduction_elems = _prod(self._shape_reduction)
        self._tail_required = self._preproc_elems + self._sub1_elems + self._reduction_elems

        # Minimal pad along last dim to fit tail, then round up to an even number for nicer alignment.
        tpad_min = int(math.ceil(self._tail_required / max(1, elems_per_last))) if self._tail_required > 0 else 0
        if tpad_min == 0:
            tpad_min = 2
        self._sub2_tpad = tpad_min if (tpad_min % 2 == 0) else (tpad_min + 1)

        self._tail_capacity = elems_per_last * self._sub2_tpad
        if self._tail_required > self._tail_capacity:
            raise ValueError(
                "Internal error: tail_required > tail_capacity after padding. "
                f"tail_required={self._tail_required}, tail_capacity={self._tail_capacity} "
                f"(C1,C2,Tpad)=({self._sub2_c1},{self._sub2_c2},{self._sub2_tpad})."
            )

        self._cache_last_channel_shape = (
            self._sub2_c1,
            self._sub2_c2,
            self._sub2_tbase + self._sub2_tpad,
        )

        # Helpful export-time info (useful when writing Triton config).
        print("=== Export state interface ===")
        print(f"skip_preprocessor={self._skip_preprocessor}")
        print(f"audio_signal fixed shape (excl batch) = {(self._feat_dim, self._signal_len) if self._skip_preprocessor else (self._signal_len, 1)}")
        print(f"cache_last_time        (B, {self._cache_last_time_shape[0]}, {self._cache_last_time_shape[1]}, {self._cache_last_time_shape[2]})  fp16")
        print(f"cache_last_channel     (B, {self._cache_last_channel_shape[0]}, {self._cache_last_channel_shape[1]}, {self._cache_last_channel_shape[2]})  fp16")
        print("cache_last_chan_len (B, 1) int32")
        print("--- Internal packing ---")
        print(f"mhsa  state (B, {self._shape_mhsa[0]}, {self._shape_mhsa[1]}, {self._shape_mhsa[2]})")
        print(f"conv  state (B, {self._shape_conv[0]}, {self._shape_conv[1]}, {self._shape_conv[2]})")
        print(f"sub2  state (B, {self._shape_sub2[0]}, {self._shape_sub2[1]}, {self._shape_sub2[2]}) -> channel head")
        print(f"tail required elems={self._tail_required}, tail capacity elems={self._tail_capacity} (pad={self._tail_capacity - self._tail_required})")
        print("================================")

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

    def _checkpoint_to_bytes(self, checkpoint: dict[str, Any]) -> IO:
        """Serializes a PyTorch checkpoint dictionary into an in-memory byte stream."""
        checkpoint_bytes = io.BytesIO()
        torch.save(checkpoint, checkpoint_bytes)
        checkpoint_bytes.seek(0)
        return checkpoint_bytes

    def _unpack_states(
        self,
        cache_last_time: torch.Tensor,
        cache_last_channel: torch.Tensor,
        cache_last_channel_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert external cache tensors into the model's internal state tensors.

        Returns:
          (state_preproc, state_mhsa, state_conv, state_sub1, state_sub2, state_reduction)
        """
        # cache_last_time: (B, L_time, H, T)
        # mhsa wants: (B, N_mhsa, T, H)
        mhsa_ht = cache_last_time[:, : self._n_mhsa, :, :]  # (B, N_mhsa, H, T)
        state_mhsa = mhsa_ht.transpose(2, 3).contiguous()   # (B, N_mhsa, T, H)

        # conv wants: (B, N_conv, H, T)
        state_conv = cache_last_time[:, self._n_mhsa :, :, :].contiguous()

        # cache_last_channel: (B, C1, C2, Tbase+Tpad)
        # sub2 is exactly the head region:
        state_sub2 = cache_last_channel[:, :, :, : self._sub2_tbase].contiguous()

        # Tail region stores flattened {preproc, sub1, reduction} with padding:
        tail = cache_last_channel[:, :, :, self._sub2_tbase :].contiguous()
        tail_flat = tail.reshape(tail.size(0), -1)  # (B, tail_capacity)

        # Slices:
        idx0 = 0
        idx1 = idx0 + self._preproc_elems
        idx2 = idx1 + self._sub1_elems
        idx3 = idx2 + self._reduction_elems

        state_preproc = tail_flat[:, idx0:idx1].reshape(-1, *self._shape_preproc).contiguous()
        state_sub1 = tail_flat[:, idx1:idx2].reshape(-1, *self._shape_sub1).contiguous()
        state_reduction = tail_flat[:, idx2:idx3].reshape(-1, *self._shape_reduction).contiguous()

        # cache_last_channel_len is handled in forward() where we adapt shape for forward_for_export
        _ = cache_last_channel_len  # explicitly unused here (handled separately)

        return state_preproc, state_mhsa, state_conv, state_sub1, state_sub2, state_reduction

    def _pack_states(
        self,
        state_preproc: torch.Tensor,
        state_mhsa: torch.Tensor,
        state_conv: torch.Tensor,
        state_sub1: torch.Tensor,
        state_sub2: torch.Tensor,
        state_reduction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert internal states into external cache tensors.

        Returns:
          (cache_last_time_next, cache_last_channel_next)
        """
        # cache_last_time_next = cat([mhsa_ht, conv], dim=1) where mhsa_ht is (B,N_mhsa,H,T)
        mhsa_ht = state_mhsa.transpose(2, 3).contiguous()  # (B,N_mhsa,H,T)
        cache_last_time_next = torch.cat([mhsa_ht, state_conv], dim=1).contiguous()

        # cache_last_channel_next = cat([sub2, tail], dim=3)
        # tail is flattened concat of (preproc, sub1, reduction) then padded to tail_capacity
        tail_flat = torch.cat(
            [
                state_preproc.flatten(1),
                state_sub1.flatten(1),
                state_reduction.flatten(1),
            ],
            dim=1,
        ).contiguous()

        pad = self._tail_capacity - tail_flat.size(1)
        if pad < 0:
            raise ValueError(
                f"Tail overflow: required={tail_flat.size(1)} capacity={self._tail_capacity}. "
                "This should be impossible if shapes are consistent."
            )
        if pad > 0:
            tail_flat = torch.cat([tail_flat, tail_flat.new_zeros((tail_flat.size(0), pad))], dim=1)

        tail = tail_flat.reshape(-1, self._sub2_c1, self._sub2_c2, self._sub2_tpad).contiguous()
        cache_last_channel_next = torch.cat([state_sub2, tail], dim=3).contiguous()

        return cache_last_time_next, cache_last_channel_next

    def forward(
        self,
        audio_signal: torch.Tensor,
        length: torch.Tensor,
        cache_last_time: torch.Tensor,
        cache_last_channel: torch.Tensor,
        cache_last_channel_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass using Triton-style multi-tensor streaming cache.

        Returns:
          decoder_logprobs,
          cache_last_time_next,
          cache_last_channel_next,
          cache_last_channel_len_next
        """
        # Unpack float16 caches into internal float16 states
        state_preproc, state_mhsa, state_conv, state_sub1, state_sub2, state_reduction = self._unpack_states(
            cache_last_time=cache_last_time,
            cache_last_channel=cache_last_channel,
            cache_last_channel_len=cache_last_channel_len,
        )

        # Normalize cache_last_channel_len to shape (B,) then adapt to (B,1) for forward_for_export,
        # because model.forward_for_export expects to do state_mhsa_len[:, 0].
        if cache_last_channel_len.dim() == 2 and cache_last_channel_len.size(1) == 1:
            cache_last_channel_len = cache_last_channel_len.squeeze(1)
        elif cache_last_channel_len.dim() != 1:
            raise ValueError(
                f"cache_last_channel_len must be (B,) or (B,1). Got {tuple(cache_last_channel_len.shape)}"
            )
        state_mhsa_len_in = cache_last_channel_len.to(dtype=torch.int64).unsqueeze(1)  # (B,1)

        with torch.amp.autocast(audio_signal.device.type, dtype=torch.float16):
            (
                decoder_logprobs,
                next_preproc,
                next_mhsa,
                next_conv,
                next_mhsa_len,
                next_sub1,
                next_sub2,
                next_reduction,
            ) = self._model.forward_for_export(
                audio_signal,
                length,
                state_preproc,
                state_mhsa,
                state_conv,
                state_mhsa_len_in,
                state_sub1,
                state_sub2,
                state_reduction,
            )

        # Pack next states back into Triton-style caches
        cache_last_time_next, cache_last_channel_next = self._pack_states(
            state_preproc=next_preproc,
            state_mhsa=next_mhsa,
            state_conv=next_conv,
            state_sub1=next_sub1,
            state_sub2=next_sub2,
            state_reduction=next_reduction,
        )

        # next_mhsa_len is (B,1) (per Tone.forward_for_export); emit (B,1) int32.
        cache_last_channel_len_next = next_mhsa_len.to(dtype=torch.int32)

        # Ensure state outputs are fp16 (consistent with current export behavior / TensorRT fp16)
        cache_last_time_next = cache_last_time_next.half()
        cache_last_channel_next = cache_last_channel_next.half()

        return decoder_logprobs, cache_last_time_next, cache_last_channel_next, cache_last_channel_len_next


def _export_onnx(model: ModelToExport) -> bytes:
    # Run one forward to lock shapes for freezing output dims.
    output_sample = model(*model.input_sample)

    # Patch LayerNorm for ONNX export stability.
    torch.nn.functional.layer_norm = layer_norm

    try:
        onnx_model_bytes = io.BytesIO()
        torch.onnx.export(
            model,
            model.input_sample,
            onnx_model_bytes,
            input_names=[
                "audio_signal",
                "length",
                "cache_last_time",
                "cache_last_channel",
                "cache_last_chan_len",
            ],
            output_names=[
                "decoder_logprobs",
                "cache_last_time_next",
                "cache_last_channel_next",
                "cache_last_chan_len_next",
            ],
            opset_version=17,
            dynamic_axes={
                "audio_signal": {0: "batch_size"},
                "length": {0: "batch_size"},
                "cache_last_time": {0: "batch_size"},
                "cache_last_channel": {0: "batch_size"},
                "cache_last_chan_len": {0: "batch_size"},
                "decoder_logprobs": {0: "batch_size"},
                "cache_last_time_next": {0: "batch_size"},
                "cache_last_channel_next": {0: "batch_size"},
                "cache_last_chan_len_next": {0: "batch_size"},
            },
        )

        onnx_model_bytes.seek(0)
        onnx_model = onnx.load(onnx_model_bytes)

        # Freeze non-batch output dims (TensorRT / Triton generally prefer fixed non-batch dims).
        # decoder_logprobs: (B, T_out, V)
        onnx_model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = int(output_sample[0].size(1))
        onnx_model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = int(output_sample[0].size(2))

        # cache_last_time_next: (B, L_time, H, T)
        for j in range(1, 4):
            onnx_model.graph.output[1].type.tensor_type.shape.dim[j].dim_value = int(output_sample[1].size(j))

        # cache_last_channel_next: (B, C1, C2, Tbase+Tpad)
        for j in range(1, 4):
            onnx_model.graph.output[2].type.tensor_type.shape.dim[j].dim_value = int(output_sample[2].size(j))

        # cache_last_chan_len_next: (B, 1)
        onnx_model.graph.output[3].type.tensor_type.shape.dim[1].dim_value = int(output_sample[3].size(1))

        out_bytes = io.BytesIO()
        onnx.save(onnx_model, out_bytes)
        return out_bytes.getvalue()
    finally:
        # Always restore original LayerNorm
        torch.nn.functional.layer_norm = _old_layer_norm


def _export_nemo(path_to_pretrained: str, output_path: AnyPath) -> None:
    try:
        from omegaconf import OmegaConf
        from nemo.core.classes import ModelPT  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("NeMo must be installed to export a .nemo model.") from exc

    from tone.nemo_wrapper import ToneCTCNemoModel

    if getattr(output_path, "is_cloud_path", False):
        raise ValueError("NeMo export only supports local output paths.")

    tone_model = ToneForCTC.from_pretrained(path_to_pretrained)
    cfg = OmegaConf.create(
        {
            "tone_config": tone_model.config.to_dict(),
            "decoding": {"strategy": "greedy"},
        }
    )
    nemo_model = ToneCTCNemoModel(cfg=cfg)
    nemo_model.tone_for_ctc.load_state_dict(tone_model.state_dict())
    nemo_model.save_to(str(output_path))


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
        help=(
            "Input chunk duration in ms (raw-audio export; also used to infer T_frames when "
            "--skip-preprocessor is set)"
        ),
    )
    parser.add_argument(
        "--skip-preprocessor",
        action="store_true",
        help="Export model expecting log-mel features instead of raw audio.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["onnx", "nemo"],
        default="onnx",
        help="Export format.",
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
    if args.output_format == "nemo":
        _export_nemo(args.path_to_pretrained, args.output_path)
    else:
        model = ModelToExport(
            args.path_to_pretrained,
            args.chunk_duration_ms,
            skip_preprocessor=args.skip_preprocessor,
        )
        model_bytes = _export_onnx(model)
        args.output_path.write_bytes(model_bytes)
