"""Module defining the T-one model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from tone.nn.modules.conformer import ConvASRDecoder, Encoder
from tone.nn.modules.feats import FilterbankFeatures
from tone.nn.torch_utils import cast_all


class Tone(nn.Module):
    """A streaming speech recognition model based on the Conformer architecture.

    This class serves as the main entry point, wrapping the preprocessor,
    encoder, and decoder into a single, cohesive module.

    Args:
        feature_extraction_params (dict[str, Any]): A dictionary of parameters
            to initialize the `FilterbankFeatures` preprocessor.
        encoder_params (dict[str, Any]): A dictionary of parameters to initialize
            the `Encoder` module.
        decoder_params (dict[str, Any]): A dictionary of parameters to initialize
            the `ConvASRDecoder` module.

    """

    def __init__(
        self,
        feature_extraction_params: dict[str, Any],
        encoder_params: dict[str, Any],
        decoder_params: dict[str, Any],
        skip_preprocessor: bool = False,
    ) -> None:
        super().__init__()
        self.preprocessor = FilterbankFeatures(**feature_extraction_params)
        self.encoder = Encoder(**encoder_params)
        self.decoder = ConvASRDecoder(**decoder_params)
        self.skip_preprocessor = skip_preprocessor

    def _as_features_bct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accept (B, C, T) or (B, T, C) and return (B, C, T).
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D features tensor, got {tuple(x.shape)}")
        feat_in = getattr(self.encoder, "_feat_in", None)
        if feat_in is None:
            return x
        if x.size(1) == feat_in:
            return x
        if x.size(2) == feat_in:
            return x.transpose(1, 2)
        raise ValueError(
            f"Cannot infer feature layout. Expected one dim == feat_in={feat_in}, got {tuple(x.shape)}"
        )

    def forward(
        self,
        input_signal: torch.Tensor,
        input_signal_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs a standard forward pass.

        Args:
            input_signal (torch.Tensor): A batch of raw audio signals of shape
                (B, T_audio).
            input_signal_length (torch.Tensor): The lengths of the signals in
                the batch, of shape (B,).

        Returns:
            A tuple containing:
            - log_probs (torch.Tensor): The output log probabilities from the
                decoder, of shape (B, T_frame, V) where V is vocabulary size.
            - encoded_len (torch.Tensor): The lengths of the encoded sequences,
                of shape (B,).

        """
        if self.skip_preprocessor:
            # log-mel features
            # input_signal: (B, C, T) or (B, T, C), where C == n_mels
            processed_signal = self._as_features_bct(input_signal)
            processed_signal_length = input_signal_length
        else:
            # raw waveform
            # input_signal: (B, T_audio), lengths in samples
            processed_signal, processed_signal_length = self.preprocessor(
                waveform=input_signal,
                waveform_lens=input_signal_length,
            )
        encoded, encoded_len = self.encoder(
            audio_signal=processed_signal,
            length=processed_signal_length,
        )
        log_probs = self.decoder(encoder_output=encoded)
        return log_probs, encoded_len

    def forward_for_export(
        self,
        inputs: torch.Tensor,
        state_preprocessing: torch.Tensor | None = None,
        state_mhsa: torch.Tensor | None = None,
        state_conv: torch.Tensor | None = None,
        state_mhsa_len: torch.Tensor | None = None,
        state_subsampling_1: torch.Tensor | None = None,
        state_subsampling_2: torch.Tensor | None = None,
        state_reduction: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Performs a streaming-compatible forward pass for ONNX export.

        This method processes a single chunk of audio and manages the model's
        state, making it suitable for real-time inference. All state
        tensors are expected to have the batch dimension first.

        Args:
            inputs (torch.Tensor): A batch of raw audio signals of shape (B, T, 1).
                B is batch size, T represents timesteps.
            state_preprocessing (torch.Tensor | None): The state for the
                `FilterbankFeatures` preprocessor.
            state_mhsa (torch.Tensor | None): The state for MHSA layers in the
                Conformer blocks, of shape (B, N, T, H). N is the number of such layers
                which need caching, H is the hidden size of activations, and T is the
                length of the state.
            state_conv (torch.Tensor | None): The state for convolutional
                layers in each Conformer block of shape (B, N, H, T).
            state_mhsa_len (torch.Tensor | None): Tensor of shape (N, B, T, H) which
                contains the state for MHSA in each Conformer block.
            state_subsampling_1 (torch.Tensor | None): State for the first
                subsampling layer.
            state_subsampling_2 (torch.Tensor | None): State for the second
                subsampling layer.
            state_reduction (torch.Tensor | None): State for the final
                reduction layer.

        Returns:
            A tuple containing the model's output and all updated states for the
            next inference step, cast to float32 for stability. The structure is:
            (log_probs, next_state_preprocessing, next_state_mhsa,
             next_state_conv, next_state_mhsa_len, next_state_subsampling_1,
             next_state_subsampling_2, next_state_reduction)

        """
        if self.skip_preprocessor:
            # log-mel features
            # inputs: (B, C, T) or (B, T, C) where C == n_mels
            feats = self._as_features_bct(inputs).to(dtype=torch.float16)
            # Preprocessor state is unused in this mode; keep it for interface compatibility.
            if state_preprocessing is None:
                state_preprocessing = feats.new_zeros((feats.size(0), self.preprocessor.state_size))
        else:
            # raw audio
            # inputs: (B, T, 1) int32 in [-32768, 32767]
            wav = inputs[:, :, 0]
            wav = (wav.float() / torch.iinfo(torch.int16).max).half()
            feats, state_preprocessing = self.preprocessor.forward_streaming(
                waveform=wav,
                state=state_preprocessing,
            )

        # Transpose states from (B, N, ...) to (N, B, ...) for internal processing
        state_mhsa = state_mhsa.transpose(0, 1)
        state_conv = state_conv.transpose(0, 1)

        # Squeeze 1D vector from state to handle Triton server issue
        # https://github.com/triton-inference-server/server/issues/3901
        state_mhsa_len = state_mhsa_len[:, 0]

        # print(f"Before encoder sub-sampling:\n{inputs.shape=}")
        encoder_output, _ = self.encoder(
            audio_signal=feats,
            state_mhsa=state_mhsa,
            state_conv=state_conv,
            state_mhsa_len=state_mhsa_len,
            state_subsampling_1=state_subsampling_1,
            state_subsampling_2=state_subsampling_2,
            state_reduction=state_reduction,
        )

        # Get state for the next chunk
        next_state = self.encoder.state.next()

        out = (
            self.decoder(encoder_output=encoder_output).float(),
            state_preprocessing,
            next_state.mhsa.transpose(0, 1),
            next_state.conv.transpose(0, 1),
            next_state.mhsa_len.unsqueeze(-1),
            next_state.subsampling[0],
            next_state.subsampling[1],
            next_state.reduction,
        )
        return out
        # return cast_all(out, from_dtype=torch.float16, to_dtype=torch.float32)

    def get_initial_state(
        self,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
        len_dtype: torch.dtype = torch.int64,
        device: torch.device | None = None,
        target: str = "export",
    ) -> tuple[torch.Tensor, ...]:
        """Generates the initial zero-states for a streaming inference pass.

        Args:
            batch_size (int, optional): The batch size for the states. Defaults to 1.
            dtype (torch.dtype, optional): The data type for the states.
                Defaults to `torch.float32`.
            len_dtype (torch.dtype, optional): The data type for length tensors.
                Defaults to `torch.int64`.
            device (torch.device | None, optional): The device to place states on.
                Defaults to None.
            target (str, optional): The target format. If "export", states are
                transposed to (B, N, ...) for ONNX compatibility.
                Defaults to "export".

        Returns:
            A tuple containing all initial state tensors required by
            `forward_for_export`.

        """
        state_preprocessing = torch.zeros(
            (batch_size, self.preprocessor.state_size),
            device=device,
            dtype=dtype,
        )
        (
            state_mhsa,
            state_conv,
            state_mhsa_len,
            state_subsampling_1,
            state_subsampling_2,
            state_reduction,
        ) = self.encoder.get_initial_state(
            batch_size=batch_size,
            dtype=dtype,
            len_dtype=len_dtype,
            device=device,
        )

        if target == "export":
            # Transpose from (N, B, ...) to (B, N, ...) for export format
            state_mhsa = state_mhsa.transpose(0, 1)
            state_conv = state_conv.transpose(0, 1)

        return (
            state_preprocessing,
            state_mhsa,
            state_conv,
            state_mhsa_len.unsqueeze(-1),
            state_subsampling_1,
            state_subsampling_2,
            state_reduction,
        )
