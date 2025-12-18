"""Module defining the core Encoder and a simple Decoder for a Conformer-based T-one model."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

import torch
from torch import nn

from tone.nn.modules.conformer_blocks import (
    CausalTemporalReduction,
    ConformerLayer,
    ConvSubsamplingPreEncode,
    EncoderState,
    TemporalUpsampling,
)


class Encoder(nn.Module):
    """The main encoder module for a Conformer-based T-one model.

    This class implements the architecture based on the paper:
    'Conformer: Convolution-augmented Transformer for Speech Recognition' by Anmol Gulati et al.
    https://arxiv.org/abs/2005.08100

    Args:
        feat_in (int): The number of input feature channels
        n_layers (int): The number of Conformer layers in the stack.
        d_model (int): The hidden dimension of the model.
        subsampling_conv_channels (int): The number of channels in the
            convolutional subsampling layers. Defaults to -1, which sets it to `d_model`.
        subsampling_kernel_size (Sequence[Sequence[int]]): Kernel sizes for the subsampling layers.
            Defaults to ((11, 21), (11, 11)).
        subsampling_strides (Sequence[Sequence[int]]): Strides for the subsampling layers.
            Defaults to ((1, 1), (3, 1)).
        reduction_position (int | list[int]] | None): The layer index after which to apply
            temporal reduction. Defaults to 6.
        reduction_factor (int): The factor by which to reduce sequence length. Must be 1 or
            a power of 2. Defaults to 1.
        reduction_kernel_size (int): The kernel size for the temporal reduction convolution.
            Defaults to 2.
        upsample_position (int | list[int]): The layer index after which to apply temporal
            upsampling to restore sequence length. Defaults to 14.
        ff_expansion_factor (int): The expansion factor in the feed-forward blocks.
            Defaults to 4.
        n_heads (int): The number of heads in multi-head attention. Defaults to 4.
        chunk_size (int): The chunk size for creating attention masks. Defaults to 10.
        mhsa_state_size (int): The size of the state cache for MHSA. Defaults to 30.
        conv_kernel_size (int): The kernel size for the depthwise convolution in Conformer
            blocks. Defaults to 31.
        dropout (float): The general dropout rate. Defaults to 0.1.
        dropout_att (float): The dropout rate for attention. Defaults to 0.0.
        mhsa_stateless_layers (int): The number of initial Conformer layers that will not use
            a state cache for MHSA. Defaults to 14.
        rope_dim (int): The feature dimension for Rotary Positional Embeddings (RoPE).
            Defaults to 64.
        should_recompute_att_scores (list[bool]): A list indicating whether to recompute
            attention scores for each layer. Defaults to all True.

    """

    def __init__(
        self,
        feat_in: int,
        n_layers: int,
        d_model: int,
        subsampling_conv_channels: int = -1,
        subsampling_kernel_size: Sequence[Sequence[int]] = ((11, 21), (11, 11)),
        subsampling_strides: Sequence[Sequence[int]] = ((1, 1), (3, 1)),
        reduction_position: int = 6,
        reduction_factor: int = 1,
        reduction_kernel_size: int = 2,
        upsample_position: int = 14,
        ff_expansion_factor: int = 4,
        n_heads: int = 4,
        chunk_size: int = 10,
        mhsa_state_size: int = 30,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        dropout_att: float = 0.0,
        mhsa_stateless_layers: int = 14,
        rope_dim: int = 64,
        should_recompute_att_scores: list[bool] = [True] * 16,
    ) -> None:
        super().__init__()
        d_ff = d_model * ff_expansion_factor
        self.d_model = d_model
        self.n_layers = n_layers
        self._feat_in = feat_in
        self.subsampling_factor = math.prod(stride[0] for stride in subsampling_strides)
        self.mhsa_stateless_layers = mhsa_stateless_layers
        self.rope_dim = rope_dim
        self.should_recompute_att_scores = should_recompute_att_scores
        self.conv_kernel_size = conv_kernel_size
        self.conv_context_size = [conv_kernel_size - 1, 0]
        self.chunk_size = chunk_size
        self.mhsa_state_size = mhsa_state_size
        self.pre_encode = ConvSubsamplingPreEncode(
            feat_in=feat_in,
            feat_out=d_model,
            conv_channels=subsampling_conv_channels,
            activation=nn.SiLU(),
            kernel_sizes=subsampling_kernel_size,
            strides=subsampling_strides,
        )
        self.reduction_factor = reduction_factor
        self.reduction_position = reduction_position
        self.upsample_position = upsample_position
        self.reduction_kernel_size = reduction_kernel_size
        self.temportal_reduction = CausalTemporalReduction(
            d_model=d_model,
            reduction_factor=reduction_factor,
            kernel_size=reduction_kernel_size,
        )
        self.temporal_upsample = TemporalUpsampling(reduction_factor)

        self.layers = nn.ModuleList(
            [
                ConformerLayer(
                    d_model=d_model,
                    d_ff=d_ff,
                    n_heads=n_heads,
                    conv_kernel_size=self.conv_kernel_size,
                    dropout=dropout,
                    dropout_att=dropout_att,
                    rope_dim=rope_dim,
                    recompute_scores=self.should_recompute_att_scores[i],
                )
                for i in range(n_layers)
            ],
        )

        self.state = EncoderState(
            num_layers=self.n_layers,
            mhsa_stateless_layers=self.mhsa_stateless_layers,
            reduction_position=self.reduction_position,
            upsample_position=self.upsample_position,
            recompute_attention_scores=self.should_recompute_att_scores,
            chunk_size=self.chunk_size,
            mhsa_state_size=self.mhsa_state_size,
            reduction_factor=self.reduction_factor,
        )

    def forward(
        self,
        audio_signal: torch.Tensor,
        length: torch.Tensor | None = None,
        state_mhsa: torch.Tensor | None = None,
        state_conv: torch.Tensor | None = None,
        state_mhsa_len: torch.Tensor | None = None,
        state_subsampling_1: torch.Tensor | None = None,
        state_subsampling_2: torch.Tensor | None = None,
        state_reduction: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Processes an audio signal through the Conformer encoder.

        This method supports both offline (full sequence) and stateful streaming
        (chunk-by-chunk) inference modes.

        Args:
            audio_signal (torch.Tensor): Input audio features of shape (B, C_in, T_in).
                B is the batch size, C_in is the feature dimension, T_in is the input
                sequence length.
            length (torch.Tensor | None, optional): The lengths of the sequences in the batch,
                of shape (B,).
            state_mhsa (torch.Tensor | None, optional): State for MHSA layers, of shape
                (N, B, L_state, D). N is the number of stateful layers, L_state is the MHSA
                state size, D is the model dimension.
            state_conv (torch.Tensor | None, optional): State for convolution layers, of shape
                (N, B, D, L_conv_state). L_conv_state is the convolution state size.
            state_mhsa_len (torch.Tensor | None, optional): Lengths of MHSA states, of shape (B,).
            state_subsampling_1 (torch.Tensor | None, optional): State for the first
                subsampling layer.
            state_subsampling_2 (torch.Tensor | None, optional): State for the second
                subsampling layer.
            state_reduction (torch.Tensor | None, optional): State for the temporal
                reduction layer.

        Returns:
            A tuple containing:
            - The encoded output tensor of shape (B, D, T_out). T_out is the output
                sequence length.
            - The lengths of the encoded sequences, of shape (B,).

        """
        # (B, C, T) -> (B, T, C) for internal processing
        audio_signal = torch.transpose(audio_signal, 1, 2)

        self.state.setup(
            mhsa=state_mhsa,
            conv=state_conv,
            mhsa_len=state_mhsa_len,
            subsampling_1=state_subsampling_1,
            subsampling_2=state_subsampling_2,
            reduction=state_reduction,
        )

        # (B, T, C) -> (B, T, D)
        audio_signal, length = self.pre_encode(
            x=audio_signal,
            lengths=length,
            state=self.state,
        )
        # print(f"After encoder sub-sampling:\n{audio_signal.shape=};{length=}")

        # Create masks for each Conformer layer according to config
        self.state.create_masks(
            length=length,
            max_audio_length=int(audio_signal.size(1)),
        )

        for i, layer in enumerate(self.layers):
            self.state.update_before_layer(layer=i)

            audio_signal = layer(x=audio_signal, state=self.state)

            if i == self.reduction_position:
                audio_signal, length = self.temportal_reduction(audio_signal, length, self.state)

            if i == self.upsample_position:
                audio_signal, length = self.temporal_upsample(audio_signal, length, self.state)

            self.state.update_after_layer(layer=i)

        # (B, T, D) -> (B, D, T) for decoder compatibility
        audio_signal = torch.transpose(audio_signal, 1, 2)
        if length is not None:
            length = length.to(dtype=torch.int64)
        return audio_signal, length

    def get_initial_state(
        self,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
        len_dtype: torch.dtype = torch.int64,
        device: torch.device | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Generates the initial zero-states for a streaming inference pass.

        Args:
            batch_size (int, optional): The batch size for the states. Defaults to 1.
            dtype (torch.dtype, optional): The data type for the states.
            len_dtype (torch.dtype, optional): The data type for length tensors.
            device (torch.device | None, optional): The device to place states on.

        Returns:
            A tuple containing all initial state tensors for the encoder.

        """
        if device is None:
            device = next(self.parameters()).device

        conv_state_size = self.conv_context_size[0]
        state_mhsa = torch.zeros(
            (
                self.n_layers - self.mhsa_stateless_layers,
                batch_size,
                self.mhsa_state_size,
                self.d_model,
            ),
            device=device,
            dtype=dtype,
        )
        state_conv = torch.zeros(
            (len(self.layers), batch_size, self.d_model, conv_state_size),
            device=device,
            dtype=dtype,
        )
        state_mhsa_len = torch.zeros(batch_size, device=device, dtype=len_dtype)

        state_subsampling_1 = None
        state_subsampling_2 = None
        if hasattr(self.pre_encode, "get_initial_states"):
            (
                state_subsampling_1,
                state_subsampling_2,
            ) = self.pre_encode.get_initial_states(
                batch_size,
                device=device,
                dtype=dtype,
            )

        if self.temportal_reduction is not None:
            state_reduction = self.temportal_reduction.get_initial_states(
                batch_size,
                device=device,
                dtype=dtype,
            )
        else:
            state_reduction = None

        return (
            state_mhsa,
            state_conv,
            state_mhsa_len,
            state_subsampling_1,
            state_subsampling_2,
            state_reduction,
        )


class ConvASRDecoder(nn.Module):
    """A simple 1x1 Convolutional Decoder for ASR.

    This decoder projects the final encoder features into log probabilities over
    the vocabulary using a single 1x1 convolution, followed by a log-softmax.

    Args:
        feat_in (int): The number of input feature channels from the encoder.
        vocabulary (List[str]): The list of characters/tokens in the vocabulary.
            The blank token is added automatically.

    """

    def __init__(self, feat_in: int, vocabulary: list[str]) -> None:
        super().__init__()

        self.__vocabulary = vocabulary
        self._feat_in = feat_in
        # Add 1 for blank char
        self._num_classes = len(vocabulary) + 1

        self.decoder_layers = torch.nn.Sequential(
            torch.nn.Conv1d(self._feat_in, self._num_classes, kernel_size=1, bias=True),
        )

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of the decoder.

        Args:
            encoder_output (torch.Tensor): The output from the encoder of shape
                (B, D, T).

        Returns:
            torch.Tensor: The log probabilities over the vocabulary of shape
                (B, T, V), where V is the number of classes including blank.

        """
        # (B, D, T) -> (B, V, T) -> (B, T, V)
        return torch.nn.functional.log_softmax(
            self.decoder_layers(encoder_output).transpose(1, 2),
            dim=-1,
        )

    @property
    def vocabulary(self) -> list[str]:
        """Returns the model's vocabulary list (excluding blank)."""
        return self.__vocabulary

    @property
    def num_classes_with_blank(self) -> int:
        """Returns the number of output classes, including the CTC blank token."""
        return self._num_classes
