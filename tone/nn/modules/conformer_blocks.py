"""Module defining the core building blocks for the Conformer encoder."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from tone.nn.modules.submodules import (
    CausalConv1D,
    MultiHeadAttention,
    RMSNorm,
    RotaryPositionalEmbeddings,
)
from tone.nn.torch_utils import avoid_float16_autocast_context


@dataclass
class LayerState:
    """A dataclass to hold the state for a single Conformer layer.

    This structure organizes the recurrent states and pre-computed masks for one
    layer within the encoder stack.

    Attributes:
        conv (torch.Tensor | None): The state for the causal convolution module.
        mhsa (torch.Tensor | None): The Key-Value cache for the MHSA module.
        pad_mask (torch.Tensor | None): The padding mask for this layer.
        att_mask (torch.Tensor | None): The attention mask for this layer.

    """

    conv: torch.Tensor = None
    mhsa: torch.Tensor = None
    pad_mask: torch.Tensor = None
    att_mask: torch.Tensor = None


class EncoderState:
    """A state manager for the Conformer encoder during streaming inference.

    This class is not an `nn.Module` but a helper that orchestrates the
    flow of all recurrent states (for convolutions, attention, etc.) and
    manages the creation of attention masks across the layers of the encoder.
    It is initialized before each forward pass.

    Args:
        num_layers (int): The total number of Conformer layers.
        mhsa_stateless_layers (int): The number of initial layers that do not
            use an MHSA state cache.
        reduction_position (int): The layer index for temporal reduction.
        upsample_position (int): The layer index for temporal upsampling.
        reduction_factor (int): The factor by which sequence length is reduced.
        recompute_attention_scores (list): A boolean list indicating whether
            to recompute attention scores for each layer.
        chunk_size (int): The chunk size for chunk-wise attention masking.
        mhsa_state_size (int): The maximum size of the MHSA state cache.

    """

    def __init__(
        self,
        num_layers: int,
        mhsa_stateless_layers: int,
        reduction_position: int,
        upsample_position: int,
        reduction_factor: int,
        recompute_attention_scores: list,
        chunk_size: int,
        mhsa_state_size: int,
    ) -> None:
        self.num_layers = num_layers
        self.reduction_position = reduction_position
        self.upsample_position = upsample_position
        self.mhsa_stateless_layers = mhsa_stateless_layers
        self.reduction_factor = reduction_factor
        self.recompute_attention_scores = recompute_attention_scores
        self.left_context_size = mhsa_state_size
        self.mhsa_state_size = mhsa_state_size
        self.chunk_size = int(chunk_size)
        self.mhsa_pad_length = max(mhsa_state_size, chunk_size)
        self.state_keep_size: int | None = None

    def setup(
        self,
        mhsa: torch.Tensor | None = None,
        conv: torch.Tensor | None = None,
        mhsa_len: torch.Tensor | None = None,
        subsampling_1: torch.Tensor | None = None,
        subsampling_2: torch.Tensor | None = None,
        reduction: torch.Tensor | None = None,
    ) -> None:
        """Initializes or resets the state for a new forward pass.

        Args:
            mhsa (torch.Tensor | None): The stacked MHSA states.
            conv (torch.Tensor | None): The stacked convolution states.
            mhsa_len (torch.Tensor | None): The lengths of the MHSA states.
            subsampling_1 (torch.Tensor | None): State for the 1st subsampling layer.
            subsampling_2 (torch.Tensor | None): State for the 2nd subsampling layer.
            reduction (torch.Tensor | None): State for the temporal reduction layer.

        """
        self.streaming = mhsa is not None
        self.mhsa = None
        self.conv = None
        self.reduction = reduction
        self.subsampling = [subsampling_1, subsampling_2] if self.streaming else None
        self.mhsa_len = mhsa_len
        self.att_scores = None
        self.residual = None
        self.att_mask = None
        self.pad_mask = None
        self.layers = [
            LayerState(
                mhsa=(mhsa[i - self.mhsa_stateless_layers] if i >= self.mhsa_stateless_layers else None),
                conv=conv[i],
            )
            if self.streaming
            else LayerState()
            for i in range(self.num_layers)
        ]

    def update_before_layer(self, layer: int) -> None:
        """Prepares the shared state before a Conformer layer is executed.

        This method loads the specific states for the given layer into the main
        attributes of this object, making them accessible to the layer.

        Args:
            layer (int): The index of the Conformer layer about to be run.

        """
        layer_state = self.layers[layer]
        self.mhsa = layer_state.mhsa
        self.conv = layer_state.conv
        self.pad_mask = layer_state.pad_mask
        self.att_mask = layer_state.att_mask

        if self.mhsa is not None:
            self.mhsa = self.mhsa[:, -self.mhsa_state_size :, :]

        # Attention scores are reused if this is false
        if self.recompute_attention_scores[layer]:
            self.att_scores = None

    def update_after_layer(self, layer: int) -> None:
        """Saves the updated states back to storage after a layer has run.

        Args:
            layer (int): The index of the Conformer layer that has just run.

        """
        if self.mhsa is not None:
            pad_size = self.mhsa_pad_length - self.mhsa.size(1)
            self.mhsa = nn.functional.pad(self.mhsa, (0, 0, pad_size, 0))

        # Copy state from self to self.layers[layer] storage
        layer_state = self.layers[layer]
        if layer_state.mhsa is not None:
            layer_state.mhsa = self.mhsa
        if layer_state.conv is not None:
            layer_state.conv = self.conv

        # Adjust state size expectations for subsequent layers
        if layer == self.reduction_position:
            self.mhsa_state_size //= self.reduction_factor
        if layer == self.upsample_position:
            self.mhsa_state_size *= self.reduction_factor

    def next(self) -> EncoderState:
        """Prepares and returns the final state for the next inference chunk.

        Gathers the states from all layers and stacks them into single tensors
        ready to be output from the model.

        Returns:
            EncoderState: The `self` object with updated, stacked states.

        """
        mhsa_stateful_layers = self.layers[self.mhsa_stateless_layers :]
        self.mhsa = torch.stack([layer.mhsa for layer in mhsa_stateful_layers], dim=0)
        self.conv = torch.stack([layer.conv for layer in self.layers], dim=0)
        self.mhsa_len = torch.clamp(self.mhsa_len + self.state_keep_size, max=self.mhsa_state_size)

        # Trim to mhsa_state_size if state is too long
        self.mhsa = self.mhsa[:, :, -self.mhsa_state_size :, :]
        return self

    def create_masks(self, length: torch.Tensor | None, max_audio_length: int) -> None:
        """Creates attention and padding masks for all Conformer layers.

        Args:
            length (torch.Tensor | None): The length of the input sequence.
            max_audio_length (int): The maximum sequence length in the batch.

        """
        if self.streaming:
            self.state_keep_size = max_audio_length
            max_audio_length = max_audio_length + self.mhsa_state_size
            padding_length = (
                torch.full_like(self.mhsa_len, max_audio_length) if length is None else length + self.mhsa_state_size
            )
        else:
            padding_length = length

        padding_length_original = padding_length
        padding_length_reduced = padding_length // self.reduction_factor

        att_mask = pad_mask = None
        for layer_index, layer in enumerate(self.layers):
            att_mask, pad_mask = self._update_masks(
                layer_index, att_mask, pad_mask, padding_length, max_audio_length,
            )
            if layer_index == self.reduction_position:
                padding_length = padding_length_reduced
            if layer_index == self.upsample_position:
                padding_length = padding_length_original

            layer.att_mask = att_mask
            layer.pad_mask = pad_mask

    def _update_masks(
        self,
        layer_index: int,
        att_mask: torch.Tensor | None,
        pad_mask: torch.Tensor | None,
        padding_length: torch.Tensor,
        max_audio_length: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Helper to compute masks for a single layer, handling reduction/upsampling."""
        mhsa_state_size = self.mhsa_state_size if self.streaming else 0
        chunk_size = self.chunk_size
        offset = mhsa_state_size - self.mhsa_len if self.streaming else None

        # Skip layers that don't require mask updates
        relevant_layers = [
            0,
            self.reduction_position + 1,
            self.upsample_position + 1,
            self.mhsa_stateless_layers,
        ]
        if layer_index not in relevant_layers:
            return att_mask, pad_mask

        # Handle time-reduced layers
        in_reduced_block = self.reduction_position < layer_index <= self.upsample_position
        if in_reduced_block:
            reduction = self.reduction_factor
            orig_state_size = mhsa_state_size
            mhsa_state_size //= reduction
            chunk_size //= reduction
            left_context = self.left_context_size // reduction

            if self.streaming:
                # Здесь всегда делался max_audio_length = math.ceil(max_audio_length / reduction)
                # как будто мы делаем правый паддинг
                # но правый паддинг мы делаем только в оффлайн режим
                # см. CausalTemporalReduction forward
                chunk_len = max_audio_length - orig_state_size
                max_audio_length = mhsa_state_size + (chunk_len // reduction)
            else:
                # Offline temporal reduction pads; keep ceil behavior
                max_audio_length = math.ceil(max_audio_length / reduction)

            offset = offset // reduction if offset is not None else None
        else:
            left_context = self.left_context_size

        # Special handling for stateless layers before mhsa_stateless_layers
        if layer_index < self.mhsa_stateless_layers:
            if self.streaming:
                return None, None
            # No left context in non-streaming for stateless layers
            left_context = 0
            padding_length = padding_length - mhsa_state_size
            max_audio_length = max_audio_length - mhsa_state_size

        # Compute masks
        pad_mask, att_mask = self._create_pad_and_attention_masks(
            mhsa_state_size=left_context,
            chunk_size=chunk_size,
            padding_length=padding_length,
            max_audio_length=max_audio_length,
            offset=offset,
        )

        # Slice state if needed (streaming with stateful layers)
        if self.streaming and layer_index >= self.mhsa_stateless_layers:
            pad_mask = pad_mask[:, mhsa_state_size:]
            att_mask = att_mask[:, mhsa_state_size:]

        return att_mask, pad_mask

    def _create_pad_and_attention_masks(
        self,
        mhsa_state_size: int,
        chunk_size: int,
        padding_length: torch.Tensor,
        max_audio_length: int,
        offset: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Core logic to create padding and chunk-wise causal attention masks."""
        device = padding_length.device
        batch_size = padding_length.size(0)

        if self.streaming:
            # Allow full attention in streaming mode
            attention_mask = torch.ones(
                (max_audio_length, max_audio_length),
                dtype=torch.bool,
                device=device,
            ).unsqueeze(0)
        else:
            # Simulate streaming attention with state and chunked window
            row_idx = torch.arange(max_audio_length, device=device).unsqueeze(1)
            col_idx = torch.arange(max_audio_length, device=device).unsqueeze(0)

            position_in_chunk = row_idx % chunk_size
            chunk_start = row_idx - position_in_chunk

            in_chunk = (col_idx >= chunk_start) & (col_idx < chunk_start + chunk_size)
            in_state = (col_idx >= chunk_start - mhsa_state_size) & (col_idx < chunk_start)

            attention_mask = (in_chunk | in_state).squeeze(1).unsqueeze(0)

        # Build padding mask: [batch_size, max_audio_length]
        time_indices = torch.arange(max_audio_length, device=device)
        time_indices = time_indices.reshape(1, -1).expand(batch_size, -1)
        padding_length = padding_length.reshape(-1, 1).expand(-1, max_audio_length)
        padding_mask = time_indices < padding_length

        if offset is not None:
            offset_mask = time_indices >= offset.reshape(-1, 1).expand(-1, max_audio_length)
            padding_mask = padding_mask & offset_mask

        # Mask out padding tokens in the attention mask as well
        expanded_padding_mask = padding_mask.unsqueeze(1) & padding_mask.unsqueeze(2)
        attention_mask = attention_mask[:, :max_audio_length, :max_audio_length]
        attention_mask = attention_mask & expanded_padding_mask

        return ~padding_mask, ~attention_mask


class ConformerConvolution(nn.Module):
    """The convolution module for the Conformer model.

    This module consists of a pointwise convolution, a gated linear unit,
    a depthwise causal convolution, and another pointwise convolution.

    Args:
        d_model (int): The hidden dimension of the model.
        kernel_size (int): The kernel size for the depthwise convolution.

    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        self.d_model = d_model
        self.kernel_size = kernel_size

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.depthwise_conv = CausalConv1D(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            groups=d_model,
            bias=True,
        )

        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x: torch.Tensor, state: EncoderState) -> torch.Tensor:
        """Forward pass for Conformer convolution.

        Args:
            x (torch.Tensor): input tensor of size (B, T, D), where B is the batch size,
                T is the sequence length, and D is the hidden model dimension.
            state (EncoderState): state of the Encoder.

        Returns:
            torch.Tensor: output tensor of size (B, T, D).

        """
        # (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)

        # (B, D, T) -> (B, 2 * D, T)
        x = self.pointwise_conv1(x)

        # (B, 2 * D, T) -> (B, D, T)
        x = nn.functional.glu(x, dim=1)

        if state.pad_mask is not None:
            x = x.float().masked_fill(state.pad_mask.unsqueeze(1), 0.0)

        x = self.depthwise_conv(x, state=state.conv)
        if state.streaming:
            x, state.conv = x

        x = self.batch_norm(x)

        x = self.activation(x)
        x = self.pointwise_conv2(x)
        # (B, D, T) -> (B, T, D)
        return x.transpose(1, 2)


class ConformerFeedForward(nn.Module):
    """The feed-forward module for the Conformer model.

    This uses a variant with a gated linear unit (GLU) structure.

    Args:
        d_model (int): The input and output model dimension.
        d_ff (int): The intermediate hidden dimension.
        dropout (float, optional): The dropout rate. Defaults to 0.
        activation (nn.Module, optional): The activation function. Defaults to nn.SiLU().

    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float | None = 0,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = activation if activation else nn.SiLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.linearv = nn.Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for Conformer feed-forward.

        Args:
            x (torch.Tensor): input tensor of size (B, T, D)

        Returns:
            torch.Tensor: output tensor of size (B, T, D)

        """
        # (B, T, D) -> (B, T, D_ff)
        gate = self.activation(self.linear1(x))

        # (B, T, D_ff) -> (B, T, D)
        return self.linear2(gate * self.linearv(x))


class ConvSubsamplingPreEncode(torch.nn.Module):
    """The initial convolutional subsampling module.

    This module reduces the sequence length and projects features into d_model
    using a stack of 2D convolutions.

    Args:
        feat_in (int): The number of input feature channels.
        feat_out (int): The target output feature dimension (d_model).
        conv_channels (List[int]): A list of output channels for each conv layer.
        activation (nn.Module, optional): The activation function. Defaults to nn.SiLU().
        kernel_sizes (Sequence[Sequence[int]], optional): Kernel sizes for each conv layer.
        strides (Sequence[Sequence[int]], optional): Strides for each conv layer.

    """

    def __init__(
        self,
        feat_in: int,
        feat_out: int,
        conv_channels: list[int],
        activation: nn.Module | None = None,
        kernel_sizes: Sequence[Sequence[int]] = ((11, 21), (11, 11)),
        strides: Sequence[Sequence[int]] = ((1, 1), (3, 1)),
    ) -> None:
        super().__init__()
        self._feat_in = feat_in
        self._feat_out = feat_out
        self.strides = strides
        self.kernel_sizes = kernel_sizes
        self.subsampling_factor = int(strides[0][0] * strides[1][0])
        self.activation = activation if activation else nn.SiLU()
        self.state_lens = [kernel_sizes[i][0] - strides[i][0] for i in range(2)]
        self.conv_channels = conv_channels
        in_channels = 1

        self.pre_norm = RMSNorm(feat_in)

        self.conv = nn.ModuleList()
        for i in range(2):
            self.conv.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels[i],
                        kernel_size=self.kernel_sizes[i],
                        stride=strides[i],
                        padding=0,
                    ),
                    nn.BatchNorm2d(conv_channels[i]),
                    self.activation,
                ).to(memory_format=torch.channels_last),
            )
            in_channels = conv_channels[i]

        self.hidden_features = []
        feat_hidden = feat_in

        for i in range(2):
            feat_hidden = feat_hidden - self.kernel_sizes[i][1]
            feat_hidden = (feat_hidden // self.strides[i][1]) + 1
            self.hidden_features.append(feat_hidden)

        self.out = nn.Linear(
            conv_channels[-1] * int(feat_hidden),
            self._feat_out,
            bias=False,
        )
        self.out_norm = RMSNorm(self._feat_out)

    def get_output_length(self, lengths: torch.Tensor) -> torch.Tensor:
        """Get the output sequence length after subsampling.

        Args:
            lengths (torch.Tensor): input sequence lengths.

        Returns:
            torch.Tensor: output sequence lengths.

        """
        for i in range(2):
            lengths = (lengths - self.kernel_sizes[i][0] + self.state_lens[i]) // self.strides[i][0] + 1

        return lengths.to(dtype=torch.int)

    def get_initial_states(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the initial states for the convolutional subsampling module.

        Args:
            batch_size (int): The batch size.
            device (torch.device): The device to use.
            dtype (torch.dtype, optional): The data type. Defaults to torch.float16.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The initial states.

        """
        state_subsampling_1 = torch.zeros(
            size=(batch_size, 1, self.state_lens[0], self._feat_in),
            device=device,
            dtype=dtype,
        )
        state_subsampling_2 = torch.zeros(
            size=(
                batch_size,
                self.conv_channels[0],
                self.state_lens[1],
                self.hidden_features[0],
            ),
            device=device,
            dtype=dtype,
        )

        return state_subsampling_1, state_subsampling_2

    def get_sampling_frames(self) -> int:
        """Returns the subsampling factor.

        Returns:
            int: The subsampling factor.

        """
        return self.subsampling_factor

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, state: EncoderState) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the convolutional subsampling module.

        Args:
            x (torch.Tensor): input signals of shape (B, T_in, F_in), where B is the batch size,
                T_in is the input sequence length, and F_in is the number of input features.
            lengths (torch.Tensor): input lengths of shape (B)
            state (EncoderState): state for the current layer of the Conformer
        Returns:
            tuple[torch.Tensor, torch.Tensor]: tuple of output tensor of shape (B, T_out, D), where
                T_out is the output sequence length and D is the hidden model dimension, and lengths
                of shape (B).

        """
        if lengths is not None:
            lengths = self.get_output_length(lengths)

        dtype = x.dtype
        x = self.pre_norm(x.contiguous()).type(dtype)

        # => (batch, 1, time_in, feat_in)
        x = x.unsqueeze(1).to(memory_format=torch.channels_last)

        for i, layer in enumerate(self.conv):
            if state.streaming:
                x = torch.cat([state.subsampling[i].to(dtype), x], dim=2)
                state.subsampling[i] = x[:, :, -self.state_lens[i] :, :]
                x = layer(x)
            else:
                # pad length dim in non-streaming mode
                x = nn.functional.pad(x, (0, 0, self.state_lens[i], 0))
                x = layer(x)
            # x => (batch, conv_channels[i], state_lens[i] + time[i], hidden_features[i-1])

        # => (batch, time_out, d_model), time_out = time_in // subsampling_factor
        x = self.out(x.transpose(1, 2).flatten(start_dim=2))
        x = x.to(memory_format=torch.contiguous_format)
        x = self.out_norm(x)

        return x, lengths


class RotaryMultiHeadAttention(MultiHeadAttention):
    """A MHA layer that incorporates Rotary Positional Embeddings (RoPE).

    Args:
        n_head (int): The number of attention heads.
        n_feat (int): The total feature dimension (d_model).
        dropout_rate (float): The dropout rate for the attention mechanism.
        rope_dim (int | None): The feature dimension for RoPE. Defaults to 64.
        recompute_scores (bool | None): If True, recomputes Q/K projections
            and attention scores. Defaults to True.

    """

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout_rate: float,
        rope_dim: int = 64,
        recompute_scores: bool = True,
    ) -> None:
        super().__init__(
            n_head=n_head,
            n_feat=n_feat,
            dropout_rate=dropout_rate,
            recompute_scores=recompute_scores,
        )
        self.recompute_scores = recompute_scores
        if recompute_scores:
            self.rotary_emb_k = RotaryPositionalEmbeddings(rope_dim)
            self.rotary_emb_q = RotaryPositionalEmbeddings(rope_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, state: EncoderState) -> torch.Tensor:
        """Computes Multi-Head Attention with rotary embeddings.

        Args:
            query (torch.Tensor): Query tensor of shape (B, T_q, D).
            key (torch.Tensor): Key tensor of shape (B, T_k, D).
            value (torch.Tensor): Value tensor of shape (B, T_v, D).
            state (EncoderState): The current encoder state object, which provides
                KV cache, attention masks, and pre-computed scores.

        Returns:
            torch.Tensor: The output context vector of shape (B, T_q, D).

        """
        key, value, query = self.update_state(
            key=key,
            value=value,
            query=query,
            state=state,
        )

        if torch.is_autocast_enabled():
            query, key, value = (
                query.to(torch.float32),
                key.to(torch.float32),
                value.to(torch.float32),
            )

        with avoid_float16_autocast_context():
            q, k, v = self.forward_qkv(query, key, value)

            if state.att_scores is None:
                q = self.rotary_emb_q(q, offset=0)
                k = self.rotary_emb_k(
                    k,
                    offset=state.mhsa_state_size if state.mhsa is not None else 0,
                )
                state.att_scores = torch.matmul(q, k.transpose(-2, -1)) / self.s_d_k
            return self.forward_attention(v, state.att_scores, state.att_mask)


class ConformerLayer(nn.Module):
    """A single, complete Conformer block.

    This block implements the "Macaron" structure:
    0.5 * FFN -> Self-Attention -> Convolution -> 0.5 * FFN -> LayerNorm

    Args:
        d_model (int): The hidden dimension of the model.
        d_ff (int): The intermediate dimension of the feed-forward layers.
        n_heads (int, optional): The number of attention heads. Defaults to 4.
        conv_kernel_size (int, optional): The kernel size for the depthwise conv. Defaults to 31.
        dropout (float, optional): The general dropout rate. Defaults to 0.1.
        dropout_att (float, optional): The attention dropout rate. Defaults to 0.1.
        rope_dim (int, optional): The feature dimension for RoPE. Defaults to 64.
        recompute_scores (bool, optional): Whether to recompute attention scores.

    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        dropout_att: float = 0.1,
        rope_dim: int = 64,
        recompute_scores: bool = True,
    ) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.fc_factor = 0.5

        # First feed forward module
        self.norm_feed_forward1 = RMSNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

        # Convolution module
        self.norm_conv = RMSNorm(d_model)
        self.conv = ConformerConvolution(
            d_model=d_model,
            kernel_size=conv_kernel_size,
        )

        # Multi-headed self-attention module
        self.norm_self_att = RMSNorm(d_model)
        self.self_attn = RotaryMultiHeadAttention(
            n_head=n_heads,
            n_feat=d_model,
            dropout_rate=dropout_att,
            rope_dim=rope_dim,
            recompute_scores=recompute_scores,
        )

        # Second feed forward module
        self.norm_feed_forward2 = RMSNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.norm_out = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, state: EncoderState) -> torch.Tensor:
        """Forward pass for a single Conformer layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, D).
            state (EncoderState): The state manager for the current layer.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, D).

        """
        residual = x.contiguous()
        x = x.contiguous()
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_self_att(residual)

        x = self.self_attn(
            query=x,
            key=x,
            value=x,
            state=state,
        )

        residual = residual + self.dropout(x)

        x = self.norm_conv(residual)
        x = self.conv(x, state=state)

        residual = residual + self.dropout(x)

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * self.fc_factor

        return self.norm_out(residual)


class CausalTemporalReduction(nn.Module):
    """A module to downsample the sequence length in a causal manner.

    Args:
        d_model (int): The number of feature channels.
        kernel_size (int, optional): The kernel size of the convolution. Defaults to 5.
        reduction_factor (int, optional): The stride of the convolution, i.e.,
            the downsampling factor. Defaults to 2.

    """

    def __init__(self, d_model: int, kernel_size: int | None = 5, reduction_factor: int | None = 2) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * 4,
            kernel_size=kernel_size,
            stride=reduction_factor,
            groups=d_model,
            bias=True,
        )
        self.conv_pw = nn.Conv1d(
            in_channels=d_model * 4,
            out_channels=d_model,
            kernel_size=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.kernel_size = kernel_size
        self.reduction_factor = reduction_factor
        assert kernel_size > reduction_factor, (
            "kernel_size must be greater than reduction_factor in temporal reduction module"
        )

    def forward(
        self,
        audio_signal: torch.Tensor,
        lengths: torch.Tensor | None,
        state: EncoderState,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Downsamples the audio signal in the time dimension using a causal convolution.

        Args:
           audio_signal (torch.Tensor): input signals of shape (B, T, D).
           lengths (torch.Tensor): input lengths of shape (B).
           state (EncoderState): state for the current layer of the Conformer.

        """
        state.residual = audio_signal
        # (B, T, D) -> (B, D, T)
        x = torch.transpose(audio_signal, 1, 2)

        # Add right padding to make input length divisible by reduction_factor
        if state.reduction is None:
            right_pad = (
                self.reduction_factor - x.size(2) % self.reduction_factor
                if x.size(2) % self.reduction_factor > 0
                else 0
            )
            x = nn.functional.pad(x, (self.kernel_size - self.reduction_factor, right_pad))
            x = self.conv(x)
            if lengths is not None:
                lengths = lengths // self.reduction_factor
        else:
            x = torch.cat([state.reduction, x], dim=-1)
            state.reduction = x[:, :, -state.reduction.size(-1) :]
            x = self.conv(x)
        x = self.conv_pw(x)

        # (B, D, T) -> (B, T, D)
        x = torch.transpose(x, 1, 2)
        return x, lengths

    def get_initial_states(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """Generates the initial zero states for the temporal reduction module.

        Args:
            batch_size (int): The batch size.
            device (torch.device): The device to use.
            dtype (torch.dtype, optional): The data type. Defaults to torch.float16.

        Returns:
            torch.Tensor: The initial states.

        """
        state_len = self.kernel_size - self.reduction_factor

        return torch.zeros(
            size=(batch_size, self.conv.in_channels, state_len),
            device=device,
            dtype=dtype,
        )


class TemporalUpsampling(nn.Module):
    """A module to upsample the sequence length.

    This is typically used to restore the sequence length after a
    `CausalTemporalReduction` block.

    Args:
        upsampling_factor (int, optional): The factor by which to upsample.
            Defaults to 2.

    """

    def __init__(self, upsampling_factor: int = 2) -> None:
        super().__init__()
        self.upsampling_factor = int(upsampling_factor)

    def forward(
        self,
        audio_signal: torch.Tensor,
        length: torch.Tensor | None,
        state: EncoderState,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Upsamples the signal by repeating elements and adds a residual.

        Args:
           audio_signal (torch.Tensor): input tensor of shape (B, T, D).
           length (torch.Tensor): input lengths of shape (B).
           state (EncoderState): state for the current layer of the Conformer.

        """
        if state.residual is None:
            raise RuntimeError("TemporalUpsampling requires state.residual to be set.")
        r = self.upsampling_factor
        # (B, T_red, D) -> (B, r*T_red, D)
        audio_signal = torch.repeat_interleave(
            audio_signal,
            repeats=self.upsampling_factor,
            dim=1,
        )
        if r > 1:
            audio_signal = F.pad(audio_signal, (0, 0, 0, r - 1))
        T_res = state.residual.shape[1]
        audio_signal = audio_signal[:, :T_res, :]
        audio_signal = audio_signal + state.residual

        if length is not None:
            length = length * r
            length = torch.clamp(length, max=T_res)

        return audio_signal, length
