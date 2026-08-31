"""Packed matrix modules and the Qwen3-MoE expert adapter."""
from __future__ import annotations

import warnings
from dataclasses import asdict
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from ..quantize import PackedTensor, QuantizationPolicy, dequantize_rows
from .cuda_backend import extension

_WARNED_REFERENCE = False


def _warn_reference() -> None:
    global _WARNED_REFERENCE
    if not _WARNED_REFERENCE:
        warnings.warn(
            "Using NHDF CPU/PyTorch reconstruction. This path verifies semantics "
            "but is not a production inference kernel.",
            RuntimeWarning,
            stacklevel=3,
        )
        _WARNED_REFERENCE = True


class PackedMatrix(nn.Module):
    """Own one packed matrix and expose row-sliced projection operations."""

    def __init__(self, packed: PackedTensor):
        super().__init__()
        if packed.policy.mode == "raw":
            raise ValueError("PackedMatrix expects a quantized tensor")
        self.name = packed.name
        self.original_shape = tuple(packed.original_shape)
        self.policy_dict = asdict(packed.policy)
        self.rows = packed.rows
        self.original_cols = packed.original_cols
        self.padded_cols = packed.padded_cols
        self.groups_per_row = packed.groups_per_row

        for name, tensor in packed.tensor_dict().items():
            self.register_buffer(name, tensor, persistent=True)

    @property
    def policy(self) -> QuantizationPolicy:
        return QuantizationPolicy(**self.policy_dict)

    def as_packed_tensor(self, *, cpu: bool = False) -> PackedTensor:
        def value(name: str) -> torch.Tensor:
            t = getattr(self, name)
            return t.detach().cpu() if cpu else t

        return PackedTensor(
            name=self.name,
            original_shape=self.original_shape,
            policy=self.policy,
            rows=self.rows,
            original_cols=self.original_cols,
            padded_cols=self.padded_cols,
            groups_per_row=self.groups_per_row,
            base_codes=value("base_codes"),
            means=value("means"),
            scales=value("scales"),
            residual_mask_words=value("residual_mask_words"),
            residual_prefix=value("residual_prefix"),
            residual_bits=value("residual_bits"),
            residual_scales=value("residual_scales"),
            log_polar_address=value("log_polar_address"),
            parity_words=value("parity_words"),
        )

    def dequantize_rows(self, rows: torch.Tensor, *, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Decode selected flattened rows."""

        ext = extension()
        if ext is not None and self.base_codes.is_cuda:
            return ext.dequantize_rows(
                rows.to(device=self.base_codes.device, dtype=torch.int64),
                self.base_codes,
                self.means,
                self.scales,
                self.residual_mask_words,
                self.residual_prefix,
                self.residual_bits,
                self.residual_scales,
                self.rows,
                self.original_cols,
                self.padded_cols,
                self.groups_per_row,
                self.policy.base_bits,
                self.policy.group_size,
            ).to(dtype=dtype)

        _warn_reference()
        decoded = dequantize_rows(self.as_packed_tensor(cpu=True), rows.detach().cpu(), dtype=dtype)
        return decoded.to(device=rows.device)

    def project(
        self,
        x: torch.Tensor,
        *,
        row_offset: int = 0,
        row_count: int | None = None,
    ) -> torch.Tensor:
        """Compute ``x @ W.T`` for a contiguous flattened row interval."""

        if x.shape[-1] != self.original_cols:
            raise ValueError(f"expected input width {self.original_cols}, got {x.shape[-1]}")
        count = self.rows - row_offset if row_count is None else row_count
        if row_offset < 0 or count < 0 or row_offset + count > self.rows:
            raise ValueError("requested row interval is outside the packed matrix")

        original_prefix = x.shape[:-1]
        flat_x = x.reshape(-1, x.shape[-1]).contiguous()
        ext = extension()
        if ext is not None and flat_x.is_cuda and self.base_codes.is_cuda:
            if flat_x.dtype not in (torch.float16, torch.float32):
                flat_x = flat_x.to(torch.float16)
            out = ext.gemv(
                flat_x,
                self.base_codes,
                self.means,
                self.scales,
                self.residual_mask_words,
                self.residual_prefix,
                self.residual_bits,
                self.residual_scales,
                self.rows,
                self.original_cols,
                self.padded_cols,
                self.groups_per_row,
                self.policy.base_bits,
                self.policy.group_size,
                row_offset,
                count,
            )
            return out.reshape(*original_prefix, count)

        _warn_reference()
        rows = torch.arange(row_offset, row_offset + count, dtype=torch.int64, device=flat_x.device)
        weight = self.dequantize_rows(rows, dtype=flat_x.dtype)
        out = F.linear(flat_x, weight)
        return out.reshape(*original_prefix, count)


class NHDFPackedLinear(nn.Module):
    """Drop-in, bias-free linear layer backed by :class:`PackedMatrix`."""

    def __init__(self, packed: PackedTensor, bias: torch.Tensor | None = None):
        super().__init__()
        if packed.rows != packed.original_shape[0] or len(packed.original_shape) != 2:
            raise ValueError("NHDFPackedLinear requires a two-dimensional tensor")
        self.in_features = packed.original_shape[1]
        self.out_features = packed.original_shape[0]
        self.matrix = PackedMatrix(packed)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.matrix.project(x)
        return y if self.bias is None else y + self.bias


class NHDFPackedEmbedding(nn.Module):
    """Embedding table decoded only for requested token rows."""

    def __init__(self, packed: PackedTensor, padding_idx: int | None = None):
        super().__init__()
        if len(packed.original_shape) != 2:
            raise ValueError("embedding weight must be two-dimensional")
        self.num_embeddings, self.embedding_dim = packed.original_shape
        self.padding_idx = padding_idx
        self.matrix = PackedMatrix(packed)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        flat = input_ids.reshape(-1)
        unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        rows = self.matrix.dequantize_rows(unique, dtype=torch.float16)
        out = rows.index_select(0, inverse)
        return out.reshape(*input_ids.shape, self.embedding_dim)


class NHDFQwen3Experts(nn.Module):
    """Operator-equivalent replacement for ``Qwen3MoeExperts``.

    The module retains the official router contract: flattened hidden states,
    top-k expert indices and normalized top-k weights.  Only rows belonging to
    experts selected by the current token batch are reconstructed/projected.
    """

    def __init__(
        self,
        gate_up: PackedTensor,
        down: PackedTensor,
        *,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
    ):
        super().__init__()
        expected_gate = (num_experts, 2 * intermediate_dim, hidden_dim)
        expected_down = (num_experts, hidden_dim, intermediate_dim)
        if tuple(gate_up.original_shape) != expected_gate:
            raise ValueError(f"gate_up shape {gate_up.original_shape} != {expected_gate}")
        if tuple(down.original_shape) != expected_down:
            raise ValueError(f"down shape {down.original_shape} != {expected_down}")
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.gate_up = PackedMatrix(gate_up)
        self.down = PackedMatrix(down)
        self.activation = activation

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError("Qwen3 expert adapter expects flattened [tokens, hidden] states")
        final_hidden_states = torch.zeros_like(hidden_states)
        # This loop mirrors the official correctness implementation.  A later
        # optimization can compact all expert-token pairs into a single launch.
        for expert_idx in torch.unique(top_k_index, sorted=True).tolist():
            token_idx, top_k_pos = torch.where(top_k_index == expert_idx)
            if token_idx.numel() == 0:
                continue
            current = hidden_states.index_select(0, token_idx)
            gate_offset = int(expert_idx) * 2 * self.intermediate_dim
            gate_up = self.gate_up.project(
                current,
                row_offset=gate_offset,
                row_count=2 * self.intermediate_dim,
            )
            gate, up = gate_up.chunk(2, dim=-1)
            activated = self.activation(gate) * up
            down_offset = int(expert_idx) * self.hidden_dim
            projected = self.down.project(
                activated,
                row_offset=down_offset,
                row_count=self.hidden_dim,
            )
            projected = projected * top_k_weights[token_idx, top_k_pos, None].to(projected.dtype)
            final_hidden_states.index_add_(0, token_idx, projected.to(final_hidden_states.dtype))
        return final_hidden_states
