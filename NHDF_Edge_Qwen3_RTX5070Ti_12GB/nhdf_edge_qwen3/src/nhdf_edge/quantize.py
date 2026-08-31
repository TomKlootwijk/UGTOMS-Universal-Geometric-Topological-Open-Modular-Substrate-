"""NHDF mixed-precision weight packer.

This is a semantics-first reference implementation.  It is designed to be
reproducible and inspectable rather than to be the fastest possible quantizer.
The production decode path can consume the same layout through the optional
CUDA GEMV extension.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from .bitpack import (
    bits_from_words,
    pack_bits,
    pack_unsigned,
    prefix_counts,
    unpack_bits,
    unpack_unsigned,
    words_from_bits,
)
from .operators import (
    branch_score,
    byte_stream_parity,
    log_polar_encode,
    select_bounded_branches,
    weighted_zero_set_residual,
)


@dataclass(frozen=True)
class QuantizationPolicy:
    """Policy for one tensor category."""

    mode: str = "quantized"  # quantized | raw
    base_bits: int = 2
    group_size: int = 256
    residual_fraction: float = 0.15
    phase_gain: float = 0.25
    gamma: float = 1.0
    radial_bins: int = 16
    angular_bins: int = 16
    iterations: int = 3
    raw_dtype: str = "float16"

    def validate(self) -> None:
        if self.mode not in {"quantized", "raw"}:
            raise ValueError("mode must be 'quantized' or 'raw'")
        if self.mode == "quantized" and self.base_bits not in (2, 4):
            raise ValueError("the reference runtime supports 2-bit or 4-bit base weights")
        if self.group_size <= 0 or self.group_size % 256 != 0:
            raise ValueError("group_size must be a positive multiple of 256 for the CUDA profile")
        if not (0.0 <= self.residual_fraction <= 1.0):
            raise ValueError("residual_fraction must be in [0, 1]")
        if self.base_bits == 4 and self.residual_fraction != 0.0:
            raise ValueError("the reference profile uses residual branches only for 2-bit tensors")
        if self.iterations < 1:
            raise ValueError("iterations must be at least one")


@dataclass
class PackedTensor:
    """In-memory representation of one packed tensor."""

    name: str
    original_shape: tuple[int, ...]
    policy: QuantizationPolicy
    rows: int
    original_cols: int
    padded_cols: int
    groups_per_row: int
    base_codes: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.uint8))
    means: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float16))
    scales: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float16))
    residual_mask_words: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    residual_prefix: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    residual_bits: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.uint8))
    residual_scales: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float16))
    log_polar_address: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.uint8))
    parity_words: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    raw: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float16))
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def total_groups(self) -> int:
        return self.rows * self.groups_per_row

    @property
    def selected_groups(self) -> int:
        return int(self.residual_scales.numel())

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "original_shape": list(self.original_shape),
            "policy": asdict(self.policy),
            "rows": self.rows,
            "original_cols": self.original_cols,
            "padded_cols": self.padded_cols,
            "groups_per_row": self.groups_per_row,
            "stats": self.stats,
        }

    def tensor_dict(self, prefix: str = "") -> dict[str, torch.Tensor]:
        p = f"{prefix}." if prefix else ""
        if self.policy.mode == "raw":
            return {f"{p}raw": self.raw.contiguous()}
        return {
            f"{p}base_codes": self.base_codes.contiguous(),
            f"{p}means": self.means.contiguous(),
            f"{p}scales": self.scales.contiguous(),
            f"{p}residual_mask_words": self.residual_mask_words.contiguous(),
            f"{p}residual_prefix": self.residual_prefix.contiguous(),
            f"{p}residual_bits": self.residual_bits.contiguous(),
            f"{p}residual_scales": self.residual_scales.contiguous(),
            f"{p}log_polar_address": self.log_polar_address.contiguous(),
            f"{p}parity_words": self.parity_words.contiguous(),
        }

    @classmethod
    def from_tensor_dict(
        cls,
        metadata: dict[str, Any],
        tensors: dict[str, torch.Tensor],
        *,
        prefix: str = "",
    ) -> "PackedTensor":
        p = f"{prefix}." if prefix else ""
        policy = QuantizationPolicy(**metadata["policy"])
        common = dict(
            name=metadata["name"],
            original_shape=tuple(metadata["original_shape"]),
            policy=policy,
            rows=int(metadata["rows"]),
            original_cols=int(metadata["original_cols"]),
            padded_cols=int(metadata["padded_cols"]),
            groups_per_row=int(metadata["groups_per_row"]),
            stats=dict(metadata.get("stats", {})),
        )
        if policy.mode == "raw":
            return cls(**common, raw=tensors[f"{p}raw"])
        return cls(
            **common,
            base_codes=tensors[f"{p}base_codes"],
            means=tensors[f"{p}means"],
            scales=tensors[f"{p}scales"],
            residual_mask_words=tensors[f"{p}residual_mask_words"],
            residual_prefix=tensors[f"{p}residual_prefix"],
            residual_bits=tensors[f"{p}residual_bits"],
            residual_scales=tensors[f"{p}residual_scales"],
            log_polar_address=tensors[f"{p}log_polar_address"],
            parity_words=tensors[f"{p}parity_words"],
        )


def _levels(bits: int, device: torch.device) -> torch.Tensor:
    if bits == 2:
        return torch.tensor([-1.5, -0.5, 0.5, 1.5], dtype=torch.float32, device=device)
    if bits == 4:
        # Mid-rise symmetric levels: -7.5, -6.5, ..., +7.5.
        return torch.arange(16, dtype=torch.float32, device=device) - 7.5
    raise ValueError(f"unsupported base bit width: {bits}")


def _prepare_groups(weight: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int, int, int, int]:
    if weight.ndim == 0:
        weight = weight.reshape(1)
    original_shape = tuple(weight.shape)
    original_cols = int(original_shape[-1]) if original_shape else 1
    rows = int(math.prod(original_shape[:-1])) if len(original_shape) > 1 else 1
    padded_cols = ((original_cols + group_size - 1) // group_size) * group_size
    matrix = weight.detach().to(device="cpu", dtype=torch.float32).reshape(rows, original_cols)
    if padded_cols != original_cols:
        matrix = torch.nn.functional.pad(matrix, (0, padded_cols - original_cols))
    groups_per_row = padded_cols // group_size
    groups = matrix.view(rows * groups_per_row, group_size)
    return groups, rows, original_cols, padded_cols, groups_per_row


def _prepare_hessian(
    hessian_diag: torch.Tensor | None,
    *,
    rows: int,
    original_cols: int,
    padded_cols: int,
    group_size: int,
) -> torch.Tensor | None:
    if hessian_diag is None:
        return None
    h = hessian_diag.detach().to(device="cpu", dtype=torch.float32)
    if h.ndim == 1 and h.numel() == original_cols:
        h = h.unsqueeze(0).expand(rows, -1)
    else:
        h = torch.broadcast_to(h, (rows, original_cols)).clone()
    if padded_cols != original_cols:
        h = torch.nn.functional.pad(h, (0, padded_cols - original_cols))
    return h.view(-1, group_size).clamp_min(0.0)


def _weighted_mean(x: torch.Tensor, h: torch.Tensor | None) -> torch.Tensor:
    if h is None:
        return x.mean(dim=-1, keepdim=True)
    return torch.sum(h * x, dim=-1, keepdim=True) / torch.sum(h, dim=-1, keepdim=True).clamp_min(1e-12)


def _weighted_scale(
    centered: torch.Tensor,
    level_values: torch.Tensor,
    h: torch.Tensor | None,
) -> torch.Tensor:
    if h is None:
        numerator = torch.sum(level_values * centered, dim=-1, keepdim=True)
        denominator = torch.sum(level_values.square(), dim=-1, keepdim=True)
    else:
        numerator = torch.sum(h * level_values * centered, dim=-1, keepdim=True)
        denominator = torch.sum(h * level_values.square(), dim=-1, keepdim=True)
    return (numerator / denominator.clamp_min(1e-12)).abs().clamp_min(1e-8)


def quantize_tensor(
    weight: torch.Tensor,
    policy: QuantizationPolicy,
    *,
    name: str = "tensor",
    hessian_diag: torch.Tensor | None = None,
) -> PackedTensor:
    """Pack one tensor according to the NHDF edge profile."""

    policy.validate()
    original_shape = tuple(weight.shape)

    if policy.mode == "raw" or weight.ndim < 2:
        dtype = torch.float16 if policy.raw_dtype == "float16" else torch.bfloat16
        raw = weight.detach().to(device="cpu", dtype=dtype).contiguous()
        cols = int(weight.shape[-1]) if weight.ndim else 1
        rows = int(math.prod(weight.shape[:-1])) if weight.ndim > 1 else 1
        return PackedTensor(
            name=name,
            original_shape=original_shape,
            policy=QuantizationPolicy(**{**asdict(policy), "mode": "raw"}),
            rows=rows,
            original_cols=cols,
            padded_cols=cols,
            groups_per_row=0,
            raw=raw,
            stats={
                "packed_bytes": int(raw.numel() * raw.element_size()),
                "original_parameters": int(weight.numel()),
                "effective_bits_per_weight": 16.0,
            },
        )

    groups, rows, original_cols, padded_cols, groups_per_row = _prepare_groups(weight, policy.group_size)
    h = _prepare_hessian(
        hessian_diag,
        rows=rows,
        original_cols=original_cols,
        padded_cols=padded_cols,
        group_size=policy.group_size,
    )
    levels = _levels(policy.base_bits, groups.device)

    mean = _weighted_mean(groups, h)
    centered = groups - mean
    # The standard-deviation initializer is robust for both 2- and 4-bit levels.
    scale = torch.sqrt(_weighted_mean(centered.square(), h)).clamp_min(1e-8)
    scale = scale / (1.05 if policy.base_bits == 2 else 3.5)

    codes = torch.empty_like(groups, dtype=torch.uint8)
    chosen = torch.empty_like(groups, dtype=torch.float32)
    for _ in range(policy.iterations):
        normalized = (groups - mean) / scale
        distances = torch.abs(normalized[..., None] - levels)
        codes = torch.argmin(distances, dim=-1).to(torch.uint8)
        chosen = levels[codes.to(torch.long)]
        # Alternating least squares for scale and the B0 mean term.
        scale = _weighted_scale(groups - mean, chosen, h)
        mean, _ = weighted_zero_set_residual(groups, scale * chosen, h)

    base_reconstruction_without_mean = scale * chosen
    mean, base_field = weighted_zero_set_residual(groups, base_reconstruction_without_mean, h)
    base_reconstruction = mean + base_reconstruction_without_mean
    base_residual = groups - base_reconstruction

    lp = log_polar_encode(
        base_residual,
        gamma=policy.gamma,
        radial_bins=policy.radial_bins,
        angular_bins=policy.angular_bins,
    )
    theta_matrix = lp.theta.view(rows, groups_per_row)
    score = branch_score(
        base_residual.view(rows, groups_per_row, policy.group_size),
        theta_matrix,
        hessian_diag=None if h is None else h.view(rows, groups_per_row, policy.group_size),
        phase_gain=policy.phase_gain,
    ).reshape(-1)
    residual_mask = select_bounded_branches(score, policy.residual_fraction).reshape(-1).cpu()

    selected_ids = torch.nonzero(residual_mask, as_tuple=False).reshape(-1)
    residual_scales = torch.empty(selected_ids.numel(), dtype=torch.float32)
    residual_sign_bits = torch.empty((selected_ids.numel(), policy.group_size), dtype=torch.uint8)

    if selected_ids.numel():
        selected_residual = base_residual[selected_ids]
        sign = torch.where(selected_residual >= 0.0, 1.0, -1.0)
        if h is None:
            rscale = torch.mean(torch.abs(selected_residual), dim=-1, keepdim=True)
        else:
            hs = h[selected_ids]
            rscale = torch.sum(hs * torch.abs(selected_residual), dim=-1, keepdim=True) / torch.sum(
                hs, dim=-1, keepdim=True
            ).clamp_min(1e-12)
        rscale = rscale.clamp_min(1e-8)
        residual_scales = rscale.squeeze(-1)
        residual_sign_bits = (sign > 0).to(torch.uint8)

        residual_term = torch.zeros_like(groups)
        residual_term[selected_ids] = sign * rscale
        mean, final_field = weighted_zero_set_residual(groups, base_reconstruction_without_mean + residual_term, h)
        final_reconstruction = mean + base_reconstruction_without_mean + residual_term
    else:
        final_field = base_field
        final_reconstruction = base_reconstruction

    final_residual = groups - final_reconstruction

    # Group-major base stream.  The group size is a multiple of 256, so every
    # group starts on a byte boundary and can be independently addressed.
    base_codes = pack_unsigned(codes.reshape(-1), policy.base_bits)
    residual_bits = pack_bits(residual_sign_bits.reshape(-1)) if selected_ids.numel() else torch.empty(0, dtype=torch.uint8)

    # A mask word covers 32 groups.  Prefix counts hold the selected-group rank
    # at the start of each word, so the CUDA kernel needs at most one popcount.
    if selected_ids.numel():
        residual_mask_words = words_from_bits(residual_mask, word_bits=32)
        residual_prefix = prefix_counts(residual_mask, block_groups=32)
    else:
        residual_mask_words = torch.empty(0, dtype=torch.int32)
        residual_prefix = torch.empty(0, dtype=torch.int32)

    bytes_per_group = policy.group_size * policy.base_bits // 8
    base_bytes_by_group = base_codes.view(-1, bytes_per_group)
    # The deployment profile stores payload parity as the one-bit gate.  A
    # topology-orientation bit can be derived from the log-polar address during
    # diagnostics, but is kept separate so integrity parity remains verifiable.
    parity = byte_stream_parity(base_bytes_by_group)
    if selected_ids.numel():
        residual_bytes_by_group = residual_bits.view(-1, policy.group_size // 8)
        residual_parity = byte_stream_parity(residual_bytes_by_group)
        parity[selected_ids] = torch.bitwise_xor(parity[selected_ids], residual_parity)
    parity_words = words_from_bits(parity.to(torch.bool), word_bits=32)

    selected_lp = lp.packed_address.reshape(-1)[selected_ids].cpu() if selected_ids.numel() else torch.empty(0, dtype=torch.uint8)

    mse = torch.mean(final_residual.square()).item()
    max_abs = torch.max(torch.abs(final_residual)).item()
    max_field = torch.max(torch.abs(final_field)).item() if final_field.numel() else 0.0

    packed = PackedTensor(
        name=name,
        original_shape=original_shape,
        policy=policy,
        rows=rows,
        original_cols=original_cols,
        padded_cols=padded_cols,
        groups_per_row=groups_per_row,
        base_codes=base_codes.cpu(),
        means=mean.squeeze(-1).to(torch.float16).cpu(),
        scales=scale.squeeze(-1).to(torch.float16).cpu(),
        residual_mask_words=residual_mask_words.cpu(),
        residual_prefix=residual_prefix.cpu(),
        residual_bits=residual_bits.cpu(),
        residual_scales=residual_scales.to(torch.float16).cpu(),
        log_polar_address=selected_lp,
        parity_words=parity_words.cpu(),
        stats={},
    )
    packed_bytes = sum(int(t.numel() * t.element_size()) for t in packed.tensor_dict().values())
    original_params = int(weight.numel())
    packed.stats.update(
        {
            "packed_bytes": packed_bytes,
            "original_parameters": original_params,
            "effective_bits_per_weight": packed_bytes * 8.0 / max(original_params, 1),
            "residual_fraction_actual": selected_ids.numel() / max(packed.total_groups, 1),
            "rho_max": lp.rho_max,
            "weighted_zero_set_max_abs": max_field,
            "reconstruction_mse": mse,
            "reconstruction_max_abs": max_abs,
        }
    )
    return packed


def residual_mask(packed: PackedTensor) -> torch.Tensor:
    """Return the boolean residual mask for a packed tensor."""

    if packed.policy.mode == "raw":
        return torch.empty(0, dtype=torch.bool)
    if packed.residual_mask_words.numel() == 0:
        return torch.zeros(packed.total_groups, dtype=torch.bool)
    return bits_from_words(packed.residual_mask_words, packed.total_groups, word_bits=32)


def stored_parity(packed: PackedTensor) -> torch.Tensor:
    """Return stored parity bits."""

    if packed.policy.mode == "raw":
        return torch.empty(0, dtype=torch.bool)
    return bits_from_words(packed.parity_words, packed.total_groups, word_bits=32)


def dequantize_tensor(packed: PackedTensor, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Reconstruct a packed tensor with the CPU reference path."""

    if packed.policy.mode == "raw":
        return packed.raw.to(dtype=dtype).reshape(packed.original_shape)

    total_values = packed.total_groups * packed.policy.group_size
    codes = unpack_unsigned(packed.base_codes, packed.policy.base_bits, total_values).to(torch.long)
    levels = _levels(packed.policy.base_bits, torch.device("cpu"))
    base = levels[codes].view(packed.total_groups, packed.policy.group_size)
    reconstruction = packed.means.to(torch.float32)[:, None] + packed.scales.to(torch.float32)[:, None] * base

    mask = residual_mask(packed)
    selected = int(mask.sum().item())
    if selected:
        signs = unpack_bits(packed.residual_bits, selected * packed.policy.group_size)
        signs = torch.where(signs.view(selected, packed.policy.group_size), 1.0, -1.0)
        reconstruction[mask] += packed.residual_scales.to(torch.float32)[:, None] * signs

    matrix = reconstruction.view(packed.rows, packed.padded_cols)[:, : packed.original_cols]
    return matrix.reshape(packed.original_shape).to(dtype=dtype)



def dequantize_rows(
    packed: PackedTensor,
    row_indices: torch.Tensor | list[int] | tuple[int, ...],
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reconstruct selected flattened rows without expanding the full tensor.

    This is the CPU reference for embedding lookup and expert slicing.  The
    production CUDA path performs the same decoding inside GEMV/GEMM kernels.
    For a 3-D expert tensor, rows are flattened across the leading dimensions
    exactly as in :func:`quantize_tensor`.
    """

    indices = torch.as_tensor(row_indices, dtype=torch.int64, device="cpu").reshape(-1)
    if indices.numel() == 0:
        return torch.empty((0, packed.original_cols), dtype=dtype)
    if torch.any(indices < 0) or torch.any(indices >= packed.rows):
        raise IndexError(f"row index outside [0, {packed.rows})")

    if packed.policy.mode == "raw":
        matrix = packed.raw.detach().to(device="cpu").reshape(packed.rows, packed.original_cols)
        return matrix.index_select(0, indices).to(dtype=dtype)

    bits = packed.policy.base_bits
    bytes_per_row = packed.padded_cols * bits // 8
    byte_rows = packed.base_codes.detach().to(device="cpu", dtype=torch.uint8).view(packed.rows, bytes_per_row)
    selected_bytes = byte_rows.index_select(0, indices).reshape(-1)
    value_count = indices.numel() * packed.padded_cols
    codes = unpack_unsigned(selected_bytes, bits, value_count).to(torch.long)
    levels = _levels(bits, torch.device("cpu"))
    base = levels[codes].view(indices.numel() * packed.groups_per_row, packed.policy.group_size)

    local_group = torch.arange(packed.groups_per_row, dtype=torch.int64)
    group_ids = (indices[:, None] * packed.groups_per_row + local_group[None, :]).reshape(-1)
    means = packed.means.detach().to(device="cpu", dtype=torch.float32).index_select(0, group_ids)
    scales = packed.scales.detach().to(device="cpu", dtype=torch.float32).index_select(0, group_ids)
    reconstruction = means[:, None] + scales[:, None] * base

    if packed.residual_mask_words.numel():
        words = packed.residual_mask_words.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        prefix = packed.residual_prefix.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        residual_bytes = packed.residual_bits.detach().to(device="cpu", dtype=torch.uint8).reshape(-1)
        residual_scales = packed.residual_scales.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        bytes_per_residual_group = packed.policy.group_size // 8
        for local_idx, group_id_tensor in enumerate(group_ids):
            group_id = int(group_id_tensor.item())
            word_index = group_id // 32
            bit_index = group_id % 32
            word = int(words[word_index].item()) & 0xFFFFFFFF
            if ((word >> bit_index) & 1) == 0:
                continue
            earlier_mask = (1 << bit_index) - 1 if bit_index else 0
            rank = int(prefix[word_index].item()) + (word & earlier_mask).bit_count()
            start = rank * bytes_per_residual_group
            encoded = residual_bytes[start : start + bytes_per_residual_group]
            signs = unpack_bits(encoded, packed.policy.group_size)
            sign_values = torch.where(signs, 1.0, -1.0)
            reconstruction[local_idx] += residual_scales[rank] * sign_values

    matrix = reconstruction.view(indices.numel(), packed.padded_cols)[:, : packed.original_cols]
    return matrix.to(dtype=dtype)

def verify_parity(packed: PackedTensor) -> dict[str, Any]:
    """Recompute stored payload parity and report mismatches."""

    if packed.policy.mode == "raw":
        return {"groups": 0, "mismatches": 0, "ok": True}

    bytes_per_group = packed.policy.group_size * packed.policy.base_bits // 8
    base_groups = packed.base_codes.view(packed.total_groups, bytes_per_group)

    # This profile uses data parity for the stored one-bit integrity gate.
    # Topology/orientation parity is a separate diagnostic derived from the
    # log-polar branch address.  Shard CRC32 remains the authoritative integrity
    # check because one-bit parity misses even-count faults.
    recomputed = byte_stream_parity(base_groups).to(torch.bool)
    mask = residual_mask(packed)
    if int(mask.sum().item()):
        selected = int(mask.sum().item())
        res_groups = packed.residual_bits.view(selected, packed.policy.group_size // 8)
        res_p = byte_stream_parity(res_groups).to(torch.bool)
        recomputed[mask] = torch.logical_xor(recomputed[mask], res_p)

    stored = stored_parity(packed)
    mismatch = torch.logical_xor(recomputed, stored)
    return {
        "groups": packed.total_groups,
        "mismatches": int(mismatch.sum().item()),
        "ok": int(mismatch.sum().item()) == 0,
        "note": "one-bit parity misses even-count faults; use shard CRC32 for full integrity verification",
    }
