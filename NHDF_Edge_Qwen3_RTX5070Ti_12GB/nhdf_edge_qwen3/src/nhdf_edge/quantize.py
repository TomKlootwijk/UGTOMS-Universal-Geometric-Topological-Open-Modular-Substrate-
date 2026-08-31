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
    byte_stream_parity,
    delta2_phase,
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
    # A static tensor has no temporal generation-to-generation phase.  Keep
    # the field for nhdf-edge-0.1 compatibility, but disable it by default;
    # any future temporal coupling needs its own measured state transition.
    phase_gain: float = 0.0
    gamma: float = 1.0
    radial_bins: int = 16
    angular_bins: int = 16
    iterations: int = 3
    chunk_groups: int = 8192
    raw_dtype: str = "float16"

    def validate(self) -> None:
        if self.mode not in {"quantized", "raw"}:
            raise ValueError("mode must be 'quantized' or 'raw'")
        if self.mode == "quantized" and self.base_bits not in (2, 4):
            raise ValueError("the reference runtime supports 2-bit or 4-bit base weights")
        # Both packed streams are byte-addressed.  A multiple of eight is the
        # actual layout requirement; the original 256-only restriction was a
        # policy assumption, not a CUDA-kernel constraint.  Exposing smaller
        # groups is necessary for the scale-aware quality calibration required
        # by the v0.3 Edge-AI profile.
        if self.group_size <= 0 or self.group_size % 8 != 0:
            raise ValueError("group_size must be a positive multiple of 8")
        if not (0.0 <= self.residual_fraction <= 1.0):
            raise ValueError("residual_fraction must be in [0, 1]")
        if self.base_bits == 4 and self.residual_fraction != 0.0:
            raise ValueError("the reference profile uses residual branches only for 2-bit tensors")
        if self.iterations < 1:
            raise ValueError("iterations must be at least one")
        if self.chunk_groups < 1:
            raise ValueError("chunk_groups must be at least one")


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

    def validate_storage(self) -> None:
        """Reject inconsistent serialized geometry before runtime dispatch."""

        self.policy.validate()
        if any(int(size) < 0 for size in self.original_shape):
            raise ValueError(f"{self.name}: original_shape contains a negative dimension")
        expected_cols = int(self.original_shape[-1]) if self.original_shape else 1
        expected_rows = int(math.prod(self.original_shape[:-1])) if len(self.original_shape) > 1 else 1
        if self.rows != expected_rows or self.original_cols != expected_cols:
            raise ValueError(f"{self.name}: stored row/column geometry disagrees with original_shape")

        if self.policy.mode == "raw":
            expected_dtype = torch.float16 if self.policy.raw_dtype == "float16" else torch.bfloat16
            if self.padded_cols != expected_cols or self.groups_per_row != 0:
                raise ValueError(f"{self.name}: raw tensor has quantized group geometry")
            if self.raw.dtype != expected_dtype or self.raw.numel() != int(math.prod(self.original_shape)):
                raise ValueError(f"{self.name}: raw tensor dtype or length mismatch")
            if not self.raw.is_contiguous():
                raise ValueError(f"{self.name}: raw tensor must be contiguous")
            return

        if len(self.original_shape) < 2 or expected_rows <= 0 or expected_cols <= 0:
            raise ValueError(f"{self.name}: quantized tensors require a non-empty matrix geometry")
        expected_padded = (
            (expected_cols + self.policy.group_size - 1) // self.policy.group_size
        ) * self.policy.group_size
        expected_groups_per_row = expected_padded // self.policy.group_size
        if self.padded_cols != expected_padded or self.groups_per_row != expected_groups_per_row:
            raise ValueError(f"{self.name}: inconsistent padded/group geometry")

        tensors = {
            "base_codes": (self.base_codes, torch.uint8),
            "means": (self.means, torch.float16),
            "scales": (self.scales, torch.float16),
            "residual_mask_words": (self.residual_mask_words, torch.int32),
            "residual_prefix": (self.residual_prefix, torch.int32),
            "residual_bits": (self.residual_bits, torch.uint8),
            "residual_scales": (self.residual_scales, torch.float16),
            "log_polar_address": (self.log_polar_address, torch.uint8),
            "parity_words": (self.parity_words, torch.int32),
        }
        devices = set()
        for tensor_name, (tensor, dtype) in tensors.items():
            if tensor.dtype != dtype or tensor.ndim != 1 or not tensor.is_contiguous():
                raise ValueError(f"{self.name}: {tensor_name} has invalid dtype, rank, or layout")
            devices.add(tensor.device)
        if len(devices) != 1:
            raise ValueError(f"{self.name}: packed buffers must share one device")

        total_groups = self.rows * self.groups_per_row
        total_values = total_groups * self.policy.group_size
        expected_base_bytes = total_values * self.policy.base_bits // 8
        expected_words = (total_groups + 31) // 32
        if self.base_codes.numel() != expected_base_bytes:
            raise ValueError(f"{self.name}: base code length mismatch")
        if self.means.numel() != total_groups or self.scales.numel() != total_groups:
            raise ValueError(f"{self.name}: mean/scale length mismatch")
        if self.parity_words.numel() != expected_words:
            raise ValueError(f"{self.name}: parity word count mismatch")

        if self.residual_mask_words.numel() == 0:
            if any(
                tensor.numel()
                for tensor in (
                    self.residual_prefix,
                    self.residual_bits,
                    self.residual_scales,
                    self.log_polar_address,
                )
            ):
                raise ValueError(f"{self.name}: residual payload exists without a residual mask")
            return

        if self.policy.base_bits != 2:
            raise ValueError(f"{self.name}: residual payload is only valid for 2-bit tensors")
        if self.residual_mask_words.numel() != expected_words or self.residual_prefix.numel() != expected_words:
            raise ValueError(f"{self.name}: residual mask/prefix word count mismatch")
        mask = bits_from_words(self.residual_mask_words, total_groups, word_bits=32)
        selected = int(mask.sum().item())
        if selected <= 0 or self.residual_scales.numel() != selected:
            raise ValueError(f"{self.name}: residual mask and scale count disagree")
        if self.log_polar_address.numel() != selected:
            raise ValueError(f"{self.name}: log-polar address count mismatch")
        if self.residual_bits.numel() != selected * self.policy.group_size // 8:
            raise ValueError(f"{self.name}: residual bitstream length mismatch")
        expected_prefix = prefix_counts(mask, block_groups=32)
        if not torch.equal(self.residual_prefix.detach().cpu(), expected_prefix):
            raise ValueError(f"{self.name}: residual prefix does not match the mask")

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
            packed = cls(**common, raw=tensors[f"{p}raw"])
        else:
            packed = cls(
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
        packed.validate_storage()
        return packed


def _levels(bits: int, device: torch.device) -> torch.Tensor:
    if bits == 2:
        return torch.tensor([-1.5, -0.5, 0.5, 1.5], dtype=torch.float32, device=device)
    if bits == 4:
        # Mid-rise symmetric levels: -7.5, -6.5, ..., +7.5.
        return torch.arange(16, dtype=torch.float32, device=device) - 7.5
    raise ValueError(f"unsupported base bit width: {bits}")


def _prepare_matrix(weight: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int, int, int, int]:
    """Return CPU source storage and grouping geometry without widening it.

    Checkpoint tensors are normally BF16/FP16.  Keeping that source dtype here
    is important: widening an entire expert or vocabulary matrix to FP32 would
    consume over a gigabyte before quantization even starts.  Individual group
    chunks are widened by :func:`_load_group_chunk` instead.
    """

    if weight.ndim == 0:
        weight = weight.reshape(1)
    original_shape = tuple(weight.shape)
    original_cols = int(original_shape[-1]) if original_shape else 1
    rows = int(math.prod(original_shape[:-1])) if len(original_shape) > 1 else 1
    padded_cols = ((original_cols + group_size - 1) // group_size) * group_size
    matrix = weight.detach().to(device="cpu").reshape(rows, original_cols)
    groups_per_row = padded_cols // group_size
    return matrix, rows, original_cols, padded_cols, groups_per_row


def _prepare_hessian_source(
    hessian_diag: torch.Tensor | None,
    *,
    rows: int,
    original_cols: int,
) -> tuple[torch.Tensor | None, bool]:
    """Prepare a broadcastable CPU Hessian view without a full-size clone."""

    if hessian_diag is None:
        return None, False
    h = hessian_diag.detach().to(device="cpu")
    if h.ndim == 1 and h.numel() == original_cols:
        return h.reshape(1, original_cols), True
    try:
        return torch.broadcast_to(h, (rows, original_cols)), False
    except RuntimeError as exc:
        raise ValueError("hessian_diag is not broadcastable to the flattened weight matrix") from exc


def _load_group_chunk(
    matrix: torch.Tensor,
    start: int,
    end: int,
    *,
    rows: int,
    original_cols: int,
    padded_cols: int,
    group_size: int,
    groups_per_row: int,
    shared_rows: bool = False,
) -> torch.Tensor:
    """Load ``[start:end]`` groups as FP32 with only chunk-sized storage.

    The contiguous, unpadded checkpoint case is a direct flat slice.  The
    indexed fallback handles row padding, broadcast Hessian diagonals and
    non-contiguous inputs while keeping all index tensors chunk bounded.
    """

    del rows  # Geometry validation is performed by the caller.
    count = end - start
    if count <= 0:
        return torch.empty((0, group_size), dtype=torch.float32)

    if not shared_rows and original_cols == padded_cols and matrix.is_contiguous():
        element_start = start * group_size
        element_end = end * group_size
        return matrix.reshape(-1)[element_start:element_end].to(torch.float32).reshape(count, group_size)

    if shared_rows and original_cols == padded_cols and matrix.is_contiguous():
        local_groups = torch.arange(start, end, dtype=torch.int64).remainder(groups_per_row)
        source_groups = matrix.reshape(groups_per_row, group_size)
        return source_groups.index_select(0, local_groups).to(torch.float32)

    group_ids = torch.arange(start, end, dtype=torch.int64)
    row_ids = torch.zeros_like(group_ids) if shared_rows else torch.div(group_ids, groups_per_row, rounding_mode="floor")
    first_cols = group_ids.remainder(groups_per_row) * group_size
    offsets = torch.arange(group_size, dtype=torch.int64)
    col_ids = first_cols[:, None] + offsets[None, :]
    valid = col_ids < original_cols
    safe_cols = col_ids.clamp_max(max(original_cols - 1, 0))
    loaded = matrix[row_ids[:, None], safe_cols].to(torch.float32)
    return loaded.masked_fill_(~valid, 0.0)


def _nearest_level_codes(normalized: torch.Tensor, bits: int) -> torch.Tensor:
    """Return the same lower-tie nearest codes without a ``[..., levels]`` tensor.

    Both supported codebooks contain unit-spaced mid-rise levels.  ``ceil`` at
    the half-step chooses the lower code on exact ties, matching ``argmin``'s
    first-index behavior.
    """

    max_code = (1 << bits) - 1
    codebook_midpoint = max_code / 2.0
    codes = torch.ceil(normalized + codebook_midpoint - 0.5)
    return codes.clamp_(0.0, float(max_code)).to(torch.uint8)


def _chosen_levels(codes: torch.Tensor, bits: int) -> torch.Tensor:
    return codes.to(torch.float32) - ((1 << bits) - 1) / 2.0


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
    """Pack one tensor according to the NHDF edge profile.

    Quantization is performed in bounded group chunks.  Only packed bytes and
    per-group scalar state persist between chunks; no full-size FP32
    reconstruction or ``weights x codebook-levels`` distance tensor is kept.
    Residual tensors use a second pass after deterministic global branch
    selection so their sign payload can also be assembled directly in packed
    form.
    """

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

    matrix, rows, original_cols, padded_cols, groups_per_row = _prepare_matrix(weight, policy.group_size)
    h_source, h_shared_rows = _prepare_hessian_source(
        hessian_diag,
        rows=rows,
        original_cols=original_cols,
    )
    total_groups = rows * groups_per_row
    total_padded_weights = total_groups * policy.group_size
    bytes_per_group = policy.group_size * policy.base_bits // 8
    base_codes = torch.empty(total_groups * bytes_per_group, dtype=torch.uint8)
    means = torch.empty(total_groups, dtype=torch.float32)
    scales = torch.empty(total_groups, dtype=torch.float32)
    parity = torch.empty(total_groups, dtype=torch.uint8)

    needs_residual_pass = policy.residual_fraction > 0.0
    if needs_residual_pass:
        rho_by_group = torch.empty(total_groups, dtype=torch.float32)
        theta_by_group = torch.empty(total_groups, dtype=torch.float32)
        base_mse_for_score = torch.empty(total_groups, dtype=torch.float32)
    else:
        rho_by_group = torch.empty(0, dtype=torch.float32)
        theta_by_group = torch.empty(0, dtype=torch.float32)
        base_mse_for_score = torch.empty(0, dtype=torch.float32)

    rho_max = 1e-12
    # Keep one deterministic scalar per group, then reduce once in canonical
    # group order.  Summing per-chunk totals makes the reported MSE depend on
    # chunk_groups even when every serialized byte is identical.
    reconstruction_sse_by_group = torch.empty(total_groups, dtype=torch.float64)
    max_abs = 0.0
    max_field = 0.0

    # First pass: quantize each chunk, emit packed base codes immediately and
    # retain only group scalars needed for global residual allocation.
    for start in range(0, total_groups, policy.chunk_groups):
        end = min(start + policy.chunk_groups, total_groups)
        groups = _load_group_chunk(
            matrix,
            start,
            end,
            rows=rows,
            original_cols=original_cols,
            padded_cols=padded_cols,
            group_size=policy.group_size,
            groups_per_row=groups_per_row,
        )
        h = None
        if h_source is not None:
            h = _load_group_chunk(
                h_source,
                start,
                end,
                rows=rows,
                original_cols=original_cols,
                padded_cols=padded_cols,
                group_size=policy.group_size,
                groups_per_row=groups_per_row,
                shared_rows=h_shared_rows,
            ).clamp_min_(0.0)

        mean = _weighted_mean(groups, h)
        centered = groups - mean
        # The standard-deviation initializer is robust for both 2- and 4-bit levels.
        scale = torch.sqrt(_weighted_mean(centered.square(), h)).clamp_min(1e-8)
        scale = scale / (1.05 if policy.base_bits == 2 else 3.5)

        for _ in range(policy.iterations):
            normalized = (groups - mean) / scale
            codes = _nearest_level_codes(normalized, policy.base_bits)
            chosen = _chosen_levels(codes, policy.base_bits)
            # Alternating least squares for scale and the B0 mean term.
            scale = _weighted_scale(groups - mean, chosen, h)
            mean, _ = weighted_zero_set_residual(groups, scale * chosen, h)

        base_reconstruction_without_mean = scale * chosen
        mean, base_field = weighted_zero_set_residual(groups, base_reconstruction_without_mean, h)
        base_residual = groups - (mean + base_reconstruction_without_mean)

        means[start:end] = mean.squeeze(-1)
        scales[start:end] = scale.squeeze(-1)

        packed_chunk = pack_unsigned(codes.reshape(-1), policy.base_bits)
        byte_start = start * bytes_per_group
        byte_end = end * bytes_per_group
        base_codes[byte_start:byte_end] = packed_chunk
        parity[start:end] = byte_stream_parity(packed_chunk.view(end - start, bytes_per_group))

        lp = log_polar_encode(
            base_residual,
            gamma=policy.gamma,
            radial_bins=policy.radial_bins,
            angular_bins=policy.angular_bins,
        )
        rho_max = max(rho_max, lp.rho_max)

        if needs_residual_pass:
            rho_by_group[start:end] = lp.rho
            theta_by_group[start:end] = lp.theta
            if h is None:
                score_mse = torch.mean(base_residual.square(), dim=-1)
            else:
                score_mse = torch.sum(h * base_residual.square(), dim=-1) / torch.sum(h, dim=-1).clamp_min(1e-12)
            base_mse_for_score[start:end] = score_mse
        else:
            reconstruction_sse_by_group[start:end] = torch.sum(
                base_residual.square(), dim=-1, dtype=torch.float64
            )
            max_abs = max(max_abs, float(torch.max(torch.abs(base_residual)).item()))
            max_field = max(max_field, float(torch.max(torch.abs(base_field)).item()))

    if needs_residual_pass:
        curvature = torch.abs(delta2_phase(theta_by_group.view(rows, groups_per_row))).reshape(-1)
        score = base_mse_for_score * (1.0 + policy.phase_gain * curvature)
        residual_mask = select_bounded_branches(score, policy.residual_fraction).reshape(-1).cpu()
    else:
        residual_mask = torch.zeros(total_groups, dtype=torch.bool)

    selected_ids = torch.nonzero(residual_mask, as_tuple=False).reshape(-1)
    selected_count = int(selected_ids.numel())
    residual_scales = torch.empty(selected_count, dtype=torch.float32)
    residual_bytes_per_group = policy.group_size // 8
    residual_bits = torch.empty(selected_count * residual_bytes_per_group, dtype=torch.uint8)

    if selected_count:
        selected_rho = rho_by_group.index_select(0, selected_ids)
        selected_theta = theta_by_group.index_select(0, selected_ids)
        rho_bin = torch.clamp(
            torch.floor(selected_rho / rho_max * policy.radial_bins),
            0,
            policy.radial_bins - 1,
        ).to(torch.uint8)
        theta_unit = (selected_theta + math.pi) / (2.0 * math.pi)
        theta_bin = torch.clamp(
            torch.floor(theta_unit * policy.angular_bins),
            0,
            policy.angular_bins - 1,
        ).to(torch.uint8)
        selected_lp = ((rho_bin.to(torch.int16) << 4) | theta_bin.to(torch.int16)).to(torch.uint8)

        selected_write = 0
        # Second pass: reconstruct one base chunk, materialize residual data only
        # for selected groups, and stream the packed residual payload to output.
        for start in range(0, total_groups, policy.chunk_groups):
            end = min(start + policy.chunk_groups, total_groups)
            groups = _load_group_chunk(
                matrix,
                start,
                end,
                rows=rows,
                original_cols=original_cols,
                padded_cols=padded_cols,
                group_size=policy.group_size,
                groups_per_row=groups_per_row,
            )
            h = None
            if h_source is not None:
                h = _load_group_chunk(
                    h_source,
                    start,
                    end,
                    rows=rows,
                    original_cols=original_cols,
                    padded_cols=padded_cols,
                    group_size=policy.group_size,
                    groups_per_row=groups_per_row,
                    shared_rows=h_shared_rows,
                ).clamp_min_(0.0)

            byte_start = start * bytes_per_group
            byte_end = end * bytes_per_group
            codes = unpack_unsigned(
                base_codes[byte_start:byte_end],
                policy.base_bits,
                (end - start) * policy.group_size,
            ).view(end - start, policy.group_size)
            chosen = _chosen_levels(codes, policy.base_bits)
            base_reconstruction_without_mean = scales[start:end, None] * chosen
            base_residual = groups - (means[start:end, None] + base_reconstruction_without_mean)

            local_ids = torch.nonzero(residual_mask[start:end], as_tuple=False).reshape(-1)
            selected_residual = base_residual.index_select(0, local_ids)
            sign = torch.where(selected_residual >= 0.0, 1.0, -1.0)
            if h is None:
                rscale = torch.mean(torch.abs(selected_residual), dim=-1, keepdim=True)
            else:
                hs = h.index_select(0, local_ids)
                rscale = torch.sum(hs * torch.abs(selected_residual), dim=-1, keepdim=True) / torch.sum(
                    hs, dim=-1, keepdim=True
                ).clamp_min(1e-12)
            rscale = rscale.clamp_min(1e-8)

            chunk_selected = int(local_ids.numel())
            next_selected_write = selected_write + chunk_selected
            residual_scales[selected_write:next_selected_write] = rscale.squeeze(-1)
            packed_residual_chunk = pack_bits((sign > 0).to(torch.uint8).reshape(-1))
            residual_byte_start = selected_write * residual_bytes_per_group
            residual_byte_end = next_selected_write * residual_bytes_per_group
            residual_bits[residual_byte_start:residual_byte_end] = packed_residual_chunk

            global_ids = local_ids + start
            residual_parity = byte_stream_parity(packed_residual_chunk.view(chunk_selected, residual_bytes_per_group))
            parity[global_ids] = torch.bitwise_xor(parity[global_ids], residual_parity)

            residual_term = torch.zeros_like(groups)
            residual_term[local_ids] = sign * rscale
            mean, final_field = weighted_zero_set_residual(
                groups,
                base_reconstruction_without_mean + residual_term,
                h,
            )
            final_residual = groups - (mean + base_reconstruction_without_mean + residual_term)
            means[start:end] = mean.squeeze(-1)
            reconstruction_sse_by_group[start:end] = torch.sum(
                final_residual.square(), dim=-1, dtype=torch.float64
            )
            max_abs = max(max_abs, float(torch.max(torch.abs(final_residual)).item()))
            max_field = max(max_field, float(torch.max(torch.abs(final_field)).item()))
            selected_write = next_selected_write
    else:
        selected_lp = torch.empty(0, dtype=torch.uint8)

    # A mask word covers 32 groups.  Prefix counts hold the selected-group rank
    # at the start of each word, so the CUDA kernel needs at most one popcount.
    if selected_count:
        residual_mask_words = words_from_bits(residual_mask, word_bits=32)
        residual_prefix = prefix_counts(residual_mask, block_groups=32)
    else:
        residual_mask_words = torch.empty(0, dtype=torch.int32)
        residual_prefix = torch.empty(0, dtype=torch.int32)

    # The deployment profile stores payload parity as the one-bit gate.  A
    # topology-orientation bit can be derived from the log-polar address during
    # diagnostics, but is kept separate so integrity parity remains verifiable.
    parity_words = words_from_bits(parity.to(torch.bool), word_bits=32)
    reconstruction_squared_sum = float(
        torch.sum(reconstruction_sse_by_group, dtype=torch.float64).item()
    )
    mse = reconstruction_squared_sum / max(total_padded_weights, 1)

    packed = PackedTensor(
        name=name,
        original_shape=original_shape,
        policy=policy,
        rows=rows,
        original_cols=original_cols,
        padded_cols=padded_cols,
        groups_per_row=groups_per_row,
        base_codes=base_codes.cpu(),
        means=means.to(torch.float16).cpu(),
        scales=scales.to(torch.float16).cpu(),
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
            "residual_fraction_actual": selected_count / max(packed.total_groups, 1),
            "rho_max": rho_max,
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
