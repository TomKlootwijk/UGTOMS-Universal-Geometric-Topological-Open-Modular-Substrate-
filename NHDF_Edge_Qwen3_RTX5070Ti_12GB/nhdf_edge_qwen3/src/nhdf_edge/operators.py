"""Semantics-first implementations of the NHDF operator chain for weights.

The mapping used by this edge-AI profile is:

ELP  -> log-polar encoding of the post-quantization residual
B0   -> local weighted zero-mean residual projection
P    -> one-bit payload/orientation parity event
RBST -> bounded residual-branch allocation ordered by log-polar key
K_T  -> deterministic forward group/tensor generation order
Scone-> on-demand reconstruction of only the tile being multiplied
Pi   -> the linear projection (GEMV/GEMM)
U    -> telemetry, integrity status and optional recalibration update

Only ELP, B0, P and RBST are numerical packing operations in this module.  The
runtime operators live in ``nhdf_edge.runtime``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LogPolarState:
    rho: torch.Tensor
    theta: torch.Tensor
    rho_bin: torch.Tensor
    theta_bin: torch.Tensor
    packed_address: torch.Tensor
    rho_max: float


def wrap_phase(phi: torch.Tensor) -> torch.Tensor:
    """Wrap angles to ``(-pi, pi]``."""

    return torch.remainder(phi + math.pi, 2.0 * math.pi) - math.pi


def delta2_phase(phi: torch.Tensor) -> torch.Tensor:
    """Second finite phase difference along the last dimension."""

    if phi.ndim == 0:
        return torch.zeros_like(phi)
    out = torch.zeros_like(phi)
    if phi.shape[-1] >= 3:
        out[..., 2:] = wrap_phase(phi[..., 2:] - 2.0 * phi[..., 1:-1] + phi[..., :-2])
    return out


def log_polar_encode(
    residual_groups: torch.Tensor,
    *,
    gamma: float = 1.0,
    radial_bins: int = 16,
    angular_bins: int = 16,
    rho_max: float | None = None,
) -> LogPolarState:
    """Encode group residuals into a compact log-polar address.

    ``residual_groups`` must have shape ``(..., group_size)``.  The angle is
    measured against deterministic sine/cosine basis vectors, avoiding a data-
    dependent PCA that would have to be stored in the model file.
    """

    if residual_groups.ndim < 1:
        raise ValueError("residual_groups must have at least one dimension")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if not (1 <= radial_bins <= 16 and 1 <= angular_bins <= 16):
        raise ValueError("the reference byte address supports at most 16x16 bins")

    r = residual_groups.to(torch.float32)
    group_size = r.shape[-1]
    idx = torch.arange(group_size, dtype=torch.float32, device=r.device)
    angle = 2.0 * math.pi * idx / max(group_size, 1)
    basis_c = torch.cos(angle)
    basis_s = torch.sin(angle)

    magnitude = torch.linalg.vector_norm(r, ord=2, dim=-1)
    rho = torch.log1p(gamma * magnitude)
    x = torch.sum(r * basis_c, dim=-1)
    y = torch.sum(r * basis_s, dim=-1)
    theta = torch.atan2(y, x)

    observed_max = float(rho.max().item()) if rho.numel() else 0.0
    max_rho = max(float(rho_max) if rho_max is not None else observed_max, 1e-12)
    rho_bin = torch.clamp(torch.floor(rho / max_rho * radial_bins), 0, radial_bins - 1).to(torch.uint8)
    theta_unit = (theta + math.pi) / (2.0 * math.pi)
    theta_bin = torch.clamp(torch.floor(theta_unit * angular_bins), 0, angular_bins - 1).to(torch.uint8)
    packed = (rho_bin.to(torch.int16) << 4) | theta_bin.to(torch.int16)

    return LogPolarState(
        rho=rho,
        theta=theta,
        rho_bin=rho_bin,
        theta_bin=theta_bin,
        packed_address=packed.to(torch.uint8),
        rho_max=max_rho,
    )


def weighted_zero_set_residual(
    original: torch.Tensor,
    reconstruction_without_mean: torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the mean term that enforces the local B0 zero-set.

    The local field is

    ``F_i(r) = sum_j h_j * r_j / sum_j h_j``.

    Setting ``mean = weighted_mean(original - reconstruction_without_mean)``
    makes ``F_i(original - (mean + reconstruction_without_mean)) == 0`` up to
    floating-point error.  The field remains non-degenerate because it is local
    to each group and has a non-zero Jacobian with respect to the mean term.
    """

    x = original.to(torch.float32)
    y = reconstruction_without_mean.to(torch.float32)
    if x.shape != y.shape:
        raise ValueError("original and reconstruction_without_mean must have equal shapes")
    if weights is None:
        h = torch.ones_like(x)
    else:
        h = weights.to(torch.float32)
        if h.shape != x.shape:
            try:
                h = torch.broadcast_to(h, x.shape)
            except RuntimeError as exc:
                raise ValueError("weights are not broadcastable to the input shape") from exc
        h = torch.clamp(h, min=0.0)

    denom = torch.sum(h, dim=-1, keepdim=True).clamp_min(eps)
    mean = torch.sum(h * (x - y), dim=-1, keepdim=True) / denom
    residual = x - (y + mean)
    field = torch.sum(h * residual, dim=-1) / denom.squeeze(-1)
    return mean, field


def branch_score(
    residual_groups: torch.Tensor,
    theta: torch.Tensor,
    *,
    hessian_diag: torch.Tensor | None = None,
    phase_gain: float = 0.0,
) -> torch.Tensor:
    """Score groups for the bounded one-bit residual branch."""

    r = residual_groups.to(torch.float32)
    if hessian_diag is None:
        mse = torch.mean(r.square(), dim=-1)
    else:
        h = torch.broadcast_to(hessian_diag.to(torch.float32), r.shape).clamp_min(0.0)
        mse = torch.sum(h * r.square(), dim=-1) / torch.sum(h, dim=-1).clamp_min(1e-12)
    curvature = torch.abs(delta2_phase(theta))
    return mse * (1.0 + phase_gain * curvature)


def select_bounded_branches(score: torch.Tensor, fraction: float) -> torch.Tensor:
    """Select exactly the highest-scoring bounded fraction of groups."""

    if not (0.0 <= fraction <= 1.0):
        raise ValueError("fraction must be in [0, 1]")
    flat = score.reshape(-1)
    mask = torch.zeros(flat.numel(), dtype=torch.bool, device=flat.device)
    if flat.numel() == 0 or fraction == 0.0:
        return mask.reshape(score.shape)
    count = min(flat.numel(), max(1, int(round(flat.numel() * fraction))))
    # Stable sorting makes the pack deterministic when scores tie.
    order = torch.argsort(flat, descending=True, stable=True)
    mask[order[:count]] = True
    return mask.reshape(score.shape)


_BYTE_PARITY = torch.tensor([int(i).bit_count() & 1 for i in range(256)], dtype=torch.uint8)


def byte_stream_parity(byte_groups: torch.Tensor, orientation_bit: torch.Tensor | None = None) -> torch.Tensor:
    """Return XOR parity for each row of bytes.

    ``byte_groups`` has shape ``(..., bytes_per_group)``.  The optional
    orientation bit is XORed into the payload parity, matching the diagnostic
    composite gate in the formal specification.
    """

    b = byte_groups.detach().to(device="cpu", dtype=torch.uint8)
    if b.ndim < 1:
        raise ValueError("byte_groups must have at least one dimension")
    table = _BYTE_PARITY
    bit_parity = table[b.to(torch.long)]
    p = torch.remainder(torch.sum(bit_parity.to(torch.int16), dim=-1), 2).to(torch.uint8)
    if orientation_bit is not None:
        orient = orientation_bit.detach().to(device="cpu", dtype=torch.uint8)
        orient = torch.broadcast_to(orient, p.shape)
        p = torch.bitwise_xor(p, orient & 1)
    return p
