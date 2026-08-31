"""Analytical memory and decode-throughput model for the selected Qwen3 MoE.

The estimates are deliberately transparent.  They are not benchmark results and
must be replaced by measurements on the target laptop before deployment claims
are made.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .config import NHDFConfig


@dataclass(frozen=True)
class Estimate:
    expert_effective_bpp: float
    sensitive_effective_bpp: float
    packed_weight_gb: float
    packed_weight_gib: float
    compression_vs_bf16: float
    compression_vs_official_int4: float
    kv_cache_gb: float
    projected_total_vram_gb: float
    projected_total_vram_gib: float
    nominal_headroom_gb: float
    active_weight_bytes_per_token_gb: float
    peak_bandwidth_roofline_tps: float
    decode_tps_by_efficiency: dict[str, float]
    fits_nominal_12gb: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def effective_bpp(cfg: NHDFConfig) -> tuple[float, float]:
    """Return modeled expert and sensitive-matrix bits per parameter.

    Expert format components:
      - low-bit base code;
      - one residual sign bit on the selected group fraction;
      - FP16 mean + scale per group (the local B0 zero-set representation);
      - FP16 residual scale and one-byte log-polar address per selected group;
      - residual mask, rank prefix, and parity metadata;
      - a small tensor-header/alignment allowance.

    Four-bit tensors have no residual branch, so only parity is retained beyond
    their FP16 mean and scale.
    """

    g = cfg.packing.group_size
    f = cfg.packing.expert_residual_fraction
    expert = (
        cfg.packing.expert_bits
        + f
        + 32.0 / g
        + 16.0 * f / g
        + 8.0 * f / g
        + 3.0 / g
        + 0.02
    )
    sensitive = cfg.packing.sensitive_bits + 32.0 / g + 1.0 / g + 0.02
    return expert, sensitive


def kv_cache_bytes(
    cfg: NHDFConfig,
    context_tokens: int | None = None,
    batch_size: int = 1,
    kv_bits: int | None = None,
) -> int:
    """Estimate decoder KV-cache storage for GQA attention."""

    m = cfg.model
    t = cfg.target
    context = context_tokens or t.default_context_tokens
    bits = kv_bits or t.kv_bits
    elements_per_token = batch_size * 2 * m.layers * m.kv_heads * m.head_dim
    if bits == 8:
        # The runtime uses Transformers' HQQ cache. Quantized elements carry
        # one FP16 scale and zero point per group, while the most recent
        # ``residual_length - 1`` tokens may remain FP16 between flushes.
        residual_tokens = min(max(context - 1, 0), max(t.kv_residual_length - 1, 0))
        quantized_tokens = context - residual_tokens
        quantized_bytes_per_element = bits / 8.0 + (
            t.kv_scale_zero_bits_per_group / 8.0 / t.kv_group_size
        )
        return int(
            quantized_tokens * elements_per_token * quantized_bytes_per_element
            + residual_tokens * elements_per_token * 2.0
        )
    bytes_per_element = bits / 8.0
    # K + V, every layer, only KV heads (GQA).
    return int(context * elements_per_token * bytes_per_element)


def estimate(cfg: NHDFConfig, *, context_tokens: int | None = None) -> Estimate:
    """Return the default analytical feasibility projection."""

    m = cfg.model
    t = cfg.target
    expert_bpp, sensitive_bpp = effective_bpp(cfg)

    raw_parameters = m.router_parameters + m.norm_parameters
    sensitive_parameters = m.total_parameters - m.expert_parameters - raw_parameters
    packed_bytes = (
        m.expert_parameters * expert_bpp / 8.0
        + sensitive_parameters * sensitive_bpp / 8.0
        + raw_parameters * 2.0
    )
    packed_gb = packed_bytes / 1e9

    kv_gb = kv_cache_bytes(cfg, context_tokens=context_tokens) / 1e9
    total_vram = packed_gb + kv_gb + t.workspace_gb + t.runtime_reserve_gb

    active_expert_parameters = m.expert_parameters * m.active_experts / m.total_experts
    active_sensitive_parameters = m.attention_parameters + m.lm_head_parameters
    active_raw_parameters = m.router_parameters + m.norm_parameters
    active_bytes = (
        active_expert_parameters * expert_bpp / 8.0
        + active_sensitive_parameters * sensitive_bpp / 8.0
        + active_raw_parameters * 2.0
    )
    active_gb = active_bytes / 1e9
    roofline = t.memory_bandwidth_gbps / active_gb
    efficiencies = (0.02, 0.03, 0.05, 0.07, 0.10)
    decode = {f"{int(e * 100)}%": roofline * e for e in efficiencies}

    return Estimate(
        expert_effective_bpp=expert_bpp,
        sensitive_effective_bpp=sensitive_bpp,
        packed_weight_gb=packed_gb,
        packed_weight_gib=packed_bytes / (1024**3),
        compression_vs_bf16=m.source_size_gb / packed_gb,
        compression_vs_official_int4=m.official_int4_size_gb / packed_gb,
        kv_cache_gb=kv_gb,
        projected_total_vram_gb=total_vram,
        projected_total_vram_gib=total_vram * 1e9 / (1024**3),
        nominal_headroom_gb=t.vram_gb_decimal - total_vram,
        active_weight_bytes_per_token_gb=active_gb,
        peak_bandwidth_roofline_tps=roofline,
        decode_tps_by_efficiency=decode,
        fits_nominal_12gb=total_vram <= t.vram_gb_decimal,
    )


def residual_fraction_sweep(cfg: NHDFConfig, fractions: list[float]) -> list[dict[str, float | bool]]:
    """Project size and decode traffic while varying the residual branch budget."""

    rows: list[dict[str, float | bool]] = []
    for fraction in fractions:
        packing = replace(cfg.packing, expert_residual_fraction=float(fraction))
        item = estimate(replace(cfg, packing=packing))
        rows.append(
            {
                "residual_fraction": float(fraction),
                "expert_bpp": item.expert_effective_bpp,
                "packed_weight_gb": item.packed_weight_gb,
                "total_vram_gb": item.projected_total_vram_gb,
                "active_weight_gb_per_token": item.active_weight_bytes_per_token_gb,
                "roofline_tps": item.peak_bandwidth_roofline_tps,
                "fits_12gb": item.fits_nominal_12gb,
            }
        )
    return rows


def context_sweep(cfg: NHDFConfig, contexts: list[int]) -> list[dict[str, float | int | bool]]:
    """Project KV cache and total VRAM for context-length choices."""

    rows: list[dict[str, float | int | bool]] = []
    for context in contexts:
        item = estimate(cfg, context_tokens=int(context))
        rows.append(
            {
                "context_tokens": int(context),
                "kv_cache_gb": item.kv_cache_gb,
                "total_vram_gb": item.projected_total_vram_gb,
                "headroom_gb": item.nominal_headroom_gb,
                "fits_12gb": item.fits_nominal_12gb,
            }
        )
    return rows
