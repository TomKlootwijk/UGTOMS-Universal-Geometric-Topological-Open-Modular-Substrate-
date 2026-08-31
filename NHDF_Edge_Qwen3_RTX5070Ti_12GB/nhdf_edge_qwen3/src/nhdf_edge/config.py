"""Configuration loading and tensor-policy routing."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .quantize import QuantizationPolicy


@dataclass(frozen=True)
class ModelProfile:
    repo_id: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    revision: str = "main"
    total_parameters: int = 30_532_122_624
    expert_parameters: int = 28_991_029_248
    attention_parameters: int = 905_969_664
    embedding_parameters: int = 311_164_928
    lm_head_parameters: int = 311_164_928
    router_parameters: int = 12_582_912
    norm_parameters: int = 210_944
    active_experts: int = 8
    total_experts: int = 128
    layers: int = 48
    hidden_size: int = 2048
    kv_heads: int = 4
    head_dim: int = 128
    source_size_gb: float = 61.1
    official_int4_size_gb: float = 16.9


@dataclass(frozen=True)
class TargetProfile:
    vram_gb_decimal: float = 12.0
    default_context_tokens: int = 8192
    kv_bits: int = 8
    kv_group_size: int = 64
    kv_residual_length: int = 128
    kv_scale_zero_bits_per_group: int = 32
    workspace_gb: float = 0.75
    runtime_reserve_gb: float = 0.90
    memory_bandwidth_gbps: float = 672.0
    cuda_cores: int = 5_888
    ai_tops: int = 992
    tgp_min_w: int = 60
    tgp_max_w: int = 115


@dataclass(frozen=True)
class PackingProfile:
    group_size: int = 256
    expert_bits: int = 2
    expert_residual_fraction: float = 0.15
    sensitive_bits: int = 4
    phase_gain: float = 0.0
    gamma: float = 1.0
    iterations: int = 3
    raw_router_and_norms: bool = True

    def expert_policy(self) -> QuantizationPolicy:
        return QuantizationPolicy(
            base_bits=self.expert_bits,
            group_size=self.group_size,
            residual_fraction=self.expert_residual_fraction,
            phase_gain=self.phase_gain,
            gamma=self.gamma,
            iterations=self.iterations,
        )

    def sensitive_policy(self) -> QuantizationPolicy:
        return QuantizationPolicy(
            base_bits=self.sensitive_bits,
            group_size=self.group_size,
            residual_fraction=0.0,
            phase_gain=self.phase_gain,
            gamma=self.gamma,
            iterations=self.iterations,
        )

    def raw_policy(self) -> QuantizationPolicy:
        return QuantizationPolicy(mode="raw", group_size=self.group_size, residual_fraction=0.0)


@dataclass(frozen=True)
class NHDFConfig:
    model: ModelProfile = field(default_factory=ModelProfile)
    target: TargetProfile = field(default_factory=TargetProfile)
    packing: PackingProfile = field(default_factory=PackingProfile)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None = None) -> NHDFConfig:
    if path is None:
        return NHDFConfig()
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return NHDFConfig(
        model=ModelProfile(**data.get("model", {})),
        target=TargetProfile(**data.get("target", {})),
        packing=PackingProfile(**data.get("packing", {})),
    )


def resolve_policy(tensor_name: str, tensor_ndim: int, cfg: NHDFConfig) -> QuantizationPolicy:
    """Map a Hugging Face state-dict name to a packing policy.

    The Qwen3-MoE router and one-dimensional normalization weights remain FP16
    because route stability is more important than the few megabytes they save.
    Expert matrices receive the low-bit residual-branch format; attention,
    embeddings and the LM head use groupwise 4-bit weights.
    """

    name = tensor_name.lower()
    if tensor_ndim < 2:
        return cfg.packing.raw_policy()
    if ".mlp.router." in name or name.endswith("router.weight") or name.endswith("gate.weight"):
        return cfg.packing.raw_policy() if cfg.packing.raw_router_and_norms else cfg.packing.sensitive_policy()
    if ".mlp.experts." in name and (
        name.endswith("gate_up_proj") or name.endswith("gate_up_proj.weight") or name.endswith("down_proj") or name.endswith("down_proj.weight")
    ):
        return cfg.packing.expert_policy()
    if "embed_tokens.weight" in name or name.endswith("lm_head.weight"):
        return cfg.packing.sensitive_policy()
    if ".self_attn." in name and name.endswith(".weight"):
        return cfg.packing.sensitive_policy()
    # Safe default for any unexpected matrix.
    return cfg.packing.sensitive_policy()
