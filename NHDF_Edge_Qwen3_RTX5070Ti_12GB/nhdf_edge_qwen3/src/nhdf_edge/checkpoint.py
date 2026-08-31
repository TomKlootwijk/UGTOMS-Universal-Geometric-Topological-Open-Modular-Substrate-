"""Streaming conversion of a Hugging Face safetensors checkpoint."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from contextlib import ExitStack, contextmanager
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open

from .calibration import load_hessian_diagonals
from .config import NHDFConfig, resolve_policy
from .format import PackWriter, crc32_file
from .quantize import quantize_tensor

_COPY_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.json",
    "LICENSE",
    "README.md",
    "NHDF_SOURCE.json",
)

_EXPERT_SOURCE_RE = re.compile(
    r"^(?P<prefix>.+\.mlp\.experts)\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
_LOGICAL_LAYOUT_VERSION = "qwen2_moe_numeric_stack_gate_then_up_v1"


@dataclass(frozen=True)
class _TensorPlan:
    """One runtime parameter and the Hub tensors needed to materialize it."""

    name: str
    kind: str
    source_names: tuple[str, ...]
    expert_count: int = 0
    component_shape: tuple[int, int] | None = None


def _read_model_config(source: Path) -> dict[str, object]:
    path = source / "config.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"model config must contain a JSON object: {path}")
    return data


def _logical_tensor_plan(
    weight_map: dict[str, str],
    model_config: dict[str, object] | None = None,
) -> dict[str, _TensorPlan]:
    """Map the released Qwen MoE layout to Transformers' runtime layout.

    Qwen's Hub checkpoint stores each expert projection separately, while
    current Transformers versions expose two stacked parameters per layer:
    ``gate_up_proj[E, 2I, H]`` and ``down_proj[E, H, I]``.  This is the same
    MergeModulelist/Concatenate conversion used by Transformers' qwen2_moe
    weight mapping.
    """

    config = model_config or {}
    configured_experts = config.get("num_experts", config.get("num_local_experts"))
    expected_experts = int(configured_experts) if configured_experts is not None else None
    hidden_size = int(config["hidden_size"]) if "hidden_size" in config else None
    moe_intermediate = (
        int(config["moe_intermediate_size"]) if "moe_intermediate_size" in config else None
    )
    expected_prefixes: set[str] | None = None
    if expected_experts and "num_hidden_layers" in config:
        layers = int(config["num_hidden_layers"])
        sparse_step = int(config.get("decoder_sparse_step", 1))
        if sparse_step <= 0:
            raise ValueError(f"decoder_sparse_step must be positive, got {sparse_step}")
        expected_prefixes = {
            f"model.layers.{layer}.mlp.experts"
            for layer in range(layers)
            if (layer + 1) % sparse_step == 0
        }

    plans: dict[str, _TensorPlan] = {}
    expert_parts: dict[str, dict[int, dict[str, str]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    matched_sources = 0

    for source_name in weight_map:
        match = _EXPERT_SOURCE_RE.match(source_name)
        if match is None:
            plans[source_name] = _TensorPlan(source_name, "direct", (source_name,))
            continue
        prefix = match.group("prefix")
        expert = int(match.group("expert"))
        projection = match.group("projection")
        if projection in expert_parts[prefix][expert]:
            raise ValueError(f"duplicate expert component: {source_name}")
        expert_parts[prefix][expert][projection] = source_name
        matched_sources += 1

    if expected_prefixes is not None and set(expert_parts) != expected_prefixes:
        missing = sorted(expected_prefixes - set(expert_parts))
        unexpected = sorted(set(expert_parts) - expected_prefixes)
        raise ValueError(
            "expert layer set does not match config: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    for prefix, experts in expert_parts.items():
        expert_ids = sorted(experts)
        count = expected_experts if expected_experts is not None else len(expert_ids)
        expected_ids = list(range(count))
        if expert_ids != expected_ids:
            raise ValueError(
                f"non-contiguous expert ids for {prefix}: expected {expected_ids}, got {expert_ids}"
            )
        required = {"gate_proj", "up_proj", "down_proj"}
        for expert in expert_ids:
            missing = required - set(experts[expert])
            if missing:
                raise ValueError(
                    f"incomplete expert {prefix}.{expert}: missing {sorted(missing)}"
                )

        gate_up_name = f"{prefix}.gate_up_proj"
        down_name = f"{prefix}.down_proj"
        if gate_up_name in plans or down_name in plans:
            raise ValueError(f"expert fusion target collides with a source tensor under {prefix}")
        gate_names = tuple(experts[i]["gate_proj"] for i in expert_ids)
        up_names = tuple(experts[i]["up_proj"] for i in expert_ids)
        down_names = tuple(experts[i]["down_proj"] for i in expert_ids)
        plans[gate_up_name] = _TensorPlan(
            gate_up_name,
            "expert_gate_up",
            gate_names + up_names,
            len(expert_ids),
            (moe_intermediate, hidden_size)
            if moe_intermediate is not None and hidden_size is not None
            else None,
        )
        plans[down_name] = _TensorPlan(
            down_name,
            "expert_down",
            down_names,
            len(expert_ids),
            (hidden_size, moe_intermediate)
            if moe_intermediate is not None and hidden_size is not None
            else None,
        )

    consumed_sources = sum(len(plan.source_names) for plan in plans.values())
    if consumed_sources != len(weight_map) or matched_sources % 3:
        raise RuntimeError(
            "internal checkpoint conversion error: not every source tensor was consumed exactly once"
        )
    return plans


def _validate_component(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"expert component {name} has shape {tuple(tensor.shape)}, expected {shape}")
    if tensor.dtype != dtype:
        raise ValueError(f"expert component {name} has dtype {tensor.dtype}, expected {dtype}")


@contextmanager
def _materialize_tensor(
    source: Path,
    weight_map: dict[str, str],
    plan: _TensorPlan,
) -> Iterator[torch.Tensor]:
    """Yield one direct or fused logical tensor while its shard maps are live."""

    shard_names = sorted({weight_map[name] for name in plan.source_names})
    with ExitStack() as stack:
        handles = {
            shard: stack.enter_context(safe_open(str(source / shard), framework="pt", device="cpu"))
            for shard in shard_names
        }

        def fetch(name: str) -> torch.Tensor:
            return handles[weight_map[name]].get_tensor(name)

        if plan.kind == "direct":
            yield fetch(plan.source_names[0])
            return

        count = plan.expert_count
        if plan.kind == "expert_gate_up":
            gate_names = plan.source_names[:count]
            up_names = plan.source_names[count:]
            first_gate = fetch(gate_names[0])
            first_up = fetch(up_names[0])
            if first_gate.ndim != 2 or first_up.ndim != 2:
                raise ValueError(f"expert projections for {plan.name} must be rank-2")
            if first_gate.shape != first_up.shape or first_gate.dtype != first_up.dtype:
                raise ValueError(
                    f"gate/up geometry mismatch for {plan.name}: "
                    f"{tuple(first_gate.shape)}/{first_gate.dtype} vs "
                    f"{tuple(first_up.shape)}/{first_up.dtype}"
                )
            rows, cols = first_gate.shape
            shape = (int(rows), int(cols))
            if plan.component_shape is not None and shape != plan.component_shape:
                raise ValueError(
                    f"expert component geometry for {plan.name} is {shape}, "
                    f"expected {plan.component_shape} from config"
                )
            fused = torch.empty((count, 2 * rows, cols), dtype=first_gate.dtype)
            for expert, (gate_name, up_name) in enumerate(zip(gate_names, up_names, strict=True)):
                gate = fetch(gate_name)
                up = fetch(up_name)
                _validate_component(gate, name=gate_name, shape=shape, dtype=fused.dtype)
                _validate_component(up, name=up_name, shape=shape, dtype=fused.dtype)
                fused[expert, :rows].copy_(gate)
                fused[expert, rows:].copy_(up)
            yield fused
            return

        if plan.kind == "expert_down":
            first = fetch(plan.source_names[0])
            if first.ndim != 2:
                raise ValueError(f"expert projections for {plan.name} must be rank-2")
            shape = tuple(first.shape)
            if plan.component_shape is not None and shape != plan.component_shape:
                raise ValueError(
                    f"expert component geometry for {plan.name} is {shape}, "
                    f"expected {plan.component_shape} from config"
                )
            fused = torch.empty((count, *shape), dtype=first.dtype)
            for expert, source_name in enumerate(plan.source_names):
                component = fetch(source_name)
                _validate_component(component, name=source_name, shape=shape, dtype=fused.dtype)
                fused[expert].copy_(component)
            yield fused
            return

        raise RuntimeError(f"unsupported tensor materialization kind: {plan.kind}")


def _weight_map(source: Path) -> dict[str, str]:
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        data = json.loads(index_path.read_text(encoding="utf-8"))
        return dict(data["weight_map"])
    files = sorted(source.glob("*.safetensors"))
    if len(files) != 1:
        raise FileNotFoundError("could not locate model.safetensors.index.json or a single safetensors file")
    with safe_open(str(files[0]), framework="pt", device="cpu") as f:
        return {key: files[0].name for key in f.keys()}


def _copy_metadata(source: Path, out: Path) -> None:
    meta = out / "hf_metadata"
    meta.mkdir(parents=True, exist_ok=True)
    for name in _COPY_FILES:
        src = source / name
        if src.exists() and src.is_file():
            shutil.copy2(src, meta / name)


def _stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_progress(path: Path, payload: dict[str, object]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def pack_checkpoint(
    source_dir: str | Path,
    output_dir: str | Path,
    cfg: NHDFConfig,
    *,
    include: str | None = None,
    exclude: str | None = None,
    max_tensors: int | None = None,
    hessian_path: str | Path | None = None,
) -> Path:
    """Convert a local Hugging Face checkpoint into an NHDF pack.

    The converter quantizes one logical runtime tensor at a time, so it never
    materializes the complete 61 GB model in RAM.  Released per-expert Qwen
    tensors are fused into the two 3-D parameters used by Transformers before
    quantization; a fused gate/up parameter is the largest CPU allocation.
    """

    source = Path(source_dir)
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()) and not (out / "pack_state.json").exists():
        raise FileExistsError(
            f"refusing to overwrite non-resumable pack output: {out}; choose a new directory"
        )
    out.mkdir(parents=True, exist_ok=True)
    weight_map = _weight_map(source)
    logical_plans = _logical_tensor_plan(weight_map, _read_model_config(source))
    include_re = re.compile(include) if include else None
    exclude_re = re.compile(exclude) if exclude else None

    selected_plans: list[_TensorPlan] = []
    for name, plan in logical_plans.items():
        if include_re and not include_re.search(name):
            continue
        if exclude_re and exclude_re.search(name):
            continue
        selected_plans.append(plan)

    # Preserve shard-streaming order.  In particular, a real partial-pack smoke
    # test can run while later download shards are still arriving.
    selected_plans.sort(
        key=lambda plan: (
            min(weight_map[source_name] for source_name in plan.source_names),
            plan.name,
        )
    )

    writer = PackWriter(out, cfg.model.repo_id, cfg.to_dict())
    hessian = load_hessian_diagonals(hessian_path)
    selected_names = [plan.name for plan in selected_plans]
    plan_identity = {
        name: {
            "kind": plan.kind,
            "source_names": plan.source_names,
            "expert_count": plan.expert_count,
            "component_shape": plan.component_shape,
        }
        for name, plan in sorted(logical_plans.items())
    }
    run_identity = {
        "source": str(source.resolve()),
        "weight_map_sha256": _stable_hash(weight_map),
        "logical_layout_version": _LOGICAL_LAYOUT_VERSION,
        "logical_plan_sha256": _stable_hash(plan_identity),
        "selected_names_sha256": _stable_hash(selected_names),
        "config_sha256": _stable_hash(cfg.to_dict()),
        "hessian": str(Path(hessian_path).resolve()) if hessian_path else None,
    }
    progress_path = out / "pack_state.json"
    processed_names: set[str] = set()
    tensor_summaries: list[dict[str, object]] = []
    if progress_path.exists():
        state = json.loads(progress_path.read_text(encoding="utf-8"))
        if state.get("run_identity") != run_identity:
            raise ValueError(
                f"in-progress pack at {progress_path} belongs to a different source/config selection"
            )
        restored = state["writer"]
        entries = dict(restored["entries"])
        for name, entry in entries.items():
            tensor_path = out / entry["file"]
            if not tensor_path.exists() or crc32_file(tensor_path) != entry["crc32"]:
                raise IOError(f"in-progress tensor failed CRC validation: {name}")
        writer.restore(entries, dict(restored["stats"]))
        processed_names = set(entries)
        tensor_summaries = list(state.get("tensor_summaries", []))

    processed = len(processed_names)

    for plan in selected_plans:
        if max_tensors is not None and processed >= max_tensors:
            break
        name = plan.name
        if name in processed_names:
            continue
        missing_shards = sorted(
            {
                str(source / weight_map[source_name])
                for source_name in plan.source_names
                if not (source / weight_map[source_name]).exists()
            }
        )
        if missing_shards:
            raise FileNotFoundError(f"missing source shard(s) for {name}: {missing_shards}")
        with _materialize_tensor(source, weight_map, plan) as tensor:
            policy = resolve_policy(name, tensor.ndim, cfg)
            packed = quantize_tensor(tensor, policy, name=name, hessian_diag=hessian.get(name))
            entry = writer.add(packed)
            tensor_summaries.append(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "mode": policy.mode,
                    "base_bits": policy.base_bits,
                    "residual_fraction": policy.residual_fraction,
                    "packed_bytes": packed.stats.get("packed_bytes", entry["file_bytes"]),
                    "effective_bits_per_weight": packed.stats.get("effective_bits_per_weight"),
                    "zero_set_max_abs": packed.stats.get("weighted_zero_set_max_abs"),
                    "source_tensor_count": len(plan.source_names),
                    "source_transform": plan.kind,
                }
            )
            processed += 1
            processed_names.add(name)
            _write_progress(
                progress_path,
                {
                    "run_identity": run_identity,
                    "writer": writer.progress_state(),
                    "tensor_summaries": tensor_summaries,
                },
            )
            print(
                f"packed {processed}/{min(len(selected_names), max_tensors or len(selected_names))}: {name}",
                flush=True,
            )
        del tensor, packed

    _copy_metadata(source, out)
    provenance_path = source / "NHDF_SOURCE.json"
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.exists()
        else None
    )
    summary_path = out / "tensor_summary.json"
    summary_path.write_text(json.dumps(tensor_summaries, indent=2), encoding="utf-8")
    manifest = writer.finalize(
        {
            "source_dir": str(source.resolve()),
            "source_provenance": provenance,
            "source_tensor_count": len(weight_map),
            "logical_tensor_count": len(logical_plans),
            "selected_tensor_count": len(selected_names),
            "source_layout_transform": (
                "qwen2_moe_individual_experts_to_stacked"
                if any(plan.kind != "direct" for plan in logical_plans.values())
                else "identity"
            ),
            "logical_layout_version": _LOGICAL_LAYOUT_VERSION,
            "packed_tensor_count": processed,
            "partial_pack": processed != len(logical_plans),
            "tensor_summary_file": summary_path.name,
            "hessian_calibration": str(Path(hessian_path).resolve()) if hessian_path else None,
        }
    )
    if progress_path.exists():
        progress_path.unlink()
    return manifest
