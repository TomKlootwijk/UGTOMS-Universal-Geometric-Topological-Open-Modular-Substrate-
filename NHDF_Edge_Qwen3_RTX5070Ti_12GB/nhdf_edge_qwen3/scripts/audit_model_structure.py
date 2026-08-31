#!/usr/bin/env python3
"""Validate pack policies and loader paths against an exact Qwen config."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

from nhdf_edge.checkpoint import _logical_tensor_plan, _read_model_config, _weight_map
from nhdf_edge.config import load_config, resolve_policy
from nhdf_edge.runtime.qwen3_loader import _get_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--config", default="configs/qwen3_30b_a3b_edge12.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    profile = load_config(args.config)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)

    tensor_counts: Counter[str] = Counter()
    parameter_counts: Counter[str] = Counter()
    expert_groups: dict[str, set[str]] = defaultdict(set)
    replacement_errors: list[dict[str, str]] = []
    state = model.state_dict()
    source_weight_map = _weight_map(model_dir)
    logical_source_plan = _logical_tensor_plan(source_weight_map, _read_model_config(model_dir))
    source_not_in_model = sorted(set(logical_source_plan) - set(state))
    model_not_in_source = sorted(set(state) - set(logical_source_plan))

    for name, tensor in state.items():
        policy = resolve_policy(name, tensor.ndim, profile)
        label = "raw" if policy.mode == "raw" else f"{policy.base_bits}bit_residual_{policy.residual_fraction:g}"
        tensor_counts[label] += 1
        parameter_counts[label] += tensor.numel()

        if name.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.gate_up_proj.weight")):
            expert_groups[name.rsplit(".gate_up_proj", 1)[0]].add("gate_up")
        elif name.endswith((".mlp.experts.down_proj", ".mlp.experts.down_proj.weight")):
            expert_groups[name.rsplit(".down_proj", 1)[0]].add("down")
        elif policy.mode != "raw" and name.endswith(".weight"):
            try:
                module = _get_path(model, name[: -len(".weight")])
                if module.__class__.__name__ not in {"Linear", "Embedding"}:
                    replacement_errors.append({"tensor": name, "module_type": module.__class__.__name__})
            except (AttributeError, IndexError, KeyError) as exc:
                replacement_errors.append({"tensor": name, "error": str(exc)})
        elif policy.mode == "raw":
            try:
                _get_path(model, name)
            except (AttributeError, IndexError, KeyError) as exc:
                replacement_errors.append({"tensor": name, "error": str(exc)})

    incomplete_experts = {
        name: sorted(parts) for name, parts in expert_groups.items() if parts != {"gate_up", "down"}
    }
    actual_parameters = sum(tensor.numel() for tensor in state.values())
    result = {
        "model_dir": str(model_dir.resolve()),
        "model_type": config.model_type,
        "architectures": config.architectures,
        "state_tensors": len(state),
        "source_checkpoint_tensors": len(source_weight_map),
        "logical_source_tensors": len(logical_source_plan),
        "source_layout_transformed_tensors": sum(
            plan.kind != "direct" for plan in logical_source_plan.values()
        ),
        "source_names_not_in_model": source_not_in_model,
        "model_names_not_in_source": model_not_in_source,
        "source_namespace_matches_model": not source_not_in_model and not model_not_in_source,
        "actual_parameters": actual_parameters,
        "configured_parameters": profile.model.total_parameters,
        "parameter_count_matches": actual_parameters == profile.model.total_parameters,
        "policy_tensor_counts": dict(tensor_counts),
        "policy_parameter_counts": dict(parameter_counts),
        "expert_blocks": len(expert_groups),
        "incomplete_expert_blocks": incomplete_experts,
        "replacement_errors": replacement_errors,
        "meta_parameter_count_before_replacement": sum(
            1 for parameter in model.parameters() if parameter.is_meta
        ),
        "meta_buffers_before_replacement": [
            name for name, buffer in model.named_buffers() if buffer.is_meta
        ],
        "status": "pass"
        if (
            not incomplete_experts
            and not replacement_errors
            and not source_not_in_model
            and not model_not_in_source
            and actual_parameters == profile.model.total_parameters
        )
        else "fail",
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
