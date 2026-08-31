"""Activation-weighted reconstruction calibration utilities.

NHDF Edge works without calibration, but optional per-input-channel second
moments approximate a diagonal Hessian and make the local zero-set and branch
selection focus on directions seen by a teacher model.  This is post-training
reconstruction distillation, not a claim of full knowledge distillation.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn


class SecondMomentCollector:
    """Collect causal input second moments for matrix-bearing modules."""

    def __init__(self) -> None:
        self.sums: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = defaultdict(int)
        self._handles: list[Any] = []

    def _accumulate(self, key: str, x: torch.Tensor) -> None:
        values = x.detach().to(device="cpu", dtype=torch.float32).reshape(-1, x.shape[-1])
        contribution = values.square().sum(dim=0)
        if key in self.sums:
            self.sums[key] += contribution
        else:
            self.sums[key] = contribution
        self.counts[key] += values.shape[0]

    def attach(self, model: nn.Module) -> None:
        """Attach hooks to standard Linear layers and Qwen3 expert inputs."""

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                key = f"{name}.weight"

                def linear_hook(_module, args, _key=key):
                    if args and isinstance(args[0], torch.Tensor):
                        self._accumulate(_key, args[0])

                self._handles.append(module.register_forward_pre_hook(linear_hook))
            elif module.__class__.__name__ == "Qwen3MoeExperts":
                # The expert input directly calibrates gate_up_proj.  The
                # down-projection activation is internal to the upstream
                # implementation and remains unweighted unless a custom teacher
                # hook is supplied.
                key = f"{name}.gate_up_proj"

                def expert_hook(_module, args, _key=key):
                    if args and isinstance(args[0], torch.Tensor):
                        self._accumulate(_key, args[0])

                self._handles.append(module.register_forward_pre_hook(expert_hook))

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def tensors(self, eps: float = 1e-8) -> dict[str, torch.Tensor]:
        return {
            key: (value / max(self.counts[key], 1)).clamp_min(eps).contiguous()
            for key, value in self.sums.items()
        }

    def save(self, path: str | Path, *, metadata: dict[str, str] | None = None) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        save_file(self.tensors(), str(target), metadata=metadata or {})
        return target


def load_hessian_diagonals(path: str | Path | None) -> dict[str, torch.Tensor]:
    if path is None:
        return {}
    return load_file(str(path), device="cpu")
