"""Streaming conversion of a Hugging Face safetensors checkpoint."""
from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from safetensors import safe_open

from .calibration import load_hessian_diagonals
from .config import NHDFConfig, resolve_policy
from .format import PackWriter
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
)


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

    The converter opens one source shard at a time and quantizes one tensor at a
    time, so it never materializes the complete 61 GB model in RAM.
    """

    source = Path(source_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    weight_map = _weight_map(source)
    include_re = re.compile(include) if include else None
    exclude_re = re.compile(exclude) if exclude else None

    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        if include_re and not include_re.search(name):
            continue
        if exclude_re and exclude_re.search(name):
            continue
        by_shard[shard].append(name)

    writer = PackWriter(out, cfg.model.repo_id, cfg.to_dict())
    hessian = load_hessian_diagonals(hessian_path)
    processed = 0
    tensor_summaries: list[dict[str, object]] = []

    for shard_name in sorted(by_shard):
        shard_path = source / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"missing source shard: {shard_path}")
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for name in sorted(by_shard[shard_name]):
                if max_tensors is not None and processed >= max_tensors:
                    break
                tensor = f.get_tensor(name)
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
                    }
                )
                processed += 1
                del tensor, packed
        if max_tensors is not None and processed >= max_tensors:
            break

    _copy_metadata(source, out)
    summary_path = out / "tensor_summary.json"
    summary_path.write_text(json.dumps(tensor_summaries, indent=2), encoding="utf-8")
    return writer.finalize(
        {
            "source_dir": str(source.resolve()),
            "source_tensor_count": len(weight_map),
            "packed_tensor_count": processed,
            "partial_pack": processed != len(weight_map),
            "tensor_summary_file": summary_path.name,
            "hessian_calibration": str(Path(hessian_path).resolve()) if hessian_path else None,
        }
    )
