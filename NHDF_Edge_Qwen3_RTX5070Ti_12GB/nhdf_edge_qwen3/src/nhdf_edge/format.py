"""Directory-level NHDF pack format with per-file CRC32."""
from __future__ import annotations

import hashlib
import json
import os
import re
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .quantize import PackedTensor

FORMAT_VERSION = "nhdf-edge-0.1"


def _safe_id(name: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")[:96]
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{digest}" if readable else digest


def crc32_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    value = 0
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            value = zlib.crc32(chunk, value)
    return f"{value & 0xFFFFFFFF:08x}"


@dataclass
class PackWriter:
    root: Path
    source_model: str
    config: dict[str, Any]

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        (self.root / "tensors").mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, Any] = {}
        self._stats = {"source_parameters": 0, "packed_bytes": 0, "tensors": 0}

    def add(self, packed: PackedTensor) -> dict[str, Any]:
        if packed.name in self._entries:
            raise ValueError(f"duplicate tensor name: {packed.name}")
        tensor_id = _safe_id(packed.name)
        relative = Path("tensors") / f"{tensor_id}.safetensors"
        path = self.root / relative
        tensors = packed.tensor_dict()
        # safetensors rejects an entirely empty dictionary; every packed object
        # contains either raw data or base codes, so this is a strong invariant.
        save_file(tensors, str(path), metadata={"format": FORMAT_VERSION, "tensor_name": packed.name})
        size = path.stat().st_size
        entry = {
            "file": relative.as_posix(),
            "crc32": crc32_file(path),
            "file_bytes": size,
            "metadata": packed.metadata(),
        }
        self._entries[packed.name] = entry
        self._stats["source_parameters"] += int(packed.stats.get("original_parameters", 0))
        self._stats["packed_bytes"] += size
        self._stats["tensors"] += 1
        return entry

    def finalize(self, extra: dict[str, Any] | None = None) -> Path:
        manifest = {
            "format": FORMAT_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_model": self.source_model,
            "config": self.config,
            "summary": self._stats,
            "tensors": self._entries,
        }
        if extra:
            manifest.update(extra)
        path = self.root / "manifest.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
        return path


class PackReader:
    def __init__(self, root: str | Path, *, verify_crc: bool = True):
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != FORMAT_VERSION:
            raise ValueError(f"unsupported pack format: {self.manifest.get('format')}")
        self.verify_crc = verify_crc

    def names(self) -> list[str]:
        return sorted(self.manifest["tensors"])

    def entry(self, name: str) -> dict[str, Any]:
        try:
            return self.manifest["tensors"][name]
        except KeyError as exc:
            raise KeyError(f"tensor not found in pack: {name}") from exc

    def load(self, name: str, *, device: str | torch.device = "cpu") -> PackedTensor:
        entry = self.entry(name)
        path = self.root / entry["file"]
        if self.verify_crc:
            actual = crc32_file(path)
            if actual != entry["crc32"]:
                raise IOError(f"CRC32 mismatch for {name}: expected {entry['crc32']}, got {actual}")
        tensors = load_file(str(path), device=str(device))
        return PackedTensor.from_tensor_dict(entry["metadata"], tensors)

    def verify_all(self) -> dict[str, Any]:
        failures: list[dict[str, str]] = []
        total = 0
        for name in self.names():
            entry = self.entry(name)
            path = self.root / entry["file"]
            total += 1
            if not path.exists():
                failures.append({"tensor": name, "error": "missing file"})
                continue
            actual = crc32_file(path)
            if actual != entry["crc32"]:
                failures.append({"tensor": name, "error": f"crc expected {entry['crc32']} got {actual}"})
        return {"files": total, "failures": failures, "ok": not failures}
