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
PACK_VALIDATION_STATUSES = frozenset(
    {"UNCALIBRATED", "QUALITY_FAILED", "RESOURCE_FAILED", "VALIDATED"}
)


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


def set_pack_validation_status(
    root: str | Path,
    status: str,
    *,
    evidence: dict[str, Any],
) -> Path:
    """Atomically record an externally measured pack disposition.

    This deliberately does not infer quality from CRC/parity or reconstruction
    statistics.  The caller must supply the measured evidence used to make the
    decision, and VALIDATED is forbidden for partial packs.
    """

    normalized = status.upper()
    if normalized not in PACK_VALIDATION_STATUSES:
        raise ValueError(
            f"unsupported pack validation status {status!r}; "
            f"expected one of {sorted(PACK_VALIDATION_STATUSES)}"
        )
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("validation evidence must be a non-empty JSON object")

    pack_root = Path(root)
    reader = PackReader(pack_root, verify_crc=False)
    if normalized == "VALIDATED" and bool(reader.manifest.get("partial_pack", False)):
        raise ValueError("a partial pack cannot be marked VALIDATED")

    manifest = dict(reader.manifest)
    previous = reader.validation_status
    record = {
        "status": normalized,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "previous_status": previous,
        "evidence": evidence,
    }
    history = list(manifest.get("validation_history", []))
    prior_record = manifest.get("validation")
    if isinstance(prior_record, dict) and prior_record.get("updated_utc"):
        history.append(prior_record)
    manifest["validation"] = record
    if history:
        manifest["validation_history"] = history

    manifest_path = pack_root / "manifest.json"
    manifest_tmp = pack_root / "manifest.json.tmp"
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    digest_path = pack_root / "manifest.sha256"
    digest_tmp = pack_root / "manifest.sha256.tmp"
    digest_tmp.write_text(digest + "\n", encoding="ascii")
    os.replace(digest_tmp, digest_path)
    return manifest_path


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

    def restore(self, entries: dict[str, Any], stats: dict[str, int]) -> None:
        """Restore a previously verified in-progress writer state."""

        if self._entries or self._stats != {"source_parameters": 0, "packed_bytes": 0, "tensors": 0}:
            raise RuntimeError("PackWriter state can only be restored before adding tensors")
        self._entries = dict(entries)
        self._stats = {
            "source_parameters": int(stats["source_parameters"]),
            "packed_bytes": int(stats["packed_bytes"]),
            "tensors": int(stats["tensors"]),
        }

    def progress_state(self) -> dict[str, Any]:
        """Return JSON-serializable state for interruption-safe conversion."""

        return {"entries": self._entries, "stats": self._stats}

    def finalize(self, extra: dict[str, Any] | None = None) -> Path:
        manifest = {
            "format": FORMAT_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_model": self.source_model,
            "config": self.config,
            "summary": self._stats,
            "tensors": self._entries,
            # Integrity is necessary but not sufficient.  A newly serialized
            # pack cannot be executed as a deployment artifact until an
            # independent quality run promotes it to VALIDATED.
            "validation": {"status": "UNCALIBRATED"},
        }
        if extra:
            manifest.update(extra)
        path = self.root / "manifest.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
        digest_path = self.root / "manifest.sha256"
        digest_tmp = self.root / "manifest.sha256.tmp"
        digest_tmp.write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n", encoding="ascii")
        os.replace(digest_tmp, digest_path)
        return path


class PackReader:
    def __init__(self, root: str | Path, *, verify_crc: bool = True):
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing manifest: {manifest_path}")
        digest_path = self.root / "manifest.sha256"
        if not digest_path.exists():
            raise FileNotFoundError(f"missing manifest digest: {digest_path}")
        expected_digest = digest_path.read_text(encoding="ascii").strip().lower()
        actual_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_digest) or actual_digest != expected_digest:
            raise IOError(
                f"manifest SHA-256 mismatch: expected {expected_digest!r}, got {actual_digest}"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != FORMAT_VERSION:
            raise ValueError(f"unsupported pack format: {self.manifest.get('format')}")
        self.verify_crc = verify_crc

    @property
    def validation_status(self) -> str:
        validation = self.manifest.get("validation", {})
        status = validation.get("status", "UNCALIBRATED") if isinstance(validation, dict) else None
        if status not in PACK_VALIDATION_STATUSES:
            raise ValueError(f"unsupported pack validation status: {status!r}")
        return status

    def require_validated(self, *, allow_unvalidated: bool = False) -> None:
        status = self.validation_status
        if status == "VALIDATED" or allow_unvalidated:
            return
        raise RuntimeError(
            f"refusing to execute pack with validation status {status}; "
            "pass allow_unvalidated=True only for an explicit experiment"
        )

    def names(self) -> list[str]:
        return sorted(self.manifest["tensors"])

    def entry(self, name: str) -> dict[str, Any]:
        try:
            return self.manifest["tensors"][name]
        except KeyError as exc:
            raise KeyError(f"tensor not found in pack: {name}") from exc

    def load(self, name: str, *, device: str | torch.device = "cpu") -> PackedTensor:
        entry = self.entry(name)
        root = self.root.resolve()
        path = (root / entry["file"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"tensor path escapes the pack root: {entry['file']}")
        if path.stat().st_size != int(entry["file_bytes"]):
            raise IOError(f"file-size mismatch for {name}")
        if self.verify_crc:
            actual = crc32_file(path)
            if actual != entry["crc32"]:
                raise IOError(f"CRC32 mismatch for {name}: expected {entry['crc32']}, got {actual}")
        tensors = load_file(str(path), device=str(device))
        # ``safetensors.torch.load_file`` may leave CPU storages backed by a
        # read-only memory mapping.  Windows then refuses to replace or mutate
        # the file for as long as the returned tensors are alive (important for
        # fault-injection and pack-repair workflows).  Detach CPU loads from
        # the mapping; CUDA loads are already copied to device and must not be
        # cloned because the edge profile has very little spare VRAM.
        if os.name == "nt" and torch.device(device).type == "cpu":
            mapped_tensors = tensors
            tensors = {key: value.clone() for key, value in mapped_tensors.items()}
            del mapped_tensors
        packed = PackedTensor.from_tensor_dict(entry["metadata"], tensors)
        if packed.name != name:
            raise ValueError(f"manifest tensor name mismatch: requested {name}, stored {packed.name}")
        return packed

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
