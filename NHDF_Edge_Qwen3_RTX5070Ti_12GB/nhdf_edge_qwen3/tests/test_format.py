from __future__ import annotations

import pytest
import torch

from nhdf_edge.format import PackReader, PackWriter, set_pack_validation_status
from nhdf_edge.quantize import QuantizationPolicy, dequantize_tensor, quantize_tensor


def test_pack_roundtrip_and_crc(tmp_path) -> None:
    weight = torch.randn(12, 512, generator=torch.Generator().manual_seed(5))
    packed = quantize_tensor(
        weight,
        QuantizationPolicy(base_bits=4, group_size=256, residual_fraction=0.0),
        name="layer.weight",
    )
    writer = PackWriter(tmp_path, "unit/test", {"profile": "test"})
    writer.add(packed)
    writer.finalize()

    reader = PackReader(tmp_path)
    assert reader.validation_status == "UNCALIBRATED"
    with pytest.raises(RuntimeError, match="validation status UNCALIBRATED"):
        reader.require_validated()
    assert reader.verify_all()["ok"]
    loaded = reader.load("layer.weight")
    assert torch.equal(dequantize_tensor(loaded), dequantize_tensor(packed))

    entry = reader.entry("layer.weight")
    tensor_path = tmp_path / entry["file"]
    data = bytearray(tensor_path.read_bytes())
    data[-1] ^= 1
    tensor_path.write_bytes(data)
    assert not reader.verify_all()["ok"]


def test_manifest_digest_rejects_geometry_tampering(tmp_path) -> None:
    weight = torch.randn(4, 256, generator=torch.Generator().manual_seed(6))
    packed = quantize_tensor(
        weight,
        QuantizationPolicy(base_bits=4, group_size=256, residual_fraction=0.0),
        name="layer.weight",
    )
    writer = PackWriter(tmp_path, "unit/test", {"profile": "test"})
    writer.add(packed)
    manifest_path = writer.finalize()

    manifest = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(manifest.replace('"original_cols": 256', '"original_cols": 512'), encoding="utf-8")
    with pytest.raises(OSError, match="manifest SHA-256 mismatch"):
        PackReader(tmp_path)


def test_pack_validation_status_requires_evidence_and_updates_digest(tmp_path) -> None:
    writer = PackWriter(tmp_path, "unit/test", {"profile": "test"})
    writer.finalize({"partial_pack": False})

    with pytest.raises(ValueError, match="non-empty JSON object"):
        set_pack_validation_status(tmp_path, "VALIDATED", evidence={})

    set_pack_validation_status(
        tmp_path,
        "QUALITY_FAILED",
        evidence={"functional_generation": "collapsed output"},
    )
    failed = PackReader(tmp_path)
    assert failed.validation_status == "QUALITY_FAILED"
    assert failed.manifest["validation"]["previous_status"] == "UNCALIBRATED"

    set_pack_validation_status(
        tmp_path,
        "VALIDATED",
        evidence={"functional_generation": "deterministic and coherent"},
    )
    validated = PackReader(tmp_path)
    validated.require_validated()
    assert validated.validation_status == "VALIDATED"


def test_partial_pack_cannot_be_marked_validated(tmp_path) -> None:
    writer = PackWriter(tmp_path, "unit/test", {"profile": "test"})
    writer.finalize({"partial_pack": True})
    with pytest.raises(ValueError, match="partial pack"):
        set_pack_validation_status(
            tmp_path,
            "VALIDATED",
            evidence={"functional_generation": "not applicable"},
        )
