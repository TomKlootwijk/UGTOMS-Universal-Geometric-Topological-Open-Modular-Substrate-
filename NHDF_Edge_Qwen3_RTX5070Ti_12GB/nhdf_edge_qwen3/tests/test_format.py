from __future__ import annotations

import torch

from nhdf_edge.format import PackReader, PackWriter
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
    assert reader.verify_all()["ok"]
    loaded = reader.load("layer.weight")
    assert torch.equal(dequantize_tensor(loaded), dequantize_tensor(packed))

    entry = reader.entry("layer.weight")
    tensor_path = tmp_path / entry["file"]
    data = bytearray(tensor_path.read_bytes())
    data[-1] ^= 1
    tensor_path.write_bytes(data)
    assert not reader.verify_all()["ok"]
