from __future__ import annotations

import pytest
import torch

from nhdf_edge.quantize import (
    PackedTensor,
    QuantizationPolicy,
    dequantize_tensor,
    quantize_tensor,
    residual_mask,
    verify_parity,
)


def test_two_bit_residual_pack_is_deterministic_and_valid() -> None:
    g = torch.Generator().manual_seed(7)
    weight = torch.randn(48, 512, generator=g)
    policy = QuantizationPolicy(base_bits=2, group_size=256, residual_fraction=0.15, iterations=3)
    first = quantize_tensor(weight, policy, name="w")
    second = quantize_tensor(weight, policy, name="w")

    assert torch.equal(first.base_codes, second.base_codes)
    assert torch.equal(first.residual_mask_words, second.residual_mask_words)
    assert first.stats["weighted_zero_set_max_abs"] < 1e-5
    assert verify_parity(first)["ok"]
    assert abs(residual_mask(first).float().mean().item() - 0.15) < 0.03

    restored = dequantize_tensor(first)
    assert restored.shape == weight.shape
    assert torch.isfinite(restored).all()
    assert torch.mean((weight - restored) ** 2).item() < torch.var(weight).item()


def test_residual_branch_improves_mse_on_same_weight() -> None:
    g = torch.Generator().manual_seed(11)
    weight = torch.randn(64, 768, generator=g)
    base = quantize_tensor(
        weight,
        QuantizationPolicy(base_bits=2, group_size=256, residual_fraction=0.0),
        name="base",
    )
    branched = quantize_tensor(
        weight,
        QuantizationPolicy(base_bits=2, group_size=256, residual_fraction=0.20),
        name="branched",
    )
    mse_base = torch.mean((weight - dequantize_tensor(base)) ** 2).item()
    mse_branched = torch.mean((weight - dequantize_tensor(branched)) ** 2).item()
    assert mse_branched < mse_base


def test_four_bit_omits_unused_residual_index() -> None:
    weight = torch.linspace(-2, 2, 32 * 256).reshape(32, 256)
    packed = quantize_tensor(
        weight,
        QuantizationPolicy(base_bits=4, group_size=256, residual_fraction=0.0),
        name="four",
    )
    assert packed.residual_mask_words.numel() == 0
    assert packed.residual_prefix.numel() == 0
    assert not residual_mask(packed).any()
    assert verify_parity(packed)["ok"]


def test_raw_tensor_tracks_source_parameters() -> None:
    weight = torch.ones(2048)
    packed = quantize_tensor(weight, QuantizationPolicy(mode="raw", group_size=256), name="norm")
    assert packed.stats["original_parameters"] == 2048
    assert torch.equal(dequantize_tensor(packed), weight)


def test_selected_row_decode_matches_full_decode() -> None:
    weight = torch.randn(16, 512, generator=torch.Generator().manual_seed(19))
    packed = quantize_tensor(
        weight,
        QuantizationPolicy(base_bits=2, group_size=256, residual_fraction=0.25),
        name="rows",
    )
    from nhdf_edge.quantize import dequantize_rows

    rows = torch.tensor([0, 3, 3, 15])
    assert torch.equal(dequantize_rows(packed, rows), dequantize_tensor(packed)[rows])


def test_chunk_boundaries_preserve_packed_output_with_padding_and_hessian() -> None:
    weight = torch.randn((5, 7, 777), generator=torch.Generator().manual_seed(112), dtype=torch.float16)
    hessian = torch.rand(777, generator=torch.Generator().manual_seed(113))
    for bits, residual_fraction in ((2, 0.20), (4, 0.0)):
        common = {
            "base_bits": bits,
            "group_size": 256,
            "residual_fraction": residual_fraction,
            "iterations": 3,
        }

        single_chunk = quantize_tensor(
            weight,
            QuantizationPolicy(**common, chunk_groups=100_000),
            name="chunked",
            hessian_diag=hessian,
        )
        many_chunks = quantize_tensor(
            weight,
            QuantizationPolicy(**common, chunk_groups=5),
            name="chunked",
            hessian_diag=hessian,
        )

        assert single_chunk.tensor_dict().keys() == many_chunks.tensor_dict().keys()
        for key, expected in single_chunk.tensor_dict().items():
            assert torch.equal(expected, many_chunks.tensor_dict()[key]), (bits, key)
        assert single_chunk.stats == many_chunks.stats


def test_serialized_geometry_and_prefix_are_validated_before_runtime() -> None:
    packed = quantize_tensor(
        torch.randn(8, 512, generator=torch.Generator().manual_seed(114)),
        QuantizationPolicy(base_bits=2, group_size=256, residual_fraction=0.25),
        name="validated.weight",
    )
    metadata = packed.metadata()
    metadata["padded_cols"] = 768
    with pytest.raises(ValueError, match="inconsistent padded/group geometry"):
        PackedTensor.from_tensor_dict(metadata, packed.tensor_dict())

    tensors = packed.tensor_dict()
    tensors["residual_prefix"] = tensors["residual_prefix"].clone()
    tensors["residual_prefix"][0] = 1
    with pytest.raises(ValueError, match="residual prefix does not match"):
        PackedTensor.from_tensor_dict(packed.metadata(), tensors)
