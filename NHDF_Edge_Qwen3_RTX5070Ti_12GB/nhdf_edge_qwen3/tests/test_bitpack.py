from __future__ import annotations

import pytest
import torch

from nhdf_edge.bitpack import (
    bits_from_words,
    pack_unsigned,
    prefix_counts,
    unpack_unsigned,
    words_from_bits,
)


@pytest.mark.parametrize("bits", [1, 2, 4, 8])
@pytest.mark.parametrize("count", [0, 1, 7, 31, 32, 33, 257])
def test_pack_roundtrip(bits: int, count: int) -> None:
    g = torch.Generator().manual_seed(1234 + bits + count)
    values = torch.randint(0, 1 << bits, (count,), generator=g, dtype=torch.int64)
    packed = pack_unsigned(values, bits)
    restored = unpack_unsigned(packed, bits, count)
    assert torch.equal(restored.to(torch.int64), values)


def test_words_roundtrip_including_sign_bit() -> None:
    bits = torch.zeros(70, dtype=torch.bool)
    bits[[0, 31, 32, 63, 69]] = True
    words = words_from_bits(bits, word_bits=32)
    assert words.dtype == torch.int32
    restored = bits_from_words(words, bits.numel(), word_bits=32)
    assert torch.equal(restored, bits)


def test_prefix_counts() -> None:
    mask = torch.tensor([1, 0, 1, 1, 0, 0, 1, 1, 1], dtype=torch.bool)
    prefix = prefix_counts(mask, block_groups=4)
    assert prefix.dtype == torch.int32
    assert prefix.tolist() == [0, 3, 5]
