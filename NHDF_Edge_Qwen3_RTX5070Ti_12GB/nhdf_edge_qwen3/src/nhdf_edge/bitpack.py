"""Bit-packing helpers used by the NHDF reference format.

The routines are intentionally simple and deterministic.  They operate on flat
unsigned integer tensors and support the base precisions used by the reference
profiles (2 and 4 bits).  The implementation is CPU-safe and has no dependency
on a custom CUDA extension; the CUDA runtime has its own unpack path.
"""
from __future__ import annotations

from typing import Final

import torch

_SUPPORTED_BITS: Final[tuple[int, ...]] = (1, 2, 4, 8)


def _validate_bits(bits: int) -> None:
    if bits not in _SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {_SUPPORTED_BITS}, got {bits}")


def pack_unsigned(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned values into a contiguous ``uint8`` tensor.

    Parameters
    ----------
    values:
        Integer tensor with every value in ``[0, 2**bits)``.  Any shape is
        accepted; the packed output is one-dimensional.
    bits:
        Number of bits per value.  Supported values are 1, 2, 4 and 8.

    Returns
    -------
    torch.Tensor
        Flat ``uint8`` tensor.  If the number of input values is not a multiple
        of the packing factor, the final byte is zero padded.
    """

    _validate_bits(bits)
    flat = values.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    if flat.numel() == 0:
        return torch.empty(0, dtype=torch.uint8)

    max_value = (1 << bits) - 1
    if torch.any(flat < 0) or torch.any(flat > max_value):
        raise ValueError(f"values must be in [0, {max_value}] for {bits}-bit packing")

    if bits == 8:
        return flat.to(torch.uint8).contiguous()

    per_byte = 8 // bits
    pad = (-flat.numel()) % per_byte
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)], dim=0)

    groups = flat.view(-1, per_byte)
    shifts = torch.arange(per_byte, dtype=torch.int64) * bits
    packed = torch.sum(groups << shifts, dim=1)
    return packed.to(torch.uint8).contiguous()


def unpack_unsigned(packed: torch.Tensor, bits: int, count: int) -> torch.Tensor:
    """Unpack a byte stream produced by :func:`pack_unsigned`.

    The returned tensor is flat and contains exactly ``count`` values.
    """

    _validate_bits(bits)
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return torch.empty(0, dtype=torch.uint8)

    data = packed.detach().to(device="cpu", dtype=torch.uint8).reshape(-1)
    if bits == 8:
        if data.numel() < count:
            raise ValueError("packed buffer is shorter than requested count")
        return data[:count].clone()

    per_byte = 8 // bits
    required = (count + per_byte - 1) // per_byte
    if data.numel() < required:
        raise ValueError("packed buffer is shorter than requested count")

    shifts = torch.arange(per_byte, dtype=torch.int64) * bits
    expanded = ((data[:required].to(torch.int64)[:, None] >> shifts[None, :]) & ((1 << bits) - 1))
    return expanded.reshape(-1)[:count].to(torch.uint8).contiguous()


def pack_bits(bits_tensor: torch.Tensor) -> torch.Tensor:
    """Pack a boolean/0-1 tensor into bytes."""

    return pack_unsigned(bits_tensor.to(torch.uint8), 1)


def unpack_bits(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Unpack bytes into a boolean tensor."""

    return unpack_unsigned(packed, 1, count).to(torch.bool)


def words_from_bits(bits_tensor: torch.Tensor, word_bits: int = 32) -> torch.Tensor:
    """Pack 0/1 values into little-endian integer words.

    This representation is convenient for CUDA because a residual mask for 32
    groups can be loaded as one ``uint32`` and queried with a population count.
    """

    if word_bits not in (8, 16, 32, 64):
        raise ValueError("word_bits must be 8, 16, 32 or 64")
    flat = bits_tensor.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    if flat.numel() == 0:
        dtype = {8: torch.uint8, 16: torch.int32, 32: torch.int32, 64: torch.int64}[word_bits]
        return torch.empty(0, dtype=dtype)
    if torch.any((flat != 0) & (flat != 1)):
        raise ValueError("bits_tensor must contain only 0 or 1")

    pad = (-flat.numel()) % word_bits
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)], dim=0)
    groups = flat.view(-1, word_bits)
    shifts = torch.arange(word_bits, dtype=torch.int64)
    words = torch.sum(groups << shifts, dim=1)
    # Signed int32 preserves the same 32-bit payload and halves the serialized
    # footprint relative to int64.  ``bits_from_words`` widens before shifting,
    # so negative values (bit 31 set) are decoded correctly.
    if word_bits <= 8:
        return words.to(torch.uint8)
    if word_bits <= 32:
        return words.to(torch.int32)
    return words.to(torch.int64)


def bits_from_words(words: torch.Tensor, count: int, word_bits: int = 32) -> torch.Tensor:
    """Inverse of :func:`words_from_bits`."""

    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return torch.empty(0, dtype=torch.bool)
    data = words.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    required = (count + word_bits - 1) // word_bits
    if data.numel() < required:
        raise ValueError("word buffer is shorter than requested count")
    shifts = torch.arange(word_bits, dtype=torch.int64)
    bits = ((data[:required, None] >> shifts[None, :]) & 1).reshape(-1)[:count]
    return bits.to(torch.bool).contiguous()


def prefix_counts(mask: torch.Tensor, block_groups: int = 32) -> torch.Tensor:
    """Return selected-group counts at the start of each mask block.

    ``prefix[j]`` equals the number of ``True`` entries in
    ``mask[: j * block_groups]``.  The array therefore permits a CUDA kernel to
    compute the rank of a selected group using one prefix load and a popcount of
    the bits earlier in the same word.
    """

    if block_groups <= 0:
        raise ValueError("block_groups must be positive")
    flat = mask.detach().to(device="cpu", dtype=torch.bool).reshape(-1)
    blocks = (flat.numel() + block_groups - 1) // block_groups
    out = torch.empty(blocks, dtype=torch.int32)
    running = 0
    for block in range(blocks):
        out[block] = running
        start = block * block_groups
        end = min(start + block_groups, flat.numel())
        running += int(flat[start:end].sum().item())
    return out
