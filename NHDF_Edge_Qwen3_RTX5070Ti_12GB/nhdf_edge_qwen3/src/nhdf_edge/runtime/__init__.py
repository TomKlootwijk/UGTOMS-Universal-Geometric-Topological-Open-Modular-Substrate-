"""Runtime modules for NHDF Edge packs.

The pure-PyTorch path is a correctness reference.  The optional CUDA extension
provides a fused low-bit GEMV and selected-row decode intended for batch-one
edge decoding.
"""

from .modules import NHDFPackedEmbedding, NHDFPackedLinear, NHDFQwen3Experts, PackedMatrix

__all__ = [
    "PackedMatrix",
    "NHDFPackedLinear",
    "NHDFPackedEmbedding",
    "NHDFQwen3Experts",
]
