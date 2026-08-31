"""NHDF Edge reference implementation.

The tensor-codec API is loaded lazily so lightweight hybrid-runtime commands
do not import PyTorch merely by importing :mod:`nhdf_edge`.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .quantize import (
        PackedTensor,
        QuantizationPolicy,
        dequantize_rows,
        dequantize_tensor,
        quantize_tensor,
    )

__all__ = [
    "PackedTensor",
    "QuantizationPolicy",
    "quantize_tensor",
    "dequantize_tensor",
    "dequantize_rows",
]

__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import quantize

    value = getattr(quantize, name)
    globals()[name] = value
    return value
