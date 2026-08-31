"""Optional CUDA-extension dispatch."""
from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def extension() -> Any | None:
    try:
        return importlib.import_module("nhdf_edge_cuda")
    except (ImportError, OSError):
        return None


def available() -> bool:
    return extension() is not None


def require() -> Any:
    module = extension()
    if module is None:
        raise RuntimeError(
            "nhdf_edge_cuda is not installed. Build it with "
            "`python setup_cuda.py build_ext --inplace` from the project root."
        )
    return module
