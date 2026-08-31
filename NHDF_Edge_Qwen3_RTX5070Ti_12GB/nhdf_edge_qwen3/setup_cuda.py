"""Build the optional CUDA extension.

Recent PyTorch/CUDA toolchains can target the RTX 50-series laptop GPU by
setting TORCH_CUDA_ARCH_LIST to the architecture accepted by that installed
toolchain.  The build intentionally does not hardcode an architecture so it can
also compile on a development GPU.
"""
from __future__ import annotations

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent

setup(
    name="nhdf-edge-cuda",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="nhdf_edge_cuda",
            sources=[str(ROOT / "csrc" / "bindings.cpp"), str(ROOT / "csrc" / "nhdf_gemv.cu")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
