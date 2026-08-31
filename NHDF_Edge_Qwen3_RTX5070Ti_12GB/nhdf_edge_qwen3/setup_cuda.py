"""Build the optional CUDA extension.

Recent PyTorch/CUDA toolchains can target the RTX 50-series laptop GPU by
setting TORCH_CUDA_ARCH_LIST to the architecture accepted by that installed
toolchain.  The build intentionally does not hardcode an architecture so it can
also compile on a development GPU.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent


class PortableBuildExtension(BuildExtension):
    """Keep generated object paths below Windows' legacy path limit."""

    def finalize_options(self) -> None:
        super().finalize_options()
        # Ninja mirrors an absolute source path below ``build_temp``.  A deeply
        # nested checkout can therefore exceed MAX_PATH even when the source
        # path itself is valid.  Honour an explicit absolute --build-temp, but
        # move setuptools' relative default under the short system temp path.
        if os.name == "nt" and self.build_temp and not Path(self.build_temp).is_absolute():
            self.build_temp = str(Path(tempfile.gettempdir()) / "nhdf_edge_cuda_build")


WINDOWS = os.name == "nt"
CXX_FLAGS = ["/O2", "/std:c++17"] if WINDOWS else ["-O3", "-std=c++17"]

setup(
    name="nhdf-edge-cuda",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="nhdf_edge_cuda",
            sources=[str(ROOT / "csrc" / "bindings.cpp"), str(ROOT / "csrc" / "nhdf_gemv.cu")],
            extra_compile_args={
                "cxx": CXX_FLAGS,
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": PortableBuildExtension},
)
