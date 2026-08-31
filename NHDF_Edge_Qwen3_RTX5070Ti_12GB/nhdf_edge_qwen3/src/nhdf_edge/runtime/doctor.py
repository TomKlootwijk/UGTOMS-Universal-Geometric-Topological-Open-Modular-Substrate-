"""Target-system diagnostics for the 12 GB laptop profile."""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

import torch

from ..config import NHDFConfig
from ..metrics import estimate
from .cuda_backend import available as cuda_extension_available


@dataclass
class DoctorReport:
    python: str
    platform: str
    torch: str
    cuda_available: bool
    cuda_runtime: str | None
    device_name: str | None
    capability: tuple[int, int] | None
    total_vram_gb: float | None
    free_vram_gb: float | None
    driver: str | None
    cuda_extension: bool
    projected_pack_vram_gb: float
    projected_headroom_gb: float
    verdict: str
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def run_doctor(cfg: NHDFConfig | None = None) -> DoctorReport:
    cfg = cfg or NHDFConfig()
    projection = estimate(cfg)
    warnings_out: list[str] = []
    cuda = torch.cuda.is_available()
    name = None
    capability = None
    total = None
    free = None
    if cuda:
        name = torch.cuda.get_device_name(0)
        capability = tuple(torch.cuda.get_device_capability(0))
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        total = total_bytes / 1e9
        free = free_bytes / 1e9
        if total < projection.projected_total_vram_gb:
            warnings_out.append("Physical VRAM is below the analytical default profile.")
        if free < projection.projected_total_vram_gb:
            warnings_out.append("Current free VRAM is below the projected runtime requirement.")
        if capability[0] < 12:
            warnings_out.append("The target profile expects an RTX 50-series-class CUDA device; use generic kernels.")
    else:
        warnings_out.append("CUDA is unavailable; only CPU reference tests can run.")

    if not cuda_extension_available():
        warnings_out.append("Optional nhdf_edge_cuda extension is not installed.")

    if cuda and free is not None and free >= projection.projected_total_vram_gb and cuda_extension_available():
        verdict = "ready-for-benchmark"
    elif cuda:
        verdict = "conditional"
    else:
        verdict = "reference-only"

    return DoctorReport(
        python=platform.python_version(),
        platform=platform.platform(),
        torch=torch.__version__,
        cuda_available=cuda,
        cuda_runtime=torch.version.cuda,
        device_name=name,
        capability=capability,
        total_vram_gb=total,
        free_vram_gb=free,
        driver=_driver_version(),
        cuda_extension=cuda_extension_available(),
        projected_pack_vram_gb=projection.projected_total_vram_gb,
        projected_headroom_gb=projection.nominal_headroom_gb,
        verdict=verdict,
        warnings=warnings_out,
    )


def main() -> None:
    print(json.dumps(run_doctor().to_dict(), indent=2))


if __name__ == "__main__":
    main()
