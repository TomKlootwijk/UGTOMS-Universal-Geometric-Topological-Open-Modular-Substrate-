from __future__ import annotations

from nhdf_edge.config import NHDFConfig
from nhdf_edge.metrics import context_sweep, estimate, residual_fraction_sweep


def test_default_projection_is_in_expected_range() -> None:
    result = estimate(NHDFConfig())
    assert 9.0 < result.packed_weight_gb < 9.5
    assert 11.0 < result.projected_total_vram_gb < 11.6
    assert result.fits_nominal_12gb
    assert 25.0 < result.decode_tps_by_efficiency["5%"] < 32.0


def test_sweeps_are_monotonic() -> None:
    cfg = NHDFConfig()
    residual = residual_fraction_sweep(cfg, [0.0, 0.1, 0.2, 0.3])
    sizes = [row["packed_weight_gb"] for row in residual]
    rooflines = [row["roofline_tps"] for row in residual]
    assert sizes == sorted(sizes)
    assert rooflines == sorted(rooflines, reverse=True)

    contexts = context_sweep(cfg, [4096, 8192, 16384])
    assert [row["kv_cache_gb"] for row in contexts] == sorted(row["kv_cache_gb"] for row in contexts)
