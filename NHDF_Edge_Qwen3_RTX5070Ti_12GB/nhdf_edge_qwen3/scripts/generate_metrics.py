#!/usr/bin/env python3
"""Generate deterministic analytical tables, smoke metrics and report figures."""
from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from nhdf_edge.config import NHDFConfig, load_config
from nhdf_edge.metrics import context_sweep, estimate, residual_fraction_sweep
from nhdf_edge.quantize import QuantizationPolicy, dequantize_tensor, quantize_tensor, verify_parity

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
FIGURES = ROOT / "figures"
METRICS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def chart_setup() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def main() -> None:
    chart_setup()
    cfg = load_config(ROOT / "configs" / "qwen3_30b_a3b_edge12.yaml")
    projection = estimate(cfg)
    fractions = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    residual_rows = residual_fraction_sweep(cfg, fractions)
    context_rows = context_sweep(cfg, [2048, 4096, 8192, 12288, 16384, 32768])
    throughput_rows = [
        {
            "effective_bandwidth_percent": percent,
            "effective_bandwidth_gbps": cfg.target.memory_bandwidth_gbps * percent / 100.0,
            "decode_tokens_per_second": projection.peak_bandwidth_roofline_tps * percent / 100.0,
        }
        for percent in range(1, 13)
    ]

    (METRICS / "analytical_projection.json").write_text(
        json.dumps(
            {
                "status": "analytical-not-measured",
                "config": cfg.to_dict(),
                "estimate": projection.to_dict(),
                "assumptions": [
                    "all packed weights resident in VRAM",
                    "batch-one autoregressive decode",
                    "int8 KV cache",
                    "published peak memory bandwidth is a roofline only",
                    "workspace and runtime/display reserves are fixed planning allowances",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(METRICS / "residual_fraction_sweep.csv", residual_rows)
    write_csv(METRICS / "context_vram_sweep.csv", context_rows)
    write_csv(METRICS / "decode_bandwidth_sensitivity.csv", throughput_rows)

    # Deterministic synthetic reconstruction experiment. This is an algorithmic
    # smoke test, not evidence of language-model accuracy.
    generator = torch.Generator().manual_seed(20260831)
    rows, cols = 512, 1024
    low_rank = torch.randn(rows, 32, generator=generator) @ torch.randn(32, cols, generator=generator)
    weight = 0.20 * low_rank + 0.65 * torch.randn(rows, cols, generator=generator)
    # Add sparse outliers to make branch allocation behavior visible.
    outlier_mask = torch.rand(rows, cols, generator=generator) < 0.002
    weight[outlier_mask] *= 7.0

    synthetic_rows: list[dict] = []
    for fraction in fractions:
        policy = QuantizationPolicy(
            base_bits=2,
            group_size=256,
            residual_fraction=fraction,
            phase_gain=0.25,
            iterations=3,
        )
        packed = quantize_tensor(weight, policy, name=f"synthetic.f{fraction:.2f}")
        restored = dequantize_tensor(packed)
        error = weight - restored
        synthetic_rows.append(
            {
                "base_bits": 2,
                "residual_fraction": fraction,
                "effective_bpp": packed.stats["effective_bits_per_weight"],
                "mse": torch.mean(error.square()).item(),
                "rmse": torch.sqrt(torch.mean(error.square())).item(),
                "max_abs": torch.max(torch.abs(error)).item(),
                "zero_set_max_abs": packed.stats["weighted_zero_set_max_abs"],
                "parity_ok": verify_parity(packed)["ok"],
            }
        )

    four = quantize_tensor(
        weight,
        QuantizationPolicy(base_bits=4, group_size=256, residual_fraction=0.0, iterations=3),
        name="synthetic.fourbit",
    )
    four_error = weight - dequantize_tensor(four)
    synthetic_rows.append(
        {
            "base_bits": 4,
            "residual_fraction": 0.0,
            "effective_bpp": four.stats["effective_bits_per_weight"],
            "mse": torch.mean(four_error.square()).item(),
            "rmse": torch.sqrt(torch.mean(four_error.square())).item(),
            "max_abs": torch.max(torch.abs(four_error)).item(),
            "zero_set_max_abs": four.stats["weighted_zero_set_max_abs"],
            "parity_ok": verify_parity(four)["ok"],
        }
    )
    write_csv(METRICS / "synthetic_reconstruction.csv", synthetic_rows)
    baseline_mse = synthetic_rows[0]["mse"]
    default_row = next(row for row in synthetic_rows if row["base_bits"] == 2 and row["residual_fraction"] == 0.15)
    smoke = {
        "status": "deterministic-synthetic-not-model-quality",
        "shape": [rows, cols],
        "default_residual_fraction": 0.15,
        "base_2bit_mse": baseline_mse,
        "default_mse": default_row["mse"],
        "default_mse_reduction_percent": 100.0 * (1.0 - default_row["mse"] / baseline_mse),
        "default_effective_bpp": default_row["effective_bpp"],
        "default_zero_set_max_abs": default_row["zero_set_max_abs"],
        "default_parity_ok": default_row["parity_ok"],
        "fourbit_mse": synthetic_rows[-1]["mse"],
    }
    (METRICS / "smoke_test.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")

    # 1. Model size comparison.
    labels = ["BF16 upstream", "FP8 upstream", "Official GPTQ-Int4", "NHDF Edge projection"]
    sizes = [61.1, 31.2, 16.9, projection.packed_weight_gb]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bars = ax.barh(labels[::-1], sizes[::-1])
    ax.axvline(12.0, linestyle="--", linewidth=1.5, label="12 GB VRAM ceiling")
    ax.set_xlabel("Checkpoint / packed-weight size (decimal GB)")
    ax.set_title("Qwen3-30B-A3B weight footprint")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    for bar, value in zip(bars, sizes[::-1]):
        ax.text(value + 0.6, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_size_comparison.png")
    plt.close(fig)

    # 2. VRAM budget.
    components = [
        ("Packed weights", projection.packed_weight_gb),
        ("8K int8 KV cache", projection.kv_cache_gb),
        ("Workspace", cfg.target.workspace_gb),
        ("Runtime/display reserve", cfg.target.runtime_reserve_gb),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    left = 0.0
    for label, value in components:
        ax.barh(["Default profile"], [value], left=left, label=f"{label}: {value:.2f} GB")
        left += value
    ax.axvline(12.0, linestyle="--", linewidth=1.5, label="Nominal 12 GB")
    ax.set_xlim(0, 12.5)
    ax.set_xlabel("VRAM (decimal GB)")
    ax.set_title(f"Modeled VRAM budget: {left:.2f} GB total, {12-left:.2f} GB nominal headroom")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "vram_budget.png")
    plt.close(fig)

    # 3. Residual budget versus size and synthetic error.
    synth2 = [row for row in synthetic_rows if row["base_bits"] == 2]
    x = np.array([100.0 * row["residual_fraction"] for row in residual_rows])
    size = np.array([row["packed_weight_gb"] for row in residual_rows])
    mse_reduction = np.array([100.0 * (1.0 - row["mse"] / baseline_mse) for row in synth2])
    fig, ax1 = plt.subplots(figsize=(8.2, 4.6))
    ax1.plot(x, size, marker="o", label="Projected packed weights")
    ax1.set_xlabel("Expert groups receiving a one-bit residual branch (%)")
    ax1.set_ylabel("Packed weights (GB)")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, mse_reduction, marker="s", linestyle="--", label="Synthetic MSE reduction")
    ax2.set_ylabel("Synthetic MSE reduction vs plain 2-bit (%)")
    ax1.axvline(15, linestyle=":", linewidth=1.2, label="Default 15%")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="center right")
    ax1.set_title("Residual-branch trade-off (synthetic quality proxy)")
    fig.tight_layout()
    fig.savefig(FIGURES / "residual_tradeoff.png")
    plt.close(fig)

    # 4. Decode sensitivity.
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(
        [row["effective_bandwidth_percent"] for row in throughput_rows],
        [row["decode_tokens_per_second"] for row in throughput_rows],
        marker="o",
    )
    ax.axvspan(3, 7, alpha=0.12, label="Engineering sensitivity band")
    ax.set_xlabel("Effective fraction of published memory bandwidth (%)")
    ax.set_ylabel("Modeled decode tokens/s")
    ax.set_title("Batch-one decode sensitivity, not a benchmark")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "decode_sensitivity.png")
    plt.close(fig)

    # 5. Context and VRAM.
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    context_k = [row["context_tokens"] / 1024 for row in context_rows]
    total = [row["total_vram_gb"] for row in context_rows]
    ax.plot(context_k, total, marker="o")
    ax.axhline(12.0, linestyle="--", label="12 GB nominal ceiling")
    ax.set_xlabel("Context length (K tokens)")
    ax.set_ylabel("Modeled total VRAM (GB)")
    ax.set_title("Context-length pressure with an int8 KV cache")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "context_vram.png")
    plt.close(fig)

    # 6. Operator pipeline diagram.
    fig, ax = plt.subplots(figsize=(11.2, 2.8))
    ax.axis("off")
    names = [
        ("ELP", "log-polar\nresidual"),
        ("B0", "local\nzero-set"),
        ("P", "1-bit\ngate"),
        ("RBST", "bounded residual\nbranch"),
        ("K_T", "causal token /\ngeneration"),
        ("Scone", "on-demand\nweight decode"),
        ("Pi", "GEMV/GEMM\nprojection"),
        ("U", "telemetry +\nfeedback"),
    ]
    xs = np.linspace(0.05, 0.95, len(names))
    for idx, (xpos, (symbol, label)) in enumerate(zip(xs, names)):
        ax.text(
            xpos,
            0.58,
            f"{symbol}\n{label}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="black", linewidth=1.0),
        )
        if idx < len(names) - 1:
            ax.annotate(
                "",
                xy=(xs[idx + 1] - 0.055, 0.58),
                xytext=(xpos + 0.055, 0.58),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", linewidth=1.2),
            )
    ax.annotate(
        "self-referential update",
        xy=(xs[0], 0.28),
        xytext=(xs[-1], 0.28),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.18", linewidth=1.2),
        ha="center",
        va="center",
    )
    ax.set_title("NHDF operator chain mapped to the edge weight runtime", pad=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "operator_pipeline.png")
    plt.close(fig)

    print(json.dumps({"estimate": projection.to_dict(), "smoke": smoke}, indent=2))


if __name__ == "__main__":
    main()
