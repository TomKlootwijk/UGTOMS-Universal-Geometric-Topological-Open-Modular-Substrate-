# NHDF Edge: Qwen3-30B-A3B on a 12 GB RTX 5070 Ti Laptop GPU

This repository is a reproducible engineering translation of Tom Klootwijk's
**Non-Euclidean Holographic Data Fields (NHDF)** formal specification into a
bounded, testable weight format and edge-inference architecture.

The selected model is `Qwen/Qwen3-30B-A3B-Instruct-2507`: a 30.5B-parameter
Mixture-of-Experts model with 3.3B active parameters per token, 128 experts and
8 routed experts per token. Its upstream BF16 repository is about 61.1 GB, and
the official GPTQ-Int4 repository is about 16.9 GB. Both exceed a 12 GB laptop
GPU. The default NHDF Edge projection is **9.23 GB of packed weights** and
**11.28 GB total modeled VRAM** at an 8K context with an int8 KV cache.

Those numbers are analytical, not benchmark results. The conversion and CPU
semantic path are implemented and tested. The included CUDA GEMV is a clear
experimental decode kernel that must be compiled, verified and optimized on the
target laptop before performance or model-quality claims are made.

## Feasibility verdict

**Conditionally feasible as an experimental, all-weights-resident decode
runtime.** The model can fit on paper with roughly 0.72 GB nominal margin, but a
real laptop must expose enough free VRAM after display/driver allocations. The
critical unknowns are low-bit quality on this exact checkpoint, fused-kernel
memory efficiency, prefill speed, thermals and integration with the installed
Transformers/CUDA versions.

Default analytical profile:

| Item | Projection |
|---|---:|
| Expert effective precision | 2.321 bits/weight |
| Attention, embeddings, LM head | 4.149 bits/weight |
| Router and norms | FP16 |
| Packed weights | 9.228 GB (8.595 GiB) |
| 8K int8 KV cache | 0.403 GB |
| Workspace | 0.750 GB |
| Driver/runtime/display reserve | 0.900 GB |
| Total | 11.281 GB |
| Active packed weight traffic/token | 1.182 GB |
| Decode model at 3% of peak bandwidth | 17.0 tok/s |
| Decode model at 5% of peak bandwidth | 28.4 tok/s |
| Decode model at 7% of peak bandwidth | 39.8 tok/s |

The token rates are a bandwidth sensitivity model using the published 672 GB/s
memory bandwidth, not measured throughput.

## NHDF operator mapping

The package preserves the formal order rather than treating the source terms as
interchangeable metaphors:

```text
ELP -> B0 -> P -> RBST -> K_Tforward -> Scone -> Pi -> U
```

| Formal operator | Edge-AI implementation |
|---|---|
| `ELP` | Log-polar address of each post-quantization residual group |
| `B0` | Per-group weighted zero-mean reconstruction residual; a non-degenerate local zero set |
| `P` | One-bit payload parity as a fast event/integrity gate; CRC32 is the stronger file check |
| `RBST` | Deterministic, bounded allocation of one-bit residual branches to the highest-error groups |
| `K_Tforward` | Monotonic tensor/group generation and autoregressive token schedule |
| `Scone` | On-demand reconstruction of only the weight rows/tiles needed by the current projection |
| `Pi` | Fused GEMV/GEMM or embedding row projection |
| `U` | Manifest, telemetry, residual metrics, optional recalibration and next-generation state |

The implementation follows the specification's crucial correction: it does
**not** use one global function `F == 0`, which would have no gradient or distance
information. Every quantization group has its own local constraint. It also does
not present one-bit parity as error correction; parity misses even-count faults.

## Packed format

The default expert format has:

- 2-bit mid-rise base codes;
- one FP16 mean and one FP16 scale per 256 weights;
- a one-bit sign residual for the top 15% of groups;
- one FP16 residual scale and one-byte log-polar address per selected group;
- a 32-group residual mask, rank prefix and parity word;
- per-tensor safetensors files protected by CRC32 in `manifest.json`.

Attention matrices, embeddings and the LM head use groupwise 4-bit weights.
Routers and normalization scales remain FP16. This component-specific choice is
consistent with published low-bit MoE work, but the NHDF residual allocation and
local zero-set constraint are specific to this reference design.

## Install and test

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
nhdf-edge smoke --output metrics/local_smoke.json
nhdf-edge estimate --config configs/qwen3_30b_a3b_edge12.yaml
```

The shipped test suite validates bit packing, local zero-set behavior,
reconstruction, residual branching, parity, CRC32, selected-row decode, packed
linear/embedding behavior, expert routing and analytical metrics.

## Pull and convert the model

The ZIP contains no model weights.

```bash
python scripts/pull_model.py \
  --output models/Qwen3-30B-A3B-Instruct-2507

nhdf-edge pack \
  models/Qwen3-30B-A3B-Instruct-2507 \
  packs/qwen3-30b-a3b-nhdf \
  --config configs/qwen3_30b_a3b_edge12.yaml

nhdf-edge verify packs/qwen3-30b-a3b-nhdf --parity-all
```

Conversion is shard-streamed and quantizes one tensor at a time, so the entire
61.1 GB checkpoint is never loaded into RAM at once. A complete conversion will
still require substantial disk space and time. Keep the source checkpoint until
the converted pack passes CRC, parity, reconstruction and model-level tests.

### Optional teacher-guided calibration

The data-free converter treats source weights as the reconstruction teacher.
For activation-weighted post-training reconstruction distillation, collect
input second moments on representative prompts and pass them to the converter:

```bash
python scripts/calibrate_teacher.py \
  models/Qwen3-30B-A3B-Instruct-2507 \
  calibration/prompts.jsonl \
  calibration/hessian.safetensors

nhdf-edge pack models/Qwen3-30B-A3B-Instruct-2507 packs/qwen3-nhdf \
  --config configs/qwen3_30b_a3b_edge12.yaml \
  --hessian calibration/hessian.safetensors
```

The BF16 teacher itself does not fit in 12 GB. Calibration therefore needs CPU
offload, enough system RAM, or a larger temporary machine. It is optional.

## Build the experimental CUDA decode path

Install a Blackwell-capable NVIDIA driver, CUDA toolkit and CUDA-enabled PyTorch
on the laptop. Then build without hardcoding unsupported PTX:

```bash
# Use the architecture spelling accepted by your installed toolchain.
export TORCH_CUDA_ARCH_LIST="12.0"
python setup_cuda.py build_ext --inplace
nhdf-edge doctor --config configs/qwen3_30b_a3b_edge12.yaml
```

On Windows PowerShell:

```powershell
$env:TORCH_CUDA_ARCH_LIST = "12.0"
python setup_cuda.py build_ext --inplace
nhdf-edge doctor --config configs/qwen3_30b_a3b_edge12.yaml
```

The included kernel fuses low-bit decode with batch-one GEMV and supports
selected expert row intervals. It does not yet provide a vendor-tuned packed
GEMM for prompt prefill. The proposed prefill fallback is chunked, layer-local
dequantization into the 0.75 GB workspace followed by cuBLAS GEMM; that path is
a design target rather than a completed optimized implementation in v0.1.

## Experimental full-model loader

After conversion and CUDA build:

```bash
python -m pip install -e ".[runtime]"
python scripts/run_qwen3.py packs/qwen3-30b-a3b-nhdf \
  --prompt "Summarize the NHDF local zero-set invariant."
```

`runtime/qwen3_loader.py` builds Qwen3 on the meta device, replaces ordinary
linear and embedding modules with packed modules, replaces each official
`Qwen3MoeExperts` block with an operator-equivalent packed expert adapter, loads
raw router/norm parameters, and rejects the model if any meta parameter remains.
Transformers internals can change, so this loader is deliberately isolated and
must be regression-tested against the installed version.

## Required validation before claiming success

1. **Pack equivalence:** per-tensor reconstruction error, local zero-set
   residual, parity and CRC32.
2. **CPU/GPU equivalence:** the fused kernel versus the CPU reference across
   2-bit, 4-bit, residual/no-residual, row offsets and boundary sizes.
3. **Model quality:** perplexity plus task benchmarks against BF16, official
   GPTQ-Int4, plain 2-bit and 4-bit ablations.
4. **Runtime:** time-to-first-token, prefill tok/s, decode tok/s, peak VRAM,
   effective memory bandwidth, power and thermal-throttling behavior.
5. **NHDF ablations:** remove log-polar scoring, B0 mean projection, parity,
   phase curvature, bounded residual routing and feedback one at a time.
6. **Failure tests:** even-count bit faults, corrupted tensor files, VRAM
   pressure, branch saturation, context growth and driver reset recovery.

If a simpler method performs equally well at equal memory, the corresponding
NHDF operator is not yet justified for this application.

## Edge applications

A successful runtime would make a comparatively large sparse model available
for private, low-connectivity workloads on one laptop: local coding and
analysis, offline document/RAG assistants, field-service copilots, research
notebooks, and local agent orchestration. The architecture is also reusable for
smaller models where the goal is lower power or longer context rather than
making an otherwise impossible model fit.

## Repository map

```text
configs/                  target and packing profiles
csrc/                     experimental fused CUDA GEMV/row decode
metrics/                  analytical projections and deterministic smoke data
scripts/                  pull, calibrate, convert, run and benchmark entry points
sources/                  NHDF formal specification used as the design basis
src/nhdf_edge/            pack format, operators, converter, runtime and CLI
tests/                    semantic and storage tests
docs/                     report, implementation plan and validation protocol
```

## Evidence boundary

This package is a research prototype. It does not establish peer review, patent
status, model-quality preservation, hardware speed, quantum/optical advantage,
SAR safety, geomagnetic-storm immunity, or zero-cost computation. The PDF report
separates source-derived NHDF requirements, established hardware/model facts,
analytical projections, implemented behavior and unverified hypotheses.

## References

- Tom Klootwijk, *Non-Euclidean Holographic Data Fields: A Formal Operator
  Specification for a Parity-Conditioned Kinematic Foliation and Implicit
  Holographic Co-Processor*, v0.1, 31 August 2026. Included in `sources/`.
- Qwen Team, `Qwen/Qwen3-30B-A3B-Instruct-2507`, Hugging Face model card and
  checkpoint tree: https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507
- NVIDIA, GeForce RTX 50 Series Laptop GPU specifications:
  https://www.nvidia.com/en-us/geforce/laptops/50-series/
- Egiazarian et al., *Extreme Compression of Large Language Models via Additive
  Quantization*, arXiv:2401.06118.
- Tseng et al., *QuIP#: Even Better LLM Quantization with Hadamard Incoherence
  and Lattice Codebooks*, arXiv:2402.04396.
- Kim, Fahim and Awadalla, *Mixture of Quantized Experts*, arXiv:2310.02410.
