# Implementation plan and engineering gates

## Gate 0 - semantic baseline (included)

- Deterministic 2-bit/4-bit bit packing.
- Per-group local zero-set mean projection.
- Log-polar residual address and second-phase-difference branch score.
- Bounded one-bit residual branch allocation.
- One-bit payload parity plus shard CRC32.
- Selected-row reconstruction for embeddings and experts.
- Qwen3 expert adapter with the upstream forward contract.
- Analytical memory/traffic model and deterministic smoke metrics.

Exit evidence: `pytest -q` passes, `nhdf-edge smoke` is deterministic, and
`nhdf-edge verify` rejects an intentionally corrupted tensor file.

## Gate 1 - target CUDA equivalence (included scaffold; target execution needed)

1. Build `nhdf_edge_cuda` with the laptop's installed driver, toolkit and
   CUDA-enabled PyTorch.
2. Run `scripts/benchmark_kernel.py` for 2-bit/4-bit, residual fractions 0 and
   0.15, dimensions that are and are not multiples of 256, and row-offset expert
   slices.
3. Require bounded error against the dequantized FP16 reference and no illegal
   memory accesses under Compute Sanitizer.
4. Profile memory transactions, occupancy, register pressure and branch
   divergence with Nsight Compute.

Exit evidence: CPU/CUDA equivalence vectors and a saved kernel benchmark JSON.

## Gate 2 - complete checkpoint conversion

1. Pull the source checkpoint and preserve its upstream license/model card.
2. Run a partial pack (`--max-tensors`) and inspect tensor policies.
3. Convert the complete checkpoint with at least 70 GB free source space plus
   10 GB destination space.
4. Verify every CRC and parity word.
5. Record actual serialized bytes and compare them with the 9.23 GB projection.

Exit evidence: complete manifest, no missing tensors, actual pack size and a
policy audit.

## Gate 3 - full-model correctness

- Load through the meta-device replacement loader.
- Confirm no meta parameter remains.
- Compare layer outputs and logits for short prompts against a dequantized
  reference or an independently quantized baseline.
- Test expert route indices and weights, tied/untied embedding behavior, KV
  cache updates, chat template and generation stopping.

Exit evidence: fixed prompt/logit vectors and deterministic greedy outputs.

## Gate 4 - quality and NHDF ablations

Compare at equal memory or node/residual budget:

- plain 2-bit groupwise;
- NHDF 2-bit plus residual branch;
- 3-bit or 4-bit baseline;
- official GPTQ-Int4;
- log-polar versus linear error ranking;
- local B0 mean projection versus ordinary zero point;
- phase-curvature term on/off;
- parity event on/off;
- residual fraction sweep.

Report perplexity, task accuracy and routing changes. A component is not
justified if a simpler equal-budget ablation matches it.

## Gate 5 - laptop performance and power

Measure cold load, time-to-first-token, prefill tok/s, steady-state decode tok/s,
peak allocated/reserved VRAM, effective bandwidth, GPU power, clock, temperature
and throttling at multiple laptop power profiles. Repeat with the internal
display and external display configurations because display reservation affects
free VRAM.

Exit evidence: raw telemetry plus medians and percentile ranges, not a single
best run.

## Gate 6 - prefill optimization

The supplied GEMV addresses decode. Implement and compare:

1. packed GEMM with fused decode;
2. layer-local dequantization into a bounded workspace followed by cuBLAS;
3. token-chunked prefill;
4. optional expert-token compaction into grouped GEMM.

Keep the 0.75 GB workspace bound and reject any hidden allocation that invalidates
the 12 GB profile.
