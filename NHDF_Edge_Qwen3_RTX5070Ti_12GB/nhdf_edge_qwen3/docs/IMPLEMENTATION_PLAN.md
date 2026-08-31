# Implementation plan and engineering gates

This plan follows the v0.3 Edge-AI evidence contract. Version 0.3 is not a
weight codec, so CCD operators are not substituted for tensor operations.

## Gate 0 - source interpretation (complete)

- Pin v0.3 and its SHA-256.
- Separate applicable Edge-AI requirements from CCD-only operators.
- Treat scale-aware calibration, bounded resources, typed status, monotone
  refinement and equal-budget ablation as the applicable contract.
- Remove static phase scoring from the default profile.

Exit evidence: `sources/README.md`, `sources/SOURCE_SHA256.txt` and the v0.3
source PDF.

## Gate 1 - legacy baseline disposition (complete, rejected)

- Preserve the 9.152 GB custom scalar pack as a reproducible baseline.
- Verify all 531 CRC/parity records and CUDA decode equivalence.
- Record full-model generation and real-expert error.
- Mark the manifest `QUALITY_FAILED`; refuse normal runtime loading.

Exit evidence: `metrics/local/default_pack_quality_gate.json` and
`VALIDATION_STATUS.md`.

## Gate 2 - comparable-budget feasibility control (complete, passed)

- Pin an independently quantized exact-model artifact below 10 GB.
- Verify exact file length and SHA-256.
- Build a pinned CUDA runtime for SM120.
- Require CPU/CUDA routed-MoE operation equivalence.
- Run deterministic exact, arithmetic, explanation and code prompts using the
  current non-thinking chat template.
- Allocate the target 8K q8 K/V context and sample total device memory.

Exit evidence:

- `metrics/local/gguf_backend_ops.json`: 8,132/8,132 pass;
- `metrics/local/gguf_iq2m_functional_gate.json`: 4/4 prompt pass;
- `metrics/local/gguf_iq2m_8k_residency_gate.json`: exact response and 10,487
  MiB peak total device usage;
- `metrics/local/gguf_iq2m_llama_bench.json`: repeated backend throughput.

This gate proves model/hardware/budget feasibility. It does not validate a
custom UGTOMS/NHDF codec.

## NHDF hybrid integration track

### Hybrid Gate H1 - fail-closed external-codec integration (complete, passed)

- Reference the exact 9,870,270,464-byte GGUF/IQ2_M payload without copying or
  relabelling its tensor encoding.
- Attribute the mixed-bit weight codec to ggml/Bartowski; NHDF makes no claim
  of authorship over IQ2_M.
- Use NHDF v0.3 for SHA-256-sealed local provenance, typed capabilities and status,
  bounded context/VRAM policy, sealed validation evidence and fail-closed
  execution.
- Seal the payload, llama.cpp b6014 runtime, execution profile and evidence by
  path, byte count and SHA-256.
- Require the functional prompt suite, complete GPU offload, allocated-context
  residency and minimum generation throughput before promotion.

Measured result: 4/4 functional prompts passed, 49/49 layers offloaded, and an
allocated 8K q8 K/V profile peaked at 10,487 MiB of 12,227 MiB, leaving 1,740
MiB measured headroom. The repeated 64-token benchmark measured 458.525658
prompt tokens/s and 102.367894 generation tokens/s. The artifact is
`VALIDATED` under this bounded functional/resource certificate and normal
execution is fail-closed on manifest, payload, runtime or evidence mismatch.

Exit evidence:
`packs/qwen3-30b-a3b-nhdf-v03-iq2m/evidence/functional_gate.json`.

The 8K result allocates 8K cache capacity while executing a short prompt; it is
not a filled-context quality measurement. Broad task accuracy, perplexity and
long-context retrieval quality remain unmeasured. This gate validates an NHDF
substrate integration around an external codec, not an NHDF-native weight
codec.

## NHDF-native codec research track

## Gate 3 - bounded replacement-codec experiment (complete, rejected)

- Capture exact layer-0 attention/router activations from disjoint prompt sets.
- Quantize one expert at a time with activation/Hessian-aware GPTQ.
- Account for physical packed bytes including scale/zero metadata.
- Compare against equal-storage optimized RTN.
- Sample experts 0 and 17 with sufficient real router hits in both splits.

Current result: absolute error thresholds pass for experts 0 and 17, but the
relative improvements over equal-storage RTN are only 2.91% and 3.91%, versus
the required 20% for each expert. Disposition: reject the current configuration
and refine it within this bounded gate.

## Gate 4 - router-weighted layer output (complete, rejected)

- Quantized all 117 experts selected by the real layer-0 top-8 routes under a
  fixed physical byte allocation.
- Compared the complete router-weighted MoE output on 656 disjoint holdout
  tokens while retaining the BF16 router, so route identities and weights were
  exact by construction.
- Compared routed-calibrated 3-bit GPTQ with a simpler equal-storage optimized
  RTN allocation; sparsely observed experts used an equal-storage RTN fallback
  without holdout leakage.

Current result: the candidate achieved routed-output NRMSE `0.155101` versus
`0.150849` for equal-storage RTN, making it 2.819% worse. Its absolute layer
quality gate passed, but its comparative-advantage gate failed. This is not
numerical collapse; it is evidence that this native candidate does not justify
itself against the simpler codec. Disposition: reject this configuration and
keep the native full-pack gate blocked.

Exit evidence: `metrics/local/gemq_router_weighted_layer0_gate.json`.

## Gate 5 - streaming full custom pack (blocked by Gates 3-4)

- Stream source shards; never materialize the 61 GB teacher.
- Use 8-bit embedding/output, at least 4-bit attention/router-sensitive paths,
  and calibrated mixed low-bit routed experts.
- Journal every tensor and keep the initial status `UNCALIBRATED`.
- Stop on disk, RAM, VRAM or projected-size budget failure.

## Gate 6 - full-model quality and promotion (blocked by Gate 5)

- Run teacher-forced logit KL/top-k agreement where BF16 comparison resources
  permit.
- Run the deterministic functional suite and a declared sampled task suite.
- Measure 8K residency, cold load, prompt/decode speed and stability.
- Compare against the same IQ2_M control at equal context/settings.
- Set `VALIDATED`, `QUALITY_FAILED` or `RESOURCE_FAILED` with a saved evidence
  object. Integrity checks alone cannot choose the status.

## Gate 7 - native-codec optimization (only after native validation)

Optimize packed prefill, sustained thermals and power only after functional
quality passes. The passed hybrid integration does not unlock optimization or
promotion of the rejected native codec. Speeding up an invalid model is not
progress toward deployment.
