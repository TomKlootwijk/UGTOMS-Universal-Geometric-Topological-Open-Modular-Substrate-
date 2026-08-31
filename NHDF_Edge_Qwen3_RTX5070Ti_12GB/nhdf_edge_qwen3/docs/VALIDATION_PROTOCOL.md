# Validation protocol

## Metrics contract

Every reproducible run should save:

```text
model revision, pack format, config hash, CUDA/PyTorch/Transformers versions,
device name, driver, power mode, prompt/context, batch, residual fraction,
actual pack bytes, free/peak VRAM, TTFT, prefill tok/s, decode tok/s, parity
rate, CRC failures, saturation/overflow counts, quality metrics, acceptance
rule, generated text, validation status and seed
```

## Correctness thresholds

- Bit-pack round trip: exact.
- Legacy local zero-set residual: `max(abs(F_i)) <= 1e-5` for CPU conversion;
  this is a reconstruction invariant, not a model-quality metric.
- Stored parity: exact for uncorrupted payloads.
- CRC32: every file must match before model load.
- CUDA kernel versus dequantized FP16: the target-laptop microbenchmark gate is
  frozen at maximum absolute error `<= 0.5`, normalized RMSE `<= 1e-3`, and
  cosine similarity `>= 0.99999`. These provisional FP16-accumulation limits
  were set after the first random boundary/4096-wide observations; changing
  them requires an explicit protocol revision, never a per-tensor relaxation.
- CPU versus GPU route selection: exact indices for deterministic inputs.
- Routed-MoE backend equivalence: every selected `MUL_MAT_ID` CPU/CUDA case
  must pass before the backend is used for a functional control.
- Exact-response prompts: compare normalized generated text, not merely process
  exit or token count.

## Pack validation states

- A new pack is `UNCALIBRATED`.
- Failed language quality is `QUALITY_FAILED`.
- A pack that cannot satisfy the declared memory/runtime envelope is
  `RESOURCE_FAILED`.
- Only a complete pack with separately recorded quality and resource evidence
  may be `VALIDATED`.

CRC, parity and manifest hashes never promote a status. Deployment loading is
fail-closed; an unvalidated pack requires an explicit research override.

`VALIDATED` is always scoped to the tests and resource envelope named by its
sealed certificate. It must not be read as a claim that unmeasured task suites,
context lengths, hardware or codecs have also passed.

## External-codec NHDF hybrid integration gate

An NHDF hybrid may use a separately attributed weight codec. Its evidence must
keep these responsibilities explicit:

- the external codec owns the low-bit tensor encoding and reconstruction;
- NHDF owns immutable artifact/runtime binding, typed validation state,
  bounded resource policy, evidence lineage and fail-closed execution;
- the manifest must reject altered payloads, runtimes, profiles or evidence;
- passing the hybrid gate never promotes an experimental NHDF-native codec.

The current `nhdf-edge-hybrid-0.1` artifact passed its bounded functional and
resource gate:

- external GGUF/IQ2_M codec, explicitly attributed to ggml/Bartowski;
- 4/4 deterministic functional prompts passed;
- 49/49 model layers offloaded to the GPU;
- allocated 8K q8 K/V capacity peaked at 10,610 MiB of 12,227 MiB, leaving
  1,617 MiB headroom;
- repeated 64-token benchmark averages were 870.026857 prompt tokens/s and
  157.141442 generation tokens/s;
- the NHDF manifest, payload, llama.cpp build 10720 runtime at commit
  `f8dbcd61893702976f9ab03be89c2b9f436d532c` and evidence are sealed, and
  execution fails closed on mismatch.

Canonical evidence:
`packs/qwen3-30b-a3b-nhdf-v03-iq2m/evidence/functional_gate.json`.

This is an NHDF substrate success using an external codec. The 8K test
allocated the full cache capacity but used a short input; it is not a filled-8K
quality test. Broad accuracy, perplexity, long-context retrieval and confidence
intervals across a representative task suite remain unmeasured.

## Bounded resident coding and repair probe

The current deployment also has a narrower executable coding probe. Its
resident runtime profile is fixed at four threads, split mode `none`, priority
2, one slot, full GPU offload, Flash Attention, 8K q8 K/V and deterministic
sampling (`temperature=0`, `top_k=1`, `seed=2026`). The server binds only to
`127.0.0.1` after full artifact verification and GPU/free-VRAM preflight.

Six small Python functions were run twice each against exact machine-scored
tests. Ten of twelve first-pass launches passed; this means 5/6 tasks passed
both first repetitions. Both failures were the same nested-input mutation in
`merge_intervals`. One separately stored machine-feedback repair per failed
launch corrected both, resulting in 12/12 final launches and 6/6 tasks passing
both final results. First-pass and final-after-repair accuracy must remain
separate fields.

Direct resident measurements were 147.139410 median generation tok/s and a
90.82995 ms median warm cached TTFT proxy. The first streamed content/byte time
is observable, but is not an exact token callback. It is also not cold startup
latency: startup to listening was separately measured at 12.644645 seconds.
Repairs deliberately disabled prefix-cache reuse; their median TTFT was 220.89
ms and median wall time was 1.178813 seconds.

Canonical optimization evidence:

- `metrics/local/runtime_optimization_20260831.json`;
- `metrics/local/coding_benchmark/run-20260831T144501Z/evidence.json`.

This probe does not satisfy the broad model-quality suite below. It does not
establish repository-scale coding, BF16 accuracy preservation, perplexity,
filled-context quality, or confidence intervals over a representative task
distribution.

## NHDF-native bounded codec-development gate

Before a full repack, test at least one common and one less-common real routed
expert on disjoint calibration/holdout prompts. The current provisional gate is:

- worst expert-output NRMSE `<= 0.35`;
- worst cosine similarity `>= 0.94`;
- median NRMSE `<= 0.30` and median cosine `>= 0.95`;
- at least 20% NRMSE improvement over a simpler equal-storage RTN baseline for
  every sampled expert.

Failure stops the full conversion. Thresholds may be revised only as a visible
protocol change, not after inspecting one failed expert.

The subsequent complete router-weighted layer-0 test is also complete and
rejected. Across 656 disjoint holdout tokens and all 117 experts selected by
their real top-8 routes, the mixed GPTQ candidate produced NRMSE `0.155101`
versus `0.150849` for equal-storage optimized RTN, or 2.819% worse. The
candidate passed the absolute-error component but failed the declared
comparative-advantage component. This is a comparative native-codec failure,
not garbage-output collapse and not a failure of the separately validated
external-codec hybrid.

## Fault injection

1. Flip one bit in a tensor file: CRC and often parity must fail.
2. Flip two payload bits in one parity group: demonstrate the one-bit parity
   blind spot while CRC still fails.
3. Delete a tensor file: loader must fail closed.
4. Change a generation tag or manifest geometry: loader must reject it.
5. Force insufficient free VRAM: doctor/loader must stop before partial load.
6. Stress residual-mask word boundaries (groups 31/32/63/64).
7. Stress row padding and expert row offsets.

## Model quality suite

Minimum suggested suite:

- WikiText-style perplexity or another declared held-out language set;
- instruction-following and reasoning tasks relevant to the intended use;
- code tasks for a coding-assistant deployment;
- long-context retrieval at 4K and 8K;
- route divergence: fraction of tokens whose top-8 experts differ from BF16;
- logit cosine similarity and top-k agreement.

Use the same tokenizer, prompts, decoding settings and context limits for every
baseline. Report confidence intervals where task size permits.

The hybrid functional certificate does not satisfy this broad suite. In
particular, allocated 8K cache residency must not be reported as filled-8K
retrieval or language-quality evidence.

## Performance suite

- Prompt lengths: 32, 512, 2048, 8192 tokens.
- Generation lengths: 32 and 256 tokens.
- Batch: 1 (primary) and 2 if memory permits.
- Power profiles: minimum, balanced and maximum TGP exposed by the laptop.
- At least five measured runs after warm-up.
- Report median, p10/p90, not just maximum throughput.

The current coding probe uses two repetitions per task and therefore remains a
bounded deployment probe, not completion of this broader performance suite.
Its three-repetition controlled 512-prompt/256-generation A/B run likewise
does not meet the five-run recommendation. Those sequential runs measured
149.441046 generation tok/s for build 10720 versus 106.236199 tok/s for the
previous b6014 profile, with Windows left on the Balanced power plan.

## Falsification rules

The functional/resource claim for a deployed profile fails if any of these
hold:

- actual pack plus required workspace cannot load with a safe VRAM margin;
- steady-state generation is unstable or repeatedly triggers driver recovery;
- quality loss exceeds the predeclared deployment threshold;
- measured decode falls below the minimum useful threshold defined by the
  intended application.

For an NHDF-native codec claim, one additional rule applies: the custom codec
must outperform a simpler equal-size baseline under its declared comparison.
Failure of that native-codec rule does not invalidate a hybrid whose external
codec, NHDF substrate role and bounded certificate are all stated explicitly.
