# Validation status

**Execution date:** 31 August 2026

**Target:** NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12,227 MiB

**Current deployable artifact:** `packs/qwen3-30b-a3b-nhdf-v03-iq2m`

**Status:** `VALIDATED`

## Disposition

The complete 30,532,122,624-parameter Qwen3-30B-A3B-Instruct-2507 model now
runs functionally through the NHDF v0.3 hybrid substrate on the target GPU.
The source checkpoint contains 61,064,245,248 BF16 tensor bytes; the referenced
payload is 9,870,270,464 bytes. This is 6.1867x smaller: 51,193,974,784 bytes,
or 83.84%, are removed relative to the BF16 tensor data. BF16 weights alone
would require 58,235 MiB, or 4.76x the GPU's reported capacity, before runtime
buffers and KV cache.

The low-bit tensor codec is explicitly attributed to Bartowski/ggml
GGUF/IQ2_M, with pinned llama.cpp build 10720 at commit
`f8dbcd61893702976f9ab03be89c2b9f436d532c` as the runtime. It is **not**
presented as an NHDF-native codec. NHDF supplies SHA-256-sealed local
provenance, typed validation and capability state, resource policy, a verified
hash-linked event and evidence chain, bounded launch profile, and fail-closed
execution decision.

The hybrid is zero-copy: its manifest references the existing verified GGUF
payload by a workspace-relative path rather than storing another 9.87 GB file.

## Fresh hybrid gate

| Gate | Requirement | Result | Status |
|---|---:|---:|---|
| Functional prompt suite | 4/4 | 4/4 | Pass |
| Full layer offload | 49/49 | 49/49 | Pass |
| Allocated-8K exact response | `OK` | `OK` | Pass |
| Peak device use | at most 12,227 MiB | 10,610 MiB | Pass |
| Device headroom | at least 512 MiB | 1,617 MiB | Pass |
| 64-token generation | at least 80 tok/s | 157.141442 tok/s | Pass |

The four functional outputs were:

- exact `OK`;
- exact arithmetic result `323`;
- a coherent explanation of why integrity does not imply model quality;
- a correct Python `is_even(n)` function.

The three-sample, 64-token llama-bench results were:

| Workload | Mean | Standard deviation |
|---|---:|---:|
| Prompt processing | 870.026857 tok/s | 137.844526 tok/s |
| Generation | 157.141442 tok/s | 2.259367 tok/s |

These are 64-token llama-bench microbenchmarks. Qwen3-30B-A3B is an MoE model:
the complete 30.532B parameter state must be stored, while about 3.3B
parameters are active per token. The generation rate is not dense-30B
throughput and should not be generalized to every interactive workload.

At the allocated 8K context, the run used a 408.00 MiB q8 K/V buffer and
peaked at 10,610 MiB total device usage on a 12,227 MiB GPU. llama.cpp reported
a 9,279.83 MiB CUDA model buffer, a 300.75 MiB CUDA compute buffer, 49/49 layer
offload, and a 127.51 MiB CPU-mapped model buffer.

This is an allocated-8K residency and short-execution measurement. It does not
measure quality after filling the context with 8K tokens.

## Optimized resident runtime and coding probe

The sealed deployment runtime is llama.cpp build 10720 / `f8dbcd61`, compiled
for CUDA architecture `120a`. The promoted profile uses four CPU threads,
split mode `none`, priority 2, full GPU offload, Flash Attention, an 8K q8 K/V
cache, one server slot, and deterministic sampling (`temperature=0`, `top_k=1`,
`seed=2026`). Startup to listening was measured at 12.644645 seconds. A resident
snapshot reported 10,299 MiB GPU memory used and 1,646 MiB free; the fresh gate
above remains the authoritative measured peak/headroom result.

The controlled 512-prompt/256-generation, three-repetition comparison measured
149.441046 generation tok/s for build 10720 versus 106.236199 tok/s for the
previous b6014 profile, a 40.67% improvement. Prompt processing improved from
1,541.754583 to 2,484.170832 tok/s in that sequential A/B comparison, but its
first-sample variance was high.

The resident executable coding probe measured:

| Measurement | Result |
|---|---:|
| Small Python tasks, two repetitions | 6 tasks / 12 launches |
| First-pass launches | 10/12 passed |
| Tasks passing both first passes | 5/6 |
| Final launches after at most one repair | 12/12 passed |
| Tasks passing both final results | 6/6 |
| Median generation rate | 147.139410 tok/s |
| Median warm cached TTFT proxy | 90.82995 ms |
| Median repair wall time | 1.178813 s |

The same `merge_intervals` implementation mutated a nested input in both first
passes. One deterministic repair using machine feedback corrected both runs
and yielded identical passing source hashes. First-pass and repaired accuracy
are reported separately: the repair result does not turn the unassisted result
into 6/6. The suite contains only six small functions, so it does not establish
broad coding accuracy, repository-scale editing, or parity with the BF16 model.

To run the resident profile in one PowerShell terminal:

```powershell
nhdf-edge serve packs\qwen3-30b-a3b-nhdf-v03-iq2m `
  --port 18080 `
  --threads 4 `
  --startup-timeout 120 `
  --request-timeout 120
```

Then run the measured coding configuration from a second terminal:

```powershell
python scripts\benchmark_hybrid_coding.py `
  packs\qwen3-30b-a3b-nhdf-v03-iq2m `
  --output-root metrics\local\coding_benchmark `
  --context 2048 `
  --max-new-tokens 384 `
  --seed 2026 `
  --repetitions 2 `
  --repair-attempts 1 `
  --server-url http://127.0.0.1:18080 `
  --server-cache-prompt `
  --quick `
  --cache-precondition-note "build 10720/f8dbcd61; t4, split none, priority 2; 8K q8 KV; cached first pass and uncached repair"
```

The TTFT value is the first streamed content/byte timestamp for an already
resident, prefix-cached request, not an exact token callback or cold-start
measurement. Repairs intentionally disabled cache reuse and had a 220.89 ms
median TTFT. The A/B runs were sequential on the Windows Balanced power plan,
not a randomized maximum-power laboratory experiment.

## Integrity, evidence, and launch policy

The hybrid manifest seals:

- exact Qwen source revision and 30,532,122,624-parameter identity;
- payload path, byte count, SHA-256, container, and codec attribution;
- llama.cpp completion, benchmark and server executables plus dependent runtime hashes;
- the NHDF v0.3 specification hash;
- source-provenance and CUDA backend-equivalence records;
- the functional, residency, resource, and throughput gate evidence;
- a hash-linked creation and validation-status event chain.

The artifact begins as `UNCALIBRATED`. A complete `gate-hybrid` pass promotes
it to `VALIDATED`; an incomplete gate records a failed disposition. `run`
verifies the artifact, requires `VALIDATED`, and checks
the declared GPU/free-memory contract before launching. Integrity alone never
establishes model quality.

```powershell
nhdf-edge verify packs\qwen3-30b-a3b-nhdf-v03-iq2m

nhdf-edge run packs\qwen3-30b-a3b-nhdf-v03-iq2m `
  --prompt "Reply with exactly the single word OK." `
  --context 512 `
  --max-new-tokens 8 `
  --text-only
```

To recreate and freshly gate the zero-copy substrate artifact:

```powershell
nhdf-edge create-hybrid packs\qwen3-30b-a3b-nhdf-v03-iq2m `
  --model models\Qwen3-30B-A3B-Instruct-2507-IQ2_M\Qwen_Qwen3-30B-A3B-Instruct-2507-IQ2_M.gguf `
  --runtime tools\llama.cpp-f8dbcd61\bin\llama-completion.exe `
  --benchmark-runtime tools\llama.cpp-f8dbcd61\bin\llama-bench.exe `
  --server-runtime tools\llama.cpp-f8dbcd61\bin\llama-server.exe `
  --runtime-revision f8dbcd61893702976f9ab03be89c2b9f436d532c `
  --runtime-build-number 10720 `
  --runtime-argument-profile current-2026 `
  --specification sources\NHDF_Formal_Specification_v0.3_General_Purpose_CCD_Tom_Klootwijk.pdf `
  --source-record models\Qwen3-30B-A3B-Instruct-2507-IQ2_M\CONTROL_SOURCE.json `
  --assurance-evidence metrics\local\gguf_backend_ops.json `
  --assurance-evidence tools\llama.cpp-f8dbcd61\SOURCE.json `
  --assurance-evidence tools\llama.cpp-f8dbcd61\LICENSE `
  --assurance-evidence metrics\local\runtime_optimization_20260831.json `
  --assurance-evidence metrics\local\coding_benchmark\run-20260831T144501Z\evidence.json

nhdf-edge gate-hybrid packs\qwen3-30b-a3b-nhdf-v03-iq2m `
  --output packs\qwen3-30b-a3b-nhdf-v03-iq2m\evidence\functional_gate.json
```

## Native-codec research status

The validated hybrid does not erase the negative native-codec result.

The complete legacy native pack contains 9,152,386,624 tensor-file bytes. It
loaded all 531 entries, passed manifest/CRC/parity integrity, fit in VRAM, and
executed CUDA generation. Its responses collapsed to repeated newlines or
`10000000`, so it is correctly marked `QUALITY_FAILED` and refused by default.

Packed-versus-BF16 layer-0 expert-output errors were:

| Expert | NRMSE | Cosine |
|---:|---:|---:|
| 0 | 0.4813 | 0.8805 |
| 17 | 0.4749 | 0.8935 |
| 127 | 0.4482 | 0.9170 |

The later GEMQ-style 3-bit GPTQ probe passed absolute isolated-expert
distortion limits but improved NRMSE over equal-byte RTN by only 2.91% and
3.91%, below the declared 20% requirement. It was therefore not expanded into
a full native checkpoint.

These results mean:

- the NHDF hybrid substrate is presently functional and validated;
- GGUF/IQ2_M supplies its working tensor encoding;
- the original NHDF-native scalar encoding remains falsified;
- the replacement native codec remains research, not a deployment artifact.

## Remaining evidence boundary

Established on the named hardware/runtime:

- complete-model functional output through the NHDF launch path;
- exact payload/runtime/specification provenance and integrity;
- 49/49 layer offload;
- measured allocated-8K fit and headroom;
- fresh prompt/generation throughput;
- warm resident request latency and throughput;
- a bounded six-task executable coding and single-repair probe;
- fail-closed validation and resource policy.

Not established:

- perplexity, broad task/coding accuracy, or quality preservation relative to BF16;
- useful output after an actually filled 8K-token prompt;
- sustained thermal, power, or long-duration behavior;
- cold-start TTFT or uncached-prefix latency generalization;
- an NHDF-native tensor-codec advantage;
- a self-contained or relocatable hybrid artifact;
- portability of the measured performance to other hardware or runtimes.
