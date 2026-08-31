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
GGUF/IQ2_M, with llama.cpp b6014 as the runtime. It is **not** presented as an
NHDF-native codec. NHDF supplies SHA-256-sealed local provenance, typed
validation and capability state, resource policy, a verified hash-linked event
and evidence chain, bounded launch
profile, and fail-closed execution decision.

The hybrid is zero-copy: its manifest references the existing verified GGUF
payload by a workspace-relative path rather than storing another 9.87 GB file.

## Fresh hybrid gate

| Gate | Requirement | Result | Status |
|---|---:|---:|---|
| Functional prompt suite | 4/4 | 4/4 | Pass |
| Full layer offload | 49/49 | 49/49 | Pass |
| Allocated-8K exact response | `OK` | `OK` | Pass |
| Peak device use | at most 12,227 MiB | 10,487 MiB | Pass |
| Device headroom | at least 512 MiB | 1,740 MiB | Pass |
| 64-token generation | at least 80 tok/s | 102.367894 tok/s | Pass |

The four functional outputs were:

- exact `OK`;
- exact arithmetic result `323`;
- a coherent explanation of why integrity does not imply model quality;
- a correct Python `is_even(n)` function.

The three-sample, 64-token llama-bench results were:

| Workload | Mean | Standard deviation |
|---|---:|---:|
| Prompt processing | 458.525658 tok/s | 8.845115 tok/s |
| Generation | 102.367894 tok/s | 3.590717 tok/s |

These are 64-token llama-bench microbenchmarks. Qwen3-30B-A3B is an MoE model:
the complete 30.532B parameter state must be stored, while about 3.3B
parameters are active per token. The generation rate is not dense-30B
throughput and should not be generalized to every interactive workload.

At the allocated 8K context, the run used a 408.00 MiB q8 K/V buffer and
peaked at 10,487 MiB total device usage on a 12,227 MiB GPU. llama.cpp reported
a 9,279.83 MiB CUDA model buffer, a 300.75 MiB CUDA compute buffer, 49/49 layer
offload, and a 127.51 MiB CPU-mapped model buffer.

This is an allocated-8K residency and short-execution measurement. It does not
measure quality after filling the context with 8K tokens.

## Integrity, evidence, and launch policy

The hybrid manifest seals:

- exact Qwen source revision and 30,532,122,624-parameter identity;
- payload path, byte count, SHA-256, container, and codec attribution;
- llama.cpp executable and dependent runtime hashes;
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
  --runtime tools\llama.cpp-b6014\bin\llama-cli.exe `
  --benchmark-runtime tools\llama.cpp-b6014\bin\llama-bench.exe `
  --specification sources\NHDF_Formal_Specification_v0.3_General_Purpose_CCD_Tom_Klootwijk.pdf `
  --source-record models\Qwen3-30B-A3B-Instruct-2507-IQ2_M\CONTROL_SOURCE.json `
  --assurance-evidence metrics\local\gguf_backend_ops.json

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
- fail-closed validation and resource policy.

Not established:

- perplexity or broad task-accuracy preservation relative to BF16;
- useful output after an actually filled 8K-token prompt;
- sustained thermal, power, or long-duration behavior;
- an NHDF-native tensor-codec advantage;
- a self-contained or relocatable hybrid artifact;
- portability of the measured performance to other hardware or runtimes.
