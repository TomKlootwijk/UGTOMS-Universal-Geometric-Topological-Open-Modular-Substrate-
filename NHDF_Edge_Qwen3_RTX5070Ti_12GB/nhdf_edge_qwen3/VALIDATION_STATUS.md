# Validation status

**Execution date:** 31 August 2026

**Target:** NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12,227 MiB

**Current model artifact:** `packs/qwen3-30b-a3b-iq2m-32k-q4kv`

| Layer | Disposition |
|---|---|
| UGTOMS kernel and registered profiles | Verified |
| 32K/q4 external-codec model artifact | `VALIDATED` |
| Bounded generic local coding-agent gate | `PASSED` |
| Live substrate-specific coding-agent gate | Pending measurement |
| Legacy native scalar codec | `QUALITY_FAILED` |

## Current disposition

The current substrate is not NHDF v0.3 alone. Its authority is
`substrate/kernel/contract.json`, kernel ID `ugtoms-kernel-v0.1`, with the
explicit selectable `nhdf-v0.1` and `sclp-foundational` profiles. The
`nhdf-v0.3-ccd` collision profile is a selectable later specialization. It
does not replace the normalized NHDF v0.1 base, the early UGTS algebra and
referential layer, or the SCLP 3.6.2 corrective role.

`nhdf-edge substrate-verify --repository .` currently validates the kernel,
its source records, and these three profile records with automatic promotion
disabled.

The deployed model result uses a separate transport layer. The complete
30,532,122,624-parameter Qwen3-30B-A3B-Instruct-2507 state is referenced as a
9,870,270,464-byte GGUF/IQ2_M payload and run by pinned llama.cpp build 10720,
commit `f8dbcd61893702976f9ab03be89c2b9f436d532c`. The source BF16 tensors are
61,064,245,248 bytes, or 58,235.4 MiB. The referenced file is 6.1867x smaller,
an 83.84% file-size reduction.

GGUF/IQ2_M is attributed to Bartowski/ggml. It is not an NHDF-native codec and
is not proof that the substrate itself compressed arbitrary model weights.
UGTOMS supplies the selected contract, provenance, typed capability and
validation state, bounded resource policy, evidence chain, and fail-closed
launch decision. The artifact is zero-copy and workspace-relative.

## Fresh sealed 32K model gate

Primary record:
`packs/qwen3-30b-a3b-iq2m-32k-q4kv/evidence/functional_gate.json`

Sealed evidence SHA-256:
`c56f79acf80f73773e80325bb3c865dc5b949c55abcdeefbf01f4fda1677baeb`.

| Gate | Requirement | Result | Status |
|---|---:|---:|---|
| Functional prompt suite | 4/4 | 4/4 | Pass |
| Full reported layer offload | 49/49 | 49/49 | Pass |
| Allocated context | 32,768 | 32,768 | Pass |
| Allocated-context exact response | `OK` | `OK` | Pass |
| Peak device use | at most 12,227 MiB | 11,064 MiB | Pass |
| Device reserve | at least 512 MiB | 1,163 MiB | Pass |
| 64-token short decode | at least 80 tok/s | 135.208586 tok/s | Pass |

The allocated 32K run used an 864 MiB q4_0/q4_0 K/V buffer, a 9,279.83 MiB
CUDA model buffer, a 300.75 MiB CUDA compute buffer, and a 127.51 MiB
CPU-mapped model buffer. The fresh three-sample short microbenchmarks were:

| Workload | Mean | Standard deviation |
|---|---:|---:|
| 64-token prompt processing | 885.013308 tok/s | 178.171701 tok/s |
| 64-token generation | 135.208586 tok/s | 2.102534 tok/s |

All 30.532B parameters must be stored, but this is a mixture-of-experts model
with approximately 3.3B active parameters per token. The decode number is not
dense-30B throughput.

The maximum-context test allocated the cache and executed a short prompt. It
did not fill 32K with meaningful content and score long-context quality. A 48K
profile remains fragile and unvalidated.

### Separate prior filled-context result

A prior, separately scoped 22,440-prompt-token measurement recorded
2,297.477 prompt tok/s and 25.189 decode tok/s. That decode rate corresponds
to about 39.70 ms per generated token, 3.97 seconds per 100 tokens, or 11.91
seconds per 300 tokens before application overhead. This is the relevant
practical long-context measurement, but it is not part of the fresh sealed 32K
microbenchmark and must not be averaged with it.

## Final generic coding-agent gate

Primary directory:
`metrics/local/coding_agent_32k/run-20260831T172748.453659Z`

Evidence SHA-256:
`a110060c30816c9d8e92d9ddc0eb0ade6c07be871371dd49942cb8de47262348`.

The generated `evidence.json` records `all_gates_passed: true` and status
`PASSED`.

| Check | Result |
|---|---:|
| Actual model-reported served context | `n_ctx = 32768` |
| Native OpenAI-compatible tool-call JSON | Passed |
| Synthetic context retrieval | Exact needle at 21,997 prompt tokens |
| Disposable Python repair run | 36.524758 s |
| Recorded tool use | Exactly one `edit` and exactly one `bash` |
| Independent final fixture test run | 4/4 passed |
| Only expected fixture source changed | Yes |
| Git HEAD unchanged | Yes |

The gate used an isolated disposable repository, a reduced environment, a
digest-verified installed `substrate/AGENT_CONTRACT.md`, disabled project
configuration, and no MCP or plugins. The event audit ran after execution. It
is evidence that the recorded tools stayed in scope; it is not a preventive
process, filesystem, or network sandbox.

The generic result proves one typed native tool-call exchange, one synthetic
approximately 22K-token retrieval, and one small repository repair. It does
not prove broad coding skill, production safety, or UGTOMS understanding. A
substrate-specific live gate remains pending until a measured run directly
tests profile selection, symbol separation, typed manifests, and bounded
next-generation behavior.

The corresponding end-user launcher is `scripts/start_local_coder.ps1`, which
runs `scripts/setup_local_coder.ps1` and `scripts/local_coder.py` against
`configs/opencode_nhdf_local.json`. Once the one-time client/model setup is
complete, inference is local and consumes no API key or paid token budget.

```powershell
.\scripts\start_local_coder.ps1 -Arguments @(
  "--quick",
  "C:\path\to\your\git-worktree"
)
```

The default permission contract denies web access, external directories,
plugins, MCP, subagents, commits, pushes, and destructive commands, and asks
before ordinary edits and shell commands. These are application controls, not
an OS sandbox.

## Substrate and application-evidence boundary

The kernel's same-generation definition graph is typed, content-addressed, and
acyclic. Any feedback is an explicit generation `n` to `n+1` record. No
unrestricted fixed-point engine or same-generation authority cycle is
implemented. Spatial knowledge-graph work remains in
`substrate/incubator/spatial-evidence-ledger.json`; it is a future boundary,
not a current validated capability.

The Grove application evidence has two distinct records:

- `kc3d392-grove-scene` is a 21,798-byte binary scene with 66 explicit nodes.
  It does not demonstrate a recipe-generated 1,024-object population.
- `kcpr392-ring-1024-display-instances` is the separate recipe result: 4,984
  bytes versus 162,274 bytes, 32.56x smaller and 96.93% reduced, with exact
  prior-prefix stability. The copies are render-only display instances, not
  independent ECS entities; they cannot independently acquire collider,
  graph, saved-object, or gameplay identity.

These records demonstrate bounded deterministic scene packing and display
generation only. They do not establish arbitrary semantic compression.

## Historical evidence, not the current profile

### Superseded 8K artifact

The former current artifact,
`packs/qwen3-30b-a3b-nhdf-v03-iq2m`, remains preserved as historical evidence.
It used an 8K q8 K/V profile.

| Historical measurement | Result |
|---|---:|
| Functional prompts | 4/4 passed |
| Reported layer offload | 49/49 |
| Allocated q8 K/V buffer | 408.00 MiB |
| Peak GPU memory | 10,610 MiB |
| Device headroom | 1,617 MiB |
| 64-token prompt microbenchmark | 870.026857 tok/s |
| 64-token short decode microbenchmark | 157.141442 tok/s |
| Controlled 512-prompt/256-generation decode | 149.441046 tok/s |
| Prior-runtime comparison decode | 106.236199 tok/s |
| Resident small-task coding median | 147.139410 tok/s |
| Resident startup to listening | 12.644645 s |
| Warm cached coding TTFT proxy, median | 90.82995 ms |
| Uncached repair TTFT, median | 220.89 ms |
| Repair wall time, median | 1.178813 s |
| Small coding launches, first pass | 10/12 passed |
| Small coding launches, after at most one repair | 12/12 passed |

That run established allocated-8K residency and short execution, not filled-8K
quality. Its speed and headroom are retained for comparison but are no longer
the current deployment claim. In its sequential runtime comparison, build
10720 improved controlled decode from 106.236199 to 149.441046 tok/s (40.67%)
and prompt processing from 1,541.754583 to 2,484.170832 tok/s, with high first-
sample prompt variance. The six-task coding probe failed both first attempts at
`merge_intervals` because the generated function mutated a nested input; one
recorded feedback repair corrected both repetitions. The warm TTFT number was
for a resident prefix-cached request, not cold startup or unrelated prompts.

### Failed native-codec research

The complete legacy native scalar pack contains 9,152,386,624 tensor-file
bytes. It loaded all 531 entries, passed manifest/CRC/parity integrity, fit in
VRAM, and executed CUDA generation. Its responses collapsed to repeated
newlines or `10000000`, so the pack remains `QUALITY_FAILED` and is refused by
default.

| Expert | Historical output NRMSE | Historical cosine |
|---:|---:|---:|
| 0 | 0.4813 | 0.8805 |
| 17 | 0.4749 | 0.8935 |
| 127 | 0.4482 | 0.9170 |

A later bounded GEMQ-style 3-bit GPTQ probe reached NRMSE/cosine of
0.2542/0.9671 for expert 0 and 0.1859/0.9830 for expert 17. The improvements
over its equal-byte RTN comparison were only 2.91% and 3.91%, below the
predeclared 20% comparative threshold. It was not expanded to a full
checkpoint. This preserves the negative result and the external-codec
boundary of the working system.

## Reproduction commands

```powershell
# Kernel, source records, and profiles
nhdf-edge substrate-verify --repository .

# Current model manifest, payload, runtime, and evidence
nhdf-edge verify .\packs\qwen3-30b-a3b-iq2m-32k-q4kv

# An evidence-bound substrate application manifest
nhdf-edge substrate-validate-app `
  .\path\to\application-manifest.json `
  --repository . `
  --evidence-root .\path\to\evidence

# Versioned report source to checked PDF plus visual-QA pages
python .\scripts\render_substrate_pdf.py `
  .\docs\UGTOMS_Local_Substrate_Coding_Agent_Guide_v0.1.md `
  --output .\output\pdf\UGTOMS_Local_Substrate_Coding_Agent_Guide_v0.1.pdf `
  --expect "UGTOMS Local Substrate Coding Agent Guide" `
  --render-pages .\output\pdf\UGTOMS_Local_Substrate_Coding_Agent_Guide_v0.1-pages
```

## Remaining evidence boundary

Established on the named hardware and runtime:

- verified kernel/profile records and explicit no-auto-promotion policy;
- current 32K/q4 cache allocation, 49/49 offload, monitored peak, reserve, and
  short functional/throughput gate;
- actual served `n_ctx = 32768`;
- one exact 21,997-token retrieval, native tool call, and isolated repair;
- exact evidence hashes and an unchanged fixture Git HEAD;
- fail-closed model artifact and resource checks.

Not established:

- NHDF-native tensor compression or arbitrary semantic compression;
- filled-32K quality, validated 48K behavior, or broad q4 K/V parity;
- broad coding accuracy, BF16 parity, or parity with a hosted frontier agent;
- a measured live substrate-specific agent pass;
- preventive OS sandboxing or production security;
- sustained thermals, power, battery, or many-hour stability;
- portability beyond the named laptop, pinned runtime, and workspace-relative
  payload layout.
