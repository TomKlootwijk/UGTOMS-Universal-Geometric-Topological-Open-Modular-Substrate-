# Validation status

**Execution date:** 31 August 2026

**Target:** NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12,227 MiB

**Current model artifact:** `packs/qwen3-30b-a3b-iq2m-32k-q4kv`

| Layer | Disposition |
|---|---|
| UGTOMS kernel and registered profiles | Verified |
| Direct deterministic substrate replay, without a model or agent | `PASSED` |
| 32K/q4 external-codec model artifact | `VALIDATED` |
| Bounded generic local coding-agent gate | `PASSED` |
| Focused one-defect SCLP repair live gate | `PASSED`, fully disclosed repair |
| Historical broad four-file substrate-authoring live gate | `FAILED` by timeout |
| Legacy native scalar codec | `QUALITY_FAILED` |

## Current disposition

The current substrate is not NHDF v0.3 alone. Its authority is
`substrate/kernel/contract.json`, semantic kernel ID `ugtoms-kernel-v0.1`, and
schema discriminator `ugtoms-kernel-contract-0.2`. The registry exposes
`nhdf-v0.1`, `sclp-foundational`, and `nhdf-v0.3-ccd` as active selectable
profiles. The direct reference application selects exactly the first two. The
later CCD specialization is registered but unselected there; it does not
replace the normalized NHDF v0.1 base, early UGTS algebra and referential
layer, or SCLP 3.6.2 corrective role.

The registry, profile, and application schemas are respectively
`ugtoms-profile-registry-0.2`, `ugtoms-profile-0.2`, and
`ugtoms-application-manifest-0.2`; reference evidence uses
`ugtoms-sclp-reference-evidence-0.2`. These schema versions are distinct from
the stable semantic kernel ID and the reference application's v0.1 semantic
version and filename.

`nhdf-edge substrate-verify --repository .` currently validates the kernel,
its source records, and these three profile records with automatic promotion
disabled.

Exact sealed substrate records are:

| Record | Bytes | SHA-256 |
|---|---:|---|
| Kernel contract | 10,187 | `9b5fa7cff4483129e80e1c234055d57e0f79a574dcd2e131c418bbe0259448c3` |
| Profile registry | 1,003 | `aa1d788808ebd624dcff435538cd4e63f13004bf84101956ffe4119f14b14152` |
| `nhdf-v0.1` | 3,429 | `f70ceec98d9029f057e7ac71187a18fe28e9108ee8e79b45130d0942a22a8625` |
| `sclp-foundational` | 5,258 | `4c21f81bf73c863065963b9296f053781079c19c6c2b22ac5783d7ea957e9d13` |
| Registered, unselected `nhdf-v0.3-ccd` | 3,514 | `56c728d74cb03642b46ff3d3854d540590b01ddd903e747b23a2d2f2d8a36c62` |

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

## Direct deterministic substrate replay

The first-party reference application is a direct executable substrate proof,
not a language-model or coding-agent measurement. Its files are:

| Record | Bytes | SHA-256 |
|---|---:|---|
| `examples/ugtoms_sclp_reference.py` | 31,380 | `9c9c2c09d5832984bbe6c4faaace9eb9fc139c6ff2c4649f718d4296b61e7120` |
| Application manifest | 13,111 | `42451914715eb1f8be85481457068f310db644c034ae1fceeadf479d8314065c` |
| Deterministic evidence | 10,154 | `1d0dc094d9649b667739fc2a202316097ae980baf5585cba22b8e33675282a42` |

Two executions produced exactly the committed evidence bytes. Manifest
validation reports one evidence record, 12 mapping records across all nine
kernel categories, and exact coverage of ten selected-profile requirements:
four for NHDF v0.1 and six for SCLP. Missing, unknown, duplicate,
wrong-profile, false, or unresolved proof coverage fails closed.

The replay executes a ten-node Merkle definition DAG, log-polar address and
metric calculations, `payload_parity_bit`, `topology_parity_bit`,
`jitter_control_bit`, and `branch_control_bit` as distinct roles, bounded
BST-T/L-system-style routing, vector kinematics, finite-cone and sphere SDFs,
an actual 32-step opposite-sign sweep interval
`[0.702539065154, 0.702539065387]`, tri-state event admission, atomic
transition/lineage, typed `n` to `n+1` feedback, both SCLP key layouts, packed
pose/motion, a shared LUT, fixed recipe, and stable display prefix.

The profile's paired-sphere support, circle/distributed-apex geometry, source
half-turn map, and radix-prefix refinement remain explicit `BYPASS` mappings.
Feedback has no extension-proposal or promotion authority. No fixed-point
engine, fuzzy logic, earliest-impact, arbitrary semantic compression, or model
weight compression is claimed by this pass.

## Fresh sealed 32K model gate

Tracked public snapshot:
`metrics/local/ugtoms_local_agent_32k/functional_gate.json`

Corresponding artifact record:
`packs/qwen3-30b-a3b-iq2m-32k-q4kv/evidence/functional_gate.json`

The sealed evidence is 32,876 bytes with SHA-256
`d56140c0a4bc97fb9fab5d3930222a494e681744570363d0f810a8e872aa01c1`.

| Gate | Requirement | Result | Status |
|---|---:|---:|---|
| Functional prompt suite | 4/4 | 4/4 | Pass |
| Full reported layer offload | 49/49 | 49/49 | Pass |
| Allocated context | 32,768 | 32,768 | Pass |
| Allocated-context exact response | `OK` | `OK` | Pass |
| Peak device use | at most 12,227 MiB | 11,068 MiB | Pass |
| Device reserve | at least 512 MiB | 1,159 MiB | Pass |
| 64-token short decode | at least 80 tok/s | 132.502673 tok/s | Pass |

The allocated 32K run used an 864 MiB q4_0/q4_0 K/V buffer, a 9,279.83 MiB
CUDA model buffer, a 300.75 MiB CUDA compute buffer, and a 127.51 MiB
CPU-mapped model buffer. The fresh three-sample short microbenchmarks were:

| Workload | Mean | Standard deviation |
|---|---:|---:|
| 64-token prompt processing | 442.151809 tok/s | 140.840844 tok/s |
| 64-token generation | 132.502673 tok/s | 22.473620 tok/s |

Prompt samples were `299.733`, `445.363`, and `581.359` tok/s. Generation
samples were `106.623`, `147.103`, and `143.782` tok/s.

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

Tracked public snapshot:
`metrics/local/ugtoms_local_agent_32k/coding_agent_gate.json`

The evidence is 10,074 bytes with SHA-256
`253fe50d62fe70ea6ca82b6a481639197e0a98988e2a8b530339ebbf518c7e6b`.

The generated `evidence.json` records `all_gates_passed: true` and status
`PASSED`.

| Check | Result |
|---|---:|
| Actual model-reported served context | `n_ctx = 32768` |
| Native OpenAI-compatible tool-call JSON | Passed |
| Synthetic context retrieval | Exact needle at 21,997 prompt tokens |
| Disposable Python repair run | 30.299467 s |
| Recorded tool use | One `edit` and one bounded `bash` |
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
not prove broad coding skill, production safety, or UGTOMS understanding. The
direct reference replay separately proves deterministic substrate execution
without a model. The focused live one-defect SCLP repair gate separately passed
the fully disclosed repair described below. The broad four-file authoring
attempt is separately `FAILED`; neither result may be inferred from this
generic pass.

The corresponding end-user desktop entry point is `START_LOCAL_CODER.cmd`;
the command-line launcher remains `scripts/start_local_coder.ps1`. Once the
one-time client/model setup is complete, inference is local and consumes no
API key or paid token budget.

```powershell
.\scripts\start_local_coder.ps1 -Arguments @(
  "--quick",
  "C:\path\to\your\git-worktree"
)
```

The default Review workflow asks before ordinary edits and shell commands. The
finite permission contract denies named web, external-directory, plugin, MCP,
delegation, commit, push, and destructive patterns. Work mode enables automatic
approval after a per-session warning, so unmatched equivalent commands may run.
These are application controls, not an OS sandbox.

## Focused substrate semantic-repair gate

Primary record:
`metrics/local/ugtoms_local_agent_32k/substrate_focused_repair/evidence.json`

Evidence size and SHA-256: 10,268 bytes,
`3a0669b40de948330a451670dd61a7baf2e03771f2d8bb128354b1da830ba4b3`.
The evidence records `status: PASSED` and `all_gates_passed: true`.

| Check | Result |
|---|---:|
| Agent wall time | 34.241295 s |
| Recorded mutations/commands | Exactly one `edit`, exactly one `bash` |
| Independent fixture tests | 3/3 passed |
| Deterministic replay | Two byte-identical independent runs |
| Changed path | Only `src/sclp_repair.py` |
| Git HEAD | Unchanged |

The fixture and prompt fully specified both the defect, same-generation
feedback `n` to `n`, and the required repair, bounded feedback `n` to `n+1`.
This establishes instruction-following and tool compliance for one declared
semantic repair. It does not establish independent diagnosis, broad substrate
understanding, broad coding competence, compression, or preventive sandboxing.

## Substrate and application-evidence boundary

The kernel's same-generation definition graph is typed, Merkle
content-addressed, and acyclic: each child identity binds dependency content
hashes; instances bind definition hashes; pipelines bind ordered step hashes,
domain, codomain, and typed adjacency; and feedback binds endpoint hashes,
named typed ports, and exactly generation `n` to `n+1`. No unrestricted
fixed-point engine or same-generation authority cycle is implemented. Spatial
knowledge-graph work remains in
`substrate/incubator/spatial-evidence-ledger.json`; it is a future boundary,
not a current validated capability.

The direct reference manifest exercises the declared subset above and binds
its evidence to the exact kernel and selected-profile hashes. It is
feedback-only: `may_propose_extensions` and `may_promote_extensions` are both
false.

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

### Failed broad substrate-authoring gate

The retained 31 August 2026 broad four-file SCLP authoring attempt passed its
server, configuration, 32K model-context, and native-tool-call prechecks;
verified the expected 4/4 failing unimplemented clean-room baseline; then timed
out after 1,200 seconds while issuing repeated oversized malformed Edit calls.
The normalized evidence is
`metrics/local/ugtoms_local_agent_32k/substrate_full_authoring_failure/evidence.json`,
2,750 bytes, SHA-256
`c0bab4824086ea9497dae5d82d519994cffced593d44b183d4fe392e797a69e8`.
It records `status: FAILED` and `all_gates_passed: false`.

This is a historical broad-authoring failure, not a pending result. It remains
separate from the passing direct replay, generic agent gate, and fully
disclosed focused repair gate.

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

# Committed direct-reference application manifest
nhdf-edge substrate-validate-app `
  .\substrate\applications\ugtoms-sclp-reference-v0.1.json `
  --repository .

# Replay twice; the focused test checks byte identity and committed evidence
python .\examples\ugtoms_sclp_reference.py --output .\reference-first.json
python .\examples\ugtoms_sclp_reference.py --output .\reference-second.json
python -m pytest .\tests\test_substrate_reference_application.py -q

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
- byte-identical direct reference replay, 12 mappings across nine categories,
  and exact coverage of all ten requirements from the two selected profiles;
- current 32K/q4 cache allocation, 49/49 offload, monitored peak, reserve, and
  short functional/throughput gate;
- actual served `n_ctx = 32768`;
- one exact 21,997-token retrieval, native tool call, and isolated repair;
- one fully disclosed focused SCLP repair in 34.241295 seconds with exactly one
  Edit and one Bash call, 3/3 tests, two identical replays, only the declared
  source changed, and unchanged HEAD;
- exact evidence hashes and an unchanged fixture Git HEAD;
- fail-closed model artifact and resource checks.

Not established:

- NHDF-native tensor compression or arbitrary semantic compression;
- filled-32K quality, validated 48K behavior, or broad q4 K/V parity;
- broad coding accuracy, BF16 parity, or parity with a hosted frontier agent;
- independent substrate diagnosis or broad substrate understanding; the
  focused live repair disclosed the answer in advance;
- broad autonomous substrate authoring; the retained live attempt failed;
- preventive OS sandboxing or production security;
- sustained thermals, power, battery, or many-hour stability;
- portability beyond the named laptop, pinned runtime, and workspace-relative
  payload layout.
