# UGTOMS substrate and a local Qwen3 coding agent on 12 GB

This repository contains a clean-room UGTOMS kernel, selectable NHDF/SCLP
profiles, and a validated local coding stack for the complete
30,532,122,624-parameter `Qwen/Qwen3-30B-A3B-Instruct-2507` model on an NVIDIA
GeForce RTX 5070 Ti Laptop GPU with 12,227 MiB of VRAM.

The current result is useful, but its boundaries matter:

- `substrate/kernel/contract.json` is the substrate authority. The registry
  exposes `nhdf-v0.1`, `sclp-foundational`, and `nhdf-v0.3-ccd` as selectable
  profiles. The first-party reference application selects exactly
  `nhdf-v0.1` and `sclp-foundational`; the later CCD profile remains unselected
  there and does not replace the base.
- `packs/qwen3-30b-a3b-iq2m-32k-q4kv` is the current validated 32K/q4 K/V
  model artifact.
- GGUF/IQ2_M is an externally produced Bartowski/ggml tensor codec. UGTOMS and
  NHDF govern provenance, typed capability, resource policy, evidence, and
  fail-closed launch; they do not claim authorship of the working weight codec.
- `START_LOCAL_CODER.cmd` launches the desktop workflow; the PowerShell and
  Python launchers remain available for command-line use. After one-time
  installation and model acquisition, inference uses no account, API key, or
  paid per-token service. Hardware, storage, power, and initial downloads are
  still real costs.

## Substrate identity

The `ugtoms-kernel-v0.1` kernel is a finite, typed, deterministic
geometric-topological generative system. It combines the normalized NHDF v0.1
closure, the early UGTS typed algebra, the UGTS 3.6 content-addressed
referential DAG, the SCLP 3.6.2
swept-cone/log-polar correction, and the UGTS-GN event-admission discipline.
Same-generation definitions must be acyclic. Feedback is an explicit
generation `n` to `n+1` edge, not an unrestricted fixed-point engine.

The semantic kernel ID remains `ugtoms-kernel-v0.1`; that is distinct from the
current breaking schema discriminator `ugtoms-kernel-contract-0.2`. The profile
registry and profile schemas are likewise `ugtoms-profile-registry-0.2` and
`ugtoms-profile-0.2`, while application manifests use
`ugtoms-application-manifest-0.2` and the reference evidence uses
`ugtoms-sclp-reference-evidence-0.2`. A `v0.1` semantic ID or application
filename therefore must not be interpreted as the older document schema.

| Layer | Current role | File |
|---|---|---|
| UGTOMS kernel | Stable typed contract and invariants | `substrate/kernel/contract.json` |
| NHDF v0.1 | Normalized base closure | `substrate/profiles/nhdf-v0.1.json` |
| SCLP 3.6.2 | Foundational cone/log-polar corrective profile | `substrate/profiles/sclp-foundational.json` |
| NHDF v0.3 CCD | Optional later collision profile | `substrate/profiles/nhdf-v0.3-ccd.json` |
| Spatial ledger | Quarantined future evidence direction | `substrate/incubator/spatial-evidence-ledger.json` |

The profile registry is `substrate/profiles/registry.json`; automatic profile
or extension promotion is disabled. Clean-room implementations live in:

- `src/nhdf_edge/substrate_contract.py`: kernel, profile, application-manifest,
  and extension validation;
- `src/nhdf_edge/substrate_graph.py`: Merkle-bound typed definitions whose
  identities include dependency content hashes, instances bound to definition
  hashes, pipelines bound to ordered step hashes and typed adjacency, and
  feedback bound to endpoint hashes, named typed ports, and `n` to `n+1`;
- `src/nhdf_edge/substrate_runtime.py`: bounded SCLP/NHDF reference mechanisms;
- `src/nhdf_edge/substrate_packing.py`: bounded clean-room display packing;
- `src/nhdf_edge/substrate_pdf.py`: deterministic PDF generation and checking.

The symbol firewall keeps linear time, modular ticks, cone slant length,
golden ratio, phase, jitter radius, spatial radius, `payload_parity_bit`,
`topology_parity_bit`, `jitter_control_bit`, `branch_control_bit`, comparison
trees, radix tries, half-turn maps, Klein gluing, implicit cones, finite SDFs,
and swept bounds from being silently conflated.

## Direct deterministic substrate replay

The first-party `ugtoms-sclp-reference` application is a direct code-level
substrate replay, not a language-model or coding-agent run. Two executions
produced byte-identical output equal to the committed 10,154-byte evidence file,
SHA-256
`1d0dc094d9649b667739fc2a202316097ae980baf5585cba22b8e33675282a42`.
Its 13,111-byte manifest has SHA-256
`42451914715eb1f8be85481457068f310db644c034ae1fceeadf479d8314065c`.

The manifest validates against:

- kernel SHA-256
  `9b5fa7cff4483129e80e1c234055d57e0f79a574dcd2e131c418bbe0259448c3`;
- `nhdf-v0.1` SHA-256
  `f70ceec98d9029f057e7ac71187a18fe28e9108ee8e79b45130d0942a22a8625`;
- `sclp-foundational` SHA-256
  `4c21f81bf73c863065963b9296f053781079c19c6c2b22ac5783d7ea957e9d13`.

Validation reports 12 mapping records across all nine kernel mapping
categories and exact coverage of ten selected-profile evidence requirements:
four from NHDF v0.1 and six from SCLP. The replay executes log-polar addressing
and metric evaluation, the four distinct bit roles, bounded BST-T/L-system-style
routing, vectors and kinematics, finite-cone and sphere SDFs, a 32-step
opposite-sign sweep bracket, tri-state event admission, a ten-node Merkle
definition DAG, atomic transition and lineage, typed `n` to `n+1` feedback,
packed pose and motion, a shared LUT, a fixed recipe, and a stable display
prefix.

Paired-sphere support, circle/distributed-apex geometry, the source half-turn
map, and radix-prefix refinement are explicit `BYPASS` mappings. The replay has
no fixed-point engine, fuzzy-logic claim, earliest-impact claim, extension
proposal or promotion authority, arbitrary semantic-compression claim, or
model-weight-compression claim.

## Current model result

The model is mixture-of-experts: all 30.532B parameters must be stored, while
approximately 3.3B are active per token. Its throughput must not be described
as dense-30B throughput.

| Storage measurement | Result |
|---|---:|
| BF16 tensor bytes | 61,064,245,248 |
| BF16 weights | 58,235.4 MiB |
| Referenced IQ2_M file | 9,870,270,464 bytes |
| File-size reduction | 6.1867x / 83.84% |
| Physical GPU capacity | 12,227 MiB |

The zero-copy hybrid manifest references the existing 9.87 GB model file; it
does not commit or create a second large payload.

### Fresh sealed 32K gate

The current artifact passed its fresh gate on 31 August 2026:

| Measurement | Result |
|---|---:|
| Functional prompts | 4/4 passed |
| Reported layer offload | 49/49 |
| Allocated context | 32,768 tokens |
| K/V cache | q4_0 / q4_0, 864 MiB |
| Peak GPU memory | 11,068 MiB |
| Headroom on 12,227 MiB GPU | 1,159 MiB |
| 64-token prompt microbenchmark | 442.151809 tok/s |
| 64-token short decode microbenchmark | 132.502673 tok/s |

The tracked public snapshot is
`metrics/local/ugtoms_local_agent_32k/functional_gate.json`; the corresponding
artifact record is
`packs/qwen3-30b-a3b-iq2m-32k-q4kv/evidence/functional_gate.json`. The
32,876-byte fresh evidence has SHA-256
`d56140c0a4bc97fb9fab5d3930222a494e681744570363d0f810a8e872aa01c1`.
The three prompt samples were `299.733`, `445.363`, and `581.359` tok/s
(standard deviation `140.840844`); decode samples were `106.623`, `147.103`,
and `143.782` tok/s (standard deviation `22.473620`).

This establishes full cache allocation, residency, a short exact response,
full layer offload, and short microbenchmark throughput. It does not establish
answer quality after filling all 32K tokens. A 48K configuration remains
fragile and unvalidated, so it is not the default.

### Separate prior filled-context measurement

A prior, separately scoped run used 22,440 prompt tokens and measured
2,297.477 prompt tok/s followed by 25.189 decode tok/s. At that decode rate,
100 generated tokens take about 3.97 seconds and 300 take about 11.91 seconds,
before application overhead. This is the practical long-context result; it is
not interchangeable with the fresh 64-token 132.502673 tok/s gate and is not
part of that sealed gate record.

## Generic local coding-agent gate

The final bounded generic agent run is retained in the tracked public snapshot:

`metrics/local/ugtoms_local_agent_32k/coding_agent_gate.json`

The 10,074-byte evidence has SHA-256
`253fe50d62fe70ea6ca82b6a481639197e0a98988e2a8b530339ebbf518c7e6b`.

| Check | Measured result |
|---|---:|
| Actual context reported by the served model | `n_ctx = 32768` |
| Synthetic needle retrieval | 21,997 prompt tokens, exact retrieval |
| Native OpenAI-compatible tool-call JSON | Passed |
| Disposable repository repair | Passed in 30.299467 s |
| Recorded mutations/commands | One `edit`, one bounded `bash` |
| Independent final fixture tests | 4/4 passed |
| Git HEAD | Unchanged |
| Overall disposition | `PASSED` |

The tool-path audit was retrospective, not a preventive OS/filesystem/network
sandbox. The launcher adds an explicit permission contract, reduced
environment, loopback-only model endpoint, no project configuration, and no
plugins or MCP, but this still does not prove production safety or broad coding
accuracy.

This was deliberately a generic coding gate. It proves one native tool call,
one approximately 22K-token retrieval, and one isolated Python repair. Do not
describe it as substrate awareness. The direct deterministic substrate replay
above passed without involving the model. A narrower live gate in which the
agent repaired one declared SCLP generation-boundary defect also passed. Its
10,268-byte evidence file is
`metrics/local/ugtoms_local_agent_32k/substrate_focused_repair/evidence.json`,
SHA-256
`3a0669b40de948330a451670dd61a7baf2e03771f2d8bb128354b1da830ba4b3`.
The 34.241295-second run used exactly one Edit and one Bash call, passed 3/3
pytest checks, reproduced the deterministic replay twice, changed only
`src/sclp_repair.py`, and left Git HEAD unchanged.

The exact defect and required `n` to `n+1` answer were fully specified in the
fixture and prompt. This narrow pass therefore proves instruction-following
and tool compliance, not independent diagnosis or broad substrate
understanding. The historical broad four-file substrate-authoring live attempt
is separately `FAILED`; it must not be merged with any passing result.

## Run the free local coding agent

The normal Windows path is the desktop GUI: double-click
`START_LOCAL_CODER.cmd`. Its five readiness cards verify the pinned OpenCode
client, model, llama.cpp plus exact CUDA 12.8 dependencies, GPU capacity, and
sealed artifact. **Install Client** expands the verified repository archive;
Node.js and npm are not required for that release path. **Download Model**
resumes the immutable-revision download and promotes it only after the exact
9,870,270,464-byte size and SHA-256 pass. Choose a Git repository, start the
resident model once, and send multiple prompts without reloading the weights.

Review mode is the default and keeps OpenCode approval prompts. Work mode is a
convenience mode that passes `--auto` after an explicit per-session warning;
it is not an OS sandbox and must be used only on a recoverable Git worktree.
See `docs/LOCAL_CODER_GUI.md` for the complete workflow and recovery steps.

The command-line path remains available. Prerequisites are a trusted Python 3.10+
with tkinter,
the repository's pinned llama.cpp runtime, the verified model at its manifest
path, the exact pinned CUDA 12.8 DLLs, a compatible NVIDIA driver, and enough
free GPU memory.

From the repository root:

```powershell
python -m pip install -e ".[dev,runtime]"
powershell -ExecutionPolicy Bypass -File .\scripts\setup_local_coder.ps1
nhdf-edge verify .\packs\qwen3-30b-a3b-iq2m-32k-q4kv
```

First trusted interactive launch, with a full payload hash check:

```powershell
.\scripts\start_local_coder.ps1 -Arguments @(
  "C:\path\to\your\git-worktree"
)
```

Compatibility invocation; strict verification is unchanged:

```powershell
.\scripts\start_local_coder.ps1 -Arguments @(
  "--quick",
  "C:\path\to\your\git-worktree"
)
```

One noninteractive request; `--run` must be last because all remaining values
are passed to OpenCode:

```powershell
.\scripts\start_local_coder.ps1 -Arguments @(
  "--quick",
  "C:\path\to\your\git-worktree",
  "--run",
  "Inspect the repository, explain the failing test, and do not edit yet."
)
```

`--quick` is retained for existing scripts. It does not skip or reduce any
verification: both forms perform the same strict final full payload hash while
the payload is locked against replacement.

The configured Review workflow asks before edits and ordinary shell commands.
Its application controls narrow the intended workflow; they are retrospective
and permission-layer controls, not a substitute for an OS sandbox. Work mode's
automatic approval makes that distinction especially important.

## Verify the substrate, manifests, and report source

Verify the committed kernel, source hashes, and all selectable profiles:

```powershell
nhdf-edge substrate-verify --repository .
```

Verify the current zero-copy model artifact and sealed evidence:

```powershell
nhdf-edge verify .\packs\qwen3-30b-a3b-iq2m-32k-q4kv
```

Validate the committed direct-reference application manifest:

```powershell
nhdf-edge substrate-validate-app `
  .\substrate\applications\ugtoms-sclp-reference-v0.1.json `
  --repository .
```

Replay it twice and compare both outputs with the committed evidence:

```powershell
python .\examples\ugtoms_sclp_reference.py --output .\reference-first.json
python .\examples\ugtoms_sclp_reference.py --output .\reference-second.json
python -m pytest .\tests\test_substrate_reference_application.py -q
```

Render the versioned guide, verify expected extractable text with pypdf, write
a SHA-256 metadata sidecar, and produce Poppler page images for visual QA:

```powershell
python .\scripts\render_substrate_pdf.py `
  .\docs\UGTOMS_Local_Substrate_Coding_Agent_Guide_v0.1.md `
  --output .\output\pdf\UGTOMS_Local_Substrate_Coding_Agent_Guide_v0.1.pdf `
  --expect "UGTOMS Local Substrate Coding Agent Guide" `
  --render-pages .\output\pdf\UGTOMS_Local_Substrate_Coding_Agent_Guide_v0.1-pages
```

## Correct application-proof scope

The validated direct reference manifest and replay are
`substrate/applications/ugtoms-sclp-reference-v0.1.json` and
`substrate/applications/evidence/ugtoms-sclp-reference-v0.1.json`. They remain
below the kernel authority boundary and exercise only their declared
implemented mappings. `substrate/evidence/application_proofs.json` separately
retains older bounded Grove and KSGP application evidence.

- The `KC3D392` Grove scene is a 21,798-byte explicit packed scene with 66
  explicit nodes. It is not the 1,024-instance recipe result.
- The separate 1,024-display-instance recipe is 4,984 bytes versus 162,274
  bytes for the comparison representation: 32.56x smaller, a 96.93% reduction,
  with exact prior-prefix stability. Its generated copies are render-only;
  they do not gain independent ECS, collider, graph, saved-object, or gameplay
  identity.
- The `KSGP1` proof reconstructs 328 recorded states with zero mismatch flags,
  maximum position delta `2.656269518315808e-7 km`, and maximum velocity delta
  `7.841169468777989e-10 km/s`. It does not place generic SGP4 mathematics in
  the kernel.

These are bounded packing/reconstruction proofs, not evidence of arbitrary
semantic compression.

## Historical results retained for comparison

### Failed broad substrate-authoring gate

The 31 August 2026 broad four-file SCLP authoring attempt passed its server,
32K context, configuration, and native-tool-call prechecks; verified the
expected 4/4 failing unimplemented clean-room baseline; then timed out after
1,200 seconds while producing repeated oversized malformed Edit calls. Its
normalized 2,750-byte evidence file is
`metrics/local/ugtoms_local_agent_32k/substrate_full_authoring_failure/evidence.json`,
SHA-256
`c0bab4824086ea9497dae5d82d519994cffced593d44b183d4fe392e797a69e8`,
with `status: FAILED` and `all_gates_passed: false`.

This is retained negative evidence for broad autonomous substrate authoring. It
does not negate the direct replay, generic-agent, or fully disclosed focused
repair passes.

### Superseded 8K deployment profile

The earlier `packs/qwen3-30b-a3b-nhdf-v03-iq2m` artifact remains historical
evidence, not the current default. It passed 4/4 functional prompts, reported
49/49 offload, allocated an 8K q8 K/V cache, peaked at 10,610 MiB with 1,617
MiB headroom, and measured 870.026857 prompt tok/s and 157.141442 short decode
tok/s. Its controlled 512-prompt/256-generation result was 149.441046 decode
tok/s; a small resident coding probe measured a 147.139410 tok/s median, 10/12
first-pass launches, and 12/12 after at most one repair. These 8K results are
preserved for comparison and must not be presented as the current gate.

### Failed native-codec research

The legacy native scalar pack contains 9,152,386,624 tensor-file bytes. It
loaded and executed on CUDA and passed all 531 CRC/parity checks, but generated
repeated newlines or `10000000`. It remains `QUALITY_FAILED` and is refused by
default. A later two-expert GEMQ-style probe improved numerical error but
missed its declared 20% comparative-advantage threshold and was not expanded
to a full checkpoint. The working deployment therefore remains explicitly on
the external GGUF/IQ2_M codec boundary.

## Evidence boundary

Established on the named laptop and pinned runtime:

- verified UGTOMS kernel/profile source records and no automatic promotion;
- a byte-deterministic direct substrate replay with 12 application mappings and
  exact coverage of all ten requirements from its two selected profiles;
- a complete-model 32K/q4 artifact with 49/49 offload and measured headroom;
- four bounded functional outputs and fresh short throughput;
- actual 32,768-token server exposure, 21,997-token retrieval, native tool-call
  JSON, and one isolated repair with independent tests;
- one fully disclosed focused SCLP repair completed in 34.241295 seconds with
  exactly one Edit and one Bash call, 3/3 tests, two identical replays, one
  changed source file, and unchanged HEAD;
- fail-closed artifact and resource checks.

Not established:

- an NHDF-native tensor-codec advantage;
- filled-32K quality, validated 48K operation, or broad q4 K/V quality parity;
- broad coding accuracy, BF16 parity, or hosted-frontier-agent parity;
- independent substrate diagnosis or broad substrate understanding; the
  focused repair answer was fully disclosed;
- broad autonomous substrate authoring; the retained attempt failed by timeout;
- preventive sandboxing, production security, sustained thermals, or
  long-duration stability;
- a self-contained artifact: the large payload remains a workspace-relative
  external file.

See `VALIDATION_STATUS.md` for the evidence disposition and
`docs/UGTOMS_Local_Substrate_Coding_Agent_Guide_v0.1.md` for the detailed
architecture and operational guide.
