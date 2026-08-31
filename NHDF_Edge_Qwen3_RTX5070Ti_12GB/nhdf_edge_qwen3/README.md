# UGTOMS substrate and a local Qwen3 coding agent on 12 GB

This repository contains a clean-room UGTOMS kernel, selectable NHDF/SCLP
profiles, and a validated local coding stack for the complete
30,532,122,624-parameter `Qwen/Qwen3-30B-A3B-Instruct-2507` model on an NVIDIA
GeForce RTX 5070 Ti Laptop GPU with 12,227 MiB of VRAM.

The current result is useful, but its boundaries matter:

- `substrate/kernel/contract.json` is the substrate authority. The explicit
  active base profiles are `nhdf-v0.1` and `sclp-foundational`; the later
  `nhdf-v0.3-ccd` profile is optional and does not replace them.
- `packs/qwen3-30b-a3b-iq2m-32k-q4kv` is the current validated 32K/q4 K/V
  model artifact.
- GGUF/IQ2_M is an externally produced Bartowski/ggml tensor codec. UGTOMS and
  NHDF govern provenance, typed capability, resource policy, evidence, and
  fail-closed launch; they do not claim authorship of the working weight codec.
- `scripts/start_local_coder.ps1` launches an isolated OpenCode client against
  the local model. After one-time installation and model acquisition, it uses
  no account, API key, or paid per-token service. Hardware, storage, power, and
  initial downloads are still real costs.

## Substrate identity

The `ugtoms-kernel-v0.1` kernel is a finite, typed, deterministic
geometric-topological generative system. It combines the normalized NHDF v0.1
closure, the early UGTS typed algebra, the UGTS 3.6 content-addressed
referential DAG, the SCLP 3.6.2
swept-cone/log-polar correction, and the UGTS-GN event-admission discipline.
Same-generation definitions must be acyclic. Feedback is an explicit
generation `n` to `n+1` edge, not an unrestricted fixed-point engine.

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
- `src/nhdf_edge/substrate_graph.py`: typed content-addressed definitions,
  instances, pipelines, topological resolution, and next-generation feedback;
- `src/nhdf_edge/substrate_runtime.py`: bounded SCLP/NHDF reference mechanisms;
- `src/nhdf_edge/substrate_packing.py`: bounded clean-room display packing;
- `src/nhdf_edge/substrate_pdf.py`: deterministic PDF generation and checking.

The symbol firewall keeps linear time, modular ticks, cone slant length,
golden ratio, phase, jitter radius, spatial radius, four distinct bit roles,
comparison trees, radix tries, half-turn maps, Klein gluing, implicit cones,
finite SDFs, and swept bounds from being silently conflated.

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
| Peak GPU memory | 11,064 MiB |
| Headroom on 12,227 MiB GPU | 1,163 MiB |
| 64-token prompt microbenchmark | 885.013308 tok/s |
| 64-token short decode microbenchmark | 135.208586 tok/s |

This establishes full cache allocation, residency, a short exact response,
full layer offload, and short microbenchmark throughput. It does not establish
answer quality after filling all 32K tokens. A 48K configuration remains
fragile and unvalidated, so it is not the default.

### Separate prior filled-context measurement

A prior, separately scoped run used 22,440 prompt tokens and measured
2,297.477 prompt tok/s followed by 25.189 decode tok/s. At that decode rate,
100 generated tokens take about 3.97 seconds and 300 take about 11.91 seconds,
before application overhead. This is the practical long-context result; it is
not interchangeable with the fresh 64-token 135.208586 tok/s gate and is not
part of that sealed gate record.

## Generic local coding-agent gate

The final bounded generic agent run passed at:

`metrics/local/coding_agent_32k/run-20260831T172748.453659Z/evidence.json`

Evidence SHA-256:
`a110060c30816c9d8e92d9ddc0eb0ade6c07be871371dd49942cb8de47262348`.

| Check | Measured result |
|---|---:|
| Actual context reported by the served model | `n_ctx = 32768` |
| Synthetic needle retrieval | 21,997 prompt tokens, exact retrieval |
| Native OpenAI-compatible tool-call JSON | Passed |
| Disposable repository repair | Passed in 36.524758 s |
| Recorded mutations/commands | Exactly one `edit`, exactly one `bash` |
| Independent final fixture tests | 4/4 passed |
| Git HEAD | Unchanged |
| Overall disposition | `PASSED` |

The tool-path audit was retrospective, not a preventive OS/filesystem/network
sandbox. The launcher adds an explicit permission contract, reduced
environment, loopback-only model endpoint, no project configuration, and no
plugins or MCP, but this still does not prove production safety or broad coding
accuracy.

This was deliberately a generic coding gate. It proves one native tool call,
one approximately 22K-token retrieval, and one isolated Python repair. A live
substrate-specific gate that tests correct profile selection, symbol-firewall
discipline, typed substrate manifests, and evidence-bound behavior is still
pending measurement. In status terms, the substrate-specific live gate is
pending. Do not describe the generic pass as substrate awareness.

## Run the free local coding agent

Prerequisites are Python 3, the repository's pinned llama.cpp/CUDA runtime,
the verified model at its manifest path, an NVIDIA driver/CUDA 12.8 runtime,
and enough free GPU memory. The one-time setup installs the pinned local
OpenCode client and may require npm network access.

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

Routine launch after a trusted full verification:

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

The configured agent asks before edits and ordinary shell commands. It denies
web access, external directories, plugins, MCP, delegation, commits, pushes,
and destructive commands. Those controls narrow the intended workflow; they
are not a substitute for an OS sandbox.

## Verify the substrate, manifests, and report source

Verify the committed kernel, source hashes, and all selectable profiles:

```powershell
nhdf-edge substrate-verify --repository .
```

Verify the current zero-copy model artifact and sealed evidence:

```powershell
nhdf-edge verify .\packs\qwen3-30b-a3b-iq2m-32k-q4kv
```

Validate an evidence-bound substrate application manifest:

```powershell
nhdf-edge substrate-validate-app `
  .\path\to\application-manifest.json `
  --repository . `
  --evidence-root .\path\to\evidence
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

`substrate/evidence/application_proofs.json` keeps application evidence below
the kernel authority boundary.

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
- a complete-model 32K/q4 artifact with 49/49 offload and measured headroom;
- four bounded functional outputs and fresh short throughput;
- actual 32,768-token server exposure, 21,997-token retrieval, native tool-call
  JSON, and one isolated repair with independent tests;
- fail-closed artifact and resource checks.

Not established:

- an NHDF-native tensor-codec advantage;
- filled-32K quality, validated 48K operation, or broad q4 K/V quality parity;
- broad coding accuracy, BF16 parity, or hosted-frontier-agent parity;
- a passing live substrate-specific agent gate;
- preventive sandboxing, production security, sustained thermals, or
  long-duration stability;
- a self-contained artifact: the large payload remains a workspace-relative
  external file.

See `VALIDATION_STATUS.md` for the evidence disposition and
`docs/UGTOMS_Local_Substrate_Coding_Agent_Guide_v0.1.md` for the detailed
architecture and operational guide.
