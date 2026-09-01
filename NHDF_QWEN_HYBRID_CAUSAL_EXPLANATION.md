# Why the NHDF Qwen hybrid worked

## Causal explanation, v0.3 lineage correction, and the sub-version worth retaining

**Status:** evidence-bound NHDF note

**Date:** 2026-09-01

**Evidence cutoff:** the pre-executable-UGTOMS NHDF hybrid work through Git commit `37f3435`; some files already carried “UGTOMS/NHDF” branding, but the executable UGTOMS/SCLP substrate integration began later at `cc4f8b2`

**Retained artifact format:** `nhdf-edge-hybrid-0.1`

## Lineage boundary

This document is deliberately about **NHDF only**.

The Qwen hybrid did not execute UGTS, UGTOMS, UGTS-GN, SCLP, or the later UGTOMS kernel/profile system. Earlier “UGTOMS/NHDF” labels were names, not evidence that those systems ran. Those later systems must not be projected backward as the reason the NHDF test worked. SCLP may remain important to the broader spatial-substrate lineage, but it was not part of the NHDF Qwen inference path and is not used here as causal evidence.

The files live today inside a parent repository whose scope later expanded. Their present location does not change their historical authority. The NHDF result is reconstructed from:

- the NHDF v0.1 and v0.3 specifications;
- the historical NHDF hybrid manifest and its functional evidence;
- the NHDF hybrid implementation as it existed at the validating commits;
- the externally attributed Qwen/IQ2_M payload; and
- the failed NHDF-native pack as a negative control.

The later 32K UGTOMS-grounded artifact is not used to explain what the earlier NHDF-labeled hybrid did. It is a later integration result.

## One-sentence conclusion

NHDF did not compress Qwen's weights; the successful NHDF-labeled hybrid implementation made an external IQ2_M model into a compactly referenced, bounded, attributed, measured, and fail-closed deployment object, and **that demonstrated hybrid integration contract—not the whole v0.3 CCD expansion—is the NHDF sub-version candidate worth retaining and formalizing**.

## What physically made the model fit

The model is Qwen3-30B-A3B-Instruct-2507:

- 30,532,122,624 total parameters;
- about 3.3 billion parameters active for each token because it is a mixture-of-experts model;
- 61,064,245,248 recorded BF16 tensor bytes; and
- a target GPU with 12,227 MiB of VRAM.

The BF16 weights alone are 58,235.402 MiB, about **4.76 times the entire physical VRAM capacity**, before allocating K/V cache, compute buffers, or runtime overhead. The normal BF16 representation could not fit.

The referenced GGUF/IQ2_M payload is 9,870,270,464 bytes, or 9,413.023 MiB. Relative to the recorded BF16 tensors:

```text
61,064,245,248 / 9,870,270,464 = 6.186684x smaller

1 - (9,870,270,464 / 61,064,245,248)
  = 83.836252% fewer stored bytes
```

That external low-bit representation is what made weight residency possible. The manifest is explicit:

```text
container:          GGUF
profile:            IQ2_M mixed-bit
owner:              ggml/Bartowski
nhdf_native_codec:  false
```

The model's MoE sparsity then helped make inference fast: roughly 3.3B parameters are active per token instead of all 30.5B. MoE sparsity did **not** make the inactive expert weights disappear from storage; all expert weights still had to be present. Therefore:

- **IQ2_M explains the stored-weight fit**;
- **MoE sparsity helps explain practical per-token execution**; and
- **NHDF must not claim authorship of either mechanism**.

## What the retained NHDF test actually measured

The first validated NHDF hybrid landed in commit `77b1ebf`. The subsequent still-NHDF optimization in `37f3435` retained the same model payload and artifact format while improving the runtime. The optimized retained 8K evidence recorded:

| Measurement | Result |
|---|---:|
| Referenced IQ2_M payload | 9,870,270,464 bytes |
| BF16 tensor baseline | 61,064,245,248 bytes |
| Allocated context | 8,192 tokens |
| K/V cache | q8_0 K and q8_0 V |
| CUDA K/V buffer at 8K | 408.00 MiB |
| CUDA model buffer | 9,279.83 MiB |
| CUDA compute buffer | 300.75 MiB |
| Full model offload | 49/49 layers |
| Peak GPU memory | 10,610 MiB |
| Physical GPU capacity | 12,227 MiB |
| Remaining headroom | 1,617 MiB |
| Functional prompts | 4/4 passed |
| Short 64-token prompt benchmark | 870.026857 tok/s |
| Short 64-token generation benchmark | 157.141442 tok/s |
| Required generation threshold | 80 tok/s |
| Validation state | `VALIDATED` |

The initial validating commit was already a success: 4/4 functional prompts, 10,487 MiB peak, 1,740 MiB headroom, and 102.367894 short-generation tok/s. The later NHDF-only optimization improved the short-generation mean to 157.141442 tok/s.

The four functional checks were deliberately simple but semantic:

1. return exactly `OK`;
2. calculate `17 * 19` as `323`;
3. explain why a checksum does not prove compressed-model answer quality; and
4. produce a valid short Python `is_even(n)` function.

This was enough to reject the claim that the retained hybrid merely emitted garbage. It was not enough to establish broad coding intelligence, frontier-model parity, or quality at a genuinely filled 8K context. The 8K gate allocated the cache and executed a short prompt. The 157.141442 tok/s number is a short engine benchmark, not a whole coding task and not filled-context decode.

## What the NHDF-labeled hybrid implementation contributed

No NHDF representation, PCKF, IHCP, geometric, routing, parity, or CCD operator caused weight fit or transformer inference in this test. The contribution demonstrated by the code carrying the NHDF hybrid label was **not tensor encoding**; it was the bounded hybrid envelope around an external tensor encoding.

The historical `NHDF_HYBRID_MANIFEST.json` is only a few kilobytes. It references the multi-gigabyte payload in place and binds the exact parts required to trust and replay one deployment:

- model identity and source revision;
- payload path, byte count, and SHA-256;
- external codec format, profile, owner, and attribution;
- NHDF specification path, byte count, and SHA-256;
- runtime implementation, revision, build, entrypoints, and sealed files;
- finite execution parameters;
- target GPU and VRAM/resource limits;
- validation status and evidence hash; and
- an append-only event chain from artifact creation to validation.

The implementation supplied four operational stages:

```text
create -> verify -> gate -> run
```

### Create

`create_hybrid_artifact` wrote a small zero-copy manifest. It did not duplicate the 9.87 GB model and did not relabel IQ2_M as an NHDF codec. The new artifact began as `UNCALIBRATED`.

### Verify

`verify_hybrid_artifact` checked the manifest, payload record, runtime files, specification record, source record, validation evidence, and event chain. A mismatch became an explicit failure rather than an ignored warning.

### Gate

`gate_hybrid_artifact` ran fresh functional prompts, an allocated-context residency check, full-offload verification, a VRAM reserve test, and a throughput test. The result could become:

```text
UNCALIBRATED
    |
    +-- VALIDATED
    +-- QUALITY_FAILED
    +-- RESOURCE_FAILED
```

The transition to `VALIDATED` required evidence matching the exact payload and artifact format. A status label alone was insufficient.

### Run

`run_hybrid_prompt` rechecked integrity and the hardware resource preflight before invoking the pinned llama.cpp runtime. An unvalidated artifact, a hash mismatch, the wrong GPU identity, or insufficient free VRAM closed the launch path by default.

This is why the word **hybrid** matters. The result combined:

- an external, already proven tensor representation;
- an external CUDA inference engine; and
- a declaration, bounding, evidence, state, and refusal layer implemented in `nhdf_edge.hybrid`.

The implementation realized an NHDF-compatible pattern in which the labeled substrate layer did not have to own every operator. It named an external mechanism, stated its responsibility boundary, bound one exact instance, constrained its use, observed its result, and refused it when the evidence failed. Whether that pattern generalizes beyond this one model/codec family still requires evidence.

## Why the failed native pack is decisive

The earlier NHDF-native scalar pack is the negative control that prevents a false success story.

It contained 9,152,386,624 packed tensor-file bytes and covered all 531 logical runtime tensors. It passed:

- 531/531 CRC checks; and
- 531/531 parity checks.

It also fit and executed through CUDA. Yet the functional output collapsed to repeated newline tokens or `10000000`. The measured layer-0 expert probes also showed material error, including normalized RMSE values from 0.4482 to 0.4813. The artifact was rejected for quality.

That experiment proves:

```text
integrity != semantic fidelity
fit       != usability
native    != correct
```

The hybrid was the breakthrough because it stopped treating “NHDF-native codec” as a condition of success. It kept the **deployment contract** in the NHDF-labeled implementation while delegating weight encoding to a mechanism that preserved usable model behavior.

This is not a semantic trick. The external codec really supplied the compression. The NHDF-labeled hybrid implementation supplied the reproducible composition and the rule that a codec cannot be trusted merely because it can be loaded.

## Why the base-to-v0.3 jump was faulty

### The base version has a compact general identity

NHDF v0.1 defines a bounded causal state machine around three layers:

- NHDF representation;
- PCKF causal dynamics; and
- IHCP runtime execution.

Its important general rules include:

- monotonic logical time and declared generation semantics;
- non-degenerate local state constraints;
- declared operator order;
- finite branch, memory, history, and output bounds;
- explicit next-generation feedback;
- visible failure when residual, saturation, parity, or resource limits are exceeded;
- deterministic replay and telemetry as recommended evidence; and
- separation of speculative AI/compression claims from core conformance.

The v0.1 specification also says that if a simpler ablation performs equally well, the corresponding NHDF operator is not justified for that application. That is directly relevant here: the Qwen test did not demonstrate that geometric, parity, routing, or CCD operators improved tensor inference.

### Version 0.3 is not a clean replacement base

NHDF v0.3 is a much larger specification focused on general-purpose continuous collision detection and a broad set of adapters. It expands provenance, attention, contact handling, numeric policy, resource rules, public-safety material, GPU concerns, and Edge AI discussion around the v0.1 base.

The document itself says:

- v0.1 remains the baseline;
- normative core, conformance profiles, and engineering hypotheses are different classes;
- profile inclusion is not proof of superiority;
- the Edge AI adapter does not make dense weights disappear;
- compression ratio, quality, latency, memory, calibration, and baseline quantization must be measured; and
- NHDF operators do not replace actual CCD certificates.

Therefore, treating “v0.3” as one universal successor created two problems.

First, it confused **chronology with semantic authority**. A later document containing more domains does not mean every application uses all those domains or that the compact base has been replaced.

Second, it created **false evidence inheritance**. The Qwen run never executed collision candidates, TOI certificates, contact manifolds, or collision response. Binding the complete CCD-focused v0.3 document to a successful inference artifact could make it appear that those mechanisms caused or validated the result. They did not.

The proposed corrected NHDF retention topology is branched:

```text
NHDF v0.1                      compact general base
|
+-- NHDF v0.3 CCD material     domain specialization; kept for CCD
|
+-- NHDF Edge Hybrid 0.1       proposed external-operator deployment sub-version;
                               its Qwen instance is evidence-backed
```

The fault is the attempted straight line:

```text
v0.1 base -> v0.3 replaces base -> Qwen success attributed to v0.3 as a whole
```

The proposed correction is:

```text
v0.1 remains the base
v0.3 CCD remains a specialization
the demonstrated hybrid contract is extracted as its own NHDF sub-version candidate
```

This is a correction to version structure and causal attribution, not a claim that all v0.3 mathematics is wrong.

## The sub-version candidate worth retaining

The repository already contains the historically validated artifact-format name:

```text
nhdf-edge-hybrid-0.1
```

That name should be retained. It should not be renamed to UGTOMS, `v0.3.1`, or `v0.4` merely because later work exists.

Its proposed normative identity, derived from the demonstrated implementation, is:

> An NHDF edge-deployment profile that binds a declared external operator and payload to a finite runtime/resource envelope, measures its semantic output, records the result as evidence, and refuses undeclared or failed execution.

The Qwen artifact is the validated evidence source from which the candidate profile is abstracted. The surrounding structure was designed so it need not require one specific model or codec; however, actual replaceability and generality beyond this model/codec family remain hypotheses until a second class of mechanism is implemented and validated.

## Formal structure to distill into NHDF

The historical manifest supplies a recorded field set from which this document proposes the following formal abstraction. No UG(TO)MS logic is needed to describe it:

```text
H_NHDF = <N, W, C, R, X, B, V, E>
```

where:

| Symbol | NHDF hybrid field |
|---|---|
| `N` | NHDF specification binding: identity, path, size, and hash. |
| `W` | Work payload: model/source identity, reference, size, and hash. |
| `C` | External codec declaration: owner, container, profile, attribution, and `nhdf_native_codec`. |
| `R` | Runtime closure: implementation, revision, build, entrypoints, and sealed runtime files. |
| `X` | Execution profile: context, cache types, offload, batching, threads, template, sampling, and limits. |
| `B` | Bounded resources: device identity, physical capacity, required free memory, reserve, and admitted context. |
| `V` | Validation record: state, evidence reference/hash, measured results, thresholds, and scope. |
| `E` | Event lineage: creation and status transitions linked to the prior event and evidence. |

This tuple is not asserted as a hidden formula from the v0.1 PDF or as a formula that the runtime executed. It is a proposed field-for-field abstraction of the recorded NHDF hybrid manifest.

### Proposed external-operator definition

The demonstrated exact-instance record suggests the following generalization without absorbing the external operator into NHDF. This is a design proposal, not a second validated result:

```text
O_ext = <domain, codomain, mechanism, owner, resources, failures, validation>
```

For the Qwen instance:

- `domain`: the declared Qwen source tensor set;
- `codomain`: a GGUF/IQ2_M payload consumable by the declared runtime;
- `mechanism`: external IQ2_M encoding, referenced rather than reimplemented;
- `owner`: ggml/Bartowski attribution boundary;
- `resources`: payload bytes, GPU memory, cache, context, compute, and reserve;
- `failures`: missing/mismatched files, bad runtime, insufficient memory, failed offload, insufficient throughput, or semantically incorrect output; and
- `validation`: exact functional prompts plus resource and performance gates bound to evidence.

This follows the v0.3 source-admission discipline: an idea becomes usable only after it receives a domain, codomain, equation or algorithm/mechanism, resource bound, failure mode, and validation requirement. It does not require importing v0.3 CCD itself.

### Demonstrated properties and proposed hybrid invariants

The following separates properties directly demonstrated by the Qwen test from requirements proposed for a generalized sub-version:

1. **Demonstrated — attribution is part of identity.** The external codec remained external even though the hybrid depended on it.
2. **Demonstrated — the exact instance is bound.** Payload, specification, runtime, source record, and evidence carry size/hash records.
3. **Demonstrated — reference is not compression authorship.** The small manifest reused a large payload without claiming its byte reduction as an NHDF codec result.
4. **Demonstrated — resources are finite before launch.** Device, required free VRAM, reserve, context, and offload expectations were declared and checked.
5. **Demonstrated — integrity and quality are separate predicates.** Hash/CRC/parity success did not replace functional-output gates.
6. **Demonstrated — validation is scoped.** `VALIDATED` applied to the bound model, codec, runtime, execution profile, hardware envelope, and tested claims.
7. **Demonstrated — state change requires evidence.** `UNCALIBRATED -> VALIDATED` bound the passing evidence; failures had explicit dispositions.
8. **Demonstrated — execution fails closed.** Mismatched, unvalidated, or resource-incompatible artifacts were refused by the declared path.
9. **Demonstrated — negative results remain part of the record.** The native pack was retained as evidence rather than rewritten as a success.
10. **Proposed — the external component is replaceable.** A different codec or mechanism must be admitted only as a new bound instance with fresh validation. This was designed for but not demonstrated by the single IQ2_M success.

### What “anything that can be formalized” should mean here

Every fact recorded in this Qwen evidence set can be represented in the proposed NHDF hybrid definition, but representation does not mean every fact becomes universal NHDF core.

| Layer | What is formalized there |
|---|---|
| NHDF base | General bounded-state, declared-order, feedback, resource, telemetry, and failure rules. |
| NHDF Edge Hybrid 0.1 candidate | Demonstrated external-operator identity, attribution, exact component binding, execution/resource contract, evidence state, and refusal behavior; proposed heterogeneous replaceability. |
| Codec instance | IQ2_M container/profile, owner, payload hash/size, and codec-specific compatibility. |
| Qwen application evidence | Model identity, functional prompts, VRAM, cache, offload, throughput, and observed outputs. |
| Rejected experiment evidence | Native pack identity, integrity results, tensor error, and failed language output. |

This preserves all formalizable information while keeping each claim at the layer where it was actually demonstrated.

## What must not be distilled into NHDF's universal identity

The following belong to this application instance, not to the general substrate definition:

- Qwen3-30B-A3B model identity and architecture;
- the 30.532B/3.3B parameter counts;
- GGUF and IQ2_M codec mathematics;
- Bartowski/ggml ownership;
- the 9,870,270,464-byte payload and its hash;
- llama.cpp and CUDA revisions;
- RTX 5070 Ti capacity;
- q8 K/V, 8K context, 49/49 offload, and measured tok/s;
- the four prompts and their answers; and
- the legacy native pack's exact failure outputs.

They remain formalized as bound evidence. They simply do not become definitions of NHDF itself.

Likewise, the following v0.3 material remains in a CCD specialization rather than being inferred from Qwen:

- broad- and narrow-phase collision handling;
- conservative geometric bounds;
- time-of-impact certificates;
- manifolds and collision response;
- CCD backend matrices; and
- CCD-specific queue, subdivision, iteration, and contact limits.

## Promotion criteria for a written NHDF Edge Hybrid specification

The Qwen instance is already worth keeping. Turning its implementation pattern into a reusable normative NHDF sub-version should require:

1. a canonical manifest schema for `H_NHDF`;
2. explicit field types, units, required/optional fields, and canonical hashing rules;
3. a reference validator reproducing create, verify, gate, and run behavior;
4. mutation tests for payload, runtime, specification, evidence, and event records;
5. quality-negative controls using the rejected native pack;
6. resource-failure controls using insufficient VRAM/reserve conditions;
7. evidence showing that a validation state cannot be changed without matching gate results;
8. equal-budget comparison against a plain unsealed launcher, so the NHDF contribution is measured rather than assumed; and
9. a second non-Qwen or non-LLM external operator proving that the profile is truly general.

Until those steps are complete, the correct status is:

- **validated Qwen application artifact**;
- **retained NHDF hybrid sub-version candidate**; and
- **not yet a universally proven NHDF core replacement**.

## Evidence map

All paths are relative to this parent repository.

| Evidence | Path or revision | Meaning |
|---|---|---|
| NHDF base specification | [`NHDF_Formal_Specification_Tom_Klootwijk.pdf`](./NHDF_Formal_Specification_Tom_Klootwijk.pdf) | v0.1 identity, bounded causal state, failure exposure, telemetry, ablations, and separation of speculative AI claims. |
| NHDF v0.3 specification | [`NHDF_Formal_Specification_v0.3_General_Purpose_CCD_Tom_Klootwijk.pdf`](./NHDF_Formal_Specification_v0.3_General_Purpose_CCD_Tom_Klootwijk.pdf) | v0.1 baseline retention, core/profile/hypothesis boundary, CCD specialization, and the Edge AI warning that compression must be measured. |
| Historical hybrid manifest | [`NHDF_HYBRID_MANIFEST.json`](./NHDF_Edge_Qwen3_RTX5070Ti_12GB/nhdf_edge_qwen3/packs/qwen3-30b-a3b-nhdf-v03-iq2m/NHDF_HYBRID_MANIFEST.json) | Exact `nhdf-edge-hybrid-0.1` object, external codec attribution, finite execution/resource contract, and validated state. |
| Historical hybrid gate | [`functional_gate.json`](./NHDF_Edge_Qwen3_RTX5070Ti_12GB/nhdf_edge_qwen3/packs/qwen3-30b-a3b-nhdf-v03-iq2m/evidence/functional_gate.json) | 4/4 semantics, 8K allocation, 49/49 offload, 10,610 MiB peak, 1,617 MiB headroom, and short throughput. |
| Native pack integrity | [`full_pack.json`](./NHDF_Edge_Qwen3_RTX5070Ti_12GB/nhdf_edge_qwen3/metrics/local/full_pack.json) | Complete 531-record pack and passing CRC/parity. |
| Native pack rejection | [`default_pack_quality_gate.json`](./NHDF_Edge_Qwen3_RTX5070Ti_12GB/nhdf_edge_qwen3/metrics/local/default_pack_quality_gate.json) | Integrity-passing pack produced unusable language output. |
| First validating NHDF commit | `77b1ebf` | Initial functional NHDF hybrid on the 12 GB GPU. |
| Optimized NHDF commit | `37f3435` | Final NHDF-only runtime optimization and retained 8K evidence used here. |
| Later lineage boundary | `cc4f8b2` and after | First executable UGTOMS substrate implementation; not used as an explanation of the NHDF experiment. |

The model payload itself remains excluded from Git because it is multi-gigabyte. The small manifest binds its byte count and SHA-256 without duplicating it.

## Final retained statement

> The NHDF Qwen result was a hybrid deployment success, not an NHDF-native tensor-compression success. External IQ2_M supplied the 6.186684x stored-weight reduction; Qwen's MoE sparsity and the pinned llama.cpp/CUDA runtime supplied practical inference. The NHDF-labeled hybrid implementation supplied a compact reference, explicit attribution, bounded resource contract, evidence-driven validation state, and fail-closed execution. The rejected native pack proves that these are not cosmetic distinctions. The faulty step was treating the whole v0.3 CCD expansion as a replacement base and as the cause of the Qwen result. The proposed correction is to retain v0.1 as the base, retain CCD as a specialization, and extract `nhdf-edge-hybrid-0.1` as an evidence-backed NHDF sub-version candidate worthy of further formalization.

That is the precise demonstrated achievement of the NHDF-labeled hybrid: **it bound one declared external mechanism, measured it, and refused unsupported use without confusing the codec's output with NHDF authorship.** The next step is to prove that this pattern is safely replaceable and reusable beyond the Qwen/IQ2_M instance.
