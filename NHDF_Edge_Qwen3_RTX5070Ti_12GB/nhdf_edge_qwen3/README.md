# UGTOMS/NHDF Edge: functional Qwen3-30B-A3B on 12 GB

This repository now has a **validated, functional NHDF v0.3 hybrid artifact**
for the complete 30,532,122,624-parameter
`Qwen/Qwen3-30B-A3B-Instruct-2507` model on an NVIDIA GeForce RTX 5070 Ti
Laptop GPU.

The result is deliberately split into two responsibilities:

- **GGUF/IQ2_M is the tensor codec.** The payload is the externally produced
  Bartowski/ggml IQ2_M artifact, executed by llama.cpp b6014. NHDF does not
  claim authorship of that codec.
- **NHDF v0.3 is the substrate.** It seals provenance, declares the capability
  and resource policy, maintains a hash-linked evidence chain, records typed
  validation state, and refuses launch when integrity, quality, or resource
  gates are not satisfied.

The hybrid artifact references the existing GGUF in place. It does not create
a second 9.87 GB copy.

## Current measured outcome

The original BF16 checkpoint contains 61,064,245,248 tensor bytes. The hybrid
payload is 9,870,270,464 bytes, or 2.5862 file-size-derived bits per parameter.
That is a 6.1867x reduction (51,193,974,784 fewer bytes, or 83.84%). BF16
weights alone would require 58,235 MiB, 4.76x the GPU's reported capacity,
before runtime buffers or KV cache.

| Measurement | Fresh result |
|---|---:|
| Complete model | 30,532,122,624 parameters |
| BF16 source tensor bytes | 61,064,245,248 |
| Referenced IQ2_M payload | 9,870,270,464 bytes |
| Functional prompts | 4/4 passed |
| llama.cpp layer offload | 49/49 |
| Physical GPU capacity | 12,227 MiB |
| Peak device use with allocated 8K context | 10,487 MiB |
| Measured device headroom | 1,740 MiB |
| 64-token prompt processing, 3 samples | 458.525658 tok/s |
| 64-token generation, 3 samples | 102.367894 tok/s |

The throughput rows are 64-token `llama-bench` microbenchmarks, not a claim
about every interactive workload. This is also a mixture-of-experts model:
all 30.532B parameters must be stored, but about 3.3B are active per token.
Its generation rate therefore must not be described as dense-30B throughput.

The functional suite produced exact `OK`, exact `323`, a coherent explanation
of integrity versus output quality, and a correct `is_even(n)` Python function.
The allocated-8K run also returned exact `OK`. llama.cpp offloaded all 49
reported layers while also reporting a 127.51 MiB CPU-mapped model buffer.

The 8K measurement establishes allocation, residency, and short-prompt
execution with an 8K q8 K/V cache. It is **not** evidence for answer quality
after filling the context with 8K tokens.

## Run the validated artifact

```powershell
# Full verification rehashes the referenced 9.87 GB payload and every sealed
# runtime/evidence component.
nhdf-edge verify packs\qwen3-30b-a3b-nhdf-v03-iq2m

# Launch is fail-closed and accepts a validated hybrid by default.
nhdf-edge run packs\qwen3-30b-a3b-nhdf-v03-iq2m `
  --prompt "Reply with exactly the single word OK." `
  --context 512 `
  --max-new-tokens 8 `
  --text-only
```

For repeated use after a trusted full verification, `verify --quick` and
`run --quick` skip rehashing the large payload but still verify the sealed
manifest and smaller components. They are convenience modes, not substitutes
for the full integrity gate after the payload changes or moves.

## Create and gate the hybrid

`create-hybrid` writes only the small NHDF manifest and references the existing
model payload by a workspace-relative path.

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

A new hybrid begins as `UNCALIBRATED`. `gate-hybrid` performs fresh integrity,
functional-output, allocated-8K residency, full-offload, VRAM-reserve, and
throughput gates. Only a complete pass promotes it to `VALIDATED`; a failure is
recorded as a failed disposition and launch remains closed.

The supported validation states are:

- `UNCALIBRATED`
- `QUALITY_FAILED`
- `RESOURCE_FAILED`
- `VALIDATED`

The default runtime verifies the sealed artifact, requires `VALIDATED`, checks
the target GPU and free-memory contract, uses q8 K/V plus Flash Attention, and
applies the pinned explicit non-thinking Qwen ChatML template. The
`--allow-unvalidated` option exists only for explicit research experiments.

## What failed in the native-codec research

The legacy `nhdf-edge-0.1` scalar codec is not the codec used by the validated
hybrid. Its full 9,152,386,624-byte tensor pack loaded and executed on CUDA,
but generated repeated newline tokens or `10000000`. Its manifest is therefore
sealed as `QUALITY_FAILED`, and the native loader refuses it by default even
though all 531 CRC and parity checks pass.

Measured packed-versus-BF16 layer-0 expert errors were:

| Expert | Output NRMSE | Cosine |
|---:|---:|---:|
| 0 | 0.4813 | 0.8805 |
| 17 | 0.4749 | 0.8935 |
| 127 | 0.4482 | 0.9170 |

A subsequent bounded GEMQ-style 3-bit GPTQ experiment was numerically much
better on two isolated layer-0 experts:

| Expert | Output NRMSE | Cosine | Improvement over equal-byte RTN |
|---:|---:|---:|---:|
| 0 | 0.2542 | 0.9671 | 2.91% |
| 17 | 0.1859 | 0.9830 | 3.91% |

Those experts passed the absolute distortion thresholds but missed the
predeclared 20% comparative-advantage requirement. No full checkpoint was
packed from this replacement experiment. It remains native-codec research,
not the current functional artifact.

## How v0.3 is used

Tom Klootwijk's *NHDF Formal Specification v0.3, General-Purpose CCD* is not a
language-model quantization specification and does not make dense weights
disappear. This project uses its Edge-AI/resource principles as an operational
substrate:

- SHA-256-sealed local source, codec, runtime, specification, and evidence records;
- bounded GPU and context allocation;
- typed capability and validation state;
- a hash-linked event/evidence chain;
- measurable quality, memory, and throughput gates;
- fail-closed execution;
- a replaceable codec boundary.

CCD collision, time-of-impact, and contact-manifold operators are not
relabeled as language-model weight operators. The validated claim is about the
NHDF substrate governing a functional external codec, not about an
NHDF-native compression advantage.

## Evidence and limitations

Primary evidence is stored in:

- `packs/qwen3-30b-a3b-nhdf-v03-iq2m/NHDF_HYBRID_MANIFEST.json`
- `packs/qwen3-30b-a3b-nhdf-v03-iq2m/evidence/functional_gate.json`
- `metrics/local/gguf_backend_ops.json`
- `VALIDATION_STATUS.md`

Established locally:

- exact model, runtime, specification, and evidence hashes;
- zero-copy payload reference with a sealed path, size, and SHA-256;
- complete-model functional generation through the NHDF hybrid launch path;
- 49/49 layer offload and measured fit on the 12,227 MiB GPU;
- allocated-8K residency with 1,740 MiB measured headroom;
- fresh three-sample prompt and generation throughput;
- fail-closed launch for unvalidated, modified, or resource-incompatible
  artifacts.

Not established:

- broad benchmark accuracy or perplexity relative to BF16;
- quality with an actually filled 8K-token prompt;
- sustained thermals, power, or long-duration stability;
- an NHDF-native tensor-codec advantage;
- a self-contained or relocatable artifact (the zero-copy manifest is workspace-bound);
- production suitability outside the measured hardware/runtime scope.

## Install and inspect

```powershell
python -m pip install -e ".[dev,runtime]"
python -m pytest -q
nhdf-edge doctor --config configs\qwen3_30b_a3b_edge12.yaml
```

## Repository map

```text
configs/                  target and native research profiles
csrc/                     experimental native CUDA decode
metrics/local/            target-machine evidence
models/                   exact BF16 source and verified IQ2_M payload
packs/                    validated hybrid and rejected native pack
scripts/                  download, calibration, and benchmark tools
sources/                  supplied NHDF specification lineage and hashes
src/nhdf_edge/            native and hybrid formats, CLI, gates, and runtimes
tests/                    semantic, integrity, policy, and loader tests
tools/                    pinned llama.cpp b6014 runtime and provenance
```

## References

- Tom Klootwijk, *NHDF Formal Specification v0.3: General-Purpose CCD*,
  included under `sources/`.
- Qwen Team, `Qwen/Qwen3-30B-A3B-Instruct-2507`.
- Bartowski, `Qwen_Qwen3-30B-A3B-Instruct-2507-IQ2_M.gguf`, immutable source
  revision and hash in `CONTROL_SOURCE.json`.
- ggml-org, llama.cpp b6014, MIT license.
- Deng et al., GEMQ, ICML 2026, MIT implementation.
