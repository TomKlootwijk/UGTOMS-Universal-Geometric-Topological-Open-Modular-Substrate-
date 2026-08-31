# Validation protocol

## Metrics contract

Every reproducible run should save:

```text
model revision, pack format, config hash, CUDA/PyTorch/Transformers versions,
device name, driver, power mode, prompt/context, batch, residual fraction,
actual pack bytes, free/peak VRAM, TTFT, prefill tok/s, decode tok/s, parity
rate, CRC failures, saturation/overflow counts, quality metrics and seed
```

## Correctness thresholds

- Bit-pack round trip: exact.
- Local zero-set residual: `max(abs(F_i)) <= 1e-5` for CPU conversion.
- Stored parity: exact for uncorrupted payloads.
- CRC32: every file must match before model load.
- CUDA kernel versus dequantized FP16: set an absolute/relative tolerance after
  observing FP16 accumulation error; do not silently loosen it per tensor.
- CPU versus GPU route selection: exact indices for deterministic inputs.

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

## Performance suite

- Prompt lengths: 32, 512, 2048, 8192 tokens.
- Generation lengths: 32 and 256 tokens.
- Batch: 1 (primary) and 2 if memory permits.
- Power profiles: minimum, balanced and maximum TGP exposed by the laptop.
- At least five measured runs after warm-up.
- Report median, p10/p90, not just maximum throughput.

## Falsification rule

The feasibility claim fails for the default profile if any of these hold:

- actual pack plus required workspace cannot load with a safe VRAM margin;
- steady-state generation is unstable or repeatedly triggers driver recovery;
- quality loss exceeds the predeclared deployment threshold;
- the NHDF branch format is not better than a simpler equal-size baseline;
- measured decode falls below the minimum useful threshold defined by the
  intended application.
