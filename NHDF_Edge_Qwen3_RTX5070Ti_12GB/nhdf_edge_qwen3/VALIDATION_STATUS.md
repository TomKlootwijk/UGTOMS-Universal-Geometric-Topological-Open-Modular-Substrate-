# Validation status

**Build date:** 31 August 2026  
**Artifact version:** 0.1.0  
**Status:** research prototype; CPU semantics validated, target-GPU execution pending.

## Completed in the build environment

- `pytest`: **41 passed**, with one expected warning indicating that the test used the CPU/PyTorch reconstruction path rather than the production CUDA kernel.
- Python source compilation: all package, script, test, and report-generator modules compiled successfully.
- Deterministic semantic smoke test: parity check passed for all groups; maximum weighted local zero-set residual was `3.73e-08`; the selected one-bit residual branches reduced synthetic reconstruction MSE by `10.42%` versus the 2-bit base in that test.
- Analytical sizing: default 8K profile projects `9.228 GB` packed weights and `11.281 GB` total nominal VRAM, leaving `0.719 GB` against a decimal 12 GB capacity.
- Report preflight: the PDF opens, is unencrypted and tagged, contains 19 Letter-size pages, and has extractable text.

## Not completed in this build environment

- The CUDA extension was **not compiled or executed**, because this container has no compatible NVIDIA laptop GPU/toolchain exposed.
- The Qwen checkpoint was **not downloaded or converted**; no model weights are included.
- End-to-end perplexity, benchmark quality, time-to-first-token, prefill speed, decode speed, power, thermals, and real peak-VRAM measurements remain to be run on the target laptop.
- The analytical token-rate table is a bandwidth-sensitivity model, not a benchmark.

## Acceptance gates on the RTX 5070 Ti Laptop target

A successful claim requires: pack CRC/parity verification; CPU/CUDA kernel equivalence; no remaining meta parameters in the model loader; quality comparison with BF16, official GPTQ-Int4, plain 2-bit and plain 4-bit baselines; measured 8K peak VRAM below the actually free device budget; and sustained decoding without thermal or driver instability. See `docs/VALIDATION_PROTOCOL.md` for the full matrix.
