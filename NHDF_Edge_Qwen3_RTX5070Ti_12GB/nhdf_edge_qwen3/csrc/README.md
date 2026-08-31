# Optional CUDA decode kernel

`nhdf_gemv.cu` implements the `Scone -> Pi` runtime mapping as a fused packed
weight decode plus GEMV. It reads 2-bit or 4-bit base codes, applies the local
FP16 mean/scale, looks up the bounded one-bit residual branch with a 32-group
mask and prefix count, and accumulates directly into the projection output.

It is intentionally a clear batch-one reference kernel, not a vendor-tuned
production kernel. It does not use unsupported PTX or claim that Tensor Cores
execute parity/topology. Optimizing prefill requires a packed GEMM or a
chunked-dequantize/cuBLAS path; the included kernel targets autoregressive
decode.

Build on the target machine:

```bash
# Use the architecture spelling supported by the installed PyTorch/CUDA pair.
# With a current Blackwell-capable toolchain this is commonly 12.0.
export TORCH_CUDA_ARCH_LIST="12.0"
python setup_cuda.py build_ext --inplace
```

Run `nhdf-edge doctor` after building. The project cannot compile this extension
on a system without an NVIDIA CUDA toolkit, so the ZIP's CI tests cover the CPU
semantic path and the target machine must perform the GPU equivalence test.
