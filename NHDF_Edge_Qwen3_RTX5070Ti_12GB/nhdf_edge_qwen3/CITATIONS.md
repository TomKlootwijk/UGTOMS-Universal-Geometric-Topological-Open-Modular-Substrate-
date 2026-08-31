# Sources and traceability

## Source specification

Tom Klootwijk. *Non-Euclidean Holographic Data Fields: A Formal Operator
Specification for a Parity-Conditioned Kinematic Foliation and Implicit
Holographic Co-Processor*. Version 0.1, 31 August 2026. The supplied PDF is
copied to `sources/NHDF_Formal_Specification_v0.1.pdf`.

Key traceability:

- Canonical operator chain and feedback: specification Sections 4.2-4.3.
- Non-degenerate local zero-set interpretation: Sections 2.2 and 5.1.
- Data versus topology parity, and parity limits: Sections 5.3 and 7.1.
- Bounded branch pool and causal timeline: Sections 6.2 and 8.3.
- Semantics-first GPU realization and memory budget: Section 12.
- Required ablations, telemetry and failure modes: Sections 15-17.

## Model and implementation facts

- Qwen Team. `Qwen/Qwen3-30B-A3B-Instruct-2507` model card, config and file tree.
  https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507
- Qwen Team. Official GPTQ-Int4 checkpoint tree.
  https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4
- Hugging Face Transformers. `modeling_qwen3_moe.py`, including the 3-D expert
  tensor layout and expert forward contract.
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
- NVIDIA. GeForce RTX 50 Series Laptop GPU specifications.
  https://www.nvidia.com/en-us/geforce/laptops/50-series/
- NVIDIA. CUDA 12.8 release notes / Blackwell support.
  https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/

## Low-bit research used for feasibility context

- Vage Egiazarian et al. *Extreme Compression of Large Language Models via
  Additive Quantization*. arXiv:2401.06118.
  https://arxiv.org/abs/2401.06118
- Albert Tseng et al. *QuIP#: Even Better LLM Quantization with Hadamard
  Incoherence and Lattice Codebooks*. arXiv:2402.04396.
  https://arxiv.org/abs/2402.04396
- Young Jin Kim, Raffy Fahim, Hany Hassan Awadalla. *Mixture of Quantized
  Experts*. arXiv:2310.02410.
  https://arxiv.org/abs/2310.02410
- Elise Frantar and Dan Alistarh et al. *Fast Inference of Mixture-of-Experts
  Language Models with Offloading*. arXiv:2312.17238.
  https://arxiv.org/abs/2312.17238

These papers support the plausibility of extreme low-bit and expert-selective
quantization. They do not validate this NHDF pack, this exact Qwen checkpoint,
or the projected RTX 5070 Ti Laptop throughput.
