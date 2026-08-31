from __future__ import annotations

import torch
from torch.nn import functional as F

from nhdf_edge.quantize import QuantizationPolicy, dequantize_tensor, quantize_tensor
from nhdf_edge.runtime.modules import NHDFPackedEmbedding, NHDFPackedLinear, NHDFQwen3Experts


def test_packed_linear_matches_dequantized_reference() -> None:
    weight = torch.randn(20, 512, generator=torch.Generator().manual_seed(31))
    packed = quantize_tensor(
        weight,
        QuantizationPolicy(base_bits=4, group_size=256, residual_fraction=0.0),
        name="linear.weight",
    )
    module = NHDFPackedLinear(packed)
    x = torch.randn(3, 512, generator=torch.Generator().manual_seed(32))
    expected = F.linear(x, dequantize_tensor(packed))
    actual = module(x)
    assert torch.allclose(actual, expected)


def test_packed_embedding_matches_selected_rows() -> None:
    weight = torch.randn(40, 256, generator=torch.Generator().manual_seed(33))
    packed = quantize_tensor(
        weight,
        QuantizationPolicy(base_bits=4, group_size=256, residual_fraction=0.0),
        name="embed.weight",
    )
    module = NHDFPackedEmbedding(packed)
    ids = torch.tensor([[1, 5, 1], [39, 0, 5]])
    expected = dequantize_tensor(packed, dtype=torch.float16)[ids]
    assert torch.equal(module(ids), expected)


def test_qwen3_expert_adapter_matches_manual_packed_math() -> None:
    num_experts = 2
    hidden = 256
    intermediate = 256
    g = torch.Generator().manual_seed(34)
    gate_up_weight = torch.randn(num_experts, 2 * intermediate, hidden, generator=g)
    down_weight = torch.randn(num_experts, hidden, intermediate, generator=g)
    policy = QuantizationPolicy(base_bits=2, group_size=256, residual_fraction=0.2)
    gate_up = quantize_tensor(gate_up_weight, policy, name="experts.gate_up_proj")
    down = quantize_tensor(down_weight, policy, name="experts.down_proj")
    module = NHDFQwen3Experts(
        gate_up,
        down,
        num_experts=num_experts,
        hidden_dim=hidden,
        intermediate_dim=intermediate,
    )

    x = torch.randn(3, hidden, generator=g)
    top_idx = torch.tensor([[0, 1], [1, 0], [0, 1]])
    top_w = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]])
    actual = module(x, top_idx, top_w)

    gu = dequantize_tensor(gate_up)
    dw = dequantize_tensor(down)
    expected = torch.zeros_like(x)
    for token in range(x.shape[0]):
        for k in range(top_idx.shape[1]):
            expert = int(top_idx[token, k])
            gate, up = F.linear(x[token], gu[expert]).chunk(2)
            y = F.linear(F.silu(gate) * up, dw[expert])
            expected[token] += top_w[token, k] * y
    # The adapter batches tokens by expert while this deliberately simple
    # reference evaluates one GEMV at a time.  BLAS may reassociate those FP32
    # reductions, so allow the observed sub-2e-3 accumulation difference.
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=5e-4)
