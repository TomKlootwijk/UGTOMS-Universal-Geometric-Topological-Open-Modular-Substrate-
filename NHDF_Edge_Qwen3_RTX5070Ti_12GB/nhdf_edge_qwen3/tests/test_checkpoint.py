from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from nhdf_edge.checkpoint import pack_checkpoint
from nhdf_edge.config import NHDFConfig
from nhdf_edge.format import PackReader, PackWriter
from nhdf_edge.quantize import QuantizationPolicy, dequantize_tensor, quantize_tensor


def test_interrupted_pack_resumes_from_verified_tensor_state(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "pack"
    source.mkdir()
    save_file(
        {
            "model.a.weight": torch.randn(4, 256, generator=torch.Generator().manual_seed(71)),
            "model.b.weight": torch.randn(4, 256, generator=torch.Generator().manual_seed(72)),
        },
        str(source / "model.safetensors"),
    )

    original_add = PackWriter.add
    calls = 0

    def interrupt_second_add(self, packed):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original_add(self, packed)

    monkeypatch.setattr(PackWriter, "add", interrupt_second_add)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        pack_checkpoint(source, output, NHDFConfig())

    state = json.loads((output / "pack_state.json").read_text(encoding="utf-8"))
    assert state["writer"]["stats"]["tensors"] == 1

    monkeypatch.setattr(PackWriter, "add", original_add)
    manifest_path = pack_checkpoint(source, output, NHDFConfig())

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["packed_tensor_count"] == 2
    assert not manifest["partial_pack"]
    assert not (output / "pack_state.json").exists()
    reader = PackReader(output)
    assert reader.names() == ["model.a.weight", "model.b.weight"]
    assert reader.verify_all()["ok"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        pack_checkpoint(source, output, NHDFConfig())


def test_pack_fuses_released_qwen_expert_layout_in_numeric_order(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "pack"
    source.mkdir()
    experts, hidden, intermediate = 11, 5, 3
    components: dict[str, torch.Tensor] = {}
    for expert in range(experts):
        prefix = f"model.layers.0.mlp.experts.{expert}"
        components[f"{prefix}.gate_proj.weight"] = torch.full(
            (intermediate, hidden), float(10 * expert + 1)
        )
        components[f"{prefix}.up_proj.weight"] = torch.full(
            (intermediate, hidden), float(10 * expert + 2)
        )
        components[f"{prefix}.down_proj.weight"] = torch.full(
            (hidden, intermediate), float(10 * expert + 3)
        )

    first_names = [name for name in components if int(name.split(".experts.")[1].split(".")[0]) < 5]
    second_names = [name for name in components if name not in first_names]
    save_file({name: components[name] for name in first_names}, str(source / "model-00001-of-00002.safetensors"))
    save_file({name: components[name] for name in second_names}, str(source / "model-00002-of-00002.safetensors"))
    weight_map = {
        **{name: "model-00001-of-00002.safetensors" for name in first_names},
        **{name: "model-00002-of-00002.safetensors" for name in second_names},
    }
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    (source / "config.json").write_text(
        json.dumps(
            {
                "num_experts": experts,
                "num_hidden_layers": 1,
                "decoder_sparse_step": 1,
                "hidden_size": hidden,
                "moe_intermediate_size": intermediate,
            }
        ),
        encoding="utf-8",
    )

    # Keep this mapping test lossless: the quantizer itself has separate
    # numerical tests, while this test must prove ordering and orientation.
    def raw_quantize(tensor, _policy, *, name, hessian_diag=None):
        del hessian_diag
        return quantize_tensor(tensor, QuantizationPolicy(mode="raw"), name=name)

    monkeypatch.setattr("nhdf_edge.checkpoint.quantize_tensor", raw_quantize)

    manifest_path = pack_checkpoint(source, output, NHDFConfig())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reader = PackReader(output)
    gate_up_name = "model.layers.0.mlp.experts.gate_up_proj"
    down_name = "model.layers.0.mlp.experts.down_proj"

    assert reader.names() == [down_name, gate_up_name]
    assert manifest["source_tensor_count"] == 3 * experts
    assert manifest["logical_tensor_count"] == 2
    assert manifest["packed_tensor_count"] == 2
    assert manifest["source_layout_transform"] == "qwen2_moe_individual_experts_to_stacked"
    assert not manifest["partial_pack"]

    expected_gate_up = torch.stack(
        [
            torch.cat(
                (
                    components[f"model.layers.0.mlp.experts.{expert}.gate_proj.weight"],
                    components[f"model.layers.0.mlp.experts.{expert}.up_proj.weight"],
                )
            )
            for expert in range(experts)
        ]
    )
    expected_down = torch.stack(
        [components[f"model.layers.0.mlp.experts.{expert}.down_proj.weight"] for expert in range(experts)]
    )
    actual_gate_up = dequantize_tensor(reader.load(gate_up_name))
    actual_down = dequantize_tensor(reader.load(down_name))
    assert actual_gate_up.shape == (experts, 2 * intermediate, hidden)
    assert actual_down.shape == (experts, hidden, intermediate)
    assert torch.equal(actual_gate_up, expected_gate_up)
    assert torch.equal(actual_down, expected_down)
