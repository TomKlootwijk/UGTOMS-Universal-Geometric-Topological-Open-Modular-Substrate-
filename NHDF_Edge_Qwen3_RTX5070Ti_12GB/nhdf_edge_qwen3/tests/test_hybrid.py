from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from nhdf_edge import hybrid


def _artifact(tmp_path):
    components = tmp_path / "components"
    components.mkdir()
    model = components / "model.gguf"
    runtime = components / "llama-cli.exe"
    benchmark = components / "llama-bench.exe"
    specification = components / "spec.pdf"
    source = components / "source.json"
    assurance = components / "backend.json"
    model.write_bytes(b"model-payload")
    runtime.write_bytes(b"runtime")
    benchmark.write_bytes(b"benchmark")
    specification.write_bytes(b"specification")
    source.write_text('{"revision": "pinned"}', encoding="utf-8")
    assurance.write_text('{"passed": true}', encoding="utf-8")
    root = tmp_path / "artifact"
    hybrid.create_hybrid_artifact(
        root,
        model=model,
        runtime=runtime,
        benchmark_runtime=benchmark,
        specification=specification,
        source_record=source,
        assurance_evidence=[assurance],
        total_parameters=100,
        target_vram_mib=12_227,
    )
    return root, model


def test_hybrid_create_is_zero_copy_and_verifies(tmp_path) -> None:
    root, model = _artifact(tmp_path)
    manifest = hybrid.load_hybrid_manifest(root)

    assert manifest["validation"]["status"] == "UNCALIBRATED"
    assert manifest["weight_codec"]["nhdf_native_codec"] is False
    assert manifest["payload"]["link_mode"] == "workspace-relative-reference"
    assert not (root / model.name).exists()
    result = hybrid.verify_hybrid_artifact(root)
    assert result["ok"]
    assert not result["deployment_loadable"]


def test_hybrid_payload_tampering_fails_integrity(tmp_path) -> None:
    root, model = _artifact(tmp_path)
    model.write_bytes(model.read_bytes() + b"tamper")

    result = hybrid.verify_hybrid_artifact(root)
    assert not result["ok"]
    assert result["failures"][0]["error"] == "byte length mismatch"


def test_hybrid_event_chain_tampering_fails_integrity(tmp_path) -> None:
    root, _ = _artifact(tmp_path)
    manifest_path = root / hybrid.HYBRID_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["events"][0]["payload"]["codec"] = "tampered"
    hybrid._write_manifest(root, manifest)

    result = hybrid.verify_hybrid_artifact(root)
    assert not result["ok"]
    assert next(item for item in result["checks"] if item["component"] == "event_chain")[
        "error"
    ] == "event 0 hash mismatch"


def test_hybrid_validation_is_evidence_gated_and_fail_closed(tmp_path) -> None:
    root, _ = _artifact(tmp_path)
    failed = tmp_path / "failed.json"
    failed.write_text('{"passed": false}', encoding="utf-8")
    with pytest.raises(ValueError, match="complete gate evidence"):
        hybrid.set_hybrid_validation(root, "VALIDATED", evidence_path=failed)
    with pytest.raises(RuntimeError, match="UNCALIBRATED"):
        hybrid.require_hybrid_validated(root)

    passed = tmp_path / "passed.json"
    passed.write_text(
        json.dumps(
            {
                "experiment": "nhdf_hybrid_full_model_functional_gate",
                "artifact_format": hybrid.HYBRID_FORMAT,
                "passed": True,
                "payload": hybrid.load_hybrid_manifest(root)["payload"],
                "aggregate": {
                    "functional_prompts_passed": 4,
                    "functional_prompts_total": 4,
                    "allocated_8k_passed": True,
                    "full_offload_passed": True,
                    "peak_gpu_memory_mib": 10_487,
                    "resource_gate_passed": True,
                    "throughput_gate_passed": True,
                },
                "benchmark": {"generation": {"average_tokens_per_second": 101.7}},
                "thresholds": {"minimum_generation_tokens_per_second": 80.0},
            }
        ),
        encoding="utf-8",
    )
    hybrid.set_hybrid_validation(root, "VALIDATED", evidence_path=passed)

    hybrid.require_hybrid_validated(root)
    result = hybrid.verify_hybrid_artifact(root, require_validated=True)
    assert result["ok"]
    assert result["deployment_loadable"]


def test_hybrid_run_uses_fixed_profile(monkeypatch, tmp_path) -> None:
    root, _ = _artifact(tmp_path)
    evidence = tmp_path / "passed.json"
    manifest = hybrid.load_hybrid_manifest(root)
    evidence.write_text(
        json.dumps(
            {
                "experiment": "nhdf_hybrid_full_model_functional_gate",
                "artifact_format": hybrid.HYBRID_FORMAT,
                "passed": True,
                "payload": manifest["payload"],
                "aggregate": {
                    "functional_prompts_passed": 4,
                    "functional_prompts_total": 4,
                    "allocated_8k_passed": True,
                    "full_offload_passed": True,
                    "resource_gate_passed": True,
                    "throughput_gate_passed": True,
                },
                "benchmark": {
                    "generation": {"average_tokens_per_second": 101.7}
                },
                "thresholds": {"minimum_generation_tokens_per_second": 80.0},
            }
        ),
        encoding="utf-8",
    )
    hybrid.set_hybrid_validation(root, "VALIDATED", evidence_path=evidence)
    monkeypatch.setattr(hybrid, "_gpu_sample", lambda: (12_227, 287, 0))
    monkeypatch.setattr(
        hybrid, "_gpu_name", lambda: "NVIDIA GeForce RTX 5070 Ti Laptop GPU"
    )
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        return SimpleNamespace(
            returncode=0,
            stdout="OK\n",
            stderr=(
                "load_tensors: offloaded 49/49 layers to GPU\n"
                "load_tensors: CUDA0 model buffer size = 9279.83 MiB\n"
                "load_tensors: CPU_Mapped model buffer size = 127.51 MiB\n"
                "llama_kv_cache: CUDA0 KV buffer size = 25.50 MiB\n"
                "llama_context: CUDA0 compute buffer size = 300.75 MiB\n"
                "prompt eval time = 1.00 ms / 1 tokens (100.00 tokens per second)\n"
                "eval time = 1.00 ms / 1 runs (100.00 tokens per second)\n"
                "load time = 1.00 ms\n"
                "total time = 2.00 ms\n"
            ),
        )

    monkeypatch.setattr(hybrid.subprocess, "run", fake_run)
    result = hybrid.run_hybrid_prompt(
        root,
        prompt="Reply with OK.",
        max_tokens=8,
        context=512,
        acceptance_rule={"kind": "exact", "value": "OK"},
        verify_payload_hash=False,
    )

    assert result["passed"]
    command = observed["command"]
    assert command[command.index("-ngl") + 1] == "999"
    assert command[command.index("-ctk") + 1] == "q8_0"
    assert command[command.index("-ctv") + 1] == "q8_0"
    assert "--temp" in command and command[command.index("--temp") + 1] == "0"
