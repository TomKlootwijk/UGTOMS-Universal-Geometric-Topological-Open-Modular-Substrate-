from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nhdf_edge import hybrid


def _artifact(tmp_path, **profile):
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
        **profile,
    )
    return root, model


def test_hybrid_create_is_zero_copy_and_verifies(tmp_path) -> None:
    root, model = _artifact(tmp_path)
    manifest = hybrid.load_hybrid_manifest(root)

    assert manifest["validation"]["status"] == "UNCALIBRATED"
    assert manifest["weight_codec"]["nhdf_native_codec"] is False
    assert manifest["payload"]["link_mode"] == "workspace-relative-reference"
    assert manifest["execution_profile"]["maximum_context_tokens"] == 8_192
    assert manifest["execution_profile"]["kv_cache_k"] == "q8_0"
    assert manifest["execution_profile"]["kv_cache_v"] == "q8_0"
    assert not (root / model.name).exists()
    result = hybrid.verify_hybrid_artifact(root)
    assert result["ok"]
    assert not result["deployment_loadable"]


def test_hybrid_create_accepts_32k_q4_profile_and_rejects_invalid_values(
    tmp_path,
) -> None:
    root, _ = _artifact(
        tmp_path,
        maximum_context_tokens=32_768,
        kv_cache_k="q4_0",
        kv_cache_v="q4_0",
    )
    manifest = hybrid.load_hybrid_manifest(root)

    assert manifest["execution_profile"]["maximum_context_tokens"] == 32_768
    assert manifest["resource_contract"]["maximum_validated_context_tokens"] == 32_768
    assert manifest["execution_profile"]["kv_cache_k"] == "q4_0"
    assert manifest["execution_profile"]["kv_cache_v"] == "q4_0"

    components = tmp_path / "components"
    common = {
        "model": components / "model.gguf",
        "runtime": components / "llama-cli.exe",
        "total_parameters": 100,
    }
    with pytest.raises(ValueError, match="positive integer"):
        hybrid.create_hybrid_artifact(
            tmp_path / "bad-context", maximum_context_tokens=0, **common
        )
    with pytest.raises(ValueError, match="validated types"):
        hybrid.create_hybrid_artifact(
            tmp_path / "bad-cache", kv_cache_v="f16", **common
        )


def test_hybrid_create_seals_optional_server_entrypoint(tmp_path) -> None:
    root, model = _artifact(tmp_path)
    components = tmp_path / "components"
    server = components / "llama-server.exe"
    server.write_bytes(b"server")

    hybrid.create_hybrid_artifact(
        root,
        model=model,
        runtime=components / "llama-cli.exe",
        benchmark_runtime=components / "llama-bench.exe",
        server_runtime=server,
        total_parameters=100,
        target_vram_mib=12_227,
    )
    manifest = hybrid.load_hybrid_manifest(root)

    assert manifest["runtime"]["server_entrypoint"].endswith("llama-server.exe")
    sealed = {Path(record["path"]).name for record in manifest["runtime"]["files"]}
    assert "llama-server.exe" in sealed
    assert hybrid.verify_hybrid_artifact(root)["ok"]


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
                "runtime_revision": hybrid.load_hybrid_manifest(root)["runtime"][
                    "revision"
                ],
                "runtime_build_number": hybrid.load_hybrid_manifest(root)["runtime"][
                    "build_number"
                ],
                "runtime_argument_profile": hybrid.load_hybrid_manifest(root)[
                    "runtime"
                ]["argument_profile"],
                "execution_profile_sha256": hybrid._execution_profile_sha256(
                    hybrid.load_hybrid_manifest(root)
                ),
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

    stale = json.loads(passed.read_text(encoding="utf-8"))
    stale["runtime_revision"] = "different-runtime"
    stale_path = tmp_path / "stale-runtime.json"
    stale_path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime_revision"):
        hybrid.set_hybrid_validation(root, "VALIDATED", evidence_path=stale_path)


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
                "runtime_revision": manifest["runtime"]["revision"],
                "runtime_build_number": manifest["runtime"]["build_number"],
                "runtime_argument_profile": manifest["runtime"]["argument_profile"],
                "execution_profile_sha256": hybrid._execution_profile_sha256(manifest),
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
    assert result["resource_monitoring_enabled"] is False
    assert result["samples"] == 0
    command = observed["command"]
    assert command[command.index("-ngl") + 1] == "999"
    assert command[command.index("-ctk") + 1] == "q8_0"
    assert command[command.index("-ctv") + 1] == "q8_0"
    assert "--temp" in command and command[command.index("--temp") + 1] == "0"


def test_current_runtime_requests_trace_logs_only_for_monitored_evidence(
    tmp_path: Path, monkeypatch
):
    root, _ = _artifact(tmp_path)
    manifest = hybrid.load_hybrid_manifest(root)
    manifest["runtime"]["argument_profile"] = "current-2026"
    hybrid._write_manifest(root, manifest)
    evidence = root / "evidence" / "gate.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text(
        json.dumps(
            {
                "experiment": "nhdf_hybrid_full_model_functional_gate",
                "artifact_format": hybrid.HYBRID_FORMAT,
                "passed": True,
                "payload": manifest["payload"],
                "runtime_revision": manifest["runtime"]["revision"],
                "runtime_build_number": manifest["runtime"]["build_number"],
                "runtime_argument_profile": manifest["runtime"]["argument_profile"],
                "execution_profile_sha256": hybrid._execution_profile_sha256(manifest),
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
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="OK\n", stderr="")

    monkeypatch.setattr(hybrid.subprocess, "run", fake_run)
    for monitored in (False, True):
        hybrid.run_hybrid_prompt(
            root,
            prompt="Reply with OK.",
            acceptance_rule={"kind": "exact", "value": "OK"},
            verify_payload_hash=False,
            monitor_resources=monitored,
        )

    assert "-lv" not in commands[0]
    assert commands[1][commands[1].index("-lv") + 1] == "4"


def test_llama_bench_uses_manifest_kv_cache_types(tmp_path, monkeypatch) -> None:
    root, _ = _artifact(
        tmp_path,
        maximum_context_tokens=32_768,
        kv_cache_k="q4_0",
        kv_cache_v="q4_0",
    )
    manifest = hybrid.load_hybrid_manifest(root)
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                [
                    {
                        "n_prompt": 64,
                        "n_gen": 0,
                        "avg_ts": 900.0,
                        "stddev_ts": 1.0,
                        "samples_ts": [900.0],
                    },
                    {
                        "n_prompt": 0,
                        "n_gen": 64,
                        "avg_ts": 150.0,
                        "stddev_ts": 1.0,
                        "samples_ts": [150.0],
                    },
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(hybrid.subprocess, "run", fake_run)
    hybrid._run_benchmark(
        root,
        manifest,
        repetitions=1,
        prompt_tokens=64,
        generation_tokens=64,
    )

    command = observed["command"]
    assert command[command.index("-ctk") + 1] == "q4_0"
    assert command[command.index("-ctv") + 1] == "q4_0"


def test_gate_evidence_names_manifest_selected_context_generically(
    tmp_path, monkeypatch
) -> None:
    root, _ = _artifact(
        tmp_path,
        maximum_context_tokens=32_768,
        kv_cache_k="q4_0",
        kv_cache_v="q4_0",
    )
    observed_contexts = []

    def fake_prompt(_root, *, context, **_kwargs):
        observed_contexts.append(context)
        return {
            "passed": True,
            "context_tokens": context,
            "peak_gpu_memory_mib": 10_000,
            "resource_preflight": {"total_mib": 12_227},
            "llama_metrics": {"offloaded_layers": [49, 49]},
        }

    monkeypatch.setattr(hybrid, "run_hybrid_prompt", fake_prompt)
    monkeypatch.setattr(
        hybrid,
        "_run_benchmark",
        lambda *_args, **_kwargs: {
            "prompt": {"average_tokens_per_second": 900.0},
            "generation": {"average_tokens_per_second": 150.0},
        },
    )

    evidence = hybrid.gate_hybrid_artifact(root)

    assert observed_contexts == [512, 512, 512, 512, 32_768]
    assert evidence["allocated_context_residency_result"]["context_tokens"] == 32_768
    assert evidence["thresholds"]["allocated_context_tokens"] == 32_768
    assert evidence["aggregate"]["allocated_context_tokens"] == 32_768
    assert evidence["aggregate"]["allocated_context_passed"] is True
    assert "allocated_8k_passed" not in evidence["aggregate"]
    assert hybrid.load_hybrid_manifest(root)["validation"]["status"] == "VALIDATED"

    stale_context = json.loads(json.dumps(evidence))
    stale_context["aggregate"]["allocated_context_tokens"] = 8_192
    stale_path = tmp_path / "stale-context.json"
    stale_path.write_text(json.dumps(stale_context), encoding="utf-8")
    with pytest.raises(ValueError, match="allocated_context"):
        hybrid.set_hybrid_validation(
            root, "VALIDATED", evidence_path=stale_path
        )
