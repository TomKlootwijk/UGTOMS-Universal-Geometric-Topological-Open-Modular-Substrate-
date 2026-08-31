from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from nhdf_edge import hybrid


@pytest.fixture(autouse=True)
def _isolate_tests_from_host_cuda(monkeypatch, tmp_path) -> None:
    """Unit fixtures must not hash or require the host's 806 MiB CUDA set."""

    cuda_bin = tmp_path / "host-cuda" / "bin"
    cuda_bin.mkdir(parents=True)
    monkeypatch.setattr(hybrid, "_windows_cuda_bin_directory", lambda: cuda_bin)
    monkeypatch.setattr(hybrid, "_cuda_execution_file_specs", lambda: ())


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


def test_cuda_dependency_identity_records_are_exact_and_read_only() -> None:
    assert hybrid.WINDOWS_CUDA_VERSION == "12.8"
    assert dict(hybrid.WINDOWS_CUDA_DEPENDENCY_RECORDS) == {
        "cudart64_12.dll": (
            573_952,
            "9d9b868955149875ac4ef43442aaaa8913d4a7b3e4d6dd60ee871626aa045768",
        ),
        "cublas64_12.dll": (
            113_712_640,
            "0a60a7ecbf906f7f3842826fecaf412f1587ff6269729339245a1ec224364161",
        ),
        "cublasLt64_12.dll": (
            692_441_600,
            "a21ddfc1c9cd090ab07d2c6aad235aa4f15fe60fe896d5db28adf7a279c09ef3",
        ),
    }
    with pytest.raises(TypeError):
        hybrid.WINDOWS_CUDA_DEPENDENCY_RECORDS["attacker.dll"] = (1, "0" * 64)  # type: ignore[index]


@pytest.mark.skipif(os.name != "nt", reason="Windows CUDA preflight")
def test_public_cuda_preflight_verifies_exact_dependency_set(
    tmp_path, monkeypatch
) -> None:
    cuda_bin = tmp_path / "CUDA" / "v12.8" / "bin"
    cuda_bin.mkdir(parents=True)
    dependency = cuda_bin / "cudart64_12.dll"
    dependency.write_bytes(b"trusted-cuda")
    spec = hybrid._LockedFileSpec(
        dependency,
        dependency.stat().st_size,
        hybrid.sha256_file(dependency),
        "test CUDA dependency",
    )
    monkeypatch.setattr(hybrid, "_windows_cuda_bin_directory", lambda: cuda_bin)
    monkeypatch.setattr(hybrid, "_cuda_execution_file_specs", lambda: (spec,))

    result = hybrid.preflight_windows_cuda_dependencies()

    assert result == hybrid.WindowsCudaDependencyPreflight(
        version="12.8",
        binary_directory=cuda_bin,
        verified_dependency_count=1,
        dependency_names=("cudart64_12.dll",),
    )
    with pytest.raises(AttributeError):
        result.version = "attacker"  # type: ignore[misc]

    bad_spec = hybrid._LockedFileSpec(
        dependency,
        dependency.stat().st_size,
        "0" * 64,
        "tampered CUDA dependency",
    )
    monkeypatch.setattr(hybrid, "_cuda_execution_file_specs", lambda: (bad_spec,))
    with pytest.raises(OSError, match="SHA-256 changed"):
        hybrid.preflight_windows_cuda_dependencies()


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

    def fake_environment(**kwargs):
        observed["environment_request"] = kwargs
        return {"PATH": "trusted-test-path"}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
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

    monkeypatch.setattr(hybrid, "_minimal_subprocess_environment", fake_environment)
    monkeypatch.setattr(hybrid.subprocess, "run", fake_run)
    result = hybrid.run_hybrid_prompt(
        root,
        prompt="Reply with OK.",
        max_tokens=8,
        context=512,
        acceptance_rule={"kind": "exact", "value": "OK"},
        verify_payload_hash=False,
        allow_self_sealed=True,
    )

    assert result["passed"]
    assert result["resource_monitoring_enabled"] is False
    assert result["samples"] == 0
    command = observed["command"]
    assert command[command.index("-ngl") + 1] == "999"
    assert command[command.index("-ctk") + 1] == "q8_0"
    assert command[command.index("-ctv") + 1] == "q8_0"
    assert "--temp" in command and command[command.index("--temp") + 1] == "0"
    assert observed["kwargs"]["cwd"] == str((tmp_path / "components").resolve())
    assert observed["environment_request"] == {
        "executable_directory": (tmp_path / "components").resolve(),
        "include_cuda": True,
    }


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
            allow_self_sealed=True,
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
    manifest, manifest_sha256 = hybrid.load_hybrid_manifest_snapshot(root)
    observed = {}
    dependency = tmp_path / "cuda" / "cudart64_12.dll"
    dependency.parent.mkdir()
    dependency.write_bytes(b"trusted-cuda")
    cuda_spec = hybrid._LockedFileSpec(
        dependency,
        dependency.stat().st_size,
        hybrid.sha256_file(dependency),
        "CUDA dependency",
    )
    monkeypatch.setattr(
        hybrid, "_cuda_execution_file_specs", lambda: (cuda_spec,)
    )

    def fake_environment(**kwargs):
        observed["environment_request"] = kwargs
        return {"PATH": "trusted-test-path"}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        if os.name == "nt":
            with pytest.raises(OSError):
                dependency.write_bytes(b"tamper")
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

    monkeypatch.setattr(hybrid, "_minimal_subprocess_environment", fake_environment)
    monkeypatch.setattr(hybrid.subprocess, "run", fake_run)
    specs = hybrid._execution_file_specs(
        root,
        manifest,
        manifest_sha256,
        required_entrypoints=("benchmark_entrypoint",),
    )
    with hybrid._ExecutionFileGuard(specs) as execution_guard:
        hybrid._run_benchmark(
            root,
            manifest,
            repetitions=1,
            prompt_tokens=64,
            generation_tokens=64,
            execution_guard=execution_guard,
        )

    command = observed["command"]
    assert command[command.index("-ctk") + 1] == "q4_0"
    assert command[command.index("-ctv") + 1] == "q4_0"
    assert observed["environment_request"] == {
        "executable_directory": (tmp_path / "components").resolve(),
        "include_cuda": True,
    }
    assert observed["kwargs"]["env"] == {"PATH": "trusted-test-path"}
    dependency.write_bytes(b"released")


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

    evidence = hybrid.gate_hybrid_artifact(root, allow_self_sealed=True)

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


def test_public_execution_and_gate_reject_self_sealed_authority_by_default(
    tmp_path,
) -> None:
    root, _ = _artifact(tmp_path)
    with pytest.raises(RuntimeError, match="self-sealed"):
        hybrid.run_hybrid_prompt(
            root,
            prompt="test",
            allow_unvalidated=True,
        )
    with pytest.raises(RuntimeError, match="self-sealed"):
        hybrid.gate_hybrid_artifact(root)


def test_final_guard_catches_runtime_mutation_after_preliminary_verification(
    tmp_path, monkeypatch
) -> None:
    root, _ = _artifact(tmp_path)
    manifest = hybrid.load_hybrid_manifest(root)
    runtime = hybrid._resolve_reference(root, manifest["runtime"]["entrypoint"])
    original = runtime.read_bytes()

    def mutate_during_preflight(_manifest, *, context):
        assert context == 512
        runtime.write_bytes(b"X" * len(original))
        return {"used_mib": 0}

    monkeypatch.setattr(hybrid, "_preflight", mutate_during_preflight)
    monkeypatch.setattr(
        hybrid.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("mutated runtime must not execute"),
    )
    with pytest.raises(OSError, match="SHA-256 changed"):
        hybrid.run_hybrid_prompt(
            root,
            prompt="test",
            allow_unvalidated=True,
            allow_self_sealed=True,
            verify_payload_hash=False,
        )


def test_gpu_queries_use_absolute_trusted_binary_and_minimal_environment(
    tmp_path, monkeypatch
) -> None:
    executable = tmp_path / "nvidia-smi.exe"
    executable.write_bytes(b"binary")
    observed = []

    def fake_run(command, **kwargs):
        observed.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="Expected GPU\n")

    monkeypatch.setenv("HTTPS_PROXY", "http://attacker.invalid")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("PROGRAMFILES", str(tmp_path / "attacker-program-files"))
    monkeypatch.setattr(hybrid, "_trusted_system_executable", lambda _name: executable)
    monkeypatch.setattr(hybrid.subprocess, "run", fake_run)

    assert hybrid._gpu_name() == "Expected GPU"
    command, kwargs = observed[0]
    assert Path(command[0]).is_absolute()
    assert command[0] == str(executable)
    assert kwargs["cwd"] == str(executable.parent)
    assert kwargs["timeout"] == 10
    assert "HTTPS_PROXY" not in kwargs["env"]
    assert "AWS_SECRET_ACCESS_KEY" not in kwargs["env"]
    if os.name == "nt":
        assert kwargs["env"]["PROGRAMFILES"] != str(
            tmp_path / "attacker-program-files"
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows Known Folder API")
def test_program_files_resolver_ignores_inherited_environment(
    tmp_path, monkeypatch
) -> None:
    attacker = tmp_path / "attacker-program-files"
    attacker.mkdir()
    monkeypatch.setenv("PROGRAMFILES", str(attacker))

    resolved = hybrid._windows_program_files_directory()

    assert resolved != attacker.resolve()
    assert resolved.is_dir()


@pytest.mark.skipif(os.name != "nt", reason="Windows environment contract")
def test_windows_environment_uses_api_program_files_and_pinned_cuda_only(
    tmp_path, monkeypatch
) -> None:
    windows = tmp_path / "Windows"
    system = windows / "System32"
    program_files = tmp_path / "Trusted Program Files"
    cuda_bin = program_files / "NVIDIA" / "CUDA" / "v12.8" / "bin"
    executable_directory = tmp_path / "runtime"
    for directory in (system, program_files, cuda_bin, executable_directory):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PROGRAMFILES", str(tmp_path / "attacker-program-files"))
    monkeypatch.setenv("PATH", str(tmp_path / "attacker-path"))
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setattr(hybrid, "_windows_system_directory", lambda: system)
    monkeypatch.setattr(
        hybrid, "_windows_program_files_directory", lambda: program_files
    )
    monkeypatch.setattr(hybrid, "_windows_cuda_bin_directory", lambda: cuda_bin)

    environment = hybrid._minimal_subprocess_environment(
        executable_directory=executable_directory,
        include_cuda=True,
    )

    assert environment["PROGRAMFILES"] == str(program_files)
    assert environment["PATH"].split(os.pathsep) == [
        str(executable_directory.resolve()),
        str(cuda_bin),
        str(system),
        str(windows),
    ]
    assert environment["NoDefaultCurrentDirectoryInExePath"] == "1"
    assert "AWS_SECRET_ACCESS_KEY" not in environment
    assert str(tmp_path / "attacker-path") not in environment["PATH"]


def test_execution_file_set_includes_pinned_cuda_dependencies(
    tmp_path, monkeypatch
) -> None:
    root, _ = _artifact(tmp_path)
    manifest, manifest_sha256 = hybrid.load_hybrid_manifest_snapshot(root)
    dependency = tmp_path / "cuda" / "cudart64_12.dll"
    dependency.parent.mkdir()
    dependency.write_bytes(b"trusted-cuda")
    cuda_spec = hybrid._LockedFileSpec(
        dependency,
        dependency.stat().st_size,
        hybrid.sha256_file(dependency),
        "CUDA dependency",
    )
    monkeypatch.setattr(
        hybrid, "_cuda_execution_file_specs", lambda: (cuda_spec,)
    )

    specs = hybrid._execution_file_specs(
        root,
        manifest,
        manifest_sha256,
        required_entrypoints=("entrypoint", "benchmark_entrypoint"),
    )

    assert cuda_spec in specs
    with hybrid._ExecutionFileGuard(specs):
        pass


@pytest.mark.skipif(os.name != "nt", reason="Win32 deny-write/delete lock semantics")
def test_one_shot_execution_locks_cuda_dependencies_for_child_lifetime(
    tmp_path, monkeypatch
) -> None:
    root, _ = _artifact(tmp_path)
    dependency = tmp_path / "cuda" / "cudart64_12.dll"
    dependency.parent.mkdir()
    dependency.write_bytes(b"trusted-cuda")
    cuda_spec = hybrid._LockedFileSpec(
        dependency,
        dependency.stat().st_size,
        hybrid.sha256_file(dependency),
        "CUDA dependency",
    )
    monkeypatch.setattr(
        hybrid, "_cuda_execution_file_specs", lambda: (cuda_spec,)
    )
    monkeypatch.setattr(
        hybrid, "_preflight", lambda *_args, **_kwargs: {"used_mib": 0}
    )
    monkeypatch.setattr(
        hybrid,
        "_minimal_subprocess_environment",
        lambda **_kwargs: {"PATH": "trusted-test-path"},
    )

    def fake_run(_command, **_kwargs):
        with pytest.raises(OSError):
            dependency.write_bytes(b"tamper")
        return SimpleNamespace(returncode=0, stdout="OK\n", stderr="")

    monkeypatch.setattr(hybrid.subprocess, "run", fake_run)

    result = hybrid.run_hybrid_prompt(
        root,
        prompt="test",
        allow_unvalidated=True,
        allow_self_sealed=True,
        verify_payload_hash=False,
    )

    assert result["exit_code"] == 0
    dependency.write_bytes(b"released")


def test_gate_evidence_rewrites_absolute_host_paths(
    tmp_path, monkeypatch
) -> None:
    root, model = _artifact(tmp_path)

    def fake_prompt(_root, *, context, **_kwargs):
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
            "runtime_records": [{"model_filename": str(model.resolve())}],
            "prompt": {"average_tokens_per_second": 900.0},
            "generation": {"average_tokens_per_second": 150.0},
        },
    )
    evidence = hybrid.gate_hybrid_artifact(root, allow_self_sealed=True)

    assert evidence["artifact"] == "."
    model_label = evidence["benchmark"]["runtime_records"][0]["model_filename"]
    assert model_label == hybrid._relative_reference(root, model)
    assert str(tmp_path.resolve()) not in json.dumps(evidence)
