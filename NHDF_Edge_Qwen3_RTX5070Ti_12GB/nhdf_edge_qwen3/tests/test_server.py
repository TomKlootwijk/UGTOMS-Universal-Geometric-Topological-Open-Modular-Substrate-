from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from nhdf_edge import cli, server


@pytest.fixture(autouse=True)
def _isolate_tests_from_host_cuda(monkeypatch, tmp_path) -> None:
    """Unit fixtures must not hash or require the host's 806 MiB CUDA set."""

    cuda_bin = tmp_path / "host-cuda" / "bin"
    cuda_bin.mkdir(parents=True)
    monkeypatch.setattr(server.hybrid, "_windows_cuda_bin_directory", lambda: cuda_bin)
    monkeypatch.setattr(server.hybrid, "_cuda_execution_file_specs", lambda: ())


class _Process:
    def __init__(self) -> None:
        self.pid = 4242
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self.wait_calls: list[float] = []

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float) -> int:
        self.wait_calls.append(timeout)
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode


def _manifest(root: Path, *, server_entrypoint: bool = True) -> dict[str, Any]:
    runtime = root / "runtime" / "llama-server.exe"
    model = root / "model.gguf"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"server")
    model.write_bytes(b"model")
    runtime_section: dict[str, Any] = {
        "revision": "b6014",
        "files": [{"path": "runtime/llama-server.exe", "bytes": 6, "sha256": "sealed"}],
    }
    if server_entrypoint:
        runtime_section["server_entrypoint"] = "runtime/llama-server.exe"
    return {
        "runtime": runtime_section,
        "payload": {"path": "model.gguf"},
        "execution_profile": {
            "gpu_layers": 999,
            "expected_offloaded_layers": [49, 49],
            "maximum_context_tokens": 8192,
            "kv_cache_k": "q8_0",
            "kv_cache_v": "q8_0",
            "flash_attention": True,
            "sampling": {"temperature": 0.0, "top_k": 1, "seed": 2026},
        },
        "resource_contract": {"maximum_validated_context_tokens": 8192},
    }


def _patch_valid_artifact(monkeypatch, tmp_path, *, manifest=None):
    root = tmp_path / "artifact"
    root.mkdir(exist_ok=True)
    value = _manifest(root) if manifest is None else manifest
    observed: dict[str, Any] = {}

    def fake_verify(path, **kwargs):
        observed["verify"] = (Path(path), kwargs)
        return {"ok": True, "deployment_loadable": True, "failures": []}

    def fake_preflight(passed_manifest, **kwargs):
        observed["preflight"] = (passed_manifest, kwargs)
        return {"gpu": "RTX", "free_mib": 11_000}

    process = _Process()

    def fake_popen(command, **kwargs):
        observed["popen"] = (command, kwargs)
        return process

    def fake_environment(**kwargs):
        observed["environment_request"] = kwargs
        return {"PATH": "trusted-test-path"}

    class FakeGuard:
        def __init__(self, _specs) -> None:
            self.active = False

        def __enter__(self):
            self.active = True
            return self

        def close(self) -> None:
            self.active = False

    monkeypatch.setattr(server.hybrid, "verify_hybrid_artifact", fake_verify)
    monkeypatch.setattr(
        server.hybrid,
        "load_hybrid_manifest_snapshot",
        lambda _path, **_kwargs: (value, "a" * 64),
    )
    monkeypatch.setattr(server.hybrid, "_ExecutionFileGuard", FakeGuard)
    monkeypatch.setattr(server.hybrid, "_execution_file_specs", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(server.hybrid, "_preflight", fake_preflight)
    monkeypatch.setattr(
        server.hybrid, "_minimal_subprocess_environment", fake_environment
    )
    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)
    monkeypatch.setenv("HTTP_PROXY", "http://attacker.invalid")
    return root, value, process, observed


def _approved_artifact(root: Path) -> tuple[server.ArtifactApproval, dict[str, Any]]:
    manifest = _manifest(root)
    runtime = root / "runtime" / "llama-server.exe"
    model = root / "model.gguf"
    runtime_digest = hashlib.sha256(runtime.read_bytes()).hexdigest()
    model_digest = hashlib.sha256(model.read_bytes()).hexdigest()
    manifest["format"] = server.hybrid.HYBRID_FORMAT
    manifest["artifact_kind"] = "external-codec-reference"
    manifest["validation"] = {"status": "VALIDATED"}
    manifest["events"] = []
    manifest["runtime"]["files"] = [
        {
            "path": "runtime/llama-server.exe",
            "bytes": runtime.stat().st_size,
            "sha256": runtime_digest,
        }
    ]
    manifest["payload"] = {
        "path": "model.gguf",
        "bytes": model.stat().st_size,
        "sha256": model_digest,
    }
    rendered = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_digest = hashlib.sha256(rendered).hexdigest()
    (root / server.hybrid.HYBRID_MANIFEST).write_bytes(rendered)
    (root / server.hybrid.HYBRID_MANIFEST_SHA256).write_text(
        manifest_digest + "\n", encoding="ascii"
    )
    runtime_file = server.ApprovedArtifactFile(
        runtime, runtime.stat().st_size, runtime_digest
    )
    approval = server.ArtifactApproval(
        artifact_dir=root,
        reference_root=root,
        manifest_sha256=manifest_digest,
        payload=server.ApprovedArtifactFile(
            model, model.stat().st_size, model_digest
        ),
        runtime_files=(runtime_file,),
        server=runtime_file,
    )
    return approval, manifest


def test_start_verifies_once_then_launches_fixed_resident_profile(monkeypatch, tmp_path) -> None:
    root, manifest, process, observed = _patch_valid_artifact(monkeypatch, tmp_path)
    runtime = server.HybridServer(
        root, port=18081, threads=20, allow_self_sealed=True
    ).start(wait_ready=False)

    assert runtime.is_running
    assert observed["verify"] == (
        root.resolve(),
        {"verify_payload_hash": True, "require_validated": True},
    )
    assert observed["preflight"] == (manifest, {"context": 8192})
    command, popen_kwargs = observed["popen"]
    assert command[0] == str((root / "runtime" / "llama-server.exe").resolve())
    assert command[command.index("-ngl") + 1] == "999"
    assert command[command.index("-sm") + 1] == "none"
    assert command[command.index("-c") + 1] == "8192"
    assert command[command.index("-ctk") + 1] == "q8_0"
    assert command[command.index("-ctv") + 1] == "q8_0"
    assert command[command.index("-np") + 1] == "1"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "18081"
    assert command[command.index("--temp") + 1] == "0"
    assert command[command.index("--top-k") + 1] == "1"
    assert command[command.index("--seed") + 1] == "2026"
    assert command[command.index("--alias") + 1] == "local-qwen3-30b-a3b"
    assert "--jinja" in command
    assert command[command.index("--reasoning-format") + 1] == "none"
    assert command[command.index("--cors-origins") + 1] == "localhost"
    assert command[command.index("--prio") + 1] == "2"
    assert command[command.index("--prio-batch") + 1] == "2"
    assert command[command.index("--poll") + 1] == "50"
    assert "-fa" in command
    assert "--no-webui" in command
    assert popen_kwargs["shell"] is False
    assert popen_kwargs["cwd"] == str((root / "runtime").resolve())
    assert "HTTP_PROXY" not in popen_kwargs["env"]
    assert "AWS_SECRET_ACCESS_KEY" not in popen_kwargs["env"]
    assert observed["environment_request"] == {
        "executable_directory": (root / "runtime").resolve(),
        "include_cuda": True,
    }
    assert runtime.preflight_result == {"gpu": "RTX", "free_mib": 11_000}
    assert process.terminated is False
    runtime.stop()


def test_self_sealed_resident_execution_is_rejected_by_default(tmp_path) -> None:
    with pytest.raises(server.HybridServerConfigurationError, match="self-sealed"):
        server.HybridServer(tmp_path).start(wait_ready=False)


def test_external_manifest_digest_authorizes_generic_resident_execution(
    monkeypatch, tmp_path
) -> None:
    root, _manifest_value, _process, observed = _patch_valid_artifact(
        monkeypatch, tmp_path
    )
    runtime = server.HybridServer(
        root,
        expected_manifest_sha256="a" * 64,
    ).start(wait_ready=False)
    assert "popen" in observed
    runtime.stop()


def test_externally_approved_start_uses_exact_snapshot_and_rehashes_before_popen(
    monkeypatch, tmp_path
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    approval, manifest = _approved_artifact(root)
    observed: dict[str, Any] = {}
    process = _Process()

    def fake_preflight(value, **kwargs):
        observed["preflight"] = (value, kwargs)
        return {"gpu": "RTX"}

    def fake_popen(command, **kwargs):
        observed["popen"] = (command, kwargs)
        return process

    monkeypatch.setattr(server.hybrid, "_preflight", fake_preflight)
    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)

    runtime = server.HybridServer(root, artifact_approval=approval)
    runtime.start(wait_ready=False)

    command, popen_kwargs = observed["popen"]
    assert command[0] == str(approval.server.path)
    assert command[command.index("-m") + 1] == str(approval.payload.path)
    assert observed["preflight"][0] == manifest
    assert "HTTP_PROXY" not in popen_kwargs["env"]
    assert runtime._execution_guard is not None
    assert runtime._execution_guard.active
    runtime.stop()
    with pytest.raises(server.HybridServerConfigurationError, match="cannot skip"):
        server.HybridServer(
            root, artifact_approval=approval, verify_payload_hash=False
        )


def test_external_approval_adds_pinned_cuda_dependencies_to_guard(
    monkeypatch, tmp_path
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    approval, _manifest = _approved_artifact(root)
    dependency = tmp_path / "cuda" / "cudart64_12.dll"
    dependency.parent.mkdir()
    dependency.write_bytes(b"trusted-cuda")
    cuda_spec = server.hybrid._LockedFileSpec(
        dependency,
        dependency.stat().st_size,
        server.hybrid.sha256_file(dependency),
        "CUDA dependency",
    )
    monkeypatch.setattr(
        server.hybrid, "_cuda_execution_file_specs", lambda: (cuda_spec,)
    )

    specs = approval.execution_file_specs()

    assert cuda_spec in specs


@pytest.mark.skipif(os.name != "nt", reason="Win32 deny-write/delete lock semantics")
def test_windows_execution_handles_block_mutation_during_popen(
    monkeypatch, tmp_path
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    approval, _manifest = _approved_artifact(root)
    runtime_path = approval.server.path
    replacement = tmp_path / "replacement.exe"
    replacement.write_bytes(b"server")
    cuda_dependency = tmp_path / "cuda" / "cudart64_12.dll"
    cuda_dependency.parent.mkdir()
    cuda_dependency.write_bytes(b"trusted-cuda")
    cuda_spec = server.hybrid._LockedFileSpec(
        cuda_dependency,
        cuda_dependency.stat().st_size,
        server.hybrid.sha256_file(cuda_dependency),
        "CUDA dependency",
    )
    process = _Process()

    monkeypatch.setattr(server.hybrid, "_preflight", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        server.hybrid, "_cuda_execution_file_specs", lambda: (cuda_spec,)
    )

    def fake_popen(_command, **_kwargs):
        with pytest.raises(OSError):
            runtime_path.write_bytes(b"tamper")
        with pytest.raises(OSError):
            os.replace(replacement, runtime_path)
        with pytest.raises(OSError):
            cuda_dependency.write_bytes(b"tamper")
        return process

    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)
    runtime = server.HybridServer(root, artifact_approval=approval)
    runtime.start(wait_ready=False)
    runtime.stop()
    cuda_dependency.write_bytes(b"released")


def test_artifact_approval_rejects_any_out_of_root_execution_file(tmp_path) -> None:
    root = tmp_path / "project"
    artifact = root / "artifact"
    runtime = root / "runtime" / "llama-server.exe"
    outside = tmp_path / "outside.gguf"
    artifact.mkdir(parents=True)
    runtime.parent.mkdir()
    runtime.write_bytes(b"server")
    outside.write_bytes(b"model")
    runtime_file = server.ApprovedArtifactFile(
        runtime, 6, hashlib.sha256(b"server").hexdigest()
    )

    with pytest.raises(server.HybridServerConfigurationError, match="outside"):
        server.ArtifactApproval(
            artifact_dir=artifact,
            reference_root=root,
            manifest_sha256="1" * 64,
            payload=server.ApprovedArtifactFile(
                outside, 5, hashlib.sha256(b"model").hexdigest()
            ),
            runtime_files=(runtime_file,),
            server=runtime_file,
        )


def test_approved_manifest_reseal_or_file_mutation_fails_before_popen(
    monkeypatch, tmp_path
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    approval, manifest = _approved_artifact(root)
    original_manifest = (root / server.hybrid.HYBRID_MANIFEST).read_bytes()
    original_sidecar = (root / server.hybrid.HYBRID_MANIFEST_SHA256).read_bytes()
    monkeypatch.setattr(server.hybrid, "_preflight", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        server.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("mutated approval must not execute"),
    )

    manifest["validation"]["rewritten"] = True
    rendered = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (root / server.hybrid.HYBRID_MANIFEST).write_bytes(rendered)
    (root / server.hybrid.HYBRID_MANIFEST_SHA256).write_text(
        hashlib.sha256(rendered).hexdigest() + "\n", encoding="ascii"
    )
    with pytest.raises(OSError, match="externally approved"):
        server.HybridServer(root, artifact_approval=approval).start(wait_ready=False)

    (root / server.hybrid.HYBRID_MANIFEST).write_bytes(original_manifest)
    (root / server.hybrid.HYBRID_MANIFEST_SHA256).write_bytes(original_sidecar)
    (root / "runtime" / "llama-server.exe").write_bytes(b"tamper")
    with pytest.raises(server.HybridServerConfigurationError, match="SHA-256 changed"):
        server.HybridServer(root, artifact_approval=approval).start(wait_ready=False)


def test_approved_manifest_mutation_during_preflight_fails_before_popen(
    monkeypatch, tmp_path
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    approval, _manifest = _approved_artifact(root)
    manifest_path = root / server.hybrid.HYBRID_MANIFEST

    def mutate_during_preflight(*_args, **_kwargs):
        manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
        return {}

    monkeypatch.setattr(server.hybrid, "_preflight", mutate_during_preflight)
    monkeypatch.setattr(
        server.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("mutated approval must not execute"),
    )

    with pytest.raises(server.HybridServerConfigurationError, match="SHA-256 changed"):
        server.HybridServer(root, artifact_approval=approval).start(wait_ready=False)


def test_start_fails_closed_before_manifest_or_process_on_bad_integrity(
    monkeypatch, tmp_path
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    monkeypatch.setattr(
        server.hybrid,
        "verify_hybrid_artifact",
        lambda *_args, **_kwargs: {
            "ok": False,
            "deployment_loadable": False,
            "failures": [{"error": "SHA-256 mismatch"}],
        },
    )
    monkeypatch.setattr(
        server.hybrid,
        "load_hybrid_manifest",
        lambda *_args: pytest.fail("manifest must not load after failed verification"),
    )
    monkeypatch.setattr(
        server.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("process must not launch"),
    )

    with pytest.raises(OSError, match="SHA-256 mismatch"):
        server.HybridServer(root, allow_self_sealed=True).start(wait_ready=False)


def test_server_entrypoint_must_be_sealed(monkeypatch, tmp_path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    manifest = _manifest(root)
    unsealed = root / "runtime" / "other" / "llama-server.exe"
    unsealed.parent.mkdir()
    unsealed.write_bytes(b"unsealed")
    manifest["runtime"]["server_entrypoint"] = "runtime/other/llama-server.exe"
    _patch_valid_artifact(monkeypatch, tmp_path, manifest=manifest)

    with pytest.raises(server.HybridServerConfigurationError, match="not present"):
        server.HybridServer(root, allow_self_sealed=True).start(wait_ready=False)


def test_unique_sealed_llama_server_is_accepted_for_legacy_manifest(
    monkeypatch, tmp_path
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    manifest = _manifest(root, server_entrypoint=False)
    _, _, _, observed = _patch_valid_artifact(monkeypatch, tmp_path, manifest=manifest)

    runtime = server.HybridServer(root, allow_self_sealed=True).start(wait_ready=False)

    assert observed["popen"][0][0].endswith("llama-server.exe")
    runtime.stop()


@pytest.mark.parametrize("host", ["localhost", "0.0.0.0", "::1", "192.168.1.2"])
def test_only_ipv4_loopback_literal_is_allowed(tmp_path, host) -> None:
    with pytest.raises(server.HybridServerConfigurationError, match="loopback literal"):
        server.HybridServer(tmp_path, host=host)


def test_manifest_profile_rejects_unvalidated_cache_type(
    monkeypatch, tmp_path
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    manifest = _manifest(root)
    manifest["execution_profile"]["kv_cache_v"] = "f16"
    _, _, _, observed = _patch_valid_artifact(monkeypatch, tmp_path, manifest=manifest)

    with pytest.raises(server.HybridServerConfigurationError, match="validated KV cache"):
        server.HybridServer(root, allow_self_sealed=True).start(wait_ready=False)
    assert "popen" not in observed


def test_start_uses_manifest_32k_q4_profile_and_quick_verification(
    monkeypatch, tmp_path
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    manifest = _manifest(root)
    manifest["execution_profile"].update(
        maximum_context_tokens=32_768,
        kv_cache_k="q4_0",
        kv_cache_v="q4_0",
    )
    manifest["resource_contract"]["maximum_validated_context_tokens"] = 32_768
    _, _, _, observed = _patch_valid_artifact(
        monkeypatch, tmp_path, manifest=manifest
    )

    runtime = server.HybridServer(
        root, verify_payload_hash=False, allow_self_sealed=True
    ).start(wait_ready=False)

    assert observed["verify"] == (
        root.resolve(),
        {"verify_payload_hash": False, "require_validated": True},
    )
    assert observed["preflight"] == (manifest, {"context": 32_768})
    command = observed["popen"][0]
    assert command[command.index("-c") + 1] == "32768"
    assert command[command.index("-ctk") + 1] == "q4_0"
    assert command[command.index("-ctv") + 1] == "q4_0"
    assert runtime._maximum_context_tokens == 32_768
    runtime.stop()


@pytest.mark.parametrize(
    ("execution_context", "contract_context"),
    [(4_096, 4_096), (32_768, 8_192)],
)
def test_server_refuses_too_small_or_not_fully_validated_context(
    monkeypatch, tmp_path, execution_context, contract_context
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    manifest = _manifest(root)
    manifest["execution_profile"]["maximum_context_tokens"] = execution_context
    manifest["resource_contract"][
        "maximum_validated_context_tokens"
    ] = contract_context
    _, _, _, observed = _patch_valid_artifact(
        monkeypatch, tmp_path, manifest=manifest
    )

    with pytest.raises(server.HybridServerConfigurationError, match="context"):
        server.HybridServer(root, allow_self_sealed=True).start(wait_ready=False)
    assert "popen" not in observed


class _Response:
    def __init__(
        self,
        value: dict[str, Any],
        status: int = 200,
        final_url: str | None = None,
    ) -> None:
        self.status = status
        self._body = json.dumps(value).encode("utf-8")
        self._final_url = final_url

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return self._body

    def geturl(self) -> str:
        assert self._final_url is not None
        return self._final_url


def _bound_runtime(tmp_path: Path, *, port: int = 19090) -> server.HybridServer:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    runtime = server.HybridServer(tmp_path, port=port)
    runtime._process = _Process()
    runtime._model_path = model.resolve()
    runtime._maximum_context_tokens = 8_192
    runtime._manifest_snapshot = {
        "runtime": {"revision": "b6014", "build_number": 6014},
        "execution_profile": {
            "sampling": {"temperature": 0.0, "top_k": 1, "seed": 2026}
        },
    }
    return runtime


def _props(runtime: server.HybridServer) -> dict[str, Any]:
    return {
        "build_info": "b6014-b6014",
        "model_path": str(runtime._model_path),
        "total_slots": 1,
        "default_generation_settings": {
            "n_ctx": 8_192,
            "params": {"temperature": 0.0, "top_k": 1, "seed": 2026},
        },
    }


def test_health_and_completion_use_loopback_and_fixed_sampling(monkeypatch, tmp_path) -> None:
    runtime = _bound_runtime(tmp_path)
    requests = []

    class Opener:
        def open(self, request, *, timeout):
            requests.append((request, timeout))
            if request.full_url.endswith("/health"):
                value = {"status": "ok"}
            elif request.full_url.endswith("/props"):
                value = _props(runtime)
            else:
                value = {"content": "answer<|im_end|>", "tokens_predicted": 1}
            return _Response(value, final_url=request.full_url)

    observed_handlers = []

    def fake_build_opener(*handlers):
        observed_handlers.extend(handlers)
        return Opener()

    monkeypatch.setenv("HTTP_PROXY", "http://attacker.invalid:8080")
    monkeypatch.setattr(server, "build_opener", fake_build_opener)
    monkeypatch.setattr(server, "_listener_owner_pids", lambda _port: {4242})

    assert runtime.health() == {"status": "ok"}
    result = runtime.completion("Write Python.", max_tokens=32)

    assert result["generated_text"] == "answer"
    assert result["sampling"] == {"temperature": 0.0, "top_k": 1, "seed": 2026}
    request, timeout = requests[-1]
    assert request.full_url == "http://127.0.0.1:19090/completion"
    assert timeout == runtime.request_timeout_seconds
    payload = json.loads(request.data)
    assert payload == {
        "prompt": server.hybrid._chatml("Write Python."),
        "n_predict": 32,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 2026,
        "cache_prompt": False,
        "reasoning_format": "none",
        "stream": False,
        "stop": ["<|im_end|>"],
    }
    assert result["cache_prompt"] is False
    proxy_handlers = [
        item for item in observed_handlers if isinstance(item, server.ProxyHandler)
    ]
    assert proxy_handlers and all(item.proxies == {} for item in proxy_handlers)


def test_unhealthy_or_malformed_completion_fails_closed(monkeypatch, tmp_path) -> None:
    runtime = _bound_runtime(tmp_path)
    monkeypatch.setattr(server, "_listener_owner_pids", lambda _port: {4242})

    class UnhealthyOpener:
        def open(self, request, **_kwargs):
            return _Response({"status": "loading"}, final_url=request.full_url)

    monkeypatch.setattr(server, "build_opener", lambda *_args: UnhealthyOpener())
    with pytest.raises(server.HybridServerUnavailableError, match="not ok"):
        runtime.health()

    class MalformedOpener:
        def open(self, request, **_kwargs):
            if request.full_url.endswith("/health"):
                value = {"status": "ok"}
            elif request.full_url.endswith("/props"):
                value = _props(runtime)
            else:
                value = {"tokens": 3}
            return _Response(value, final_url=request.full_url)

    monkeypatch.setattr(server, "build_opener", lambda *_args: MalformedOpener())
    with pytest.raises(server.HybridServerUnavailableError, match="text content"):
        runtime.completion("prompt")


def test_health_rejects_redirect_wrong_pid_and_wrong_model(monkeypatch, tmp_path) -> None:
    runtime = _bound_runtime(tmp_path)

    class RedirectOpener:
        def open(self, request, **_kwargs):
            return _Response(
                {"status": "ok"},
                final_url="http://attacker.invalid/health",
            )

    monkeypatch.setattr(server, "build_opener", lambda *_args: RedirectOpener())
    with pytest.raises(server.HybridServerUnavailableError, match="redirected"):
        runtime.health()

    class BoundOpener:
        def open(self, request, **_kwargs):
            value = {"status": "ok"} if request.full_url.endswith("/health") else _props(runtime)
            return _Response(value, final_url=request.full_url)

    monkeypatch.setattr(server, "build_opener", lambda *_args: BoundOpener())
    monkeypatch.setattr(server, "_listener_owner_pids", lambda _port: {9999})
    with pytest.raises(server.HybridServerUnavailableError, match="child PID"):
        runtime.health()

    monkeypatch.setattr(server, "_listener_owner_pids", lambda _port: {4242})

    class WrongModelOpener:
        def open(self, request, **_kwargs):
            if request.full_url.endswith("/health"):
                value = {"status": "ok"}
            else:
                value = _props(runtime)
                wrong = tmp_path / "wrong.gguf"
                wrong.write_bytes(b"wrong")
                value["model_path"] = str(wrong.resolve())
            return _Response(value, final_url=request.full_url)

    monkeypatch.setattr(server, "build_opener", lambda *_args: WrongModelOpener())
    with pytest.raises(server.HybridServerUnavailableError, match="model mismatch"):
        runtime.health()


def test_stop_terminates_owned_process(monkeypatch, tmp_path) -> None:
    runtime = server.HybridServer(tmp_path)
    process = _Process()
    runtime._process = process

    runtime.stop(timeout_seconds=3.0)

    assert process.terminated
    assert process.wait_calls == [3.0]
    assert not runtime.is_running


def test_stop_escalates_after_timeout(monkeypatch, tmp_path) -> None:
    runtime = server.HybridServer(tmp_path)

    class StubbornProcess(_Process):
        def wait(self, timeout: float) -> int:
            self.wait_calls.append(timeout)
            if not self.killed:
                raise subprocess.TimeoutExpired("llama-server", timeout)
            return -9

    process = StubbornProcess()
    runtime._process = process

    runtime.stop(timeout_seconds=0.5)

    assert process.terminated
    assert process.killed
    assert process.wait_calls == [0.5, 0.5]


def test_failed_shutdown_retains_process_and_file_guard(tmp_path) -> None:
    runtime = server.HybridServer(tmp_path)

    class FailingProcess(_Process):
        def terminate(self) -> None:
            raise OSError("access denied")

    class Guard:
        active = True

        def close(self) -> None:
            self.active = False

    process = FailingProcess()
    guard = Guard()
    runtime._process = process
    runtime._execution_guard = guard

    with pytest.raises(OSError, match="access denied"):
        runtime.stop()
    assert runtime._process is process
    assert runtime._execution_guard is guard
    assert guard.active is True


def test_cli_preserves_default_profile_and_exposes_large_context_quick_mode() -> None:
    parser = cli.build_parser()
    defaults = parser.parse_args(
        ["create-hybrid", "artifact", "--model", "model", "--runtime", "runtime"]
    )
    configured = parser.parse_args(
        [
            "create-hybrid",
            "artifact",
            "--model",
            "model",
            "--runtime",
            "runtime",
            "--maximum-context",
            "32768",
            "--kv-cache-k",
            "q4_0",
            "--kv-cache-v",
            "q4_0",
        ]
    )
    serve = parser.parse_args(["serve", "artifact", "--quick"])

    assert defaults.maximum_context == 8_192
    assert defaults.kv_cache_k == defaults.kv_cache_v == "q8_0"
    assert configured.maximum_context == 32_768
    assert configured.kv_cache_k == configured.kv_cache_v == "q4_0"
    assert serve.quick is True
    assert serve.manifest_sha256 is None
    secured = parser.parse_args(
        ["serve", "artifact", "--manifest-sha256", "a" * 64]
    )
    assert secured.manifest_sha256 == "a" * 64
