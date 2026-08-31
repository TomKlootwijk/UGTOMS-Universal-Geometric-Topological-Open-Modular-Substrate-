from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from nhdf_edge import cli, server


class _Process:
    def __init__(self) -> None:
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

    monkeypatch.setattr(server.hybrid, "verify_hybrid_artifact", fake_verify)
    monkeypatch.setattr(server.hybrid, "load_hybrid_manifest", lambda _path: value)
    monkeypatch.setattr(server.hybrid, "_preflight", fake_preflight)
    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)
    return root, value, process, observed


def test_start_verifies_once_then_launches_fixed_resident_profile(monkeypatch, tmp_path) -> None:
    root, manifest, process, observed = _patch_valid_artifact(monkeypatch, tmp_path)
    runtime = server.HybridServer(root, port=18081, threads=20).start(wait_ready=False)

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
    assert runtime.preflight_result == {"gpu": "RTX", "free_mib": 11_000}
    assert process.terminated is False


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
        server.HybridServer(root).start(wait_ready=False)


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
        server.HybridServer(root).start(wait_ready=False)


def test_unique_sealed_llama_server_is_accepted_for_legacy_manifest(
    monkeypatch, tmp_path
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    manifest = _manifest(root, server_entrypoint=False)
    _, _, _, observed = _patch_valid_artifact(monkeypatch, tmp_path, manifest=manifest)

    server.HybridServer(root).start(wait_ready=False)

    assert observed["popen"][0][0].endswith("llama-server.exe")


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
        server.HybridServer(root).start(wait_ready=False)
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

    runtime = server.HybridServer(root, verify_payload_hash=False).start(
        wait_ready=False
    )

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
        server.HybridServer(root).start(wait_ready=False)
    assert "popen" not in observed


class _Response:
    def __init__(self, value: dict[str, Any], status: int = 200) -> None:
        self.status = status
        self._body = json.dumps(value).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_health_and_completion_use_loopback_and_fixed_sampling(monkeypatch, tmp_path) -> None:
    runtime = server.HybridServer(tmp_path, port=19090)
    runtime._process = _Process()
    requests = []

    def fake_urlopen(request, *, timeout):
        requests.append((request, timeout))
        if request.full_url.endswith("/health"):
            return _Response({"status": "ok"})
        return _Response({"content": "answer<|im_end|>", "tokens_predicted": 1})

    monkeypatch.setattr(server, "urlopen", fake_urlopen)

    assert runtime.health() == {"status": "ok"}
    result = runtime.completion("Write Python.", max_tokens=32)

    assert result["generated_text"] == "answer"
    assert result["sampling"] == {"temperature": 0.0, "top_k": 1, "seed": 2026}
    request, timeout = requests[1]
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


def test_unhealthy_or_malformed_completion_fails_closed(monkeypatch, tmp_path) -> None:
    runtime = server.HybridServer(tmp_path)
    runtime._process = _Process()
    monkeypatch.setattr(server, "urlopen", lambda *_args, **_kwargs: _Response({"status": "loading"}))
    with pytest.raises(server.HybridServerUnavailableError, match="not ok"):
        runtime.health()

    monkeypatch.setattr(server, "urlopen", lambda *_args, **_kwargs: _Response({"tokens": 3}))
    with pytest.raises(server.HybridServerUnavailableError, match="text content"):
        runtime.completion("prompt")


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
