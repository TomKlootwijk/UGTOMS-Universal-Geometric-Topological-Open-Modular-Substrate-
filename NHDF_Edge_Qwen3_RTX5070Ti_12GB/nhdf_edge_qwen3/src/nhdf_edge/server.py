"""Persistent, fail-closed llama.cpp server for validated hybrid artifacts.

The one-shot hybrid runner intentionally verifies every sealed component before
each process launch.  This module keeps the same integrity and resource gates,
but pays their cost once before starting a resident ``llama-server`` process.
Every completion request remains pinned to the fixed sampling profile used by
the validated artifact. Prompt-cache reuse is opt-in because different CUDA
batch shapes can change greedy floating-point outcomes.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from . import hybrid


LOOPBACK_HOST = "127.0.0.1"
DEFAULT_PORT = 18080
VALIDATED_CONTEXT_TOKENS = 8_192
MODEL_ALIAS = "local-qwen3-30b-a3b"
DETERMINISTIC_TEMPERATURE = 0.0
DETERMINISTIC_TOP_K = 1
DETERMINISTIC_SEED = 2026


class HybridServerError(RuntimeError):
    """Base error for the persistent hybrid runtime."""


class HybridServerConfigurationError(HybridServerError):
    """Raised when a manifest cannot support the validated server profile."""


class HybridServerUnavailableError(HybridServerError):
    """Raised when the resident process or its loopback API is unavailable."""


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left.resolve())) == os.path.normcase(str(right.resolve()))


def _require_loopback(host: str) -> None:
    # A hostname such as ``localhost`` can be redirected through the hosts file.
    # Requiring the IPv4 loopback literal makes accidental network exposure
    # impossible without changing this implementation.
    if host != LOOPBACK_HOST:
        raise HybridServerConfigurationError(
            f"server host must be the loopback literal {LOOPBACK_HOST!r}, got {host!r}"
        )


def _require_validated_profile(manifest: dict[str, Any]) -> None:
    """Reject any profile that differs from the validated resident settings."""

    try:
        runtime = manifest["runtime"]
        profile = manifest["execution_profile"]
        contract = manifest["resource_contract"]
        sampling = profile["sampling"]
        expected_offload = profile["expected_offloaded_layers"]
    except (KeyError, TypeError) as exc:
        raise HybridServerConfigurationError(
            f"hybrid manifest is missing a required server field: {exc}"
        ) from exc

    problems: list[str] = []
    if not isinstance(runtime.get("revision"), str) or not runtime["revision"].strip():
        problems.append("runtime revision is not pinned")
    if int(profile.get("gpu_layers", -1)) != 999:
        problems.append("gpu_layers must be 999 for full GPU offload")
    if (
        not isinstance(expected_offload, list)
        or len(expected_offload) != 2
        or expected_offload[0] != expected_offload[1]
        or not isinstance(expected_offload[0], int)
        or isinstance(expected_offload[0], bool)
        or expected_offload[0] <= 0
    ):
        problems.append("expected_offloaded_layers must declare a complete nonzero offload")
    for field_name in ("kv_cache_k", "kv_cache_v"):
        cache_type = profile.get(field_name)
        if (
            not isinstance(cache_type, str)
            or cache_type not in hybrid.VALIDATED_KV_CACHE_TYPES
        ):
            supported = ", ".join(sorted(hybrid.VALIDATED_KV_CACHE_TYPES))
            problems.append(
                f"{field_name} must use a validated KV cache type ({supported})"
            )
    if profile.get("flash_attention") is not True:
        problems.append("flash attention must be enabled")
    context = profile.get("maximum_context_tokens")
    contract_context = contract.get("maximum_validated_context_tokens")
    valid_context = (
        isinstance(context, int)
        and not isinstance(context, bool)
        and context >= VALIDATED_CONTEXT_TOKENS
    )
    valid_contract_context = (
        isinstance(contract_context, int)
        and not isinstance(contract_context, bool)
        and contract_context >= VALIDATED_CONTEXT_TOKENS
    )
    if not valid_context:
        problems.append(f"execution context must cover {VALIDATED_CONTEXT_TOKENS} tokens")
    if not valid_contract_context:
        problems.append(f"resource contract must validate {VALIDATED_CONTEXT_TOKENS} tokens")
    elif valid_context and contract_context < context:
        problems.append(
            "resource contract does not validate the full execution context "
            f"({contract_context} < {context})"
        )
    if sampling.get("temperature") != DETERMINISTIC_TEMPERATURE:
        problems.append("sampling temperature must be 0")
    if (
        sampling.get("top_k") != DETERMINISTIC_TOP_K
        or isinstance(sampling.get("top_k"), bool)
    ):
        problems.append("sampling top_k must be 1")
    if (
        sampling.get("seed") != DETERMINISTIC_SEED
        or isinstance(sampling.get("seed"), bool)
    ):
        problems.append(f"sampling seed must be {DETERMINISTIC_SEED}")
    if problems:
        raise HybridServerConfigurationError(
            "manifest is outside the validated persistent-server profile: "
            + "; ".join(problems)
        )


def _sealed_server_entrypoint(root: Path, manifest: dict[str, Any]) -> Path:
    """Resolve one explicitly sealed llama-server executable.

    ``server_entrypoint`` is the preferred manifest field.  For compatibility
    with artifacts created before that field existed, a unique ``llama-server``
    record in ``runtime.files`` is also accepted.  In both cases the executable
    must be part of the runtime file set already checked by
    :func:`hybrid.verify_hybrid_artifact`.
    """

    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        raise HybridServerConfigurationError("manifest runtime section is missing")
    records = runtime.get("files")
    if not isinstance(records, list):
        raise HybridServerConfigurationError("manifest runtime.files is missing")

    sealed_paths: list[Path] = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise HybridServerConfigurationError("manifest contains an invalid runtime file")
        sealed_paths.append(hybrid._resolve_reference(root, record["path"]))

    declared = runtime.get("server_entrypoint")
    if declared is not None:
        if not isinstance(declared, str) or not declared:
            raise HybridServerConfigurationError("runtime.server_entrypoint is invalid")
        server = hybrid._resolve_reference(root, declared)
    else:
        candidates = [
            path
            for path in sealed_paths
            if path.name.lower() in {"llama-server", "llama-server.exe"}
        ]
        if len(candidates) != 1:
            raise HybridServerConfigurationError(
                "manifest must declare runtime.server_entrypoint or seal exactly one "
                "llama-server executable"
            )
        server = candidates[0]

    if server.name.lower() not in {"llama-server", "llama-server.exe"}:
        raise HybridServerConfigurationError(
            f"server entrypoint is not llama-server: {server.name!r}"
        )
    if not any(_same_path(server, sealed) for sealed in sealed_paths):
        raise HybridServerConfigurationError(
            "runtime.server_entrypoint is not present in the sealed runtime file list"
        )
    if not server.is_file():
        raise HybridServerConfigurationError(f"sealed server entrypoint is missing: {server}")
    return server


def _request_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout_seconds: float,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="GET" if data is None else "POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status = int(getattr(response, "status", 200))
            body = response.read()
    except HTTPError as exc:
        raise HybridServerUnavailableError(
            f"llama-server HTTP request failed with status {exc.code}"
        ) from exc
    except (URLError, OSError, TimeoutError) as exc:
        raise HybridServerUnavailableError(f"llama-server request failed: {exc}") from exc
    if status != 200:
        raise HybridServerUnavailableError(
            f"llama-server returned unexpected HTTP status {status}"
        )
    try:
        result = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HybridServerUnavailableError("llama-server returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise HybridServerUnavailableError("llama-server returned a non-object JSON response")
    return result


@dataclass
class HybridServer:
    """Own one verified, loopback-only, persistent llama-server process."""

    artifact_dir: str | Path
    host: str = LOOPBACK_HOST
    port: int = DEFAULT_PORT
    startup_timeout_seconds: float = 120.0
    request_timeout_seconds: float = 120.0
    threads: int | None = None
    verify_payload_hash: bool = True
    _process: subprocess.Popen[Any] | None = field(default=None, init=False, repr=False)
    _command: tuple[str, ...] | None = field(default=None, init=False, repr=False)
    _preflight_result: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _maximum_context_tokens: int = field(
        default=VALIDATED_CONTEXT_TOKENS, init=False, repr=False
    )
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.artifact_dir = Path(self.artifact_dir).resolve()
        _require_loopback(self.host)
        if isinstance(self.port, bool) or not isinstance(self.port, int):
            raise HybridServerConfigurationError("server port must be an integer")
        if not 1 <= self.port <= 65_535:
            raise HybridServerConfigurationError("server port must be between 1 and 65535")
        if self.startup_timeout_seconds <= 0 or self.request_timeout_seconds <= 0:
            raise HybridServerConfigurationError("server timeouts must be positive")
        if not isinstance(self.verify_payload_hash, bool):
            raise HybridServerConfigurationError("verify_payload_hash must be a boolean")
        if self.threads is None:
            # Measured on the target Core Ultra 7 255HX: four generation
            # threads outperformed 2/8/12/20 while the model was fully offloaded.
            self.threads = min(4, max(1, os.cpu_count() or 1))
        if (
            isinstance(self.threads, bool)
            or not isinstance(self.threads, int)
            or self.threads <= 0
        ):
            raise HybridServerConfigurationError("server thread count must be positive")

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def command(self) -> tuple[str, ...] | None:
        return self._command

    @property
    def preflight_result(self) -> dict[str, Any] | None:
        return dict(self._preflight_result) if self._preflight_result is not None else None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def _require_running(self) -> subprocess.Popen[Any]:
        with self._lock:
            process = self._process
            if process is None:
                raise HybridServerUnavailableError("llama-server has not been started")
            return_code = process.poll()
        if return_code is not None:
            raise HybridServerUnavailableError(
                f"llama-server exited unexpectedly with code {return_code}"
            )
        return process

    def start(self, *, wait_ready: bool = True) -> "HybridServer":
        """Verify once, perform the resource preflight, and launch the server."""

        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise HybridServerError("llama-server is already running")

            verification = hybrid.verify_hybrid_artifact(
                self.artifact_dir,
                verify_payload_hash=self.verify_payload_hash,
                require_validated=True,
            )
            if not verification.get("ok") or not verification.get("deployment_loadable"):
                raise OSError(
                    "refusing to launch hybrid server because artifact verification failed: "
                    f"{verification.get('failures', [])}"
                )

            manifest = hybrid.load_hybrid_manifest(self.artifact_dir)
            _require_validated_profile(manifest)
            profile = manifest["execution_profile"]
            context = int(profile["maximum_context_tokens"])
            server = _sealed_server_entrypoint(self.artifact_dir, manifest)
            model = hybrid._resolve_reference(self.artifact_dir, manifest["payload"]["path"])
            runtime_profile = manifest["runtime"].get("argument_profile", "b6014")
            if runtime_profile == "b6014":
                flash_arguments = ["-fa"]
            elif runtime_profile == "current-2026":
                flash_arguments = ["-fa", "on"]
            else:
                raise HybridServerConfigurationError(
                    f"unsupported llama.cpp argument profile: {runtime_profile!r}"
                )
            preflight = hybrid._preflight(manifest, context=context)
            command = [
                str(server),
                "-m",
                str(model),
                "-ngl",
                "999",
                "-sm",
                "none",
                "-c",
                str(context),
                "-ctk",
                str(profile["kv_cache_k"]),
                "-ctv",
                str(profile["kv_cache_v"]),
                *flash_arguments,
                "-np",
                "1",
                "-t",
                str(self.threads),
                "-tb",
                str(self.threads),
                "-b",
                "2048",
                "-ub",
                "512",
                "--prio",
                "2",
                "--prio-batch",
                "2",
                "--poll",
                "50",
                "--temp",
                "0",
                "--top-k",
                "1",
                "--seed",
                str(DETERMINISTIC_SEED),
                "--alias",
                MODEL_ALIAS,
                "--jinja",
                "--reasoning-format",
                "none",
                "--cors-origins",
                "localhost",
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--no-webui",
                "--metrics",
                "--slots",
            ]
            process = subprocess.Popen(
                command,
                cwd=str(server.parent),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._process = process
            self._command = tuple(command)
            self._preflight_result = preflight
            self._maximum_context_tokens = context

        if wait_ready:
            try:
                self.wait_until_ready(timeout_seconds=self.startup_timeout_seconds)
            except BaseException:
                self.stop()
                raise
        return self

    def wait_until_ready(
        self,
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = 0.1,
    ) -> dict[str, Any]:
        """Wait until the process reports a healthy, fully loaded model."""

        if timeout_seconds is None:
            timeout_seconds = self.startup_timeout_seconds
        if timeout_seconds <= 0 or poll_interval_seconds <= 0:
            raise ValueError("readiness timeout and poll interval must be positive")
        deadline = time.monotonic() + timeout_seconds
        last_error: HybridServerUnavailableError | None = None
        while time.monotonic() < deadline:
            self._require_running()
            try:
                return self.health(timeout_seconds=min(2.0, timeout_seconds))
            except HybridServerUnavailableError as exc:
                last_error = exc
            time.sleep(min(poll_interval_seconds, max(0.0, deadline - time.monotonic())))
        detail = f": {last_error}" if last_error is not None else ""
        raise HybridServerUnavailableError(
            f"llama-server did not become healthy within {timeout_seconds:g} seconds{detail}"
        )

    def health(self, *, timeout_seconds: float = 2.0) -> dict[str, Any]:
        """Return the loopback health response or fail closed."""

        self._require_running()
        result = _request_json(
            f"{self.base_url}/health", timeout_seconds=timeout_seconds
        )
        if result.get("status") != "ok":
            raise HybridServerUnavailableError(
                f"llama-server health status is not ok: {result.get('status')!r}"
            )
        return result

    def completion(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        cache_prompt: bool = False,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Run one fixed-sampling completion against the resident model.

        Prefix caching defaults off for the accuracy-first library path. It can
        be enabled explicitly when latency is more important than bitwise
        repeatability across different cache/batch states.
        """

        self._require_running()
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not 1 <= max_tokens <= self._maximum_context_tokens
        ):
            raise ValueError(
                f"max_tokens must be between 1 and {self._maximum_context_tokens}"
            )
        if not isinstance(cache_prompt, bool):
            raise ValueError("cache_prompt must be a boolean")
        timeout = self.request_timeout_seconds if timeout_seconds is None else timeout_seconds
        if timeout <= 0:
            raise ValueError("completion timeout must be positive")
        request_payload: dict[str, Any] = {
            "prompt": hybrid._chatml(prompt),
            "n_predict": max_tokens,
            "temperature": DETERMINISTIC_TEMPERATURE,
            "top_k": DETERMINISTIC_TOP_K,
            "seed": DETERMINISTIC_SEED,
            "cache_prompt": cache_prompt,
            "reasoning_format": "none",
            "stream": False,
            "stop": ["<|im_end|>"],
        }
        result = _request_json(
            f"{self.base_url}/completion",
            payload=request_payload,
            timeout_seconds=timeout,
        )
        content = result.get("content")
        if not isinstance(content, str):
            raise HybridServerUnavailableError(
                "llama-server completion response does not contain text content"
            )
        normalized = dict(result)
        normalized["generated_text"] = hybrid._clean_generation(content)
        normalized["sampling"] = {
            "temperature": DETERMINISTIC_TEMPERATURE,
            "top_k": DETERMINISTIC_TOP_K,
            "seed": DETERMINISTIC_SEED,
        }
        normalized["cache_prompt"] = cache_prompt
        return normalized

    def stop(self, *, timeout_seconds: float = 10.0) -> None:
        """Stop the owned process, escalating to kill only after a timeout."""

        if timeout_seconds < 0:
            raise ValueError("stop timeout must not be negative")
        with self._lock:
            process = self._process
            self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=timeout_seconds)

    def __enter__(self) -> "HybridServer":
        return self.start()

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.stop()


def start_hybrid_server(
    artifact_dir: str | Path,
    **kwargs: Any,
) -> HybridServer:
    """Construct and start a :class:`HybridServer` for CLI/library callers."""

    return HybridServer(artifact_dir, **kwargs).start()
