#!/usr/bin/env python3
"""Run a bounded, reproducible local coding-agent acceptance gate.

This gate checks one local OpenAI-compatible server and one pinned OpenCode
executable.  It proves only the measured capabilities exercised here: native
tool-call JSON, synthetic long-context needle retrieval, and repair of one
small disposable Python repository.  It is not a broad coding benchmark, a
general accuracy claim, or evidence that the model understands the substrate.  No
legacy/archive corpus is loaded by this script.

The tool-event checks are deliberately retrospective.  They strengthen the
recorded evidence but are not a preventive process, filesystem, or network
sandbox.

Only the Python standard library is used by the harness itself.  The disposable
fixture deliberately invokes pytest because that is part of the agent workflow
being tested.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CODER_SCRIPT = PROJECT_ROOT / "scripts" / "local_coder.py"


def _load_launcher_contract() -> Any:
    """Load the canonical launcher definitions without executing its CLI."""

    module_name = "_ugtoms_local_coder_contract"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, LOCAL_CODER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load canonical launcher from {LOCAL_CODER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_LAUNCHER = _load_launcher_contract()

FORMAT = "ugtoms-local-coder-agent-gate-0.2"
MODEL_ID = _LAUNCHER.MODEL_ID
PROVIDER_ID, MODEL_ALIAS = MODEL_ID.split("/", 1)
PINNED_OPENCODE_VERSION = _LAUNCHER.PINNED_OPENCODE_VERSION
SOURCE_BASE_URL_PLACEHOLDER = "{env:UGTOMS_LOCAL_CODER_BASE_URL}/v1"
MIN_CONTEXT_TOKENS = 32_768
LONG_CONTEXT_TARGET_TOKENS = 22_000
LONG_CONTEXT_TOLERANCE_TOKENS = 500
LONG_CONTEXT_NEEDLE = "UGTOMS_LOCAL_GATE_KEY_7F3A91C2D8B4"
EXPECTED_SOURCE = "src/intervals.py"
EXPECTED_FINAL_PYTEST_TESTS = 4
ALLOWED_TOOL_NAMES = frozenset(
    {"read", "grep", "glob", "edit", "bash", "todowrite"}
)
READ_TOOLS = frozenset({"read"})
SEARCH_TOOLS = frozenset({"grep", "glob"})
EDIT_TOOLS = frozenset({"edit"})
BASH_TOOLS = frozenset({"bash"})
PATH_ARGUMENT_KEYS = frozenset(
    {"path", "filepath", "directory", "workdir", "cwd", "root", "rootdir"}
)
GLOB_ARGUMENT_KEYS = frozenset(
    {"include", "glob", "fileglob", "filepattern", "pathpattern"}
)
ALLOWED_PARENT_ENVIRONMENT = frozenset(
    {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "SYSTEMDRIVE",
        "OS",
        "PROCESSOR_ARCHITECTURE",
        "PROCESSOR_IDENTIFIER",
        "NUMBER_OF_PROCESSORS",
        "LANG",
        "LC_ALL",
        "TERM",
        "COLORTERM",
    }
)


class GateError(RuntimeError):
    """A measured gate condition was not met."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _public_path(path: Path, *, fallback: str) -> str:
    """Represent a host path without publishing host-specific directory names."""

    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return fallback
    return f"<PROJECT_ROOT>/{relative.as_posix()}"


def _public_failure_message(
    message: str, *, private_paths: Mapping[str | Path, str]
) -> str:
    """Replace known private roots in a diagnostic before it enters public JSON."""

    replacements: list[tuple[str, str]] = [
        (str(PROJECT_ROOT.resolve()), "<PROJECT_ROOT>"),
        (str(Path(tempfile.gettempdir()).resolve()), "<TEMP_ROOT>"),
    ]
    for private, public in private_paths.items():
        try:
            candidate = Path(private).expanduser()
        except (TypeError, ValueError):
            continue
        if candidate.is_absolute():
            replacements.append((str(candidate.resolve()), public))
    result = message
    for private, public in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        result = result.replace(private, public)
        result = result.replace(private.replace("\\", "/"), public)
    return result


def _verified_pytest_pass_count(stdout: bytes, stderr: bytes, *, expected: int) -> int:
    """Return an exact pytest pass count or fail closed on missing/ambiguous output."""

    output = (stdout + b"\n" + stderr).decode("utf-8", errors="replace")
    counts = {int(value) for value in re.findall(r"(?<!\d)(\d+) passed\b", output)}
    if counts != {expected}:
        raise GateError(
            f"independent final pytest reported pass counts {sorted(counts)!r}; "
            f"expected exactly {expected}/{expected}"
        )
    return expected


def _normalize_server_url(value: str) -> str:
    """Return a canonical loopback HTTP origin, optionally stripping /v1."""

    parsed = urllib.parse.urlsplit(value.strip())
    if parsed.scheme.lower() != "http":
        raise GateError("--server-url must use plain HTTP on loopback")
    if parsed.username or parsed.password:
        raise GateError("--server-url must not contain credentials")
    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise GateError("--server-url must resolve literally to a loopback hostname")
    if parsed.query or parsed.fragment:
        raise GateError("--server-url must not contain a query or fragment")
    path = parsed.path.rstrip("/")
    if path not in {"", "/v1"}:
        raise GateError("--server-url may contain only the optional /v1 path")
    try:
        port = parsed.port
    except ValueError as exc:
        raise GateError(f"invalid --server-url port: {exc}") from exc
    host_display = f"[{host}]" if ":" in host else host
    return f"http://{host_display}{f':{port}' if port is not None else ''}"


def _http_json(
    base_url: str,
    route: str,
    *,
    method: str = "GET",
    payload: Mapping[str, Any] | None = None,
    timeout: float = 120.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Perform one JSON request and return its body plus measured transport data."""

    raw_request = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        raw_request = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        base_url.rstrip("/") + "/" + route.lstrip("/"),
        data=raw_request,
        headers=headers,
        method=method,
    )
    started = time.perf_counter()
    try:
        with opener(request, timeout=timeout) as response:
            status = int(response.getcode())
            raw_response = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:2000]
        raise GateError(f"HTTP {exc.code} from {route}: {body}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise GateError(f"request to {route} failed: {exc}") from exc
    elapsed = time.perf_counter() - started
    if status < 200 or status >= 300:
        raise GateError(f"HTTP {status} from {route}")
    try:
        decoded = json.loads(raw_response.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError(f"{route} did not return valid UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise GateError(f"{route} returned a non-object JSON value")
    return decoded, {
        "elapsed_seconds": elapsed,
        "request_bytes": len(raw_request or b""),
        "response_bytes": len(raw_response),
        "status": status,
    }


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.isdigit() and int(value) > 0:
        return int(value)
    return None


def _validate_models_payload(
    payload: Mapping[str, Any],
    *,
    alias: str = MODEL_ALIAS,
    minimum_context: int = MIN_CONTEXT_TOKENS,
) -> dict[str, Any]:
    """Select the required alias and validate reported model context metadata."""

    data = payload.get("data")
    if not isinstance(data, list):
        raise GateError("/v1/models response has no data array")
    selected: Mapping[str, Any] | None = None
    for candidate in data:
        if not isinstance(candidate, Mapping):
            continue
        aliases = candidate.get("aliases")
        reported_aliases = aliases if isinstance(aliases, list) else []
        if candidate.get("id") == alias or alias in reported_aliases:
            selected = candidate
            break
    if selected is None:
        ids = [item.get("id") for item in data if isinstance(item, Mapping)]
        raise GateError(f"/v1/models does not report required alias {alias!r}; ids={ids!r}")

    candidates: list[tuple[str, int]] = []
    for container_name, container in (("model", selected), ("meta", selected.get("meta"))):
        if not isinstance(container, Mapping):
            continue
        for key in (
            "n_ctx",
            "context_length",
            "max_context_length",
            "max_model_len",
        ):
            parsed = _positive_int(container.get(key))
            if parsed is not None:
                candidates.append((f"{container_name}.{key}", parsed))
    if not candidates:
        training_context = None
        metadata = selected.get("meta")
        if isinstance(metadata, Mapping):
            training_context = _positive_int(metadata.get("n_ctx_train"))
        detail = (
            f"; training context {training_context} is not served allocation"
            if training_context is not None
            else ""
        )
        raise GateError(
            "required model entry has no served context-length metadata" + detail
        )
    field, reported = max(candidates, key=lambda item: item[1])
    if reported < minimum_context:
        raise GateError(
            f"/v1/models reports only {reported} context tokens via {field}; "
            f"need at least {minimum_context}"
        )
    return {
        "alias": alias,
        "context_field": field,
        "reported_context_tokens": reported,
        "minimum_context_tokens": minimum_context,
        "passed": True,
    }


def _tool_probe_payload() -> dict[str, Any]:
    return {
        "model": MODEL_ALIAS,
        "messages": [
            {
                "role": "system",
                "content": (
                    "This is a native tool-call protocol test. Call the forced function "
                    "exactly once with the exact values requested; do not answer in prose."
                ),
            },
            {
                "role": "user",
                "content": "Call record_local_probe with nonce 20260831 and label gate-ready.",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "record_local_probe",
                    "description": "Record the deterministic local compatibility probe.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "nonce": {"type": "integer"},
                            "label": {"type": "string"},
                        },
                        "required": ["nonce", "label"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": {
            "type": "function",
            "function": {"name": "record_local_probe"},
        },
        "parallel_tool_calls": False,
        "temperature": 0,
        "seed": 20260831,
        "max_tokens": 128,
        "stream": False,
    }


def _evaluate_tool_probe(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Require an OpenAI-native tool call whose arguments are a JSON string."""

    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise GateError("tool probe did not return exactly one choice")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise GateError("tool probe choice is not an object")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise GateError("tool probe choice has no assistant message")
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise GateError("tool probe did not return exactly one native tool_call")
    call = calls[0]
    function = call.get("function") if isinstance(call, Mapping) else None
    if not isinstance(function, Mapping) or function.get("name") != "record_local_probe":
        raise GateError("tool probe called the wrong function")
    arguments_raw = function.get("arguments")
    if not isinstance(arguments_raw, str):
        raise GateError("native tool arguments were not an OpenAI JSON string")
    try:
        arguments = json.loads(arguments_raw)
    except json.JSONDecodeError as exc:
        raise GateError("native tool arguments are not valid JSON") from exc
    expected = {"nonce": 20260831, "label": "gate-ready"}
    if arguments != expected:
        raise GateError(f"native tool arguments differ: expected {expected!r}, got {arguments!r}")
    finish_reason = choice.get("finish_reason")
    if finish_reason not in {"tool_calls", "stop"}:
        raise GateError(f"unexpected tool-probe finish_reason: {finish_reason!r}")
    return {
        "passed": True,
        "function": "record_local_probe",
        "arguments": arguments,
        "arguments_were_json_string": True,
        "finish_reason": finish_reason,
    }


_WORDS_A = ("amber", "cobalt", "juniper", "marble", "silver", "topaz", "violet")
_WORDS_B = ("bridge", "circuit", "harbor", "ledger", "orbit", "signal", "vector")


def _long_context_messages(record_count: int) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Build deterministic synthetic context with one needle near the middle."""

    if record_count < 1:
        raise ValueError("record_count must be positive")
    lines: list[str] = []
    needle_line = record_count // 2
    for index in range(record_count):
        if index == needle_line:
            lines.append(
                "AUTHORITATIVE NEEDLE: the exact retrieval key is "
                f"{LONG_CONTEXT_NEEDLE}. Return that key when asked.\n"
            )
        a = _WORDS_A[(index * 5 + 1) % len(_WORDS_A)]
        b = _WORDS_B[(index * 3 + 2) % len(_WORDS_B)]
        checksum = (index * 7919 + 104729) % 1_000_003
        lines.append(
            f"Synthetic record {index:05d}: {a} {b}; checksum {checksum:06d}; "
            "ordinary deterministic reference material.\n"
        )
    context = "".join(lines)
    if context.count(LONG_CONTEXT_NEEDLE) != 1:
        raise AssertionError("synthetic context must contain exactly one needle")
    user = (
        "Read the synthetic reference block. Ignore ordinary records and return only "
        "the exact retrieval key from the AUTHORITATIVE NEEDLE line.\n\n"
        "--- SYNTHETIC REFERENCE START ---\n"
        f"{context}"
        "--- SYNTHETIC REFERENCE END ---"
    )
    messages = [
        {
            "role": "system",
            "content": "Perform exact retrieval. Return only the requested key, with no explanation.",
        },
        {"role": "user", "content": user},
    ]
    return messages, {
        "record_count": record_count,
        "context_characters": len(context),
        "message_characters": sum(len(item["content"]) for item in messages),
        "needle": LONG_CONTEXT_NEEDLE,
        "needle_character_offset": context.index(LONG_CONTEXT_NEEDLE),
        "needle_position_fraction": context.index(LONG_CONTEXT_NEEDLE) / len(context),
        "synthetic_context_sha256": _sha256_bytes(context.encode("utf-8")),
    }


def _select_near_target(
    measure: Callable[[int], int],
    *,
    target: int = LONG_CONTEXT_TARGET_TOKENS,
    tolerance: int = LONG_CONTEXT_TOLERANCE_TOKENS,
    maximum_records: int = 8192,
) -> tuple[int, int, list[dict[str, int]]]:
    """Binary-search a monotonic token counter and return its closest sample."""

    if target <= 0 or tolerance < 0 or maximum_records < 1:
        raise ValueError("invalid long-context calibration bounds")
    samples: dict[int, int] = {}

    def sample(records: int) -> int:
        if records not in samples:
            value = measure(records)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise GateError("tokenization returned an invalid token count")
            samples[records] = value
        return samples[records]

    low, high = 1, 1024
    sample(low)
    while sample(high) < target and high < maximum_records:
        low = high
        high = min(maximum_records, high * 2)
    if sample(high) < target - tolerance:
        raise GateError(
            f"could not construct approximately {target} tokens within {maximum_records} records"
        )
    while low <= high:
        middle = (low + high) // 2
        tokens = sample(middle)
        if tokens < target:
            low = middle + 1
        elif tokens > target:
            high = middle - 1
        else:
            break
    best_records, best_tokens = min(samples.items(), key=lambda item: abs(item[1] - target))
    if abs(best_tokens - target) > tolerance:
        raise GateError(
            f"closest deterministic prompt has {best_tokens} tokens; "
            f"target is {target} +/- {tolerance}"
        )
    trace = [
        {"record_count": records, "formatted_prompt_tokens": tokens}
        for records, tokens in sorted(samples.items())
    ]
    return best_records, best_tokens, trace


def _formatted_prompt_measurement(
    server_url: str,
    messages: Sequence[Mapping[str, str]],
    *,
    timeout: float,
) -> tuple[int, dict[str, Any]]:
    applied, apply_timing = _http_json(
        server_url,
        "/apply-template",
        method="POST",
        payload={"messages": list(messages)},
        timeout=timeout,
    )
    prompt = applied.get("prompt")
    if not isinstance(prompt, str):
        raise GateError("/apply-template did not return a prompt string")
    tokenized, tokenize_timing = _http_json(
        server_url,
        "/tokenize",
        method="POST",
        payload={
            "content": prompt,
            "add_special": False,
            "parse_special": True,
            "with_pieces": False,
        },
        timeout=timeout,
    )
    tokens = tokenized.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
        raise GateError("/tokenize did not return a token-id array")
    return len(tokens), {
        "formatted_prompt_characters": len(prompt),
        "formatted_prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
        "apply_template": apply_timing,
        "tokenize": tokenize_timing,
    }


def _evaluate_long_context_response(
    response: Mapping[str, Any], *, exact_prompt_tokens: int
) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise GateError("long-context probe did not return exactly one choice")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, str) or LONG_CONTEXT_NEEDLE not in content:
        raise GateError("long-context response did not contain the exact retrieval key")
    usage = response.get("usage")
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, Mapping) else None
    if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool):
        raise GateError("long-context response did not report exact prompt token usage")
    if prompt_tokens != exact_prompt_tokens:
        raise GateError(
            f"token count mismatch: /tokenize={exact_prompt_tokens}, usage={prompt_tokens}"
        )
    return {
        "passed": True,
        "exact_formatted_prompt_tokens": exact_prompt_tokens,
        "api_usage_prompt_tokens": prompt_tokens,
        "completion_tokens": usage.get("completion_tokens"),
        "response_content": content,
        "needle_present": True,
        "finish_reason": choice.get("finish_reason") if isinstance(choice, Mapping) else None,
    }


def _walk_config_values(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = path + (str(key),)
            yield child_path, child
            yield from _walk_config_values(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_config_values(child, path + (str(index),))


def _validate_opencode_config(config: Mapping[str, Any], server_url: str) -> dict[str, Any]:
    """Validate the canonical source config and its single allowed substitution."""

    normalized_server = _normalize_server_url(server_url)
    try:
        _LAUNCHER._validate_config_contract(
            dict(config),
            expected_base_url=SOURCE_BASE_URL_PLACEHOLDER,
            source_config=True,
        )
    except _LAUNCHER.LocalCoderError as exc:
        raise GateError(f"OpenCode config violates the canonical launcher contract: {exc}") from exc

    endpoint_values: list[tuple[tuple[str, ...], str]] = []
    for path, value in _walk_config_values(config):
        key = path[-1].lower().replace("_", "") if path else ""
        if key in {"baseurl", "endpoint", "apiurl"} and isinstance(value, str):
            endpoint_values.append((path, value))
    expected_path = (
        "provider",
        PROVIDER_ID,
        "options",
        "baseURL",
    )
    if endpoint_values != [(expected_path, SOURCE_BASE_URL_PLACEHOLDER)]:
        raise GateError(
            "OpenCode config must contain exactly the canonical UGTOMS loopback "
            f"placeholder at {'.'.join(expected_path)}"
        )
    resolved_endpoint = normalized_server + "/v1"
    resolved = json.loads(json.dumps(config))
    resolved["provider"][PROVIDER_ID]["options"]["baseURL"] = resolved_endpoint
    try:
        _LAUNCHER._validate_config_contract(
            resolved,
            expected_base_url=resolved_endpoint,
            source_config=False,
        )
    except _LAUNCHER.LocalCoderError as exc:  # pragma: no cover - source validation should dominate
        raise GateError(f"resolved OpenCode config is unsafe: {exc}") from exc
    return {
        "passed": True,
        "provider": PROVIDER_ID,
        "model": MODEL_ID,
        "source_base_url_placeholder": SOURCE_BASE_URL_PLACEHOLDER,
        "resolved_provider_endpoint": resolved_endpoint,
        "only_canonical_placeholder_resolved": True,
        "plugins_disabled": True,
        "mcp_disabled": True,
    }


def _load_pinned_config(path: str | Path) -> tuple[Path, bytes, dict[str, Any]]:
    """Reuse the launcher's digest and schema check for the benchmark input."""

    try:
        validated = _LAUNCHER.validate_config(Path(path))
        resolved = Path(validated.path)
        raw = bytes(validated.raw)
        value = json.loads(raw.decode("utf-8"))
    except (_LAUNCHER.LocalCoderError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateError(f"benchmark requires the pinned launcher config: {exc}") from exc
    if not isinstance(value, dict):  # pragma: no cover - launcher already checks this
        raise GateError("pinned launcher config must contain an object")
    return resolved, raw, value


def _fixture_files() -> dict[str, str]:
    return {
        ".gitignore": "__pycache__/\n.pytest_cache/\n*.py[cod]\n",
        "README.md": (
            "# Interval helper fixture\n\n"
            "`coalesce_periods` accepts integer `[start, end]` periods, sorts them, "
            "and merges overlapping or touching periods. Touching means the next start "
            "equals the current end. It must return fresh lists and never mutate input.\n"
        ),
        "src/__init__.py": "",
        "src/intervals.py": (
            "def coalesce_periods(periods):\n"
            "    \"\"\"Merge overlapping or touching periods without mutating input.\"\"\"\n"
            "    ordered = sorted((list(period) for period in periods), key=lambda item: item[0])\n"
            "    if not ordered:\n"
            "        return []\n"
            "    merged = [ordered[0]]\n"
            "    for start, end in ordered[1:]:\n"
            "        # BUG: touching periods are incorrectly kept separate.\n"
            "        if start < merged[-1][1]:\n"
            "            merged[-1][1] = max(merged[-1][1], end)\n"
            "        else:\n"
            "            merged.append([start, end])\n"
            "    return merged\n"
        ),
        "tests/test_intervals.py": (
            "from src.intervals import coalesce_periods\n\n\n"
            "def test_empty_and_disjoint():\n"
            "    assert coalesce_periods([]) == []\n"
            "    assert coalesce_periods([[5, 7], [1, 2]]) == [[1, 2], [5, 7]]\n\n\n"
            "def test_overlapping_and_nested():\n"
            "    assert coalesce_periods([[1, 5], [2, 3], [4, 8]]) == [[1, 8]]\n\n\n"
            "def test_touching_periods_merge_transitively():\n"
            "    assert coalesce_periods([[8, 10], [1, 3], [3, 6], [6, 8]]) == [[1, 10]]\n\n\n"
            "def test_input_and_nested_lists_are_not_mutated():\n"
            "    periods = [[3, 5], [1, 3]]\n"
            "    before = [item[:] for item in periods]\n"
            "    assert coalesce_periods(periods) == [[1, 5]]\n"
            "    assert periods == before\n"
        ),
    }


def _run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout: float = 120.0,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            list(command),
            cwd=str(cwd),
            env=dict(env) if env is not None else None,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise GateError(f"command timed out after {timeout:.1f}s: {command[0]}") from exc
    except OSError as exc:
        raise GateError(f"could not execute {command[0]!r}: {exc}") from exc


def _run_git(fixture: Path, arguments: Sequence[str], *, check: bool = True) -> bytes:
    result = _run_process(["git", *arguments], cwd=fixture, timeout=30.0)
    if check and result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")[:2000]
        raise GateError(f"git {' '.join(arguments)} failed: {stderr}")
    return result.stdout


def _create_fixture(root: Path) -> dict[str, Any]:
    files = _fixture_files()
    for relative, content in files.items():
        _write_text(root / relative, content)
    _run_git(root, ["init", "--quiet"])
    _run_git(root, ["config", "user.name", "UGTOMS Local Gate"])
    _run_git(root, ["config", "user.email", "local-gate@invalid.example"])
    _run_git(root, ["config", "commit.gpgsign", "false"])
    _run_git(root, ["add", "--", *sorted(files)])
    _run_git(
        root,
        ["-c", f"core.hooksPath={os.devnull}", "commit", "--quiet", "--no-gpg-sign", "-m", "fixture baseline"],
    )
    head = _run_git(root, ["rev-parse", "HEAD"]).decode("ascii").strip()
    return {
        "initial_head": head,
        "tracked_files": sorted(files),
        "expected_source": EXPECTED_SOURCE,
        "expected_source_sha256_before": _sha256_file(root / EXPECTED_SOURCE),
        "tests_sha256_before": _sha256_file(root / "tests/test_intervals.py"),
    }


def _parse_json_events(raw: bytes | str) -> list[dict[str, Any]]:
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GateError("OpenCode stdout is not valid UTF-8 JSONL") from exc
    else:
        text = raw
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GateError(f"OpenCode stdout line {line_number} is not JSON") from exc
        if not isinstance(event, dict):
            raise GateError(f"OpenCode stdout line {line_number} is not a JSON object")
        events.append(event)
    if not events:
        raise GateError("OpenCode emitted no JSON events")
    return events


def _tool_events(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") not in {"tool_use", "tool"}:
            continue
        part = event.get("part")
        source = part if isinstance(part, Mapping) else event
        name = source.get("tool") or source.get("name")
        state = source.get("state")
        state_map = state if isinstance(state, Mapping) else {}
        arguments = state_map.get("input", source.get("input", source.get("arguments", {})))
        tools.append(
            {
                "name": str(name).strip().lower() if name is not None else "",
                "status": str(state_map.get("status", source.get("status", "unknown"))).lower(),
                "input": arguments if isinstance(arguments, Mapping) else {},
            }
        )
    return tools


def _resolve_tool_path(value: str, fixture: Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = fixture / candidate
    return candidate.resolve(strict=False)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _walk_argument_values(
    value: Any, path: tuple[str, ...] = ()
) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = path + (str(key),)
            yield child_path, child
            yield from _walk_argument_values(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_argument_values(child, path + (str(index),))


def _normalized_argument_key(path: tuple[str, ...]) -> str:
    return path[-1].lower().replace("_", "") if path else ""


def _validate_tool_path(
    value: object,
    *,
    fixture: Path,
    label: str,
    exact_path: Path | None = None,
) -> Path:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise GateError(f"{label} must be a non-empty path string")
    if "\x00" in value or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value):
        raise GateError(f"{label} is not a local fixture path: {value!r}")
    if any(character in value for character in "*?[]{}"):
        raise GateError(f"{label} must be a concrete path, not a pattern: {value!r}")
    resolved = _resolve_tool_path(value, fixture)
    fixture_resolved = fixture.resolve()
    if not _is_relative_to(resolved, fixture_resolved):
        raise GateError(f"{label} escaped the disposable fixture: {value!r}")
    if exact_path is not None and resolved != exact_path.resolve():
        raise GateError(f"{label} targeted a file other than {EXPECTED_SOURCE}: {value!r}")
    return resolved


def _validate_glob_pattern(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise GateError(f"{label} must be a non-empty glob string")
    if len(value) > 4096 or "\x00" in value or "\r" in value or "\n" in value:
        raise GateError(f"{label} is not a bounded single-line glob")
    normalized = value.replace("\\", "/")
    if (
        normalized.startswith("/")
        or re.match(r"^[A-Za-z]:", normalized)
        or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", normalized)
        or ".." in normalized.split("/")
    ):
        raise GateError(f"{label} can only match paths inside the disposable fixture: {value!r}")
    return value


def _validate_content_pattern(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 4096 or "\x00" in value:
        raise GateError(f"{label} must be a non-empty bounded search pattern")
    return value


def _validate_recorded_tool_arguments(
    name: str,
    arguments: Mapping[str, Any],
    *,
    fixture: Path,
    expected_source: str,
) -> None:
    """Audit every recorded path/pattern exposed by a supported tool event."""

    walked = list(_walk_argument_values(arguments))
    path_values = [
        (path, value)
        for path, value in walked
        if _normalized_argument_key(path) in PATH_ARGUMENT_KEYS
    ]
    for path, value in path_values:
        _validate_tool_path(
            value,
            fixture=fixture,
            label=f"{name} argument {'.'.join(path)}",
            exact_path=(fixture / expected_source) if name in EDIT_TOOLS else None,
        )

    if name in READ_TOOLS | EDIT_TOOLS and not path_values:
        raise GateError(f"{name} tool event did not expose its target path")

    pattern_values = [
        (path, value)
        for path, value in walked
        if _normalized_argument_key(path) == "pattern"
    ]
    glob_values = [
        (path, value)
        for path, value in walked
        if _normalized_argument_key(path) in GLOB_ARGUMENT_KEYS
    ]
    if name == "glob":
        if len(pattern_values) != 1:
            raise GateError("glob tool event must expose exactly one pattern")
        _validate_glob_pattern(pattern_values[0][1], label="glob argument pattern")
    elif name == "grep":
        if len(pattern_values) != 1:
            raise GateError("grep tool event must expose exactly one search pattern")
        _validate_content_pattern(pattern_values[0][1], label="grep argument pattern")
    for path, value in glob_values:
        _validate_glob_pattern(value, label=f"{name} argument {'.'.join(path)}")


def _validate_pytest_command(command: str) -> None:
    """Accept only the exact command requested by this disposable fixture."""

    if command != "python -m pytest -q":
        raise GateError(
            "Bash command must be exactly 'python -m pytest -q'; "
            f"recorded {command!r}"
        )


def _evaluate_tool_events(
    events: Sequence[Mapping[str, Any]], fixture: Path, expected_source: str = EXPECTED_SOURCE
) -> dict[str, Any]:
    fixture_resolved = fixture.resolve()
    expected_resolved = (fixture / expected_source).resolve()
    if not _is_relative_to(expected_resolved, fixture_resolved):
        raise GateError("expected source path escapes the disposable fixture")
    tools = _tool_events(events)
    if not tools:
        raise GateError("OpenCode emitted no completed tool-use events")
    names = [tool["name"] for tool in tools]
    unknown = sorted(set(names) - ALLOWED_TOOL_NAMES)
    if unknown:
        raise GateError(f"OpenCode used disallowed/destructive/external tools: {unknown!r}")
    failed = [tool["name"] for tool in tools if tool["status"] not in {"completed", "success"}]
    if failed:
        raise GateError(f"OpenCode emitted failed or incomplete tool events: {failed!r}")
    requirements = {
        "read": any(name in READ_TOOLS for name in names),
        "search": any(name in SEARCH_TOOLS for name in names),
        "edit": any(name in EDIT_TOOLS for name in names),
        "bash": any(name in BASH_TOOLS for name in names),
    }
    missing = [name for name, present in requirements.items() if not present]
    if missing:
        raise GateError(f"OpenCode did not exercise required tool categories: {missing!r}")
    edit_count = sum(name in EDIT_TOOLS for name in names)
    bash_count = sum(name in BASH_TOOLS for name in names)
    if edit_count != 1 or bash_count != 1:
        raise GateError(
            "OpenCode must use exactly one Edit and one Bash call; "
            f"recorded edit={edit_count}, bash={bash_count}"
        )

    for tool in tools:
        arguments = tool["input"]
        _validate_recorded_tool_arguments(
            tool["name"],
            arguments,
            fixture=fixture,
            expected_source=expected_source,
        )
        if tool["name"] in BASH_TOOLS:
            command = arguments.get("command")
            if not isinstance(command, str):
                raise GateError("Bash tool event did not expose a command string")
            _validate_pytest_command(command)
    return {
        "passed": True,
        "tool_event_count": len(tools),
        "tool_names_in_order": names,
        "required_categories": requirements,
        "exact_edit_calls": edit_count,
        "exact_bash_calls": bash_count,
        "allowed_tool_names": sorted(ALLOWED_TOOL_NAMES),
        "recorded_bash_command_was_exact": True,
        "recorded_tool_paths_and_patterns_within_fixture": True,
        "audit_timing": "POST_RUN_RETROSPECTIVE",
        "preventive_sandbox": False,
        "scope_note": (
            "This result audits recorded tool events after execution; it is evidence, "
            "not a preventive process, filesystem, or network sandbox."
        ),
    }


def _parse_porcelain_z(raw: bytes) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    records = raw.split(b"\0")
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        if len(record) < 4 or record[2:3] != b" ":
            raise GateError("could not parse git porcelain status")
        status = record[:2].decode("ascii", errors="strict")
        path = record[3:].decode("utf-8", errors="surrogateescape")
        entry = {"status": status, "path": path}
        if "R" in status or "C" in status:
            if index >= len(records) or not records[index]:
                raise GateError("truncated rename/copy record in git status")
            entry["source_path"] = records[index].decode("utf-8", errors="surrogateescape")
            index += 1
        entries.append(entry)
    return entries


def _pinned_contract_bytes() -> bytes:
    try:
        return _LAUNCHER._read_pinned_file(
            _LAUNCHER.SUBSTRATE_CONTRACT,
            _LAUNCHER.EXPECTED_CONTRACT_SHA256,
            "substrate contract",
        )
    except _LAUNCHER.LocalCoderError as exc:
        raise GateError(f"could not load the pinned substrate contract: {exc}") from exc


def _install_isolated_contract(config_home: Path) -> Path:
    data = _pinned_contract_bytes()
    destination = config_home / "opencode" / "AGENTS.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    if _sha256_file(destination) != _LAUNCHER.EXPECTED_CONTRACT_SHA256:
        raise GateError("isolated substrate contract failed its post-write digest check")
    return destination


def _agent_environment(
    base: Mapping[str, str],
    isolated_root: Path,
    copied_config: Path,
    *,
    server_url: str,
) -> dict[str, str]:
    normalized_server = _normalize_server_url(server_url)
    env = {
        key: value
        for key, value in base.items()
        if key.upper() in ALLOWED_PARENT_ENVIRONMENT
    }
    home = isolated_root / "home"
    config_home = isolated_root / "config"
    cache_home = isolated_root / "cache"
    data_home = isolated_root / "data"
    state_home = isolated_root / "state"
    appdata = isolated_root / "appdata"
    local_appdata = isolated_root / "local-appdata"
    temporary = isolated_root / "temp"
    config_directory = isolated_root / "config-dir"
    for directory in (
        home,
        config_home,
        cache_home,
        data_home,
        state_home,
        appdata,
        local_appdata,
        temporary,
        config_directory,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    installed_contract = _install_isolated_contract(config_home)
    try:
        inline_config = json.loads(copied_config.read_text(encoding="utf-8"))
        _validate_opencode_config(inline_config, normalized_server)
        inline_config["provider"][PROVIDER_ID]["options"]["baseURL"] = (
            normalized_server + "/v1"
        )
        inline_config["instructions"] = [str(installed_contract)]
        _LAUNCHER._validate_config_contract(
            inline_config,
            expected_base_url=normalized_server + "/v1",
            expected_instructions=[str(installed_contract)],
            source_config=False,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise GateError(f"could not prepare isolated OpenCode config: {exc}") from exc
    except _LAUNCHER.LocalCoderError as exc:
        raise GateError(f"isolated OpenCode config is unsafe: {exc}") from exc
    env.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "APPDATA": str(appdata),
            "LOCALAPPDATA": str(local_appdata),
            "TEMP": str(temporary),
            "TMP": str(temporary),
            "HOMEDRIVE": home.drive,
            "HOMEPATH": str(home)[len(home.drive) :],
            "XDG_CONFIG_HOME": str(config_home),
            "XDG_CACHE_HOME": str(cache_home),
            "XDG_DATA_HOME": str(data_home),
            "XDG_STATE_HOME": str(state_home),
            "OPENCODE_CONFIG": str(copied_config),
            "OPENCODE_CONFIG_CONTENT": json.dumps(
                inline_config, separators=(",", ":")
            ),
            "OPENCODE_CONFIG_DIR": str(config_directory),
            "OPENCODE_DISABLE_CLAUDE_CODE": "1",
            "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1",
            "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
            "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
            "OPENCODE_DISABLE_AUTOUPDATE": "1",
            "OPENCODE_DISABLE_SHARE": "1",
            "OPENCODE_DISABLE_TELEMETRY": "1",
            "OPENCODE_PURE": "1",
            "UGTOMS_LOCAL_CODER_BASE_URL": normalized_server,
            "UGTOMS_SUBSTRATE_CONTRACT_SHA256": _LAUNCHER.EXPECTED_CONTRACT_SHA256,
            "CI": "true",
            "NO_COLOR": "1",
            "DO_NOT_TRACK": "1",
            "NO_PROXY": "127.0.0.1,localhost,::1",
            "no_proxy": "127.0.0.1,localhost,::1",
        }
    )
    return env


def _validated_opencode_version(result: subprocess.CompletedProcess[bytes]) -> str:
    if result.returncode != 0:
        raise GateError("pinned OpenCode executable failed its --version preflight")
    stdout = result.stdout.decode("utf-8", errors="replace").strip().splitlines()
    stderr = result.stderr.decode("utf-8", errors="replace").strip().splitlines()
    version = (stdout or stderr or [""])[0].strip()
    if version != PINNED_OPENCODE_VERSION:
        raise GateError(
            f"OpenCode {PINNED_OPENCODE_VERSION} is required; found {version or 'unknown'}"
        )
    return version


def _agent_prompt() -> str:
    return (
        "Repair this disposable offline Python repository. This is a bounded test. "
        "Work only inside the current repository. First use Read to inspect README.md, "
        "the implementation, and tests. Use Grep or Glob for repository search. Diagnose "
        "the failing behavior, then use Edit exactly once to change only src/intervals.py. "
        "If the intended replacement is already present, do not call Edit again. Finally "
        "use Bash exactly once to run `python -m pytest -q`; when it passes, stop tool use "
        "immediately and give the final response. Do not "
        "change tests or configuration; do not create files; do not stage or commit; do "
        "not use network, web, package installation, task delegation, or any tool other "
        "than Read, Grep/Glob, Edit, Bash, and the local TodoWrite planner."
    )


def _write_sha256sums(run_dir: Path) -> None:
    checksum_path = run_dir / "SHA256SUMS"
    files = sorted(
        path for path in run_dir.rglob("*") if path.is_file() and path != checksum_path
    )
    lines = [f"{_sha256_file(path)}  {path.relative_to(run_dir).as_posix()}" for path in files]
    _write_text(checksum_path, "\n".join(lines) + ("\n" if lines else ""))


def _resolve_executable(value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        located = shutil.which(value)
        if located:
            candidate = Path(located)
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise GateError(f"pinned OpenCode executable does not exist: {candidate}")
    return candidate


def _run_gate(args: argparse.Namespace) -> tuple[int, Path]:
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S.%fZ")
    run_dir = output_root / stamp
    run_dir.mkdir(parents=False, exist_ok=False)
    stdout_path = run_dir / "opencode.stdout.jsonl"
    stderr_path = run_dir / "opencode.stderr.txt"
    stdout_path.write_bytes(b"")
    stderr_path.write_bytes(b"")
    evidence: dict[str, Any] = {
        "format": FORMAT,
        "started_at_utc": _utc_now(),
        "status": "FAILED",
        "scope": {
            "classification": "bounded_local_acceptance_gate",
            "proves": [
                "one native OpenAI tool-call JSON exchange",
                "one synthetic approximately 22K-token needle retrieval",
                "one disposable Python repository repair with required local tools",
            ],
            "does_not_prove": [
                "broad coding accuracy",
                "production safety or universal general-purpose capability",
                "UGTOMS substrate awareness",
                "understanding of any legacy or archive material",
                "preventive process, filesystem, or network sandboxing",
            ],
            "legacy_or_archive_material_loaded": False,
            "post_run_tool_audit_is_retrospective": True,
            "preventive_sandbox": False,
        },
    }

    try:
        server_url = _normalize_server_url(args.server_url)
        executable = _resolve_executable(args.opencode_exe)
        config_path, config_bytes, config = _load_pinned_config(args.config)
        config_gate = _validate_opencode_config(config, server_url)
        evidence["inputs"] = {
            "server_url": server_url,
            "opencode_executable": _public_path(
                executable, fallback="<PINNED_OPENCODE_EXECUTABLE>"
            ),
            "opencode_executable_sha256": _sha256_file(executable),
            "config_path": _public_path(
                config_path, fallback="<PINNED_LOCAL_CODER_CONFIG>"
            ),
            "config_sha256": _sha256_bytes(config_bytes),
            "output_directory": _public_path(
                run_dir, fallback="<OUTPUT_ROOT>/<RUN_DIRECTORY>"
            ),
        }
        evidence["config_gate"] = config_gate

        models, models_timing = _http_json(
            server_url, "/v1/models", timeout=args.http_timeout_seconds
        )
        model_gate = _validate_models_payload(models)
        model_gate["request"] = models_timing
        model_gate["raw_response"] = models
        evidence["model_gate"] = model_gate

        tool_response, tool_timing = _http_json(
            server_url,
            "/v1/chat/completions",
            method="POST",
            payload=_tool_probe_payload(),
            timeout=args.http_timeout_seconds,
        )
        tool_gate = _evaluate_tool_probe(tool_response)
        tool_gate["request"] = tool_timing
        tool_gate["response_model"] = tool_response.get("model")
        tool_gate["usage"] = tool_response.get("usage")
        evidence["native_tool_call_gate"] = tool_gate

        calibration_details: dict[int, dict[str, Any]] = {}

        def measure(record_count: int) -> int:
            messages, _ = _long_context_messages(record_count)
            count, details = _formatted_prompt_measurement(
                server_url, messages, timeout=args.http_timeout_seconds
            )
            calibration_details[record_count] = details
            return count

        selected_records, exact_tokens, calibration_trace = _select_near_target(measure)
        long_messages, long_metadata = _long_context_messages(selected_records)
        # Reuse the exact measurement made during calibration; messages are deterministic.
        selected_measurement = calibration_details[selected_records]
        long_payload = {
            "model": MODEL_ALIAS,
            "messages": long_messages,
            "temperature": 0,
            "seed": 20260831,
            "max_tokens": 48,
            "stream": False,
        }
        long_response, long_timing = _http_json(
            server_url,
            "/v1/chat/completions",
            method="POST",
            payload=long_payload,
            timeout=args.long_context_timeout_seconds,
        )
        long_gate = _evaluate_long_context_response(
            long_response, exact_prompt_tokens=exact_tokens
        )
        long_gate.update(long_metadata)
        long_gate.update(selected_measurement)
        long_gate.update(
            {
                "target_tokens": LONG_CONTEXT_TARGET_TOKENS,
                "tolerance_tokens": LONG_CONTEXT_TOLERANCE_TOKENS,
                "calibration_trace": calibration_trace,
                "inference_request": long_timing,
                "response_model": long_response.get("model"),
            }
        )
        evidence["long_context_gate"] = long_gate

        with tempfile.TemporaryDirectory(prefix="ugtoms-local-agent-gate-") as temporary:
            temporary_root = Path(temporary)
            fixture = temporary_root / "fixture"
            fixture.mkdir()
            fixture_evidence = _create_fixture(fixture)
            baseline = _run_process(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=fixture,
                timeout=args.pytest_timeout_seconds,
            )
            (run_dir / "fixture.baseline-pytest.stdout.txt").write_bytes(baseline.stdout)
            (run_dir / "fixture.baseline-pytest.stderr.txt").write_bytes(baseline.stderr)
            if baseline.returncode == 0:
                raise GateError("disposable fixture unexpectedly passed before agent repair")
            fixture_evidence["baseline_pytest"] = {
                "returncode": baseline.returncode,
                "passed": False,
                "stdout_sha256": _sha256_bytes(baseline.stdout),
                "stderr_sha256": _sha256_bytes(baseline.stderr),
            }

            isolated = temporary_root / "xdg"
            copied_config = isolated / "config" / "opencode" / "opencode.json"
            copied_config.parent.mkdir(parents=True, exist_ok=True)
            copied_config.write_bytes(config_bytes)
            agent_env = _agent_environment(
                os.environ,
                isolated,
                copied_config,
                server_url=server_url,
            )
            version = _run_process(
                [str(executable), "--version"],
                cwd=fixture,
                env=agent_env,
                timeout=30.0,
            )
            version_text = _validated_opencode_version(version)

            command = [
                str(executable),
                "--pure",
                "run",
                "--auto",
                "--model",
                MODEL_ID,
                "--agent",
                "local-coder",
                "--format",
                "json",
                _agent_prompt(),
            ]
            agent_started = time.perf_counter()
            try:
                agent = subprocess.run(
                    command,
                    cwd=str(fixture),
                    env=agent_env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=args.agent_timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                stdout_path.write_bytes(exc.stdout or b"")
                stderr_path.write_bytes(exc.stderr or b"")
                raise GateError(
                    f"OpenCode agent timed out after {args.agent_timeout_seconds:.1f}s"
                ) from exc
            agent_elapsed = time.perf_counter() - agent_started
            stdout_path.write_bytes(agent.stdout)
            stderr_path.write_bytes(agent.stderr)
            if agent.returncode != 0:
                raise GateError(f"OpenCode exited with status {agent.returncode}")
            events = _parse_json_events(agent.stdout)
            event_gate = _evaluate_tool_events(events, fixture)

            final = _run_process(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=fixture,
                timeout=args.pytest_timeout_seconds,
            )
            (run_dir / "fixture.final-pytest.stdout.txt").write_bytes(final.stdout)
            (run_dir / "fixture.final-pytest.stderr.txt").write_bytes(final.stderr)
            if final.returncode != 0:
                raise GateError("independent final pytest did not pass after agent repair")
            final_passed_tests = _verified_pytest_pass_count(
                final.stdout,
                final.stderr,
                expected=EXPECTED_FINAL_PYTEST_TESTS,
            )

            final_head = _run_git(fixture, ["rev-parse", "HEAD"]).decode("ascii").strip()
            head_unchanged = final_head == fixture_evidence["initial_head"]
            if not head_unchanged:
                raise GateError("agent changed Git HEAD or created a commit")
            status_raw = _run_git(fixture, ["status", "--porcelain=v1", "-z"])
            (run_dir / "fixture.git-status.porcelain-z.bin").write_bytes(status_raw)
            status = _parse_porcelain_z(status_raw)
            changed_paths = [entry["path"].replace("\\", "/") for entry in status]
            if changed_paths != [EXPECTED_SOURCE]:
                raise GateError(
                    f"agent changed files outside the expected source: {changed_paths!r}"
                )
            if status[0]["status"] == "??":
                raise GateError("expected source became untracked instead of being edited")
            source_after = _sha256_file(fixture / EXPECTED_SOURCE)
            if source_after == fixture_evidence["expected_source_sha256_before"]:
                raise GateError("expected source file was not materially changed")
            if _sha256_file(fixture / "tests/test_intervals.py") != fixture_evidence["tests_sha256_before"]:
                raise GateError("fixture tests changed despite the source-only rule")
            diff = _run_git(fixture, ["diff", "--binary", "HEAD", "--", EXPECTED_SOURCE])
            (run_dir / "fixture.source-change.patch").write_bytes(diff)
            if not diff:
                raise GateError("Git recorded no source diff after agent repair")

            fixture_evidence.update(
                {
                    "final_head": final_head,
                    "head_unchanged": head_unchanged,
                    "git_status": status,
                    "only_expected_source_changed": True,
                    "expected_source_sha256_after": source_after,
                    "final_pytest": {
                        "returncode": final.returncode,
                        "passed": True,
                        "passed_tests": final_passed_tests,
                        "total_tests": EXPECTED_FINAL_PYTEST_TESTS,
                        "result": (
                            f"{final_passed_tests}/{EXPECTED_FINAL_PYTEST_TESTS}"
                        ),
                        "stdout_sha256": _sha256_bytes(final.stdout),
                        "stderr_sha256": _sha256_bytes(final.stderr),
                    },
                }
            )
            evidence["opencode_agent_gate"] = {
                "passed": True,
                "version": version_text,
                "command_arguments": command[1:-1],
                "prompt_sha256": _sha256_bytes(_agent_prompt().encode("utf-8")),
                "elapsed_seconds": agent_elapsed,
                "returncode": agent.returncode,
                "stdout_sha256": _sha256_bytes(agent.stdout),
                "stderr_sha256": _sha256_bytes(agent.stderr),
                "json_event_count": len(events),
                "isolated_environment": {
                    "xdg_config": True,
                    "xdg_cache": True,
                    "xdg_data": True,
                    "xdg_state": True,
                    "home": True,
                    "project_config_disabled": True,
                    "autoupdate_disabled": True,
                    "parent_environment_reduced_to_operational_allowlist": True,
                    "credential_like_environment_inherited": False,
                    "digest_verified_substrate_contract_installed": True,
                },
                "tool_gate": event_gate,
                "preventive_sandbox": False,
                "fixture": fixture_evidence,
            }

        evidence["status"] = "PASSED"
        evidence["all_gates_passed"] = True
        exit_code = 0
    except (GateError, ValueError) as exc:
        evidence["status"] = "FAILED"
        evidence["all_gates_passed"] = False
        evidence["failure"] = {
            "type": type(exc).__name__,
            "message": _public_failure_message(
                str(exc),
                private_paths={
                    output_root: "<OUTPUT_ROOT>",
                    run_dir: "<OUTPUT_ROOT>/<RUN_DIRECTORY>",
                    args.opencode_exe: "<OPENCODE_EXECUTABLE>",
                    args.config: "<CONFIG_PATH>",
                },
            ),
        }
        exit_code = 1
    except Exception as exc:  # Preserve evidence for unexpected harness defects.
        evidence["status"] = "ERROR"
        evidence["all_gates_passed"] = False
        evidence["failure"] = {
            "type": type(exc).__name__,
            "message": _public_failure_message(
                str(exc),
                private_paths={
                    output_root: "<OUTPUT_ROOT>",
                    run_dir: "<OUTPUT_ROOT>/<RUN_DIRECTORY>",
                    args.opencode_exe: "<OPENCODE_EXECUTABLE>",
                    args.config: "<CONFIG_PATH>",
                },
            ),
        }
        exit_code = 2
    evidence["finished_at_utc"] = _utc_now()
    _write_json(run_dir / "evidence.json", evidence)
    _write_sha256sums(run_dir)
    return exit_code, run_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded local coding-agent gate; this is not a broad accuracy or "
            "substrate-awareness benchmark."
        )
    )
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--opencode-exe", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--http-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--long-context-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--agent-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--pytest-timeout-seconds", type=float, default=60.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for name in (
        "http_timeout_seconds",
        "long_context_timeout_seconds",
        "agent_timeout_seconds",
        "pytest_timeout_seconds",
    ):
        if getattr(args, name) <= 0:
            _parser().error(f"--{name.replace('_', '-')} must be positive")
    exit_code, run_dir = _run_gate(args)
    print(
        json.dumps(
            {
                "status": "PASSED" if exit_code == 0 else "FAILED",
                "evidence": _public_path(
                    run_dir, fallback="<OUTPUT_ROOT>/<RUN_DIRECTORY>"
                ),
            }
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
