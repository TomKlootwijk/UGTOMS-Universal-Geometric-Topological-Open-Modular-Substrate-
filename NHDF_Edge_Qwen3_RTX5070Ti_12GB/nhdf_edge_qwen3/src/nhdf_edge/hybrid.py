"""Fail-closed local model transport for a substrate-grounded coding agent.

The model transport is not the geometric-topological substrate and its GGUF
codec is not substrate compression.  This module keeps that boundary explicit
while sealing integrity, capability, resource, validation, and execution
policy around a GGUF/IQ2_M payload and pinned llama.cpp runtime.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


HYBRID_FORMAT = "nhdf-edge-hybrid-0.1"
HYBRID_MANIFEST = "NHDF_HYBRID_MANIFEST.json"
HYBRID_MANIFEST_SHA256 = "NHDF_HYBRID_MANIFEST.sha256"
HYBRID_VALIDATION_STATUSES = frozenset(
    {"UNCALIBRATED", "QUALITY_FAILED", "RESOURCE_FAILED", "VALIDATED"}
)
VALIDATED_KV_CACHE_TYPES = frozenset({"q8_0", "q4_0"})

FUNCTIONAL_PROMPTS: tuple[dict[str, Any], ...] = (
    {
        "id": "exact_ok",
        "user": "Reply with exactly the single word OK.",
        "max_tokens": 8,
        "accept": {"kind": "exact", "value": "OK"},
    },
    {
        "id": "arithmetic",
        "user": "Compute 17 multiplied by 19. Reply with only the integer.",
        "max_tokens": 12,
        "accept": {"kind": "regex", "value": r"^323\.?$"},
    },
    {
        "id": "integrity_vs_quality",
        "user": (
            "In one concise sentence, explain why a checksum passing does not prove "
            "that a compressed language model still produces useful answers."
        ),
        "max_tokens": 64,
        "accept": {
            "kind": "terms",
            "all_groups": [
                ["checksum", "integrity"],
                ["quality", "accur", "semantic", "useful"],
            ],
        },
    },
    {
        "id": "code",
        "user": (
            "Write only a short Python function named is_even(n) that returns whether "
            "an integer is even."
        ),
        "max_tokens": 64,
        "accept": {
            "kind": "terms",
            "all_groups": [["def is_even"], ["% 2", "& 1"]],
        },
    },
)


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _execution_profile_sha256(manifest: dict[str, Any]) -> str:
    """Bind gate evidence to the exact execution policy it exercised."""

    return hashlib.sha256(_canonical_bytes(manifest["execution_profile"])).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _write_manifest(root: Path, manifest: dict[str, Any]) -> Path:
    manifest_path = root / HYBRID_MANIFEST
    rendered = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write(manifest_path, rendered)
    _atomic_write(
        root / HYBRID_MANIFEST_SHA256,
        (hashlib.sha256(rendered).hexdigest() + "\n").encode("ascii"),
    )
    return manifest_path


def _relative_reference(root: Path, path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), root.resolve())).as_posix()


def _resolve_reference(root: Path, reference: str) -> Path:
    path = Path(reference)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sealed_file(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": _relative_reference(root, path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _append_event(manifest: dict[str, Any], kind: str, payload: dict[str, Any]) -> None:
    events = manifest.setdefault("events", [])
    previous = events[-1]["hash"] if events else "0" * 64
    event_body = {"kind": kind, "previous": previous, "payload": payload}
    events.append({**event_body, "hash": hashlib.sha256(_canonical_bytes(event_body)).hexdigest()})


def create_hybrid_artifact(
    artifact_dir: str | Path,
    *,
    model: str | Path,
    runtime: str | Path,
    benchmark_runtime: str | Path | None = None,
    server_runtime: str | Path | None = None,
    specification: str | Path | None = None,
    source_record: str | Path | None = None,
    assurance_evidence: Iterable[str | Path] = (),
    model_id: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    source_revision: str = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
    runtime_revision: str = "b6014",
    runtime_build_number: int = 6014,
    runtime_argument_profile: str = "b6014",
    total_parameters: int = 30_532_122_624,
    target_gpu: str = "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
    target_vram_mib: int = 12_227,
    maximum_context_tokens: int = 8_192,
    kv_cache_k: str = "q8_0",
    kv_cache_v: str = "q8_0",
) -> Path:
    """Create a zero-copy hybrid manifest and seal every referenced component."""

    if (
        isinstance(maximum_context_tokens, bool)
        or not isinstance(maximum_context_tokens, int)
        or maximum_context_tokens <= 0
    ):
        raise ValueError("maximum context tokens must be a positive integer")
    invalid_kv_types = [
        cache_type
        for cache_type in (kv_cache_k, kv_cache_v)
        if not isinstance(cache_type, str)
        or cache_type not in VALIDATED_KV_CACHE_TYPES
    ]
    if invalid_kv_types:
        supported = ", ".join(sorted(VALIDATED_KV_CACHE_TYPES))
        raise ValueError(
            "KV cache type must be one of the validated types "
            f"({supported}); got {invalid_kv_types!r}"
        )

    root = Path(artifact_dir).resolve()
    model_path = Path(model).resolve()
    runtime_path = Path(runtime).resolve()
    benchmark_path = Path(benchmark_runtime).resolve() if benchmark_runtime else None
    server_path = Path(server_runtime).resolve() if server_runtime else None
    specification_path = Path(specification).resolve() if specification else None
    source_record_path = Path(source_record).resolve() if source_record else None
    required = [model_path, runtime_path]
    required.extend(
        path
        for path in (benchmark_path, server_path, specification_path, source_record_path)
        if path
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"hybrid component(s) not found: {missing}")

    if root.exists() and any(root.iterdir()) and not (root / HYBRID_MANIFEST).is_file():
        raise FileExistsError(f"refusing to overwrite non-hybrid directory: {root}")
    root.mkdir(parents=True, exist_ok=True)

    runtime_files = [runtime_path]
    if benchmark_path is not None and benchmark_path != runtime_path:
        runtime_files.append(benchmark_path)
    if server_path is not None and server_path not in runtime_files:
        runtime_files.append(server_path)
    for dll in sorted(runtime_path.parent.glob("*.dll")):
        if dll not in runtime_files:
            runtime_files.append(dll)

    evidence_records = []
    for evidence in assurance_evidence:
        path = Path(evidence).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"assurance evidence not found: {path}")
        evidence_records.append(_sealed_file(root, path))

    payload = _sealed_file(root, model_path)
    payload.update(
        {
            "link_mode": "workspace-relative-reference",
            "ggml_tensor_bytes": 9_864_300_544,
            "file_size_bits_per_parameter": payload["bytes"] * 8.0 / total_parameters,
        }
    )
    manifest: dict[str, Any] = {
        "format": HYBRID_FORMAT,
        "created_at_utc": _utc_now(),
        "artifact_kind": "external-codec-reference",
        "model": {
            "id": model_id,
            "source_revision": source_revision,
            "parameters": total_parameters,
            "source_bf16_tensor_bytes": 61_064_245_248,
        },
        "weight_codec": {
            "owner": "ggml/Bartowski",
            "container": "GGUF",
            "profile": "IQ2_M mixed-bit",
            "nhdf_native_codec": False,
            "attribution": (
                "The low-bit tensor encoding is GGUF/IQ2_M. NHDF does not claim "
                "authorship of this codec."
            ),
        },
        "payload": payload,
        "runtime": {
            "implementation": "llama.cpp",
            "revision": runtime_revision,
            "build_number": runtime_build_number,
            "argument_profile": runtime_argument_profile,
            "files": [_sealed_file(root, path) for path in runtime_files],
            "entrypoint": _relative_reference(root, runtime_path),
            "benchmark_entrypoint": (
                _relative_reference(root, benchmark_path) if benchmark_path else None
            ),
            "server_entrypoint": (
                _relative_reference(root, server_path) if server_path else None
            ),
        },
        "substrate": {
            "name": "UGTOMS-grounded local-agent transport",
            "role": [
                "bind a selected substrate contract without claiming a substrate-native tensor codec",
                "SHA-256-sealed local provenance and evidence chain",
                "typed capability and validation state",
                "bounded GPU/context allocation",
                "fail-closed execution policy",
                "replaceable tensor-codec boundary",
            ],
            "specification": (
                _sealed_file(root, specification_path) if specification_path else None
            ),
        },
        "source_record": _sealed_file(root, source_record_path) if source_record_path else None,
        "assurance_evidence": evidence_records,
        "execution_profile": {
            "model_family": "qwen3moe",
            "gpu_layers": 999,
            "split_mode": "none",
            "threads": 4,
            "threads_batch": 4,
            "batch": 2048,
            "ubatch": 512,
            "priority": 2,
            "priority_batch": 2,
            "poll": 50,
            "expected_offloaded_layers": [49, 49],
            "maximum_context_tokens": maximum_context_tokens,
            "kv_cache_k": kv_cache_k,
            "kv_cache_v": kv_cache_v,
            "flash_attention": True,
            "sampling": {"temperature": 0.0, "top_k": 1, "seed": 2026},
            "prompt_template": "explicit-current-nonthinking-qwen-chatml",
            "minimum_generation_tokens_per_second": 80.0,
        },
        "resource_contract": {
            "target_gpu": target_gpu,
            "target_vram_mib": target_vram_mib,
            # The allocated-8K gate measured a 10,200 MiB increase above the
            # idle desktop baseline.  Requiring that increase plus the full
            # reserve keeps the declared margin intact even when the desktop
            # baseline is higher at launch time.
            "measured_incremental_peak_vram_mib": 10_200,
            "required_free_vram_mib": 10_712,
            "reserve_vram_mib": 512,
            "source_bf16_weight_mib": 61_064_245_248 / (1024**2),
            "maximum_validated_context_tokens": maximum_context_tokens,
        },
        "validation": {
            "status": "UNCALIBRATED",
            "deployment_loadable": False,
            "hardware_scope": target_gpu,
            "evidence": None,
        },
        "events": [],
    }
    _append_event(
        manifest,
        "ARTIFACT_CREATED",
        {
            "payload_sha256": payload["sha256"],
            "payload_bytes": payload["bytes"],
            "codec": "GGUF/IQ2_M",
            "validation_status": "UNCALIBRATED",
        },
    )
    return _write_manifest(root, manifest)


def load_hybrid_manifest(
    artifact_dir: str | Path, *, verify_manifest: bool = True
) -> dict[str, Any]:
    root = Path(artifact_dir).resolve()
    manifest_path = root / HYBRID_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"hybrid manifest not found: {manifest_path}")
    rendered = manifest_path.read_bytes()
    if verify_manifest:
        digest_path = root / HYBRID_MANIFEST_SHA256
        if not digest_path.is_file():
            raise OSError(f"hybrid manifest digest not found: {digest_path}")
        expected = digest_path.read_text(encoding="ascii").strip().lower()
        actual = hashlib.sha256(rendered).hexdigest()
        if actual != expected:
            raise OSError("hybrid manifest SHA-256 mismatch")
    manifest = json.loads(rendered)
    if manifest.get("format") != HYBRID_FORMAT:
        raise ValueError(f"unsupported hybrid format: {manifest.get('format')!r}")
    status = manifest.get("validation", {}).get("status")
    if status not in HYBRID_VALIDATION_STATUSES:
        raise ValueError(f"unsupported hybrid validation status: {status!r}")
    return manifest


def _verify_record(root: Path, record: dict[str, Any], label: str, *, hash_file: bool) -> dict[str, Any]:
    path = _resolve_reference(root, record["path"])
    result: dict[str, Any] = {"component": label, "path": str(path), "ok": True}
    if not path.is_file():
        result.update(ok=False, error="file missing")
        return result
    actual_bytes = path.stat().st_size
    if actual_bytes != int(record["bytes"]):
        result.update(ok=False, error="byte length mismatch", actual_bytes=actual_bytes)
        return result
    if hash_file:
        actual_hash = sha256_file(path)
        if actual_hash.lower() != str(record["sha256"]).lower():
            result.update(ok=False, error="SHA-256 mismatch", actual_sha256=actual_hash)
    return result


def _verify_event_chain(manifest: dict[str, Any]) -> dict[str, Any]:
    previous = "0" * 64
    for index, event in enumerate(manifest.get("events", [])):
        event_body = {
            "kind": event.get("kind"),
            "previous": event.get("previous"),
            "payload": event.get("payload"),
        }
        expected = hashlib.sha256(_canonical_bytes(event_body)).hexdigest()
        if event.get("previous") != previous:
            return {
                "component": "event_chain",
                "ok": False,
                "error": f"event {index} previous hash mismatch",
            }
        if event.get("hash") != expected:
            return {
                "component": "event_chain",
                "ok": False,
                "error": f"event {index} hash mismatch",
            }
        previous = expected
    return {"component": "event_chain", "ok": True, "events": len(manifest.get("events", []))}


def verify_hybrid_artifact(
    artifact_dir: str | Path,
    *,
    verify_payload_hash: bool = True,
    require_validated: bool = False,
) -> dict[str, Any]:
    root = Path(artifact_dir).resolve()
    manifest = load_hybrid_manifest(root)
    checks = [
        _verify_record(root, manifest["payload"], "model payload", hash_file=verify_payload_hash),
        _verify_event_chain(manifest),
    ]
    checks.extend(
        _verify_record(root, record, f"runtime:{index}", hash_file=True)
        for index, record in enumerate(manifest["runtime"]["files"])
    )
    optional_records: list[tuple[str, dict[str, Any] | None]] = [
        ("specification", manifest.get("substrate", {}).get("specification")),
        ("source_record", manifest.get("source_record")),
    ]
    optional_records.extend(
        (f"assurance_evidence:{index}", record)
        for index, record in enumerate(manifest.get("assurance_evidence", []))
    )
    validation_record = manifest.get("validation", {}).get("evidence")
    optional_records.append(("validation_evidence", validation_record))
    checks.extend(
        _verify_record(root, record, label, hash_file=True)
        for label, record in optional_records
        if record is not None
    )
    failures = [check for check in checks if not check["ok"]]
    status = manifest["validation"]["status"]
    if require_validated and status != "VALIDATED":
        failures.append(
            {
                "component": "validation",
                "ok": False,
                "error": f"validation status is {status}, not VALIDATED",
            }
        )
    return {
        "artifact": str(root),
        "format": HYBRID_FORMAT,
        "artifact_kind": manifest["artifact_kind"],
        "codec": manifest["weight_codec"],
        "validation_status": status,
        "deployment_loadable": status == "VALIDATED" and not failures,
        "payload_hash_checked": verify_payload_hash,
        "checks": checks,
        "failures": failures,
        "ok": not failures,
    }


def require_hybrid_validated(artifact_dir: str | Path, *, allow_unvalidated: bool = False) -> None:
    status = load_hybrid_manifest(artifact_dir)["validation"]["status"]
    if status != "VALIDATED" and not allow_unvalidated:
        raise RuntimeError(
            f"refusing to execute hybrid artifact with validation status {status}; "
            "run the functional gate or pass the explicit research-only override"
        )


def _gpu_name() -> str | None:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode or not completed.stdout.strip():
        return None
    return completed.stdout.strip().splitlines()[0].strip()


def _gpu_sample() -> tuple[int, int, int] | None:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode or not completed.stdout.strip():
        return None
    fields = [field.strip() for field in completed.stdout.strip().splitlines()[0].split(",")]
    return int(fields[0]), int(fields[1]), int(fields[2])


def _sample_until(stop: threading.Event, samples: list[dict[str, float | int]]) -> None:
    while not stop.is_set():
        sample = _gpu_sample()
        if sample is not None:
            total, used, utilization = sample
            samples.append(
                {
                    "monotonic_seconds": time.perf_counter(),
                    "total_memory_mib": total,
                    "used_memory_mib": used,
                    "utilization_percent": utilization,
                }
            )
        stop.wait(0.10)


def _chatml(user: str) -> str:
    return (
        "<|im_start|>system\nYou are a precise assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _clean_generation(stdout: str) -> str:
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", stdout)
    text = text.replace("[end of text]", "")
    return text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()


def _accepted(text: str, rule: dict[str, Any]) -> bool:
    normalized = text.strip()
    if rule["kind"] == "exact":
        return normalized == rule["value"]
    if rule["kind"] == "regex":
        return re.fullmatch(rule["value"], normalized, flags=re.IGNORECASE) is not None
    if rule["kind"] == "terms":
        lowered = normalized.lower()
        return all(
            any(term.lower() in lowered for term in group) for group in rule["all_groups"]
        )
    raise ValueError(f"unsupported acceptance rule: {rule['kind']}")


def _metric(pattern: str, stderr: str) -> float | None:
    match = re.search(pattern, stderr)
    return float(match.group(1)) if match else None


def _offload_metric(stderr: str) -> list[int] | None:
    match = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers", stderr)
    return [int(match.group(1)), int(match.group(2))] if match else None


def _preflight(manifest: dict[str, Any], *, context: int) -> dict[str, Any]:
    profile = manifest["execution_profile"]
    if context <= 0 or context > int(profile["maximum_context_tokens"]):
        raise ValueError(
            f"context {context} exceeds validated maximum {profile['maximum_context_tokens']}"
        )
    sample = _gpu_sample()
    if sample is None:
        raise RuntimeError("nvidia-smi resource preflight failed")
    actual_gpu = _gpu_name()
    expected_gpu = str(manifest["resource_contract"]["target_gpu"])
    if actual_gpu is None:
        raise RuntimeError("nvidia-smi GPU identity preflight failed")
    if actual_gpu != expected_gpu:
        raise RuntimeError(
            f"GPU identity mismatch: detected {actual_gpu!r}, profile requires {expected_gpu!r}"
        )
    total, used, _ = sample
    target_total = int(manifest["resource_contract"]["target_vram_mib"])
    if total < target_total:
        raise RuntimeError(
            f"GPU capacity mismatch: detected {total} MiB, profile requires at least {target_total} MiB"
        )
    free = total - used
    required = int(manifest["resource_contract"]["required_free_vram_mib"])
    if free < required:
        raise RuntimeError(
            f"insufficient free VRAM: {free} MiB free, {required} MiB required by profile"
        )
    return {
        "gpu": actual_gpu,
        "total_mib": total,
        "used_mib": used,
        "free_mib": free,
        "required_mib": required,
    }


def run_hybrid_prompt(
    artifact_dir: str | Path,
    *,
    prompt: str,
    max_tokens: int = 64,
    context: int = 512,
    seed: int = 2026,
    acceptance_rule: dict[str, Any] | None = None,
    allow_unvalidated: bool = False,
    verify_payload_hash: bool = True,
    monitor_resources: bool = False,
) -> dict[str, Any]:
    root = Path(artifact_dir).resolve()
    verification = verify_hybrid_artifact(
        root, verify_payload_hash=verify_payload_hash, require_validated=False
    )
    if not verification["ok"]:
        raise OSError(f"hybrid artifact integrity failed: {verification['failures']}")
    require_hybrid_validated(root, allow_unvalidated=allow_unvalidated)
    manifest = load_hybrid_manifest(root)
    preflight = _preflight(manifest, context=context)
    runtime = _resolve_reference(root, manifest["runtime"]["entrypoint"])
    model = _resolve_reference(root, manifest["payload"]["path"])
    runtime_profile = manifest["runtime"].get("argument_profile", "b6014")
    if runtime_profile == "b6014":
        flash_arguments = ["-fa"]
        single_turn_arguments = ["--no-conversation"]
        evidence_log_arguments: list[str] = []
    elif runtime_profile == "current-2026":
        flash_arguments = ["-fa", "on"]
        single_turn_arguments = ["--no-conversation"]
        # Current llama.cpp moved tensor-placement details from INFO to TRACE.
        # Request TRACE only for monitored evidence runs so the validation gate
        # can prove the declared 49/49 GPU offload without adding noise to normal
        # inference.
        evidence_log_arguments = ["-lv", "4"] if monitor_resources else []
    else:
        raise ValueError(f"unsupported llama.cpp argument profile: {runtime_profile!r}")
    command = [
        str(runtime),
        "-m",
        str(model),
        "-ngl",
        str(manifest["execution_profile"]["gpu_layers"]),
        "-sm",
        str(manifest["execution_profile"].get("split_mode", "none")),
        "-c",
        str(context),
        "-n",
        str(max_tokens),
        "-ctk",
        manifest["execution_profile"]["kv_cache_k"],
        "-ctv",
        manifest["execution_profile"]["kv_cache_v"],
        *flash_arguments,
        "-t",
        str(manifest["execution_profile"].get("threads", 4)),
        "-tb",
        str(manifest["execution_profile"].get("threads_batch", 4)),
        "-b",
        str(manifest["execution_profile"].get("batch", 2048)),
        "-ub",
        str(manifest["execution_profile"].get("ubatch", 512)),
        "--prio",
        str(manifest["execution_profile"].get("priority", 2)),
        "--prio-batch",
        str(manifest["execution_profile"].get("priority_batch", 2)),
        "--poll",
        str(manifest["execution_profile"].get("poll", 50)),
        "--temp",
        "0",
        "--top-k",
        "1",
        "-s",
        str(seed),
        "--no-display-prompt",
        "--simple-io",
        *single_turn_arguments,
        *evidence_log_arguments,
        "--no-warmup",
        "-p",
        _chatml(prompt),
    ]
    samples: list[dict[str, float | int]] = []
    stop = threading.Event()
    sampler = (
        threading.Thread(target=_sample_until, args=(stop, samples), daemon=True)
        if monitor_resources
        else None
    )
    start = time.perf_counter()
    if sampler is not None:
        sampler.start()
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    elapsed = time.perf_counter() - start
    if sampler is not None:
        stop.set()
        sampler.join(timeout=2)
    generated = _clean_generation(process.stdout)
    peak = max((int(sample["used_memory_mib"]) for sample in samples), default=None)
    peak_utilization = max(
        (int(sample["utilization_percent"]) for sample in samples), default=None
    )
    baseline = preflight["used_mib"]
    passed = process.returncode == 0
    if acceptance_rule is not None:
        passed = passed and _accepted(generated, acceptance_rule)
    return {
        "prompt": prompt,
        "acceptance_rule": acceptance_rule,
        "passed": passed,
        "exit_code": process.returncode,
        "generated_text": generated,
        "context_tokens": context,
        "max_new_tokens": max_tokens,
        "elapsed_seconds": elapsed,
        "resource_preflight": preflight,
        "baseline_gpu_memory_mib": baseline,
        "peak_gpu_memory_mib": peak,
        "incremental_peak_gpu_memory_mib": peak - baseline if peak is not None else None,
        "peak_gpu_utilization_percent": peak_utilization,
        "resource_monitoring_enabled": monitor_resources,
        "samples": len(samples),
        "llama_metrics": {
            "load_ms": _metric(r"load time\s*=\s*([0-9.]+) ms", process.stderr),
            "prompt_tokens_per_second": _metric(
                r"prompt eval time.*?([0-9.]+) tokens per second", process.stderr
            ),
            "decode_tokens_per_second": _metric(
                r"(?<!prompt )eval time.*?([0-9.]+) tokens per second", process.stderr
            ),
            "total_ms": _metric(r"total time\s*=\s*([0-9.]+) ms", process.stderr),
            "cuda_model_buffer_mib": _metric(
                r"CUDA0 model buffer size\s*=\s*([0-9.]+) MiB", process.stderr
            ),
            "cpu_mapped_model_buffer_mib": _metric(
                r"CPU_Mapped model buffer size\s*=\s*([0-9.]+) MiB", process.stderr
            ),
            "cuda_kv_buffer_mib": _metric(
                r"CUDA0 KV buffer size\s*=\s*([0-9.]+) MiB", process.stderr
            ),
            "cuda_compute_buffer_mib": _metric(
                r"CUDA0 compute buffer size\s*=\s*([0-9.]+) MiB", process.stderr
            ),
            "offloaded_layers": _offload_metric(process.stderr),
        },
        "stderr_tail": process.stderr[-4000:],
    }


def _run_benchmark(
    root: Path,
    manifest: dict[str, Any],
    *,
    repetitions: int,
    prompt_tokens: int,
    generation_tokens: int,
) -> dict[str, Any]:
    entrypoint = manifest["runtime"].get("benchmark_entrypoint")
    if not entrypoint:
        raise ValueError("hybrid manifest does not declare a benchmark entrypoint")
    runtime = _resolve_reference(root, entrypoint)
    model = _resolve_reference(root, manifest["payload"]["path"])
    completed = subprocess.run(
        [
            str(runtime),
            "-m",
            str(model),
            "-p",
            str(prompt_tokens),
            "-n",
            str(generation_tokens),
            "-r",
            str(repetitions),
            "-ctk",
            str(manifest["execution_profile"]["kv_cache_k"]),
            "-ctv",
            str(manifest["execution_profile"]["kv_cache_v"]),
            "-fa",
            "1",
            "-ngl",
            str(manifest["execution_profile"]["gpu_layers"]),
            "-sm",
            str(manifest["execution_profile"].get("split_mode", "none")),
            "-b",
            str(manifest["execution_profile"].get("batch", 2048)),
            "-ub",
            str(manifest["execution_profile"].get("ubatch", 512)),
            "-t",
            str(manifest["execution_profile"].get("threads", 4)),
            "--prio",
            str(manifest["execution_profile"].get("priority", 2)),
            "--poll",
            str(manifest["execution_profile"].get("poll", 50)),
            "-o",
            "json",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode:
        raise RuntimeError(f"llama-bench failed with exit code {completed.returncode}")
    records = json.loads(completed.stdout)
    prompt_record = next(record for record in records if int(record["n_prompt"]) > 0)
    generation_record = next(record for record in records if int(record["n_gen"]) > 0)
    return {
        "runtime_records": records,
        "prompt": {
            "tokens": int(prompt_record["n_prompt"]),
            "average_tokens_per_second": float(prompt_record["avg_ts"]),
            "standard_deviation_tokens_per_second": float(prompt_record["stddev_ts"]),
            "samples_tokens_per_second": prompt_record["samples_ts"],
        },
        "generation": {
            "tokens": int(generation_record["n_gen"]),
            "average_tokens_per_second": float(generation_record["avg_ts"]),
            "standard_deviation_tokens_per_second": float(generation_record["stddev_ts"]),
            "samples_tokens_per_second": generation_record["samples_ts"],
        },
    }


def set_hybrid_validation(
    artifact_dir: str | Path,
    status: str,
    *,
    evidence_path: str | Path,
) -> Path:
    if status not in HYBRID_VALIDATION_STATUSES:
        raise ValueError(f"unsupported validation status: {status}")
    root = Path(artifact_dir).resolve()
    source = Path(evidence_path).resolve()
    evidence = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("validation evidence must be a non-empty JSON object")
    manifest = load_hybrid_manifest(root)
    if status == "VALIDATED":
        aggregate = evidence.get("aggregate", {})
        benchmark = evidence.get("benchmark", {})
        generation = benchmark.get("generation", {})
        thresholds = evidence.get("thresholds", {})
        required_generation_tps = thresholds.get("minimum_generation_tokens_per_second")
        measured_generation_tps = generation.get("average_tokens_per_second")
        if "allocated_context_passed" in aggregate:
            allocated_context_passed = (
                aggregate.get("allocated_context_passed") is True
                and aggregate.get("allocated_context_tokens")
                == manifest["execution_profile"]["maximum_context_tokens"]
            )
        else:
            # Compatibility for evidence sealed before context-generic names
            # were introduced. Its execution-profile digest still binds the
            # evidence to the manifest-selected maximum context.
            allocated_context_passed = aggregate.get("allocated_8k_passed") is True
        promotion_checks = {
            "passed": evidence.get("passed") is True,
            "gate_kind": evidence.get("experiment")
            == "nhdf_hybrid_full_model_functional_gate",
            "artifact_format": evidence.get("artifact_format") == HYBRID_FORMAT,
            "payload": evidence.get("payload", {}).get("sha256")
            == manifest["payload"]["sha256"],
            "runtime_revision": evidence.get("runtime_revision")
            == manifest["runtime"]["revision"],
            "runtime_build_number": evidence.get("runtime_build_number")
            == manifest["runtime"].get("build_number"),
            "runtime_argument_profile": evidence.get("runtime_argument_profile")
            == manifest["runtime"].get("argument_profile"),
            "execution_profile": evidence.get("execution_profile_sha256")
            == _execution_profile_sha256(manifest),
            "functional_prompts": int(aggregate.get("functional_prompts_passed", -1))
            == int(aggregate.get("functional_prompts_total", -2))
            and int(aggregate.get("functional_prompts_total", 0)) > 0,
            "allocated_context": allocated_context_passed,
            "full_offload": aggregate.get("full_offload_passed") is True,
            "resource": aggregate.get("resource_gate_passed") is True,
            "throughput": aggregate.get("throughput_gate_passed") is True,
            "generation_threshold": isinstance(required_generation_tps, (int, float))
            and isinstance(measured_generation_tps, (int, float))
            and measured_generation_tps >= required_generation_tps,
        }
        failed_checks = [name for name, passed in promotion_checks.items() if not passed]
        if failed_checks:
            raise ValueError(
                "VALIDATED requires complete gate evidence; failed checks: "
                + ", ".join(failed_checks)
            )
    destination = root / "evidence" / "functional_gate.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source != destination.resolve():
        shutil.copy2(source, destination)
    previous = manifest["validation"]["status"]
    manifest["validation"] = {
        "status": status,
        "previous_status": previous,
        "deployment_loadable": status == "VALIDATED",
        "hardware_scope": manifest["resource_contract"]["target_gpu"],
        "evidence": _sealed_file(root, destination),
        "evidence_summary": {
            "passed": evidence.get("passed"),
            "functional_prompts_passed": evidence.get("aggregate", {}).get(
                "functional_prompts_passed"
            ),
            "peak_gpu_memory_mib": evidence.get("aggregate", {}).get(
                "peak_gpu_memory_mib"
            ),
            "generation_tokens_per_second": evidence.get("benchmark", {})
            .get("generation", {})
            .get("average_tokens_per_second"),
        },
    }
    _append_event(
        manifest,
        "VALIDATION_STATUS_CHANGED",
        {
            "previous_status": previous,
            "status": status,
            "evidence_sha256": manifest["validation"]["evidence"]["sha256"],
        },
    )
    return _write_manifest(root, manifest)


def gate_hybrid_artifact(
    artifact_dir: str | Path,
    *,
    output: str | Path | None = None,
    seed: int = 2026,
    benchmark_repetitions: int = 3,
    minimum_generation_tokens_per_second: float = 80.0,
    reserve_vram_mib: int = 512,
    verify_payload_hash: bool = True,
) -> dict[str, Any]:
    root = Path(artifact_dir).resolve()
    verification = verify_hybrid_artifact(root, verify_payload_hash=verify_payload_hash)
    if not verification["ok"]:
        raise OSError(f"hybrid artifact integrity failed: {verification['failures']}")
    manifest = load_hybrid_manifest(root)
    contract_reserve = int(manifest["resource_contract"]["reserve_vram_mib"])
    if reserve_vram_mib < contract_reserve:
        raise ValueError(
            f"reserve {reserve_vram_mib} MiB is below the artifact contract "
            f"({contract_reserve} MiB)"
        )
    contract_minimum_tps = float(
        manifest["execution_profile"]["minimum_generation_tokens_per_second"]
    )
    if minimum_generation_tokens_per_second < contract_minimum_tps:
        raise ValueError(
            f"generation threshold {minimum_generation_tokens_per_second} tok/s is below "
            f"the artifact contract ({contract_minimum_tps} tok/s)"
        )
    results = []
    for prompt in FUNCTIONAL_PROMPTS:
        result = run_hybrid_prompt(
            root,
            prompt=prompt["user"],
            max_tokens=int(prompt["max_tokens"]),
            context=512,
            seed=seed,
            acceptance_rule=prompt["accept"],
            allow_unvalidated=True,
            verify_payload_hash=False,
            monitor_resources=True,
        )
        result["id"] = prompt["id"]
        results.append(result)
    validated_context = int(manifest["execution_profile"]["maximum_context_tokens"])
    residency = run_hybrid_prompt(
        root,
        prompt=FUNCTIONAL_PROMPTS[0]["user"],
        max_tokens=int(FUNCTIONAL_PROMPTS[0]["max_tokens"]),
        context=validated_context,
        seed=seed,
        acceptance_rule=FUNCTIONAL_PROMPTS[0]["accept"],
        allow_unvalidated=True,
        verify_payload_hash=False,
        monitor_resources=True,
    )
    residency["id"] = "allocated_context_exact_ok"
    benchmark = _run_benchmark(
        root,
        manifest,
        repetitions=benchmark_repetitions,
        prompt_tokens=64,
        generation_tokens=64,
    )
    peaks = [
        int(result["peak_gpu_memory_mib"])
        for result in [*results, residency]
        if result["peak_gpu_memory_mib"] is not None
    ]
    peak = max(peaks) if peaks else None
    total_vram = int(residency["resource_preflight"]["total_mib"])
    contract_target_vram = int(manifest["resource_contract"]["target_vram_mib"])
    expected_offload = manifest["execution_profile"]["expected_offloaded_layers"]
    offload_ok = all(
        result["llama_metrics"]["offloaded_layers"] == expected_offload
        for result in [*results, residency]
    )
    prompts_passed = sum(bool(result["passed"]) for result in results)
    resource_passed = peak is not None and peak <= total_vram - reserve_vram_mib
    benchmark_passed = (
        benchmark["generation"]["average_tokens_per_second"]
        >= minimum_generation_tokens_per_second
    )
    passed = (
        prompts_passed == len(results)
        and bool(residency["passed"])
        and resource_passed
        and benchmark_passed
        and offload_ok
    )
    evidence: dict[str, Any] = {
        "experiment": "nhdf_hybrid_full_model_functional_gate",
        "generated_at_utc": _utc_now(),
        "scope": (
            "complete Qwen model through a sealed external-codec transport for a "
            "substrate-grounded agent; GGUF/IQ2_M remains the attributed tensor codec"
        ),
        "artifact": str(root),
        "artifact_format": HYBRID_FORMAT,
        "payload": manifest["payload"],
        "codec": manifest["weight_codec"],
        "runtime_revision": manifest["runtime"]["revision"],
        "runtime_build_number": manifest["runtime"].get("build_number"),
        "runtime_argument_profile": manifest["runtime"].get("argument_profile"),
        "execution_profile_sha256": _execution_profile_sha256(manifest),
        "functional_results": results,
        "allocated_context_residency_result": residency,
        "benchmark": benchmark,
        "thresholds": {
            "functional_prompts_required": len(results),
            "allocated_context_tokens": validated_context,
            "allocated_context_exact_response_required": True,
            "full_offload_required": expected_offload,
            "reserve_vram_mib": reserve_vram_mib,
            "minimum_generation_tokens_per_second": minimum_generation_tokens_per_second,
        },
        "aggregate": {
            "functional_prompts_passed": prompts_passed,
            "functional_prompts_total": len(results),
            "allocated_context_tokens": validated_context,
            "allocated_context_passed": bool(residency["passed"]),
            "full_offload_passed": offload_ok,
            "peak_gpu_memory_mib": peak,
            "target_vram_mib": total_vram,
            "contract_target_vram_mib": contract_target_vram,
            "headroom_mib": total_vram - peak if peak is not None else None,
            "resource_gate_passed": resource_passed,
            "throughput_gate_passed": benchmark_passed,
        },
        "passed": passed,
        "status": "functional-hybrid-pass" if passed else "functional-hybrid-fail",
        "limitations": [
            "The maximum-context run allocates the cache and executes a short prompt; "
            "it is not a filled-context quality test.",
            "The weight codec is external GGUF/IQ2_M, not an NHDF-native tensor codec.",
        ],
    }
    destination = Path(output).resolve() if output else root / "evidence" / "functional_gate.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(destination, (json.dumps(evidence, indent=2) + "\n").encode("utf-8"))
    if passed:
        disposition = "VALIDATED"
    elif not resource_passed or not benchmark_passed or not offload_ok:
        disposition = "RESOURCE_FAILED"
    else:
        disposition = "QUALITY_FAILED"
    set_hybrid_validation(root, disposition, evidence_path=destination)
    return evidence
