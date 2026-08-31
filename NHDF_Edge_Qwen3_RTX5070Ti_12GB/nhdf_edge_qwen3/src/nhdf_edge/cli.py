"""Command-line interface for conversion, verification and feasibility checks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HYBRID_MANIFEST = "NHDF_HYBRID_MANIFEST.json"
PACK_VALIDATION_STATUSES = frozenset(
    {"UNCALIBRATED", "QUALITY_FAILED", "RESOURCE_FAILED", "VALIDATED"}
)


def _json(data: object) -> None:
    print(json.dumps(data, indent=2, sort_keys=True), flush=True)


def command_estimate(args: argparse.Namespace) -> int:
    from .config import load_config
    from .metrics import context_sweep, estimate, residual_fraction_sweep

    cfg = load_config(args.config)
    result = estimate(cfg, context_tokens=args.context)
    payload = {
        "estimate": result.to_dict(),
        "residual_fraction_sweep": residual_fraction_sweep(cfg, [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]),
        "context_sweep": context_sweep(cfg, [4096, 8192, 16384, 32768]),
        "status": "analytical-not-measured",
    }
    _json(payload)
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


def command_pack(args: argparse.Namespace) -> int:
    from .checkpoint import pack_checkpoint
    from .config import load_config

    cfg = load_config(args.config)
    manifest = pack_checkpoint(
        args.source,
        args.output,
        cfg,
        include=args.include,
        exclude=args.exclude,
        max_tensors=args.max_tensors,
        hessian_path=args.hessian,
    )
    print(manifest)
    return 0


def command_verify(args: argparse.Namespace) -> int:
    if (Path(args.pack) / HYBRID_MANIFEST).is_file():
        from .hybrid import verify_hybrid_artifact

        result = verify_hybrid_artifact(
            args.pack,
            verify_payload_hash=not args.quick,
            require_validated=False,
        )
        _json(result)
        return 0 if result["ok"] else 2
    from .format import PackReader
    from .quantize import verify_parity

    reader = PackReader(args.pack, verify_crc=True)
    result = reader.verify_all()
    partial = bool(reader.manifest.get("partial_pack", False))
    result["partial_pack"] = partial
    result["complete"] = not partial
    result["validation_status"] = reader.validation_status
    result["deployment_loadable"] = reader.validation_status == "VALIDATED" and not partial
    if partial and not args.allow_partial:
        result["failures"].append(
            {
                "tensor": "<manifest>",
                "error": "pack is partial (pass --allow-partial for an integrity-only smoke check)",
            }
        )
        result["ok"] = False
    parity_failures = []
    names = reader.names() if args.parity_all else reader.names()[: args.parity_sample]
    for name in names:
        check = verify_parity(reader.load(name))
        if not check["ok"]:
            parity_failures.append({"tensor": name, **check})
    result["parity_checked"] = len(names)
    result["parity_failures"] = parity_failures
    result["ok"] = bool(result["ok"] and not parity_failures)
    _json(result)
    return 0 if result["ok"] else 2


def command_create_hybrid(args: argparse.Namespace) -> int:
    from .hybrid import create_hybrid_artifact

    manifest = create_hybrid_artifact(
        args.output,
        model=args.model,
        runtime=args.runtime,
        benchmark_runtime=args.benchmark_runtime,
        server_runtime=args.server_runtime,
        specification=args.specification,
        source_record=args.source_record,
        assurance_evidence=args.assurance_evidence or (),
        model_id=args.model_id,
        source_revision=args.source_revision,
        runtime_revision=args.runtime_revision,
        runtime_build_number=args.runtime_build_number,
        runtime_argument_profile=args.runtime_argument_profile,
        total_parameters=args.total_parameters,
        target_gpu=args.target_gpu,
        target_vram_mib=args.target_vram_mib,
        maximum_context_tokens=args.maximum_context,
        kv_cache_k=args.kv_cache_k,
        kv_cache_v=args.kv_cache_v,
    )
    _json(
        {
            "manifest": str(manifest),
            "artifact": str(Path(args.output).resolve()),
            "validation_status": "UNCALIBRATED",
            "next": "run `nhdf-edge gate-hybrid ARTIFACT`",
        }
    )
    return 0


def command_gate_hybrid(args: argparse.Namespace) -> int:
    from .hybrid import gate_hybrid_artifact

    evidence = gate_hybrid_artifact(
        args.artifact,
        output=args.output,
        seed=args.seed,
        benchmark_repetitions=args.repetitions,
        minimum_generation_tokens_per_second=args.minimum_generation_tps,
        reserve_vram_mib=args.reserve_vram_mib,
        verify_payload_hash=not args.quick,
    )
    _json(
        {
            "artifact": str(Path(args.artifact).resolve()),
            "passed": evidence["passed"],
            "status": evidence["status"],
            "aggregate": evidence["aggregate"],
            "benchmark": {
                "prompt": evidence["benchmark"]["prompt"],
                "generation": evidence["benchmark"]["generation"],
            },
        }
    )
    return 0 if evidence["passed"] else 2


def command_run(args: argparse.Namespace) -> int:
    from .hybrid import run_hybrid_prompt

    result = run_hybrid_prompt(
        args.artifact,
        prompt=args.prompt,
        max_tokens=args.max_new_tokens,
        context=args.context,
        seed=args.seed,
        allow_unvalidated=args.allow_unvalidated,
        verify_payload_hash=not args.quick,
        monitor_resources=args.monitor_resources,
    )
    if args.text_only:
        print(result["generated_text"])
    else:
        _json(result)
    return 0 if result["exit_code"] == 0 else 2


def command_serve(args: argparse.Namespace) -> int:
    import time

    from .server import HybridServer

    runtime = HybridServer(
        args.artifact,
        port=args.port,
        threads=args.threads,
        startup_timeout_seconds=args.startup_timeout,
        request_timeout_seconds=args.request_timeout,
        verify_payload_hash=not args.quick,
    )
    runtime.start()
    _json(
        {
            "status": "ready",
            "url": runtime.base_url,
            "artifact": str(Path(args.artifact).resolve()),
            "preflight": runtime.preflight_result,
            "command": runtime.command,
            "note": "Press Ctrl+C to stop the resident model.",
        }
    )
    try:
        while runtime.is_running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    finally:
        runtime.stop()
    raise RuntimeError("resident llama-server exited unexpectedly")


def command_set_validation(args: argparse.Namespace) -> int:
    from .format import PackReader, set_pack_validation_status

    evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        raise ValueError("validation evidence file must contain a JSON object")
    manifest = set_pack_validation_status(
        args.pack,
        args.status,
        evidence=evidence,
    )
    reader = PackReader(args.pack, verify_crc=False)
    _json(
        {
            "manifest": str(manifest),
            "validation_status": reader.validation_status,
            "evidence": reader.manifest["validation"]["evidence"],
        }
    )
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    from .config import load_config
    from .runtime.doctor import run_doctor

    report = run_doctor(load_config(args.config))
    payload = report.to_dict()
    _json(payload)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if report.verdict == "ready-for-benchmark" else 1


def command_smoke(args: argparse.Namespace) -> int:
    import torch

    from .quantize import (
        QuantizationPolicy,
        dequantize_tensor,
        quantize_tensor,
        verify_parity,
    )

    generator = torch.Generator().manual_seed(args.seed)
    weight = torch.randn(args.rows, args.cols, generator=generator)
    base_policy = QuantizationPolicy(
        base_bits=2,
        group_size=256,
        residual_fraction=0.0,
        iterations=3,
    )
    branch_policy = QuantizationPolicy(
        base_bits=2,
        group_size=256,
        residual_fraction=args.residual_fraction,
        iterations=3,
    )
    base = quantize_tensor(weight, base_policy, name="smoke.base")
    branch = quantize_tensor(weight, branch_policy, name="smoke.branch")
    base_mse = torch.mean((weight - dequantize_tensor(base)) ** 2).item()
    branch_mse = torch.mean((weight - dequantize_tensor(branch)) ** 2).item()
    payload = {
        "shape": list(weight.shape),
        "base_mse": base_mse,
        "branch_mse": branch_mse,
        "mse_reduction_percent": 100.0 * (1.0 - branch_mse / base_mse),
        "branch_effective_bpp": branch.stats["effective_bits_per_weight"],
        "zero_set_max_abs": branch.stats["weighted_zero_set_max_abs"],
        "parity": verify_parity(branch),
    }
    _json(payload)
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


def command_substrate_verify(args: argparse.Namespace) -> int:
    from .substrate_contract import load_contract_bundle

    bundle = load_contract_bundle(
        args.repository,
        kernel_contract_path=args.kernel,
        profile_registry_path=args.profiles,
        verify_sources=not args.skip_source_hashes,
    )
    _json(
        {
            "ok": True,
            "kernel_id": bundle.kernel["kernel_id"],
            "kernel_sha256": bundle.kernel_sha256,
            "profiles": [
                {
                    "profile_id": profile_id,
                    "sha256": bundle.profile_sha256s[profile_id],
                }
                for profile_id in sorted(bundle.profiles)
            ],
            "automatic_promotion": False,
        }
    )
    return 0


def command_substrate_validate_application(args: argparse.Namespace) -> int:
    from .substrate_contract import validate_application_manifest

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    result = validate_application_manifest(
        manifest,
        repository_root=args.repository,
        evidence_root=args.evidence_root,
        kernel_contract_path=args.kernel,
        profile_registry_path=args.profiles,
        verify_files=not args.skip_evidence_hashes,
    )
    _json(result)
    return 0


def command_substrate_validate_extension(args: argparse.Namespace) -> int:
    from .substrate_contract import validate_extension_proposal

    proposal = json.loads(Path(args.proposal).read_text(encoding="utf-8"))
    result = validate_extension_proposal(
        proposal,
        repository_root=args.repository,
        evidence_root=args.evidence_root,
        kernel_contract_path=args.kernel,
        profile_registry_path=args.profiles,
        verify_files=not args.skip_evidence_hashes,
    )
    _json(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nhdf-edge")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("estimate", help="print analytical size/throughput projections")
    p.add_argument("--config")
    p.add_argument("--context", type=int)
    p.add_argument("--output")
    p.set_defaults(func=command_estimate)

    p = sub.add_parser("pack", help="stream-convert a local Hugging Face checkpoint")
    p.add_argument("source")
    p.add_argument("output")
    p.add_argument("--config")
    p.add_argument("--include")
    p.add_argument("--exclude")
    p.add_argument("--max-tensors", type=int)
    p.add_argument("--hessian", help="optional safetensors file of input second moments")
    p.set_defaults(func=command_pack)

    p = sub.add_parser("verify", help="verify CRC32 and one-bit payload parity")
    p.add_argument("pack")
    p.add_argument("--parity-sample", type=int, default=16)
    p.add_argument("--parity-all", action="store_true")
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="verify a deliberately partial smoke-test pack without treating incompleteness as failure",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="for hybrid artifacts, verify sealed metadata/runtime but skip rehashing the large payload",
    )
    p.set_defaults(func=command_verify)

    p = sub.add_parser(
        "create-hybrid",
        help="create a zero-copy sealed transport around an external model codec",
    )
    p.add_argument("output")
    p.add_argument("--model", required=True, help="verified GGUF model payload")
    p.add_argument(
        "--runtime", required=True, help="pinned llama completion executable"
    )
    p.add_argument("--benchmark-runtime", help="matching llama-bench executable")
    p.add_argument("--server-runtime", help="matching llama-server executable")
    p.add_argument("--specification", help="selected substrate grounding contract or source")
    p.add_argument("--source-record", help="immutable upstream provenance JSON")
    p.add_argument(
        "--assurance-evidence",
        action="append",
        help="additional evidence file to seal (repeatable)",
    )
    p.add_argument("--model-id", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    p.add_argument(
        "--source-revision",
        default="0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
    )
    p.add_argument("--runtime-revision", default="b6014")
    p.add_argument("--runtime-build-number", type=int, default=6014)
    p.add_argument(
        "--runtime-argument-profile",
        choices=["b6014", "current-2026"],
        default="b6014",
    )
    p.add_argument("--total-parameters", type=int, default=30_532_122_624)
    p.add_argument("--target-gpu", default="NVIDIA GeForce RTX 5070 Ti Laptop GPU")
    p.add_argument("--target-vram-mib", type=int, default=12_227)
    p.add_argument("--maximum-context", type=int, default=8_192)
    p.add_argument(
        "--kv-cache-k",
        choices=["q8_0", "q4_0"],
        default="q8_0",
        help="validated key-cache type recorded in the execution profile",
    )
    p.add_argument(
        "--kv-cache-v",
        choices=["q8_0", "q4_0"],
        default="q8_0",
        help="validated value-cache type recorded in the execution profile",
    )
    p.set_defaults(func=command_create_hybrid)

    p = sub.add_parser(
        "gate-hybrid",
        help="run fresh functional, maximum-context, resource, and throughput gates",
    )
    p.add_argument("artifact")
    p.add_argument("--output")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--minimum-generation-tps", type=float, default=80.0)
    p.add_argument("--reserve-vram-mib", type=int, default=512)
    p.add_argument("--quick", action="store_true", help="skip the large-payload rehash")
    p.set_defaults(func=command_gate_hybrid)

    p = sub.add_parser("run", help="run a validated NHDF hybrid artifact")
    p.add_argument("artifact")
    p.add_argument("--prompt", required=True)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--text-only", action="store_true")
    p.add_argument(
        "--monitor-resources",
        action="store_true",
        help="sample GPU telemetry during generation (adds subprocess overhead)",
    )
    p.add_argument("--quick", action="store_true", help="skip the large-payload rehash")
    p.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help="explicit research-only override for an unvalidated hybrid artifact",
    )
    p.set_defaults(func=command_run)

    p = sub.add_parser(
        "serve",
        help="verify once and keep a validated NHDF hybrid resident on loopback",
    )
    p.add_argument("artifact")
    p.add_argument("--port", type=int, default=18080)
    p.add_argument("--threads", type=int)
    p.add_argument("--startup-timeout", type=float, default=120.0)
    p.add_argument("--request-timeout", type=float, default=120.0)
    p.add_argument(
        "--quick",
        action="store_true",
        help="verify sealed metadata/runtime but skip rehashing the large model payload",
    )
    p.set_defaults(func=command_serve)

    p = sub.add_parser(
        "set-validation",
        help="record a measured pack disposition; never inferred from integrity checks",
    )
    p.add_argument("pack")
    p.add_argument("status", choices=sorted(PACK_VALIDATION_STATUSES))
    p.add_argument("--evidence", required=True, help="JSON object containing measured evidence")
    p.set_defaults(func=command_set_validation)

    p = sub.add_parser("doctor", help="inspect CUDA/VRAM readiness")
    p.add_argument("--config")
    p.add_argument("--output")
    p.set_defaults(func=command_doctor)

    p = sub.add_parser("smoke", help="run a deterministic synthetic quantization test")
    p.add_argument("--rows", type=int, default=256)
    p.add_argument("--cols", type=int, default=1024)
    p.add_argument("--residual-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output")
    p.set_defaults(func=command_smoke)

    p = sub.add_parser(
        "substrate-verify",
        help="verify the compact UGTOMS kernel, source provenance, and selectable profiles",
    )
    p.add_argument("--repository", default=".")
    p.add_argument("--kernel", default="substrate/kernel/contract.json")
    p.add_argument("--profiles", default="substrate/profiles/registry.json")
    p.add_argument(
        "--skip-source-hashes",
        action="store_true",
        help="schema-only development check; do not use as release evidence",
    )
    p.set_defaults(func=command_substrate_verify)

    p = sub.add_parser(
        "substrate-validate-app",
        help="validate a substrate application manifest and its evidence bindings",
    )
    p.add_argument("manifest")
    p.add_argument("--repository", default=".")
    p.add_argument("--kernel", default="substrate/kernel/contract.json")
    p.add_argument("--profiles", default="substrate/profiles/registry.json")
    p.add_argument("--evidence-root")
    p.add_argument("--skip-evidence-hashes", action="store_true")
    p.set_defaults(func=command_substrate_validate_application)

    p = sub.add_parser(
        "substrate-validate-extension",
        help="validate that an extension proposal remains bounded and quarantined",
    )
    p.add_argument("proposal")
    p.add_argument("--repository", default=".")
    p.add_argument("--kernel", default="substrate/kernel/contract.json")
    p.add_argument("--profiles", default="substrate/profiles/registry.json")
    p.add_argument("--evidence-root")
    p.add_argument("--skip-evidence-hashes", action="store_true")
    p.set_defaults(func=command_substrate_validate_extension)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
