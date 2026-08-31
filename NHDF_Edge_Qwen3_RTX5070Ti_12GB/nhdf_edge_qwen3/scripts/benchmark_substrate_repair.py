#!/usr/bin/env python3
"""Run a focused, substrate-aware local-agent repair acceptance gate.

This gate measures one deliberately narrow capability: can the canonical local
OpenCode agent inspect the committed UGTOMS kernel/SCLP profile and repair one
semantic next-generation feedback defect in one disposable Python source file?
It is not a broad substrate-authoring benchmark, a compression proof, or a
claim of general coding competence.  Recorded tool/path/network checks are
retrospective evidence, not a preventive sandbox.

The harness uses only the Python standard library.  The fixture invokes pytest
because the exact offline test command is part of the measured repair task.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERIC_GATE_PATH = PROJECT_ROOT / "scripts" / "benchmark_local_coder_agent.py"


def _load_generic_gate() -> Any:
    """Load the hardened launcher/config/isolation benchmark helpers."""

    name = "_ugtoms_generic_gate_for_substrate_repair"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, GENERIC_GATE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load shared gate helpers from {GENERIC_GATE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_BASE = _load_generic_gate()
GateError = _BASE.GateError

FORMAT = "ugtoms-substrate-repair-gate-0.1"
REPLAY_FORMAT = "ugtoms-sclp-repair-replay-0.1"
MODEL_ID = _BASE.MODEL_ID
MODEL_ALIAS = _BASE.MODEL_ALIAS
PINNED_OPENCODE_VERSION = _BASE.PINNED_OPENCODE_VERSION
MIN_CONTEXT_TOKENS = _BASE.MIN_CONTEXT_TOKENS
EXPECTED_SOURCE = "src/sclp_repair.py"
PYTEST_COMMAND = "python -m pytest -q"
EXPECTED_FINAL_PYTEST_TESTS = 3
PROFILE_ID = "sclp-foundational"
KEY_WIDTHS = {"rho": 20, "theta": 18, "time": 14, "phi": 12}

COPIED_PROJECT_FILES = (
    "src/nhdf_edge/substrate_graph.py",
    "src/nhdf_edge/substrate_runtime.py",
    "substrate/kernel/contract.json",
    "substrate/profiles/registry.json",
    "substrate/profiles/sclp-foundational.json",
)
REQUIRED_READ_PATHS = frozenset(
    {
        "README.md",
        "src/sclp_repair.py",
        "tests/test_sclp_repair.py",
        "substrate/kernel/contract.json",
        "substrate/profiles/registry.json",
        "substrate/profiles/sclp-foundational.json",
    }
)
ALLOWED_TOOL_NAMES = frozenset(
    {"read", "grep", "glob", "edit", "bash", "todowrite"}
)
READ_TOOLS = frozenset({"read"})
SEARCH_TOOLS = frozenset({"grep", "glob"})
EDIT_TOOLS = frozenset({"edit"})
BASH_TOOLS = frozenset({"bash"})
FORBIDDEN_PATH_COMPONENTS = frozenset({"legacy", "archive", "archives"})

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_TAGGED_HEX64 = re.compile(r"^sha256:[0-9a-f]{64}$")
_BROKEN_FEEDBACK_LINE = (
    "        target_generation=source_generation,  # must cross exactly one generation"
)
_FIXED_FEEDBACK_LINE = (
    "        target_generation=source_generation + 1,  # must cross exactly one generation"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _claim_scope() -> dict[str, Any]:
    """Return the narrow, machine-readable claim boundary for this gate."""

    return {
        "classification": "focused_substrate_semantic_repair_acceptance_gate",
        "proves": [
            "one local agent repaired the one declared n-to-n feedback defect",
            "the repaired program built a content-addressed graph with explicit n-to-n+1 feedback",
            "the repaired deterministic SCLP replay passed twice",
            "only one declared source file changed and exact Git HEAD stayed unchanged",
        ],
        "does_not_prove": [
            "independent diagnosis or broad substrate understanding",
            "broad substrate authoring or broad coding competence",
            "NHDF, model, tensor, or semantic compression",
            "general intelligence, production safety, or universal substrate correctness",
            "preventive process, filesystem, or network sandboxing",
        ],
        "legacy_or_archive_material_loaded": False,
        "post_run_audits_are_retrospective": True,
        "preventive_sandbox": False,
        "answer_fully_disclosed": True,
        "independent_diagnosis_demonstrated": False,
    }


def _repository_root(path: Path) -> Path:
    result = _BASE._run_process(
        ["git", "rev-parse", "--show-toplevel"], cwd=path, timeout=30.0
    )
    if result.returncode != 0:
        raise GateError("the clean-room source must be inside a Git repository")
    try:
        return Path(result.stdout.decode("utf-8").strip()).resolve(strict=True)
    except (UnicodeDecodeError, OSError) as exc:
        raise GateError(f"could not resolve the parent Git repository: {exc}") from exc


def _safe_relative(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise GateError(f"{label} must be a non-empty normalized relative path")
    normalized = value.replace("\\", "/")
    candidate = Path(normalized)
    lowered = {part.lower() for part in candidate.parts}
    if (
        candidate.is_absolute()
        or candidate.drive
        or normalized.startswith(("/", "//"))
        or re.match(r"^[A-Za-z]:", normalized)
        or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", normalized)
        or ".." in candidate.parts
        or lowered & FORBIDDEN_PATH_COMPONENTS
    ):
        raise GateError(f"{label} is outside the focused clean-room allowlist: {value!r}")
    return normalized


def _assert_committed_source(source: Path, repository_root: Path) -> str:
    try:
        relative = source.resolve(strict=True).relative_to(repository_root).as_posix()
    except (OSError, ValueError) as exc:
        raise GateError(f"copy source is not inside the parent repository: {source}") from exc
    _safe_relative(relative, label="copy source")
    tracked = _BASE._run_process(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=repository_root,
        timeout=30.0,
    )
    if tracked.returncode != 0:
        raise GateError(f"clean-room source is not committed/tracked: {relative}")
    clean = _BASE._run_process(
        ["git", "diff", "--quiet", "HEAD", "--", relative],
        cwd=repository_root,
        timeout=30.0,
    )
    if clean.returncode != 0:
        raise GateError(f"clean-room source differs from committed HEAD: {relative}")
    return relative


def _copy_clean_room_sources(
    fixture: Path, *, require_tracked: bool = True
) -> list[dict[str, Any]]:
    repository_root = _repository_root(PROJECT_ROOT)
    copied: list[dict[str, Any]] = []
    for relative in COPIED_PROJECT_FILES:
        destination_text = _safe_relative(relative, label="copy destination")
        source = PROJECT_ROOT / relative
        try:
            resolved = source.resolve(strict=True)
        except OSError as exc:
            raise GateError(f"required clean-room source is missing: {source}") from exc
        if not resolved.is_file():
            raise GateError(f"required clean-room source is not a file: {resolved}")
        if require_tracked:
            repository_relative = _assert_committed_source(resolved, repository_root)
        else:
            try:
                repository_relative = resolved.relative_to(repository_root).as_posix()
            except ValueError as exc:
                raise GateError(f"copy source escaped the parent repository: {resolved}") from exc
            _safe_relative(repository_relative, label="copy source")
        destination = fixture / destination_text
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved, destination)
        digest = _sha256_file(destination)
        if digest != _sha256_file(resolved):
            raise GateError(f"copied source digest mismatch for {destination_text}")
        copied.append(
            {
                "source": repository_relative,
                "destination": destination_text,
                "bytes": destination.stat().st_size,
                "sha256": digest,
            }
        )
    _validate_copied_contract_profile(fixture)
    return copied


def _load_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateError(f"{label} is not valid UTF-8 JSON: {exc}") from exc


def _validate_copied_contract_profile(fixture: Path) -> dict[str, Any]:
    kernel_path = fixture / "substrate/kernel/contract.json"
    registry_path = fixture / "substrate/profiles/registry.json"
    profile_path = fixture / "substrate/profiles/sclp-foundational.json"
    kernel = _load_json(kernel_path, label="copied kernel contract")
    registry = _load_json(registry_path, label="copied profile registry")
    profile = _load_json(profile_path, label="copied SCLP profile")
    try:
        entry = next(
            item
            for item in registry["profiles"]
            if isinstance(item, Mapping) and item.get("profile_id") == PROFILE_ID
        )
    except (KeyError, TypeError, StopIteration) as exc:
        raise GateError("copied registry does not contain sclp-foundational") from exc
    kernel_digest = _sha256_file(kernel_path)
    profile_digest = _sha256_file(profile_path)
    if kernel.get("kernel_id") != "ugtoms-kernel-v0.1":
        raise GateError("copied kernel has the wrong identity")
    if registry.get("automatic_promotion") is not False:
        raise GateError("copied profile registry permits automatic promotion")
    if registry.get("kernel", {}).get("sha256") != kernel_digest:
        raise GateError("copied registry does not bind the kernel digest")
    if entry.get("path") != "substrate/profiles/sclp-foundational.json":
        raise GateError("copied registry has a noncanonical SCLP profile path")
    if entry.get("sha256") != profile_digest:
        raise GateError("copied registry does not bind the SCLP profile digest")
    if profile.get("profile_id") != PROFILE_ID or profile.get("kernel_sha256") != kernel_digest:
        raise GateError("copied SCLP profile does not bind the selected kernel")
    if profile.get("mappings", {}).get("packing", "").find("20/18/14/12") < 0:
        raise GateError("copied SCLP profile omits the fixed packing widths")
    return {
        "kernel_id": kernel["kernel_id"],
        "kernel_sha256": kernel_digest,
        "profile_id": PROFILE_ID,
        "profile_sha256": profile_digest,
        "automatic_promotion": False,
    }


def _application_source() -> str:
    return '''#!/usr/bin/env python3
"""Mostly-correct bounded SCLP repair fixture with one feedback defect."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nhdf_edge.substrate_graph import (
    DefinitionNode,
    FeedbackEdge,
    SubstrateGraph,
    canonical_hash,
)
from nhdf_edge.substrate_runtime import LogPolarLUT, SCLPKeyLayout64, deterministic_bit


REPLAY_FORMAT = "ugtoms-sclp-repair-replay-0.1"


def build_definition_graph() -> SubstrateGraph:
    """Build the content-addressed SCLP DAG and its delayed feedback edge."""
    source_generation = 0
    definitions = (
        DefinitionNode(
            id="sclp-input",
            kind="typed-input",
            domain="SCLPState",
            codomain="SCLPState",
            evaluation_phase=0,
            parameters={"profile": "sclp-foundational"},
            bounds={"generation_min": 0, "generation_max": 4},
            failures=("invalid-or-unbounded-state",),
            provenance={"source_refs": ["substrate/kernel/contract.json"]},
            input_ports={"residual": "SCLPState"},
            output_ports={"state": "SCLPState"},
        ),
        DefinitionNode(
            id="sclp-observable",
            kind="bounded-log-polar-observable",
            domain="SCLPState",
            codomain="SCLPState",
            dependencies=("sclp-input",),
            evaluation_phase=1,
            parameters={"packing_widths": {"rho": 20, "theta": 18, "time": 14, "phi": 12}},
            bounds={"key_bits": 64},
            failures=("key-saturation", "indeterminate-input"),
            provenance={"source_refs": ["substrate/profiles/sclp-foundational.json"]},
            input_ports={"state": "SCLPState"},
            output_ports={"observable": "SCLPState"},
        ),
    )
    feedback = FeedbackEdge(
        id="next-generation-feedback",
        source_ref="sclp-observable",
        target_ref="sclp-input",
        source_port="observable",
        target_port="residual",
        source_generation=source_generation,
        target_generation=source_generation,  # must cross exactly one generation
        provenance={
            "source_refs": [
                "substrate/kernel/contract.json",
                "substrate/profiles/sclp-foundational.json",
            ]
        },
    )
    return SubstrateGraph(definitions, feedback_edges=(feedback,))


def run_replay(*, generations: int = 4, seed: int = 20260831) -> dict:
    """Return a deterministic replay generated with the copied real primitives."""
    if isinstance(generations, bool) or not isinstance(generations, int) or not 1 <= generations <= 5:
        raise ValueError("generations must be an integer from one through five")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    graph = build_definition_graph()
    edge = graph.feedback_edges[0]
    lut = LogPolarLUT(radial_bins=32, angular_bins=64, time_bins=16)
    layout = SCLPKeyLayout64()
    rows = []
    for generation in range(generations):
        address = lut.encode((generation + 1.0, generation + 0.5), forward_step=generation)
        state = layout.quantize(address.rho, address.theta, generation, generation * 0.125)
        contiguous = layout.pack_contiguous(state)
        morton = layout.pack_morton(state)
        state_record = {
            "generation": generation,
            "feedback": {
                "source_generation": generation,
                "target_generation": generation + 1,
            },
            "address": {
                "kind": "log-polar-lut",
                "rho": address.rho,
                "theta": address.theta,
                "radial_bin": address.radial_bin,
                "angular_bin": address.angular_bin,
            },
            "packing": {
                "widths": dict(layout.WIDTHS),
                "contiguous": contiguous,
                "morton": morton,
                "contiguous_round_trip": layout.unpack_contiguous(contiguous) == state,
                "morton_round_trip": layout.unpack_morton(morton) == state,
            },
            "jitter_bit": deterministic_bit(seed, generation, "jitter-control"),
        }
        rows.append({**state_record, "state_digest": canonical_hash(state_record)})
    return {
        "format": REPLAY_FORMAT,
        "profile_id": "sclp-foundational",
        "requested_generations": generations,
        "definition_graph": {
            "content_hash": graph.content_hash,
            "definition_hashes": [node.content_hash for node in graph.definitions],
            "all_definitions_verified": all(node.verify_content_hash() for node in graph.definitions),
            "feedback": {
                "content_hash": edge.content_hash,
                "source_generation": edge.source_generation,
                "target_generation": edge.target_generation,
                "semantics": edge.semantics,
                "fixed_point_claim": edge.fixed_point_claim,
            },
        },
        "generations": rows,
        "self_reference": {"may_propose": True, "may_promote": False},
    }


def write_replay(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\\n",
        encoding="utf-8",
        newline="\\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_replay(Path(args.output), run_replay(generations=args.generations, seed=args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _broken_application() -> str:
    source = _application_source()
    if source.count(_BROKEN_FEEDBACK_LINE) != 1:
        raise AssertionError("repair fixture must contain exactly one marked defect")
    return source


def _fixed_application() -> str:
    return _broken_application().replace(_BROKEN_FEEDBACK_LINE, _FIXED_FEEDBACK_LINE)


def _fixture_tests() -> str:
    return '''from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sclp_repair import build_definition_graph, run_replay


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_WIDTHS = {"rho": 20, "theta": 18, "time": 14, "phi": 12}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_copied_kernel_registry_and_profile_are_bound() -> None:
    kernel_path = ROOT / "substrate/kernel/contract.json"
    registry = json.loads((ROOT / "substrate/profiles/registry.json").read_text(encoding="utf-8"))
    profile_path = ROOT / "substrate/profiles/sclp-foundational.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    entry = next(item for item in registry["profiles"] if item["profile_id"] == "sclp-foundational")
    assert registry["automatic_promotion"] is False
    assert registry["kernel"]["sha256"] == _sha256(kernel_path)
    assert entry["path"] == "substrate/profiles/sclp-foundational.json"
    assert entry["sha256"] == _sha256(profile_path)
    assert profile["kernel_sha256"] == _sha256(kernel_path)
    assert "20/18/14/12" in profile["mappings"]["packing"]


def test_graph_is_content_addressed_with_exact_next_generation_feedback() -> None:
    graph = build_definition_graph()
    assert graph.fixed_point_engine is False
    assert graph.content_hash.startswith("sha256:")
    assert all(node.verify_content_hash() for node in graph.definitions)
    assert len({node.content_hash for node in graph.definitions}) == len(graph.definitions)
    assert len(graph.feedback_edges) == 1
    edge = graph.feedback_edges[0]
    assert edge.source_generation == 0
    assert edge.target_generation == edge.source_generation + 1
    assert edge.semantics == "referential-next-generation"
    assert edge.fixed_point_claim is False


def test_replay_is_deterministic_and_uses_real_sclp_primitives() -> None:
    first = run_replay(generations=4, seed=20260831)
    second = run_replay(generations=4, seed=20260831)
    assert first == second
    assert first["format"] == "ugtoms-sclp-repair-replay-0.1"
    assert first["profile_id"] == "sclp-foundational"
    assert first["definition_graph"]["all_definitions_verified"] is True
    for generation, row in enumerate(first["generations"]):
        assert row["generation"] == generation
        assert row["feedback"] == {
            "source_generation": generation,
            "target_generation": generation + 1,
        }
        assert row["address"]["kind"] == "log-polar-lut"
        assert row["packing"]["widths"] == EXPECTED_WIDTHS
        assert row["packing"]["contiguous_round_trip"] is True
        assert row["packing"]["morton_round_trip"] is True
        assert row["jitter_bit"] in (0, 1)
        assert row["state_digest"].startswith("sha256:")
'''


def _readme() -> str:
    return '''# Focused UGTOMS SCLP repair fixture

This disposable clean-room repository contains committed copies of the real
UGTOMS graph/runtime modules, the base kernel contract, the profile registry,
and the selected `sclp-foundational` profile.  `src/sclp_repair.py` is already
complete except for exactly one obvious semantic defect in the declared
feedback edge: it targets generation `n` instead of the required `n + 1`.

Inspect the contract, registry, selected profile, implementation, and tests.
Make the smallest possible edit in `src/sclp_repair.py`: repair only that
feedback target from `n` to `n + 1`, without reformatting or changing anything
else.  Then run exactly `python -m pytest -q` once and stop when it passes.

This is a focused repair acceptance gate.  It is not broad substrate
authoring, NHDF/model compression, or general coding competence.  Tool, path,
network, and Git checks performed by the outer harness are retrospective
evidence and are not a preventive sandbox.  Do not inspect legacy/archive
material, use network or package installation, stage, commit, or create files.
'''


def _snapshot_files(root: Path, *, excluded: Iterable[str] = ()) -> dict[str, str]:
    exclusions = {item.replace("\\", "/") for item in excluded}
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        if relative not in exclusions:
            result[relative] = _sha256_file(path)
    return result


def _create_fixture(root: Path, *, require_tracked: bool = True) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    copied = _copy_clean_room_sources(root, require_tracked=require_tracked)
    generated = {
        ".gitignore": "__pycache__/\n*.py[cod]\n.pytest_cache/\n",
        "README.md": _readme(),
        "pytest.ini": "[pytest]\npythonpath = src\naddopts = -p no:cacheprovider\n",
        "src/nhdf_edge/__init__.py": '"""Focused clean-room substrate package."""\n',
        EXPECTED_SOURCE: _broken_application(),
        "tests/test_sclp_repair.py": _fixture_tests(),
    }
    for relative, content in generated.items():
        _write_text(root / relative, content)
    immutable_before = _snapshot_files(root, excluded={EXPECTED_SOURCE})
    source_before = _sha256_file(root / EXPECTED_SOURCE)
    _BASE._run_git(root, ["init", "--quiet"])
    _BASE._run_git(root, ["config", "user.name", "UGTOMS Repair Gate"])
    _BASE._run_git(root, ["config", "user.email", "repair-gate@invalid.example"])
    _BASE._run_git(root, ["config", "commit.gpgsign", "false"])
    _BASE._run_git(root, ["add", "--", "."])
    _BASE._run_git(
        root,
        [
            "-c",
            f"core.hooksPath={os.devnull}",
            "commit",
            "--quiet",
            "--no-gpg-sign",
            "-m",
            "focused substrate repair baseline",
        ],
    )
    head = _BASE._run_git(root, ["rev-parse", "HEAD"]).decode("ascii").strip()
    return {
        "initial_head": head,
        "copied_sources": copied,
        "contract_profile_binding": _validate_copied_contract_profile(root),
        "immutable_sha256_before": immutable_before,
        "source_sha256_before": source_before,
        "tracked_files": sorted(
            line.decode("utf-8")
            for line in _BASE._run_git(root, ["ls-files", "-z"]).split(b"\0")
            if line
        ),
        "defect": {
            "kind": "same_generation_feedback",
            "broken": "n -> n",
            "required": "n -> n + 1",
            "defect_count": 1,
        },
        "legacy_or_archive_material_loaded": False,
    }


def _agent_prompt() -> str:
    return (
        "Repair this disposable focused UGTOMS SCLP fixture. This measures only one "
        "semantic repair, not broad substrate authoring, compression, or general coding "
        "competence. Follow the installed digest-pinned UGTOMS contract and work only "
        "inside the current Git repository. Use Read to inspect README.md, "
        "substrate/kernel/contract.json, substrate/profiles/registry.json, "
        "substrate/profiles/sclp-foundational.json, src/sclp_repair.py, and "
        "tests/test_sclp_repair.py. Use Grep or Glob to trace the copied graph/runtime "
        "APIs. The program is otherwise complete; diagnose the one marked feedback "
        "defect and use Edit exactly once on src/sclp_repair.py. Make only the smallest "
        "line edit that changes the feedback target from source generation n to exactly "
        "n + 1; do not reformat or change any other text. Then use Bash exactly once to "
        f"run `{PYTEST_COMMAND}`. Stop tool use immediately when it passes. Do not edit "
        "tests, copied modules, contracts, profiles, README, or configuration. Do not "
        "create files, stage, commit, use Git commands, network/web/package installation, "
        "delegation, or legacy/archive material. Use no tools other than Read, "
        "Grep/Glob, Edit, Bash, and optional local TodoWrite. The outer audit is "
        "retrospective evidence, not a preventive sandbox."
    )


def _forbidden_path_reference(value: str) -> bool:
    normalized = value.replace("\\", "/")
    pieces = {
        piece.lower()
        for piece in re.split(r"[/\[\]{}*?]+", normalized)
        if piece and piece not in {".", "**"}
    }
    return bool(pieces & FORBIDDEN_PATH_COMPONENTS)


def _evaluate_tool_events(
    events: Sequence[Mapping[str, Any]], fixture: Path
) -> dict[str, Any]:
    tools = _BASE._tool_events(events)
    if not tools:
        raise GateError("OpenCode emitted no recorded tool-use events")
    names = [tool["name"] for tool in tools]
    unknown = sorted(set(names) - ALLOWED_TOOL_NAMES)
    if unknown:
        raise GateError(f"agent used disallowed, destructive, or external tools: {unknown!r}")
    failed = [tool["name"] for tool in tools if tool["status"] not in {"completed", "success"}]
    if failed:
        raise GateError(f"agent emitted failed or incomplete tool events: {failed!r}")
    requirements = {
        "read": any(name in READ_TOOLS for name in names),
        "search": any(name in SEARCH_TOOLS for name in names),
        "edit": any(name in EDIT_TOOLS for name in names),
        "bash": any(name in BASH_TOOLS for name in names),
    }
    missing = sorted(name for name, present in requirements.items() if not present)
    if missing:
        raise GateError(f"agent did not exercise required tool categories: {missing!r}")
    if sum(name in EDIT_TOOLS for name in names) != 1:
        raise GateError("agent must emit exactly one recorded Edit call")
    if sum(name in BASH_TOOLS for name in names) != 1:
        raise GateError("agent must emit exactly one recorded Bash call")
    if names[-1] not in BASH_TOOLS:
        raise GateError("the exact pytest Bash call must be the final recorded tool event")

    read_paths: set[str] = set()
    for tool in tools:
        arguments = tool["input"]
        _BASE._validate_recorded_tool_arguments(
            tool["name"],
            arguments,
            fixture=fixture,
            expected_source=EXPECTED_SOURCE,
        )
        walked = list(_BASE._walk_argument_values(arguments))
        for path, value in walked:
            key = _BASE._normalized_argument_key(path)
            if (
                key in _BASE.PATH_ARGUMENT_KEYS | _BASE.GLOB_ARGUMENT_KEYS
                or (tool["name"] == "glob" and key == "pattern")
            ):
                if isinstance(value, str) and _forbidden_path_reference(value):
                    raise GateError("recorded tool attempted a legacy/archive path")
            if tool["name"] == "read" and key in _BASE.PATH_ARGUMENT_KEYS:
                resolved = _BASE._validate_tool_path(
                    value,
                    fixture=fixture,
                    label="read target",
                )
                read_paths.add(resolved.relative_to(fixture.resolve()).as_posix())
        if tool["name"] == "bash":
            command = arguments.get("command")
            if not isinstance(command, str):
                raise GateError("Bash tool event did not expose a command string")
            _BASE._validate_pytest_command(command)
    omitted_reads = sorted(REQUIRED_READ_PATHS - read_paths)
    if omitted_reads:
        raise GateError(f"agent did not inspect required fixture paths: {omitted_reads!r}")
    return {
        "passed": True,
        "tool_event_count": len(tools),
        "tool_names_in_order": names,
        "required_categories": requirements,
        "required_read_paths": sorted(read_paths & REQUIRED_READ_PATHS),
        "exact_edit_calls": 1,
        "exact_bash_calls": 1,
        "pytest_was_final_recorded_tool": True,
        "recorded_bash_command_was_exact": True,
        "recorded_paths_and_patterns_within_fixture": True,
        "recorded_network_package_git_commands": False,
        "recorded_legacy_or_archive_access": False,
        "audit_timing": "POST_RUN_RETROSPECTIVE",
        "preventive_sandbox": False,
        "scope_note": (
            "Recorded tool arguments are audited after execution; this is evidence, "
            "not a preventive process, filesystem, or network sandbox."
        ),
    }


def _hex_digest(value: Any, *, label: str, tagged: bool = False) -> str:
    pattern = _TAGGED_HEX64 if tagged else _HEX64
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        prefix = "sha256:" if tagged else ""
        raise GateError(f"{label} must be {prefix}<64 lowercase hexadecimal characters>")
    return value


def _validate_replay_payload(
    payload: object, *, expected_generations: int = 4
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise GateError("repair replay must be a JSON object")
    if payload.get("format") != REPLAY_FORMAT or payload.get("profile_id") != PROFILE_ID:
        raise GateError("repair replay has the wrong format or selected profile")
    if payload.get("requested_generations") != expected_generations:
        raise GateError("repair replay has the wrong bounded generation count")
    graph = payload.get("definition_graph")
    if not isinstance(graph, Mapping):
        raise GateError("repair replay has no content-addressed definition graph")
    _hex_digest(graph.get("content_hash"), label="graph content hash", tagged=True)
    hashes = graph.get("definition_hashes")
    if (
        not isinstance(hashes, list)
        or len(hashes) < 2
        or len(hashes) != len(set(hashes))
    ):
        raise GateError("definition graph hashes are missing or not unique")
    for index, digest in enumerate(hashes):
        _hex_digest(digest, label=f"definition hash {index}", tagged=True)
    if graph.get("all_definitions_verified") is not True:
        raise GateError("definition graph does not report verified content hashes")
    feedback = graph.get("feedback")
    if not isinstance(feedback, Mapping):
        raise GateError("definition graph has no explicit feedback edge")
    _hex_digest(feedback.get("content_hash"), label="feedback content hash", tagged=True)
    if (
        feedback.get("source_generation") != 0
        or feedback.get("target_generation") != 1
        or feedback.get("semantics") != "referential-next-generation"
        or feedback.get("fixed_point_claim") is not False
    ):
        raise GateError("definition graph feedback must be explicit generation 0 to 1")

    rows = payload.get("generations")
    if not isinstance(rows, list) or len(rows) != expected_generations:
        raise GateError("repair replay rows do not match the generation bound")
    for generation, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("generation") != generation:
            raise GateError(f"repair replay generation {generation} is absent or out of order")
        row_feedback = row.get("feedback")
        if not isinstance(row_feedback, Mapping) or row_feedback != {
            "source_generation": generation,
            "target_generation": generation + 1,
        }:
            raise GateError(f"generation {generation} is not handed to n + 1")
        address = row.get("address")
        if not isinstance(address, Mapping) or address.get("kind") != "log-polar-lut":
            raise GateError(f"generation {generation} lacks a log-polar address")
        packing = row.get("packing")
        if not isinstance(packing, Mapping) or packing.get("widths") != KEY_WIDTHS:
            raise GateError(f"generation {generation} changed the 20/18/14/12 packing")
        if (
            packing.get("contiguous_round_trip") is not True
            or packing.get("morton_round_trip") is not True
        ):
            raise GateError(f"generation {generation} lacks both packing round trips")
        for key in ("contiguous", "morton"):
            packed = packing.get(key)
            if isinstance(packed, bool) or not isinstance(packed, int) or not 0 <= packed < 2**64:
                raise GateError(f"generation {generation} {key} key is not unsigned 64-bit")
        if row.get("jitter_bit") not in (0, 1) or isinstance(row.get("jitter_bit"), bool):
            raise GateError(f"generation {generation} has an invalid jitter bit")
        _hex_digest(row.get("state_digest"), label=f"generation {generation} state hash", tagged=True)
    self_reference = payload.get("self_reference")
    if not isinstance(self_reference, Mapping) or self_reference.get("may_promote") is not False:
        raise GateError("repair replay permits self-reference promotion")
    return {
        "passed": True,
        "generations": len(rows),
        "definition_count": len(hashes),
        "graph_content_hash": graph["content_hash"],
        "explicit_next_generation_feedback": True,
        "exact_key_widths": dict(KEY_WIDTHS),
    }


def _fixture_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    return environment


def _run_replay(
    fixture: Path, output: Path, *, timeout: float, environment: Mapping[str, str]
) -> dict[str, Any]:
    result = _BASE._run_process(
        [
            sys.executable,
            EXPECTED_SOURCE,
            "--generations",
            "4",
            "--seed",
            "20260831",
            "--output",
            str(output),
        ],
        cwd=fixture,
        env=environment,
        timeout=timeout,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")[:2000]
        raise GateError(f"independent repair replay failed: {stderr}")
    return _load_json(output, label="independent repair replay")


def _evaluate_replay(
    fixture: Path,
    temporary: Path,
    *,
    timeout: float,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    temporary.mkdir(parents=True, exist_ok=False)
    run_environment = _fixture_environment(environment)
    first_path = temporary / "replay-a.json"
    second_path = temporary / "replay-b.json"
    first = _run_replay(
        fixture, first_path, timeout=timeout, environment=run_environment
    )
    second = _run_replay(
        fixture, second_path, timeout=timeout, environment=run_environment
    )
    gate = _validate_replay_payload(first)
    _validate_replay_payload(second)
    if first != second or first_path.read_bytes() != second_path.read_bytes():
        raise GateError("the repaired SCLP replay was not byte-for-byte deterministic twice")
    return {
        **gate,
        "independent_runs": 2,
        "byte_deterministic_twice": True,
        "replay_sha256": _sha256_file(first_path),
    }


def _evaluate_git_state(fixture: Path, baseline: Mapping[str, Any]) -> dict[str, Any]:
    head = _BASE._run_git(fixture, ["rev-parse", "HEAD"]).decode("ascii").strip()
    if head != baseline["initial_head"]:
        raise GateError("agent changed exact Git HEAD or created a commit")
    cached = _BASE._run_process(
        ["git", "diff", "--cached", "--quiet", "--exit-code"],
        cwd=fixture,
        timeout=30.0,
    )
    if cached.returncode != 0:
        raise GateError("agent staged fixture changes")
    entries = _BASE._parse_porcelain_z(
        _BASE._run_git(
            fixture, ["status", "--porcelain=v1", "--untracked-files=all", "-z"]
        )
    )
    if entries != [{"status": " M", "path": EXPECTED_SOURCE}]:
        raise GateError(
            "fixture must contain exactly one unstaged source modification; "
            f"recorded {entries!r}"
        )
    source = fixture / EXPECTED_SOURCE
    if source.read_text(encoding="utf-8") != _fixed_application():
        raise GateError("source change was not exactly the one-line n to n + 1 repair")
    if _sha256_file(source) == baseline["source_sha256_before"]:
        raise GateError("the defective source was not materially repaired")
    immutable_after = _snapshot_files(fixture, excluded={EXPECTED_SOURCE})
    if immutable_after != baseline["immutable_sha256_before"]:
        before = baseline["immutable_sha256_before"]
        changed = sorted(
            set(before) ^ set(immutable_after)
            | {
                path
                for path in set(before) & set(immutable_after)
                if before[path] != immutable_after[path]
            }
        )
        raise GateError(f"immutable or undeclared fixture files changed: {changed!r}")
    return {
        "passed": True,
        "initial_head": baseline["initial_head"],
        "final_head": head,
        "exact_head_unchanged": True,
        "staged_changes": False,
        "changed_paths": [EXPECTED_SOURCE],
        "exactly_one_allowed_source_changed": True,
        "exact_one_line_semantic_repair": True,
        "immutable_file_hashes_unchanged": True,
        "untracked_or_ignored_extra_files": False,
    }


def _resolve_canonical_opencode(value: str | Path) -> Path:
    try:
        candidate = Path(value).expanduser().resolve(strict=True)
        expected = _BASE._LAUNCHER.OPENCODE_EXE.resolve(strict=True)
    except OSError as exc:
        raise GateError(f"could not resolve canonical OpenCode executable: {exc}") from exc
    if candidate != expected:
        raise GateError(f"only the canonical project-local OpenCode is permitted: {expected}")
    try:
        _BASE._LAUNCHER.validate_local_install(candidate)
    except _BASE._LAUNCHER.LocalCoderError as exc:
        raise GateError(f"canonical OpenCode identity check failed: {exc}") from exc
    return candidate


def _load_canonical_config(
    value: str | Path,
) -> tuple[Path, bytes, dict[str, Any]]:
    try:
        candidate = Path(value).expanduser().resolve(strict=True)
        expected = _BASE._LAUNCHER.DEFAULT_CONFIG.resolve(strict=True)
    except OSError as exc:
        raise GateError(f"could not resolve canonical local-coder config: {exc}") from exc
    if candidate != expected:
        raise GateError(f"only the canonical local-coder config is permitted: {expected}")
    try:
        validated = _BASE._LAUNCHER.validate_config(candidate)
        if hasattr(validated, "path") and hasattr(validated, "raw"):
            resolved = Path(validated.path)
            raw = bytes(validated.raw)
        else:  # Compatibility with the launcher's earlier Path return contract.
            resolved = Path(validated)
            raw = resolved.read_bytes()
        config = json.loads(raw.decode("utf-8"))
    except (
        _BASE._LAUNCHER.LocalCoderError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
    ) as exc:
        raise GateError(f"repair gate requires the canonical launcher config: {exc}") from exc
    if resolved.resolve(strict=True) != expected or not isinstance(config, dict):
        raise GateError("canonical launcher config validation returned an unsafe snapshot")
    return resolved, raw, config


def _write_sha256sums(run_dir: Path) -> None:
    checksum = run_dir / "SHA256SUMS"
    paths = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != checksum)
    _write_text(
        checksum,
        "".join(
            f"{_sha256_file(path)}  {path.relative_to(run_dir).as_posix()}\n"
            for path in paths
        ),
    )


def _run_gate(args: argparse.Namespace) -> tuple[int, Path]:
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S.%fZ")
    run_dir.mkdir(parents=False, exist_ok=False)
    stdout_path = run_dir / "opencode.stdout.jsonl"
    stderr_path = run_dir / "opencode.stderr.txt"
    stdout_path.write_bytes(b"")
    stderr_path.write_bytes(b"")
    evidence: dict[str, Any] = {
        "format": FORMAT,
        "started_at_utc": _utc_now(),
        "status": "FAILED",
        "scope": _claim_scope(),
    }
    try:
        server_url = _BASE._normalize_server_url(args.server_url)
        executable = _resolve_canonical_opencode(args.opencode_exe)
        config_path, config_bytes, config = _load_canonical_config(args.config)
        config_gate = _BASE._validate_opencode_config(config, server_url)
        evidence["inputs"] = {
            "server_url": server_url,
            "opencode_executable": _BASE._public_path(
                executable, fallback="<PINNED_OPENCODE_EXECUTABLE>"
            ),
            "opencode_executable_sha256": _sha256_file(executable),
            "expected_opencode_sha256": _BASE._LAUNCHER.EXPECTED_OPENCODE_SHA256,
            "config_path": _BASE._public_path(
                config_path, fallback="<PINNED_LOCAL_CODER_CONFIG>"
            ),
            "config_sha256": _sha256_bytes(config_bytes),
            "model": MODEL_ID,
            "output_directory": _BASE._public_path(
                run_dir, fallback="<OUTPUT_ROOT>/<RUN_DIRECTORY>"
            ),
        }
        evidence["config_gate"] = config_gate

        models, models_timing = _BASE._http_json(
            server_url, "/v1/models", timeout=args.http_timeout_seconds
        )
        model_gate = _BASE._validate_models_payload(
            models, minimum_context=MIN_CONTEXT_TOKENS
        )
        model_gate["request"] = models_timing
        evidence["model_gate"] = model_gate
        tool_response, tool_timing = _BASE._http_json(
            server_url,
            "/v1/chat/completions",
            method="POST",
            payload=_BASE._tool_probe_payload(),
            timeout=args.http_timeout_seconds,
        )
        native_gate = _BASE._evaluate_tool_probe(tool_response)
        native_gate["request"] = tool_timing
        evidence["native_tool_call_gate"] = native_gate

        with tempfile.TemporaryDirectory(prefix="ugtoms-substrate-repair-gate-") as temporary:
            temp_root = Path(temporary)
            fixture = temp_root / "fixture"
            baseline = _create_fixture(fixture, require_tracked=True)
            fixture_environment = _fixture_environment()
            initial_test = _BASE._run_process(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=fixture,
                env=fixture_environment,
                timeout=args.fixture_timeout_seconds,
            )
            (run_dir / "fixture.baseline-pytest.stdout.txt").write_bytes(initial_test.stdout)
            (run_dir / "fixture.baseline-pytest.stderr.txt").write_bytes(initial_test.stderr)
            diagnostic = (initial_test.stdout + initial_test.stderr).decode(
                "utf-8", errors="replace"
            )
            if initial_test.returncode == 0:
                raise GateError("repair fixture unexpectedly passed before the agent repair")
            if "feedback must cross exactly one generation" not in diagnostic:
                raise GateError("repair fixture failed for something other than the declared defect")

            isolated = temp_root / "isolated"
            copied_config = isolated / "config/opencode/opencode.json"
            copied_config.parent.mkdir(parents=True, exist_ok=True)
            copied_config.write_bytes(config_bytes)
            agent_environment = _BASE._agent_environment(
                os.environ, isolated, copied_config, server_url=server_url
            )
            agent_environment.update(
                {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}
            )
            version_result = _BASE._run_process(
                [str(executable), "--version"],
                cwd=fixture,
                env=agent_environment,
                timeout=30.0,
            )
            version = _BASE._validated_opencode_version(version_result)
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
            started = time.perf_counter()
            try:
                agent = subprocess.run(
                    command,
                    cwd=str(fixture),
                    env=agent_environment,
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
            elapsed = time.perf_counter() - started
            stdout_path.write_bytes(agent.stdout)
            stderr_path.write_bytes(agent.stderr)
            if agent.returncode != 0:
                raise GateError(f"OpenCode exited with status {agent.returncode}")
            events = _BASE._parse_json_events(agent.stdout)
            tool_gate = _evaluate_tool_events(events, fixture)
            if _sha256_file(executable) != _BASE._LAUNCHER.EXPECTED_OPENCODE_SHA256:
                raise GateError("canonical OpenCode executable changed during the repair run")

            final_test = _BASE._run_process(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=fixture,
                env=fixture_environment,
                timeout=args.fixture_timeout_seconds,
            )
            (run_dir / "fixture.final-pytest.stdout.txt").write_bytes(final_test.stdout)
            (run_dir / "fixture.final-pytest.stderr.txt").write_bytes(final_test.stderr)
            if final_test.returncode != 0:
                raise GateError("independent focused repair pytest did not pass")
            final_passed_tests = _BASE._verified_pytest_pass_count(
                final_test.stdout,
                final_test.stderr,
                expected=EXPECTED_FINAL_PYTEST_TESTS,
            )
            replay_gate = _evaluate_replay(
                fixture,
                temp_root / "independent-replays",
                timeout=args.fixture_timeout_seconds,
                environment=fixture_environment,
            )
            for replay_name in ("replay-a.json", "replay-b.json"):
                shutil.copy2(
                    temp_root / "independent-replays" / replay_name,
                    run_dir / f"fixture.{replay_name}",
                )
            git_gate = _evaluate_git_state(fixture, baseline)
            diff = _BASE._run_git(
                fixture, ["diff", "--binary", "HEAD", "--", EXPECTED_SOURCE]
            )
            (run_dir / "fixture.repair.patch").write_bytes(diff)

            evidence["agent_gate"] = {
                "passed": True,
                "opencode_version": version,
                "opencode_sha256_rechecked_after_run": True,
                "command_arguments": command[1:-1],
                "prompt_sha256": _sha256_bytes(_agent_prompt().encode("utf-8")),
                "elapsed_seconds": elapsed,
                "json_event_count": len(events),
                "tool_audit": tool_gate,
                "isolated_environment": {
                    "canonical_digest_pinned_contract_installed": True,
                    "canonical_pinned_config_validated": True,
                    "parent_environment_reduced_to_operational_allowlist": True,
                    "credential_like_environment_inherited": False,
                    "plugins_mcp_skills_project_config_autoupdate_share_telemetry_disabled": True,
                },
            }
            evidence["fixture"] = {
                **baseline,
                "baseline_pytest_returncode": initial_test.returncode,
                "baseline_failure_was_exact_declared_defect": True,
                "final_pytest": {
                    "passed": True,
                    "returncode": final_test.returncode,
                    "passed_tests": final_passed_tests,
                    "total_tests": EXPECTED_FINAL_PYTEST_TESTS,
                    "result": f"{final_passed_tests}/{EXPECTED_FINAL_PYTEST_TESTS}",
                    "stdout_sha256": _sha256_bytes(final_test.stdout),
                    "stderr_sha256": _sha256_bytes(final_test.stderr),
                },
                "replay_gate": replay_gate,
                "git_gate": git_gate,
            }
        evidence["status"] = "PASSED"
        evidence["all_gates_passed"] = True
        exit_code = 0
    except (GateError, ValueError) as exc:
        evidence["status"] = "FAILED"
        evidence["all_gates_passed"] = False
        evidence["failure"] = {
            "type": type(exc).__name__,
            "message": _BASE._public_failure_message(
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
    except Exception as exc:
        evidence["status"] = "ERROR"
        evidence["all_gates_passed"] = False
        evidence["failure"] = {
            "type": type(exc).__name__,
            "message": _BASE._public_failure_message(
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
            "Run the focused SCLP semantic-repair gate. This is not broad substrate "
            "authoring, compression, or general competence; audits are retrospective."
        )
    )
    parser.add_argument("--server-url", required=True, help="existing loopback local server URL")
    parser.add_argument(
        "--opencode-exe",
        default=str(_BASE._LAUNCHER.OPENCODE_EXE),
        help=f"canonical project-local OpenCode {PINNED_OPENCODE_VERSION} executable",
    )
    parser.add_argument(
        "--config",
        default=str(_BASE._LAUNCHER.DEFAULT_CONFIG),
        help="canonical digest-pinned local-coder config",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--http-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--agent-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--fixture-timeout-seconds", type=float, default=120.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    for name in (
        "http_timeout_seconds",
        "agent_timeout_seconds",
        "fixture_timeout_seconds",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    exit_code, run_dir = _run_gate(args)
    print(
        json.dumps(
            {
                "status": "PASSED" if exit_code == 0 else "FAILED",
                "evidence": _BASE._public_path(
                    run_dir, fallback="<OUTPUT_ROOT>/<RUN_DIRECTORY>"
                ),
            }
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
