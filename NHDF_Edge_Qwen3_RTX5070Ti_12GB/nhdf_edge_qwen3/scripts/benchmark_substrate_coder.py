#!/usr/bin/env python3
"""Run the bounded UGTOMS literal-substrate coding acceptance gate.

This gate asks one pinned local OpenCode agent to finish one disposable,
clean-room SCLP application.  It measures only the exact substrate primitives,
determinism, evidence, and repository discipline checked below.  It is not a
broad-intelligence benchmark, a model-compression proof, or a preventive
process/filesystem/network sandbox.  Tool and Git auditing is retrospective.

The harness itself uses only the Python standard library.  The disposable
fixture intentionally exercises the repository's PDF stack and pytest because
those are part of the bounded application being accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
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
    """Load the already-hardened launcher/config/isolation helpers."""

    name = "_ugtoms_generic_local_coder_gate"
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

FORMAT = "ugtoms-substrate-coder-gate-0.2"
MODEL_ID = _BASE.MODEL_ID
MODEL_ALIAS = _BASE.MODEL_ALIAS
PINNED_OPENCODE_VERSION = _BASE.PINNED_OPENCODE_VERSION
APPLICATION_ID = "ugtoms-sclp-bounded-acceptance"
APPLICATION_VERSION = "0.2.0"
APPLICATION_FORMAT = "ugtoms-application-manifest-0.2"
REPLAY_FORMAT = "ugtoms-sclp-replay-0.2"
PROFILE_ID = "sclp-foundational"
PROFILE_REQUIREMENT_IDS = (
    "finite-cone-reference-vector",
    "packed-key-round-trips",
    "jitter-margin-certificate",
    "metric-kinematic-reference-vector",
    "grammar-budget-trace",
    "sweep-interval",
)
EXPECTED_MAPPING_ROWS = 12

MAPPING_CATEGORIES = (
    "typed",
    "vector",
    "kinematic",
    "geometry",
    "topology",
    "packing",
    "predicate",
    "operator",
    "self_reference",
)
KEY_WIDTHS = {"rho": 20, "theta": 18, "time": 14, "phi": 12}
EXPECTED_TEXT = (
    "UGTOMS SCLP Acceptance Application",
    "20/18/14/12",
    "bounded acceptance gate",
)

COPIED_PROJECT_FILES = (
    "src/nhdf_edge/substrate_runtime.py",
    "src/nhdf_edge/substrate_graph.py",
    "src/nhdf_edge/substrate_contract.py",
    "src/nhdf_edge/substrate_pdf.py",
    "scripts/render_substrate_pdf.py",
    "substrate/kernel/contract.json",
    "substrate/profiles/registry.json",
    "substrate/profiles/nhdf-v0.1.json",
    "substrate/profiles/nhdf-v0.3-ccd.json",
    "substrate/profiles/sclp-foundational.json",
    "sources/NHDF_Formal_Specification_v0.1.pdf",
    "sources/NHDF_Formal_Specification_v0.3_General_Purpose_CCD_Tom_Klootwijk.pdf",
)
PARENT_PROVENANCE_PDF = (
    "1bit-parity-bit-lower-case-phi-jitter-log-encoded-polar-LUT-analytic-"
    "sweeping-cone-T-a-side-view-of-the-pyramid-a-circle-a-sphere-the-apex-binary-.pdf"
)

MUTABLE_TRACKED_PATHS = frozenset(
    {
        "app/application-manifest.json",
        "evidence/replay.json",
        "report/substrate-report-v0.1.md",
        "src/substrate_app.py",
    }
)
GENERATED_PATHS = frozenset(
    {
        "output/substrate-report.pdf",
        "output/substrate-report.pdf.metadata.json",
    }
)
EXPECTED_CHANGED_PATHS = MUTABLE_TRACKED_PATHS | GENERATED_PATHS
ALLOWED_TOOL_NAMES = frozenset({"read", "grep", "glob", "edit", "bash", "todowrite"})
READ_TOOLS = frozenset({"read"})
SEARCH_TOOLS = frozenset({"grep", "glob"})
EDIT_TOOLS = frozenset({"edit"})
BASH_TOOLS = frozenset({"bash"})

APPLICATION_COMMAND = (
    "python src/substrate_app.py --generations 5 --output evidence/replay.json "
    "--manifest app/application-manifest.json"
)
RENDER_COMMAND = (
    'python scripts/render_substrate_pdf.py report/substrate-report-v0.1.md '
    '--output output/substrate-report.pdf '
    '--expect "UGTOMS SCLP Acceptance Application" '
    '--expect "20/18/14/12" --expect "bounded acceptance gate"'
)
PYTEST_COMMAND = "python -m pytest -q"
VALIDATE_COMMAND = "python scripts/validate_fixture.py"
ALLOWED_BASH_COMMANDS = frozenset(
    {APPLICATION_COMMAND, RENDER_COMMAND, PYTEST_COMMAND, VALIDATE_COMMAND}
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_TAGGED_HEX64 = re.compile(r"^sha256:[0-9a-f]{64}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _repository_root(path: Path) -> Path:
    result = _BASE._run_process(
        ["git", "rev-parse", "--show-toplevel"], cwd=path, timeout=30.0
    )
    if result.returncode != 0:
        raise GateError("the clean-room source must be inside a Git repository")
    try:
        resolved = Path(result.stdout.decode("utf-8").strip()).resolve(strict=True)
    except (UnicodeDecodeError, OSError) as exc:
        raise GateError(f"could not resolve parent Git repository: {exc}") from exc
    return resolved


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
        or ".." in candidate.parts
        or "legacy" in lowered
        or "archive" in lowered
        or "archives" in lowered
    ):
        raise GateError(f"{label} is outside the bounded clean-room allowlist: {value!r}")
    return normalized


def _copy_plan() -> tuple[tuple[Path, str], ...]:
    repo_root = _repository_root(PROJECT_ROOT)
    rows = [(PROJECT_ROOT / relative, _safe_relative(relative, label="copy target")) for relative in COPIED_PROJECT_FILES]
    rows.append((repo_root / PARENT_PROVENANCE_PDF, _safe_relative(PARENT_PROVENANCE_PDF, label="copy target")))
    destinations = [row[1] for row in rows]
    if len(destinations) != len(set(destinations)):
        raise GateError("clean-room copy allowlist contains duplicate destinations")
    return tuple(rows)


def _assert_tracked_source(source: Path, repository_root: Path) -> str:
    try:
        relative = source.resolve(strict=True).relative_to(repository_root).as_posix()
    except (OSError, ValueError) as exc:
        raise GateError(f"copy source is not inside the parent repository: {source}") from exc
    result = _BASE._run_process(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=repository_root,
        timeout=30.0,
    )
    if result.returncode != 0:
        raise GateError(f"clean-room copy source is not committed/tracked: {relative}")
    return relative


def _copy_clean_room_sources(
    fixture: Path, *, require_tracked: bool = True
) -> list[dict[str, Any]]:
    repository_root = _repository_root(PROJECT_ROOT)
    copied: list[dict[str, Any]] = []
    for source, destination_text in _copy_plan():
        try:
            resolved = source.resolve(strict=True)
        except OSError as exc:
            raise GateError(f"required clean-room source is missing: {source}") from exc
        if not resolved.is_file():
            raise GateError(f"required clean-room source is not a file: {resolved}")
        if require_tracked:
            repository_relative = _assert_tracked_source(resolved, repository_root)
        else:
            try:
                repository_relative = resolved.relative_to(repository_root).as_posix()
            except ValueError as exc:
                raise GateError(f"copy source escaped the parent repository: {resolved}") from exc
        _safe_relative(repository_relative, label="copy source")
        destination = fixture / destination_text
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved, destination)
        if _sha256_file(destination) != _sha256_file(resolved):
            raise GateError(f"copied source digest mismatch for {destination_text}")
        copied.append(
            {
                "source": repository_relative,
                "destination": destination_text,
                "bytes": destination.stat().st_size,
                "sha256": _sha256_file(destination),
            }
        )
    return copied


def _manifest_template(fixture: Path) -> dict[str, Any]:
    kernel_path = fixture / "substrate/kernel/contract.json"
    registry_path = fixture / "substrate/profiles/registry.json"
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        profile = next(
            item for item in registry["profiles"] if item.get("profile_id") == PROFILE_ID
        )
        profile_path = fixture / profile["path"]
        profile_document = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, StopIteration, TypeError) as exc:
        raise GateError(f"could not bind fixture manifest template: {exc}") from exc
    kernel_digest = _sha256_file(kernel_path)
    if registry.get("kernel", {}).get("sha256") != kernel_digest:
        raise GateError("profile registry does not bind the copied kernel digest")
    if _sha256_file(profile_path) != profile.get("sha256"):
        raise GateError("profile registry does not bind the copied SCLP profile digest")
    requirements = profile_document.get("evidence_requirements")
    if not isinstance(requirements, list):
        raise GateError("SCLP profile does not declare machine-readable evidence requirements")
    requirement_ids = tuple(
        row.get("id") for row in requirements if isinstance(row, Mapping)
    )
    if requirement_ids != PROFILE_REQUIREMENT_IDS:
        raise GateError(
            "SCLP profile evidence requirements differ from the bounded authoring gate"
        )
    bounds = [
        {
            "id": "bound-generations",
            "resource": "closed deterministic generations",
            "limit": 5,
            "unit": "generations",
            "enforcement": "Reject a request outside one through five generations.",
        },
        {
            "id": "bound-lineage",
            "resource": "retained lineage records",
            "limit": 5,
            "unit": "records",
            "enforcement": "Allocate a five-record NoveltyLog and reject overflow.",
        },
    ]
    failures = [
        {
            "id": "failure-invalid-input",
            "condition": "An input is non-finite, ill-typed, or outside a declared bound.",
            "observable": "The application exits nonzero or returns an explicit indeterminate status.",
            "handling": "Stop without silently promoting, repairing, or inventing state.",
        },
        {
            "id": "failure-evidence-mismatch",
            "condition": "Replay, contract, profile, lineage, or PDF evidence has a digest mismatch.",
            "observable": "Offline validation fails closed with the mismatching record named.",
            "handling": "Reject the application manifest and preserve the prior committed baseline.",
        },
    ]
    mappings: dict[str, list[dict[str, Any]]] = {}
    for category in MAPPING_CATEGORIES:
        mappings[category] = [
            {
                "id": f"application-{category}",
                "domain": [f"bounded {category} input"],
                "codomain": [f"evidence-bearing {category} output"],
                "definition": (
                    f"The fixture implements the copied kernel's {category} primitive "
                    "with deterministic replay, explicit bounds, and observable failure."
                ),
                "primitive_refs": [category],
                "bound_refs": ["bound-generations", "bound-lineage"],
                "failure_refs": ["failure-invalid-input", "failure-evidence-mismatch"],
                "evidence_refs": ["replay-evidence"],
                "disposition": "IMPLEMENTED",
            }
        ]
    mappings["geometry"].append(
        {
            "id": "application-additional-geometry-bypass",
            "domain": ["unexercised retained geometry mechanisms"],
            "codomain": ["explicit bypass record"],
            "definition": (
                "BYPASS: paired-sphere support, circle relations, and distributed-apex "
                "geometry are not executed or claimed by this bounded fixture."
            ),
            "primitive_refs": ["geometry"],
            "bound_refs": ["bound-generations", "bound-lineage"],
            "failure_refs": ["failure-invalid-input", "failure-evidence-mismatch"],
            "evidence_refs": ["replay-evidence"],
            "disposition": "BYPASS",
        }
    )
    mappings["topology"].append(
        {
            "id": "application-half-turn-bypass",
            "domain": ["unexercised source half-turn topology"],
            "codomain": ["explicit bypass record"],
            "definition": (
                "BYPASS: the source half-turn bundle map is not executed or inferred "
                "from any separate topology operation."
            ),
            "primitive_refs": ["topology"],
            "bound_refs": ["bound-generations", "bound-lineage"],
            "failure_refs": ["failure-invalid-input", "failure-evidence-mismatch"],
            "evidence_refs": ["replay-evidence"],
            "disposition": "BYPASS",
        }
    )
    mappings["operator"].append(
        {
            "id": "application-radix-bypass",
            "domain": ["unexercised radix-prefix operator"],
            "codomain": ["explicit bypass record"],
            "definition": (
                "BYPASS: radix-prefix refinement is not executed; the fixture uses "
                "only its explicitly bounded route."
            ),
            "primitive_refs": ["operator"],
            "bound_refs": ["bound-generations", "bound-lineage"],
            "failure_refs": ["failure-invalid-input", "failure-evidence-mismatch"],
            "evidence_refs": ["replay-evidence"],
            "disposition": "BYPASS",
        }
    )
    return {
        "format": APPLICATION_FORMAT,
        "application_id": APPLICATION_ID,
        "application_version": APPLICATION_VERSION,
        "kernel": {
            "kernel_id": "ugtoms-kernel-v0.1",
            "path": "substrate/kernel/contract.json",
            "sha256": kernel_digest,
        },
        "profiles": [{"profile_id": PROFILE_ID, "sha256": profile["sha256"]}],
        "profile_selection_rationale": (
            "The SCLP foundational profile is selected explicitly for its finite-cone, "
            "certified translational sign bracket, log-polar, typed-bit-role, bounded "
            "routing, and 20/18/14/12 packing corrections. Paired-sphere, half-turn, "
            "circle/apex, and radix mechanisms are explicit BYPASS mappings."
        ),
        "mappings": mappings,
        "resource_bounds": bounds,
        "failure_modes": failures,
        "evidence": [
            {
                "id": "replay-evidence",
                "role": "REPLAY",
                "path": "evidence/replay.json",
                "bytes": 1,
                "claim_coverage": [
                    {
                        "profile_id": PROFILE_ID,
                        "requirement_id": requirement_id,
                        "proof_pointer": (
                            f"/proof_inventory/{PROFILE_ID}/{requirement_id}"
                        ),
                    }
                    for requirement_id in PROFILE_REQUIREMENT_IDS
                ],
                "sha256": "0" * 64,
                "title": "Deterministic bounded SCLP replay",
            }
        ],
        "self_reference": {
            "enabled": True,
            "bounded_generations": 5,
            "may_propose_extensions": False,
            "may_promote_extensions": False,
            "proposal_disposition": "QUARANTINED",
        },
    }


def _stub_application() -> str:
    return '''#!/usr/bin/env python3
"""TODO: bounded deterministic SCLP application for the acceptance fixture."""
from __future__ import annotations

import argparse
from pathlib import Path

from nhdf_edge.substrate_graph import SubstrateGraph


def build_definition_graph() -> SubstrateGraph:
    """Return a content-addressed typed DAG with explicit n -> n + 1 feedback."""
    raise NotImplementedError("implement the bounded definition graph")


def run_application(*, generations: int = 5, seed: int = 20260831) -> dict:
    """Return a canonical JSON-compatible replay using the copied primitives."""
    raise NotImplementedError("implement the bounded SCLP replay")


def write_replay_and_manifest(
    replay: dict, *, output: Path, manifest: Path | None = None
) -> None:
    """Write replay evidence and, when requested, seal the manifest evidence row."""
    raise NotImplementedError("write canonical replay and complete the manifest")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    args = parser.parse_args()
    replay = run_application(generations=args.generations, seed=args.seed)
    write_replay_and_manifest(
        replay,
        output=Path(args.output),
        manifest=Path(args.manifest) if args.manifest else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _fixture_tests() -> str:
    return '''from __future__ import annotations

import json
from pathlib import Path

from nhdf_edge.substrate_graph import SubstrateGraph
from substrate_app import build_definition_graph, run_application


EXPECTED_WIDTHS = {"rho": 20, "theta": 18, "time": 14, "phi": 12}
BIT_ROLES = {
    "payload_parity_bit",
    "topology_parity_bit",
    "jitter_control_bit",
    "branch_control_bit",
}
PROFILE_REQUIREMENTS = {
    "finite-cone-reference-vector",
    "packed-key-round-trips",
    "jitter-margin-certificate",
    "metric-kinematic-reference-vector",
    "grammar-budget-trace",
    "sweep-interval",
}


def test_content_addressed_dag_and_next_generation_feedback() -> None:
    graph = build_definition_graph()
    assert isinstance(graph, SubstrateGraph)
    assert len(graph.definitions) >= 4
    assert all(node.verify_content_hash() for node in graph.definitions)
    assert len({node.content_hash for node in graph.definitions}) == len(graph.definitions)
    assert graph.content_hash.startswith("sha256:")
    assert graph.fixed_point_engine is False
    assert graph.feedback_edges
    assert all(edge.target_generation == edge.source_generation + 1 for edge in graph.feedback_edges)
    assert all(edge.fixed_point_claim is False for edge in graph.feedback_edges)


def test_replay_is_deterministic_and_prefix_stable() -> None:
    short_a = run_application(generations=3, seed=20260831)
    short_b = run_application(generations=3, seed=20260831)
    long = run_application(generations=5, seed=20260831)
    assert short_a == short_b
    assert short_a["generations"] == long["generations"][:3]
    assert short_a["format"] == "ugtoms-sclp-replay-0.2"
    assert short_a["key_layout"]["widths"] == EXPECTED_WIDTHS


def test_replay_exercises_real_distinct_primitives() -> None:
    replay = run_application(generations=5, seed=20260831)
    rows = replay["generations"]
    assert len(rows) == 5
    statuses = {row["event"]["status"] for row in rows}
    assert "VERIFIED" in statuses and "INDETERMINATE" in statuses
    for generation, row in enumerate(rows):
        assert row["generation"] == generation
        assert row["feedback"]["source_generation"] == generation
        assert row["feedback"]["target_generation"] == generation + 1
        assert row["address"]["kind"] == "log-polar-lut"
        assert row["packing"]["widths"] == EXPECTED_WIDTHS
        assert row["packing"]["contiguous_round_trip"] is True
        assert row["packing"]["morton_round_trip"] is True
        assert set(row["bits"]) == BIT_ROLES
        assert set(row["bit_role_provenance"]) == BIT_ROLES
        assert len(set(row["bit_role_provenance"].values())) == 4
        assert {arrow["role"] for arrow in row["vectors"]} == {"velocity", "acceleration"}
        assert row["geometry"]["finite_cone"]["kind"] == "exact-finite-cone-sdf"
        assert row["geometry"]["sphere"]["kind"] == "exact-sphere-sdf"
        sweep = row["geometry"]["sweep_interval"]
        assert sweep["certified"] is True
        assert sweep["earliest_impact_claim"] is False
        assert sweep["parameter_interval"][0] < sweep["parameter_interval"][1]
        assert sweep["endpoint_distances"][0] > 0.0
        assert sweep["endpoint_distances"][1] <= 0.0
        jitter = row["jitter_certificate"]
        assert jitter["safe_under_margin"] is True
        assert 0.0 <= jitter["amplitude"] < jitter["guard_margin"]
        routing = row["routing_budget"]
        assert routing["bounded"] is True
        assert routing["used_depth"] <= routing["maximum_depth"]
        assert routing["used_active_branches"] <= routing["maximum_active_branches"]
        assert row["state_digest"].startswith("sha256:")
        assert len(row["lineage_digest"]) == 64
    proofs = replay["proof_inventory"]["sclp-foundational"]
    assert set(proofs) == PROFILE_REQUIREMENTS
    assert all(proof["passed"] is True for proof in proofs.values())


def test_application_rejects_unbounded_generation_requests() -> None:
    for invalid in (0, 6):
        try:
            run_application(generations=invalid, seed=1)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"generation bound accepted {invalid}")
'''


def _validator_script() -> str:
    expects = repr(EXPECTED_TEXT)
    return f'''#!/usr/bin/env python3
"""Offline validator for the disposable SCLP fixture."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhdf_edge.substrate_contract import validate_application_manifest
from nhdf_edge.substrate_pdf import validate_pdf_text


def main() -> int:
    manifest_path = ROOT / "app/application-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = validate_application_manifest(
        manifest,
        repository_root=ROOT,
        evidence_root=ROOT,
        verify_files=True,
    )
    pdf = validate_pdf_text(ROOT / "output/substrate-report.pdf", {expects})
    replay = json.loads((ROOT / "evidence/replay.json").read_text(encoding="utf-8"))
    if replay.get("format") != "{REPLAY_FORMAT}" or len(replay.get("generations", [])) != 5:
        raise ValueError("replay does not contain the required five bounded generations")
    print(json.dumps({{"manifest": contract, "pdf_pages": pdf.page_count, "ok": True}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _report_stub() -> str:
    return '''---
title: UGTOMS SCLP Acceptance Application
version: 0.1.0
subtitle: Bounded literal-substrate evidence report
status: incomplete-fixture
---

# TODO: complete the bounded application report

Replace this stub with a concise account of the implemented log-polar address,
typed one-bit roles, fixed packing, vectors and kinematics, geometry, tri-state
admission, content-addressed DAG, lineage, and explicit next-generation feedback.

State plainly that this is a bounded acceptance gate, not broad intelligence or
a compression proof, and that tool auditing is retrospective rather than a
preventive sandbox.
'''


def _readme() -> str:
    return '''# Disposable UGTOMS SCLP acceptance fixture

This clean-room fixture contains copied, digest-bound substrate primitives and
contracts. Finish `src/substrate_app.py`, seal `app/application-manifest.json`
against `evidence/replay.json`, and complete the versioned report source.

The implementation must use the real copied primitives for log-polar addressing;
four separately typed parity/orientation/jitter/branch roles; 20/18/14/12-bit
contiguous and Morton packing; vectors and constant-acceleration kinematics;
exact finite-cone and sphere SDF samples; tri-state event admission; a
content-addressed definition DAG; a bounded lineage log; and explicit feedback
from generation n to exactly n + 1. It must record a real finite-cone sweep
sign bracket, jitter-margin certificate, bounded routing trace, both key-layout
round trips, and profile-qualified proof inventory. Feedback has no extension
proposal or promotion authority in this fixture.

This is a bounded acceptance gate. It is not a broad-intelligence or compression
proof. The outer harness's tool/path/network audit is retrospective evidence,
not a preventive sandbox. No legacy or archive code belongs in this fixture.
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
        "src/nhdf_edge/__init__.py": "\"\"\"Clean-room substrate acceptance package.\"\"\"\n",
        "src/substrate_app.py": _stub_application(),
        "tests/test_substrate_app.py": _fixture_tests(),
        "scripts/validate_fixture.py": _validator_script(),
        "report/substrate-report-v0.1.md": _report_stub(),
        "evidence/replay.json": "{}\n",
    }
    for relative, content in generated.items():
        _write_text(root / relative, content)
    template = _manifest_template(root)
    _write_json(root / "app/application-manifest.template.json", template)
    _write_json(root / "app/application-manifest.json", template)

    immutable_before = _snapshot_files(root, excluded=MUTABLE_TRACKED_PATHS)
    mutable_before = {
        relative: _sha256_file(root / relative) for relative in MUTABLE_TRACKED_PATHS
    }
    _BASE._run_git(root, ["init", "--quiet"])
    _BASE._run_git(root, ["config", "user.name", "UGTOMS Substrate Gate"])
    _BASE._run_git(root, ["config", "user.email", "substrate-gate@invalid.example"])
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
            "clean-room substrate fixture baseline",
        ],
    )
    head = _BASE._run_git(root, ["rev-parse", "HEAD"]).decode("ascii").strip()
    return {
        "initial_head": head,
        "copied_sources": copied,
        "immutable_sha256_before": immutable_before,
        "mutable_sha256_before": mutable_before,
        "tracked_file_count": len(_BASE._run_git(root, ["ls-files"]).splitlines()),
        "legacy_or_archive_material_loaded": False,
    }


def _agent_prompt() -> str:
    return (
        "Complete this disposable clean-room UGTOMS substrate fixture. This is a bounded "
        "acceptance gate, not a broad-intelligence or compression proof. Follow the "
        "installed digest-pinned UGTOMS contract in your system instructions. Work only "
        "inside the current Git repository. First use Read to inspect README.md, "
        "substrate/kernel/contract.json, substrate/profiles/registry.json, "
        "substrate/profiles/sclp-foundational.json, the four copied substrate modules, "
        "the manifest template, stub application, tests, validator, and report source. "
        "Use Grep or Glob to trace the real APIs. Implement a bounded deterministic SCLP "
        "application in src/substrate_app.py using real log-polar addressing; separate "
        "payload parity, topology parity, jitter control, and branch control roles; "
        "exact 20/18/14/12 contiguous and Morton packing; vectors and kinematics; "
        "exact finite-cone and sphere SDF support; a certified finite-cone sweep sign "
        "bracket; a jitter-margin certificate; a bounded routing/grammar trace; "
        "profile-qualified proof inventory for every declared SCLP requirement; "
        "tri-state event admission; a "
        "content-addressed definition DAG; bounded lineage; and explicit n to n+1 "
        "feedback. Feedback has no extension-proposal or promotion authority. Complete "
        "app/application-manifest.json from the immutable versioned "
        "template, bound to the copied kernel and sclp-foundational hashes and the replay "
        "evidence. Complete report/substrate-report-v0.1.md with limitations and state "
        "that auditing is retrospective, not a preventive sandbox. Use Edit only for "
        "src/substrate_app.py, app/application-manifest.json, evidence/replay.json, or "
        "report/substrate-report-v0.1.md. Use Bash only for these exact standalone "
        "commands; an exact command may be rerun while correcting the fixture, but finish "
        f"with this sequence in order: `{APPLICATION_COMMAND}`, `{RENDER_COMMAND}`, "
        f"`{PYTEST_COMMAND}`, `{VALIDATE_COMMAND}`. Do not combine commands. Finish only "
        "when both validations pass. Do not edit copied modules, contracts, profiles, "
        "tests, scripts, README, configuration, or the manifest template. Do not create "
        "anything except the declared PDF and its metadata sidecar. Do not stage, commit, "
        "use network/web/package installation/delegation, or use tools other than Read, "
        "Grep/Glob, Edit, Bash, and local TodoWrite. Never inspect or copy legacy/archive."
    )


def _validate_bash_command(command: object) -> str:
    if not isinstance(command, str) or command not in ALLOWED_BASH_COMMANDS:
        raise GateError(
            "Bash command was not one of the four exact offline fixture commands: "
            f"{command!r}"
        )
    return command


def _validate_recorded_tool_arguments(
    name: str, arguments: Mapping[str, Any], *, fixture: Path
) -> None:
    walked = list(_BASE._walk_argument_values(arguments))
    path_values = [
        (path, value)
        for path, value in walked
        if _BASE._normalized_argument_key(path) in _BASE.PATH_ARGUMENT_KEYS
    ]
    for path, value in path_values:
        resolved = _BASE._validate_tool_path(
            value, fixture=fixture, label=f"{name} argument {'.'.join(path)}"
        )
        if name in EDIT_TOOLS:
            relative = resolved.relative_to(fixture.resolve()).as_posix()
            if relative not in MUTABLE_TRACKED_PATHS:
                raise GateError(f"Edit targeted immutable fixture path {relative!r}")
    if name in READ_TOOLS | EDIT_TOOLS and not path_values:
        raise GateError(f"{name} tool event did not expose its target path")

    patterns = [
        (path, value)
        for path, value in walked
        if _BASE._normalized_argument_key(path) == "pattern"
    ]
    globs = [
        (path, value)
        for path, value in walked
        if _BASE._normalized_argument_key(path) in _BASE.GLOB_ARGUMENT_KEYS
    ]
    if name == "glob":
        if len(patterns) != 1:
            raise GateError("glob tool event must expose exactly one pattern")
        _BASE._validate_glob_pattern(patterns[0][1], label="glob argument pattern")
    elif name == "grep":
        if len(patterns) != 1:
            raise GateError("grep tool event must expose exactly one search pattern")
        _BASE._validate_content_pattern(patterns[0][1], label="grep argument pattern")
    for path, value in globs:
        _BASE._validate_glob_pattern(value, label=f"{name} argument {'.'.join(path)}")

    if name in BASH_TOOLS:
        _validate_bash_command(arguments.get("command"))


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
    commands: list[str] = []
    for tool in tools:
        arguments = tool["input"]
        _validate_recorded_tool_arguments(tool["name"], arguments, fixture=fixture)
        if tool["name"] in BASH_TOOLS:
            commands.append(str(arguments["command"]))
    final_validation_sequence = [
        APPLICATION_COMMAND,
        RENDER_COMMAND,
        PYTEST_COMMAND,
        VALIDATE_COMMAND,
    ]
    if commands[-4:] != final_validation_sequence:
        raise GateError(
            "recorded Bash commands must finish with the four declared validations in order; "
            f"recorded {commands!r}"
        )
    return {
        "passed": True,
        "tool_event_count": len(tools),
        "tool_names_in_order": names,
        "bash_commands_in_order": commands,
        "required_categories": requirements,
        "all_recorded_paths_and_patterns_within_fixture": True,
        "edits_limited_to_declared_mutable_paths": True,
        "network_capable_commands_recorded": False,
        "audit_timing": "POST_RUN_RETROSPECTIVE",
        "preventive_sandbox": False,
        "scope_note": (
            "Recorded tool arguments are audited after execution; this is evidence, "
            "not a preventive process, filesystem, or network sandbox."
        ),
    }


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise GateError(f"{label} must be finite")
    return result


def _hex_digest(value: Any, *, label: str, tagged: bool = False) -> str:
    pattern = _TAGGED_HEX64 if tagged else _HEX64
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        prefix = "sha256:" if tagged else ""
        raise GateError(f"{label} must be {prefix}<64 lowercase hexadecimal characters>")
    return value


def _validate_replay_payload(
    payload: object, *, expected_generations: int
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise GateError("replay must be a JSON object")
    if payload.get("format") != REPLAY_FORMAT:
        raise GateError(f"replay format must be {REPLAY_FORMAT!r}")
    if payload.get("application_id") != APPLICATION_ID:
        raise GateError("replay application_id does not identify the bounded fixture")
    if payload.get("requested_generations") != expected_generations:
        raise GateError("replay requested_generations differs from the evaluated run")
    key_layout = payload.get("key_layout")
    if not isinstance(key_layout, Mapping) or key_layout.get("widths") != KEY_WIDTHS:
        raise GateError("replay does not declare the exact 20/18/14/12 key widths")
    if key_layout.get("total_bits") != 64:
        raise GateError("replay key layout must total exactly 64 bits")
    graph = payload.get("definition_graph")
    if not isinstance(graph, Mapping):
        raise GateError("replay has no content-addressed definition_graph record")
    _hex_digest(graph.get("content_hash"), label="definition graph hash", tagged=True)
    definition_hashes = graph.get("definition_hashes")
    if (
        not isinstance(definition_hashes, list)
        or len(definition_hashes) < 4
        or len(set(definition_hashes)) != len(definition_hashes)
    ):
        raise GateError("definition graph must expose at least four unique definition hashes")
    for index, digest in enumerate(definition_hashes):
        _hex_digest(digest, label=f"definition hash {index}", tagged=True)
    feedback = graph.get("feedback")
    if not isinstance(feedback, Mapping) or feedback.get("source_generation") != 0 or feedback.get("target_generation") != 1:
        raise GateError("definition graph feedback must explicitly cross generation 0 to 1")
    if feedback.get("fixed_point_claim") is not False:
        raise GateError("definition graph must not claim an unrestricted fixed point")

    rows = payload.get("generations")
    if not isinstance(rows, list) or len(rows) != expected_generations:
        raise GateError("replay has the wrong bounded generation count")
    statuses: set[str] = set()
    lineages: list[str] = []
    bit_roles = {
        "payload_parity_bit",
        "topology_parity_bit",
        "jitter_control_bit",
        "branch_control_bit",
    }
    for generation, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("generation") != generation:
            raise GateError(f"replay generation {generation} is absent or out of order")
        feedback_row = row.get("feedback")
        if not isinstance(feedback_row, Mapping) or feedback_row.get("source_generation") != generation or feedback_row.get("target_generation") != generation + 1:
            raise GateError(f"generation {generation} lacks explicit n to n+1 feedback")
        address = row.get("address")
        if not isinstance(address, Mapping) or address.get("kind") != "log-polar-lut":
            raise GateError(f"generation {generation} lacks a real log-polar address")
        for key in ("rho", "theta"):
            _number(address.get(key), label=f"generation {generation} address {key}")
        packing = row.get("packing")
        if not isinstance(packing, Mapping) or packing.get("widths") != KEY_WIDTHS:
            raise GateError(f"generation {generation} packing widths changed")
        if packing.get("contiguous_round_trip") is not True or packing.get("morton_round_trip") is not True:
            raise GateError(f"generation {generation} does not prove both key round trips")
        for key in ("contiguous", "morton"):
            packed = packing.get(key)
            if isinstance(packed, bool) or not isinstance(packed, int) or not 0 <= packed < 2**64:
                raise GateError(f"generation {generation} {key} key is not unsigned 64-bit")
        bits = row.get("bits")
        roles = row.get("bit_role_provenance")
        if not isinstance(bits, Mapping) or set(bits) != bit_roles:
            raise GateError(f"generation {generation} collapsed or omitted a typed bit role")
        if any(value not in (0, 1) or isinstance(value, bool) for value in bits.values()):
            raise GateError(f"generation {generation} contains a non-bit role value")
        if not isinstance(roles, Mapping) or set(roles) != bit_roles or len(set(roles.values())) != 4:
            raise GateError(f"generation {generation} does not distinguish bit-role provenance")
        vectors = row.get("vectors")
        if not isinstance(vectors, list) or {item.get("role") for item in vectors if isinstance(item, Mapping)} != {"velocity", "acceleration"}:
            raise GateError(f"generation {generation} lacks typed velocity/acceleration arrows")
        kinematics = row.get("kinematics")
        if not isinstance(kinematics, Mapping):
            raise GateError(f"generation {generation} lacks kinematics")
        for field in ("position", "velocity", "acceleration"):
            vector = kinematics.get(field)
            if not isinstance(vector, list) or len(vector) != 3:
                raise GateError(f"generation {generation} kinematic {field} is not a 3-vector")
            for axis, value in enumerate(vector):
                _number(value, label=f"generation {generation} {field}[{axis}]")
        geometry = row.get("geometry")
        cone = geometry.get("finite_cone") if isinstance(geometry, Mapping) else None
        sphere = geometry.get("sphere") if isinstance(geometry, Mapping) else None
        if not isinstance(cone, Mapping) or cone.get("kind") != "exact-finite-cone-sdf":
            raise GateError(f"generation {generation} lacks the exact finite cone SDF")
        if not isinstance(sphere, Mapping) or sphere.get("kind") != "exact-sphere-sdf":
            raise GateError(f"generation {generation} lacks exact sphere support")
        _number(cone.get("distance"), label=f"generation {generation} cone distance")
        _number(sphere.get("distance"), label=f"generation {generation} sphere distance")
        sweep = geometry.get("sweep_interval") if isinstance(geometry, Mapping) else None
        if not isinstance(sweep, Mapping) or sweep.get("certified") is not True:
            raise GateError(f"generation {generation} lacks a certified sweep interval")
        if sweep.get("earliest_impact_claim") is not False:
            raise GateError(f"generation {generation} sweep makes an unsupported earliest-impact claim")
        parameter_interval = sweep.get("parameter_interval")
        endpoint_distances = sweep.get("endpoint_distances")
        if not isinstance(parameter_interval, list) or len(parameter_interval) != 2:
            raise GateError(f"generation {generation} sweep parameter interval is not a pair")
        if not isinstance(endpoint_distances, list) or len(endpoint_distances) != 2:
            raise GateError(f"generation {generation} sweep endpoint distances are not a pair")
        lower_parameter = _number(
            parameter_interval[0], label=f"generation {generation} sweep lower parameter"
        )
        upper_parameter = _number(
            parameter_interval[1], label=f"generation {generation} sweep upper parameter"
        )
        lower_distance = _number(
            endpoint_distances[0], label=f"generation {generation} sweep lower distance"
        )
        upper_distance = _number(
            endpoint_distances[1], label=f"generation {generation} sweep upper distance"
        )
        if not (0.0 <= lower_parameter < upper_parameter <= 1.0):
            raise GateError(f"generation {generation} sweep interval is not ordered inside [0,1]")
        if not (lower_distance > 0.0 and upper_distance <= 0.0):
            raise GateError(f"generation {generation} sweep endpoints do not retain a sign bracket")
        jitter = row.get("jitter_certificate")
        if not isinstance(jitter, Mapping) or jitter.get("safe_under_margin") is not True:
            raise GateError(f"generation {generation} lacks a passing jitter-margin certificate")
        amplitude = _number(jitter.get("amplitude"), label=f"generation {generation} jitter amplitude")
        guard_margin = _number(
            jitter.get("guard_margin"), label=f"generation {generation} jitter guard margin"
        )
        jitter_interval = jitter.get("interval")
        if not isinstance(jitter_interval, list) or len(jitter_interval) != 2:
            raise GateError(f"generation {generation} jitter interval is not a pair")
        interval_low = _number(
            jitter_interval[0], label=f"generation {generation} jitter interval lower"
        )
        interval_high = _number(
            jitter_interval[1], label=f"generation {generation} jitter interval upper"
        )
        if not (0.0 <= amplitude < guard_margin and interval_low < interval_high):
            raise GateError(f"generation {generation} jitter certificate exceeds its guard margin")
        routing = row.get("routing_budget")
        if not isinstance(routing, Mapping) or routing.get("bounded") is not True:
            raise GateError(f"generation {generation} lacks a bounded routing trace")
        for used_name, maximum_name in (
            ("used_depth", "maximum_depth"),
            ("used_active_branches", "maximum_active_branches"),
        ):
            used = routing.get(used_name)
            maximum = routing.get(maximum_name)
            if (
                isinstance(used, bool)
                or not isinstance(used, int)
                or isinstance(maximum, bool)
                or not isinstance(maximum, int)
                or used < 0
                or maximum < 1
                or used > maximum
            ):
                raise GateError(
                    f"generation {generation} routing {used_name} exceeds {maximum_name}"
                )
        event = row.get("event")
        if not isinstance(event, Mapping):
            raise GateError(f"generation {generation} lacks event admission evidence")
        status = event.get("status")
        if status not in {"NO_SUPPORT", "INCOMPATIBLE", "NO_CROSSING", "VERIFIED", "INDETERMINATE"}:
            raise GateError(f"generation {generation} has an invalid event status")
        statuses.add(str(status))
        for predicate in ("support", "compatibility", "crossing"):
            if event.get(predicate) not in {"TRUE", "FALSE", "INDETERMINATE"}:
                raise GateError(f"generation {generation} has invalid tri-state {predicate}")
        _hex_digest(row.get("state_digest"), label=f"generation {generation} state digest", tagged=True)
        lineages.append(_hex_digest(row.get("lineage_digest"), label=f"generation {generation} lineage"))
    if "VERIFIED" not in statuses or "INDETERMINATE" not in statuses:
        raise GateError("bounded replay must exercise both verified and indeterminate admission")
    if len(set(lineages)) != len(lineages):
        raise GateError("lineage digest did not advance uniquely per generation")
    if payload.get("lineage_head") != lineages[-1]:
        raise GateError("replay lineage_head does not bind the final generation")
    proof_inventory = payload.get("proof_inventory")
    proofs = proof_inventory.get(PROFILE_ID) if isinstance(proof_inventory, Mapping) else None
    if not isinstance(proofs, Mapping) or set(proofs) != set(PROFILE_REQUIREMENT_IDS):
        raise GateError("replay proof inventory does not exactly cover the selected SCLP profile")
    for requirement_id in PROFILE_REQUIREMENT_IDS:
        proof = proofs.get(requirement_id)
        if (
            not isinstance(proof, Mapping)
            or proof.get("profile_id") != PROFILE_ID
            or proof.get("requirement_id") != requirement_id
            or proof.get("passed") is not True
            or not isinstance(proof.get("evidence_paths"), list)
            or not proof["evidence_paths"]
            or any(
                not isinstance(pointer, str) or not pointer.startswith("/")
                for pointer in proof["evidence_paths"]
            )
        ):
            raise GateError(f"replay proof inventory has invalid coverage for {requirement_id}")
    distinctions = payload.get("distinctions")
    if (
        not isinstance(distinctions, Mapping)
        or distinctions.get("self_reference_may_propose_extensions") is not False
        or distinctions.get("self_reference_may_promote") is not False
    ):
        raise GateError("replay feedback must have no extension proposal or promotion authority")
    return {
        "passed": True,
        "generations": len(rows),
        "statuses": sorted(statuses),
        "definition_count": len(definition_hashes),
        "lineage_head": lineages[-1],
        "exact_key_widths": KEY_WIDTHS,
        "profile_requirement_count": len(proofs),
    }


def _load_json_file(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateError(f"{label} is not valid UTF-8 JSON: {exc}") from exc


def _run_replay(fixture: Path, generations: int, output: Path, *, timeout: float) -> dict[str, Any]:
    result = _BASE._run_process(
        [
            sys.executable,
            "src/substrate_app.py",
            "--generations",
            str(generations),
            "--output",
            str(output),
        ],
        cwd=fixture,
        timeout=timeout,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")[:2000]
        raise GateError(f"independent replay command failed: {stderr}")
    return _load_json_file(output, label=f"{generations}-generation replay")


def _evaluate_replay(fixture: Path, temporary: Path, *, timeout: float) -> dict[str, Any]:
    temporary.mkdir(parents=True, exist_ok=False)
    committed = _load_json_file(fixture / "evidence/replay.json", label="committed replay evidence")
    committed_gate = _validate_replay_payload(committed, expected_generations=5)
    short_a_path = temporary / "replay-short-a.json"
    short_b_path = temporary / "replay-short-b.json"
    long_path = temporary / "replay-long.json"
    short_a = _run_replay(fixture, 3, short_a_path, timeout=timeout)
    short_b = _run_replay(fixture, 3, short_b_path, timeout=timeout)
    long_replay = _run_replay(fixture, 5, long_path, timeout=timeout)
    _validate_replay_payload(short_a, expected_generations=3)
    _validate_replay_payload(short_b, expected_generations=3)
    _validate_replay_payload(long_replay, expected_generations=5)
    if short_a_path.read_bytes() != short_b_path.read_bytes() or short_a != short_b:
        raise GateError("same-seed independent replay was not byte-for-byte deterministic")
    if short_a["generations"] != long_replay["generations"][:3]:
        raise GateError("longer replay changed the already-produced generation prefix")
    if committed != long_replay:
        raise GateError("committed replay evidence differs from an independent five-generation replay")
    return {
        **committed_gate,
        "byte_deterministic": True,
        "prefix_stable": True,
        "committed_replay_matches_independent_run": True,
        "replay_sha256": _sha256_file(fixture / "evidence/replay.json"),
    }


def _evaluate_pdf(fixture: Path) -> dict[str, Any]:
    pdf = fixture / "output/substrate-report.pdf"
    sidecar_path = fixture / "output/substrate-report.pdf.metadata.json"
    try:
        header = pdf.read_bytes()[:8]
    except OSError as exc:
        raise GateError(f"generated PDF cannot be reopened: {exc}") from exc
    if not header.startswith(b"%PDF-"):
        raise GateError("generated report does not have a PDF header")
    sidecar = _load_json_file(sidecar_path, label="PDF metadata sidecar")
    output_record = sidecar.get("output") if isinstance(sidecar, Mapping) else None
    validation = sidecar.get("validation") if isinstance(sidecar, Mapping) else None
    if not isinstance(output_record, Mapping) or output_record.get("sha256") != _sha256_file(pdf):
        raise GateError("PDF metadata sidecar does not bind the report digest")
    if not isinstance(validation, Mapping) or validation.get("passed") is not True:
        raise GateError("PDF metadata sidecar does not record passed text validation")
    pages = output_record.get("pages")
    if isinstance(pages, bool) or not isinstance(pages, int) or pages < 1:
        raise GateError("PDF metadata sidecar does not record a positive page count")
    source = sidecar.get("source")
    if not isinstance(source, Mapping) or source.get("sha256") != _sha256_file(fixture / "report/substrate-report-v0.1.md"):
        raise GateError("PDF metadata sidecar does not bind the report source")
    return {
        "passed": True,
        "reopened": True,
        "text_validation_recorded": True,
        "expected_text": list(EXPECTED_TEXT),
        "pages": pages,
        "pdf_sha256": _sha256_file(pdf),
        "source_sha256": source["sha256"],
    }


def _evaluate_manifest_and_pdf_cli(
    fixture: Path, *, timeout: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = _BASE._run_process(
        [sys.executable, "scripts/validate_fixture.py"], cwd=fixture, timeout=timeout
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")[:3000]
        raise GateError(f"independent contract/PDF validator failed: {stderr}")
    try:
        payload = json.loads(result.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError("independent contract/PDF validator did not return JSON") from exc
    manifest = payload.get("manifest") if isinstance(payload, Mapping) else None
    if not isinstance(manifest, Mapping) or manifest.get("ok") is not True:
        raise GateError("substrate_contract did not validate the application manifest")
    if manifest.get("selected_profiles") != [PROFILE_ID]:
        raise GateError("application manifest did not select exactly sclp-foundational")
    if manifest.get("mapping_count") != EXPECTED_MAPPING_ROWS:
        raise GateError("application manifest did not bind every implemented and bypass mapping")
    if manifest.get("profile_requirement_count") != len(PROFILE_REQUIREMENT_IDS):
        raise GateError("application manifest did not cover every selected-profile requirement")
    if manifest.get("self_reference_enabled") is not True:
        raise GateError("application manifest omitted bounded self-reference")
    return dict(manifest), _evaluate_pdf(fixture)


def _evaluate_git_state(fixture: Path, baseline: Mapping[str, Any]) -> dict[str, Any]:
    head = _BASE._run_git(fixture, ["rev-parse", "HEAD"]).decode("ascii").strip()
    if head != baseline["initial_head"]:
        raise GateError("agent changed exact Git HEAD or created a commit")
    cached = _BASE._run_process(
        ["git", "diff", "--cached", "--quiet", "--exit-code"], cwd=fixture, timeout=30.0
    )
    if cached.returncode != 0:
        raise GateError("agent staged fixture changes")
    raw = _BASE._run_git(
        fixture, ["status", "--porcelain=v1", "--untracked-files=all", "-z"]
    )
    entries = _BASE._parse_porcelain_z(raw)
    actual: dict[str, str] = {}
    for entry in entries:
        relative = entry["path"].replace("\\", "/")
        if relative in actual:
            raise GateError(f"duplicate Git status entry for {relative!r}")
        actual[relative] = entry["status"]
    if set(actual) != EXPECTED_CHANGED_PATHS:
        raise GateError(
            "fixture changed/created paths differ from the exact allowlist: "
            f"expected {sorted(EXPECTED_CHANGED_PATHS)!r}, got {sorted(actual)!r}"
        )
    for relative in MUTABLE_TRACKED_PATHS:
        if actual[relative] != " M":
            raise GateError(f"tracked mutable path {relative!r} must be modified but unstaged")
    for relative in GENERATED_PATHS:
        if actual[relative] != "??":
            raise GateError(f"generated path {relative!r} must be untracked")
    immutable_after = _snapshot_files(fixture, excluded=EXPECTED_CHANGED_PATHS)
    if immutable_after != baseline["immutable_sha256_before"]:
        before = baseline["immutable_sha256_before"]
        changed = sorted(
            set(before) ^ set(immutable_after)
            | {path for path in set(before) & set(immutable_after) if before[path] != immutable_after[path]}
        )
        raise GateError(f"immutable clean-room fixture files changed: {changed!r}")
    for relative, digest in baseline["mutable_sha256_before"].items():
        if _sha256_file(fixture / relative) == digest:
            raise GateError(f"required mutable artifact was not materially completed: {relative}")
    return {
        "passed": True,
        "initial_head": baseline["initial_head"],
        "final_head": head,
        "exact_head_unchanged": True,
        "staged_changes": False,
        "git_status": entries,
        "changed_paths": sorted(actual),
        "only_expected_paths_changed_or_created": True,
        "immutable_file_hashes_unchanged": True,
    }


def _write_sha256sums(run_dir: Path) -> None:
    checksum = run_dir / "SHA256SUMS"
    paths = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != checksum)
    _write_text(
        checksum,
        "".join(f"{_sha256_file(path)}  {path.relative_to(run_dir).as_posix()}\n" for path in paths),
    )


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
        "scope": {
            "classification": "bounded_literal_substrate_acceptance_gate",
            "proves": [
                "one local model can complete the exact clean-room SCLP fixture",
                "the completed fixture passes the named deterministic primitive checks",
                "the application manifest binds the sealed kernel/profile/replay evidence",
                "the repository PDF renderer produces reopenable text-validated evidence",
            ],
            "does_not_prove": [
                "broad intelligence or general coding competence",
                "model compression, NHDF compression, or semantic compression",
                "production safety, universal substrate correctness, or convergence",
                "preventive process, filesystem, or network sandboxing",
            ],
            "legacy_or_archive_material_loaded": False,
            "post_run_audits_are_retrospective": True,
            "preventive_sandbox": False,
        },
    }
    try:
        server_url = _BASE._normalize_server_url(args.server_url)
        executable = _resolve_executable(args.opencode_exe)
        config_path, config_bytes, config = _BASE._load_pinned_config(args.config)
        config_gate = _BASE._validate_opencode_config(config, server_url)
        evidence["inputs"] = {
            "server_url": server_url,
            "opencode_executable": str(executable),
            "opencode_executable_sha256": _sha256_file(executable),
            "config_path": str(config_path),
            "config_sha256": _sha256_bytes(config_bytes),
            "model": MODEL_ID,
        }
        evidence["config_gate"] = config_gate

        models, models_timing = _BASE._http_json(
            server_url, "/v1/models", timeout=args.http_timeout_seconds
        )
        model_gate = _BASE._validate_models_payload(models)
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

        with tempfile.TemporaryDirectory(prefix="ugtoms-substrate-coder-gate-") as temp_text:
            temp_root = Path(temp_text)
            fixture = temp_root / "fixture"
            baseline = _create_fixture(fixture, require_tracked=True)
            initial_test = _BASE._run_process(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=fixture,
                timeout=args.fixture_timeout_seconds,
            )
            (run_dir / "fixture.baseline-pytest.stdout.txt").write_bytes(initial_test.stdout)
            (run_dir / "fixture.baseline-pytest.stderr.txt").write_bytes(initial_test.stderr)
            if initial_test.returncode == 0:
                raise GateError("substrate fixture unexpectedly passed before agent implementation")

            isolated = temp_root / "isolated"
            copied_config = isolated / "config/opencode/opencode.json"
            copied_config.parent.mkdir(parents=True, exist_ok=True)
            copied_config.write_bytes(config_bytes)
            agent_env = _BASE._agent_environment(
                os.environ, isolated, copied_config, server_url=server_url
            )
            version_result = _BASE._run_process(
                [str(executable), "--version"], cwd=fixture, env=agent_env, timeout=30.0
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
                raise GateError(f"OpenCode agent timed out after {args.agent_timeout_seconds:.1f}s") from exc
            elapsed = time.perf_counter() - started
            stdout_path.write_bytes(agent.stdout)
            stderr_path.write_bytes(agent.stderr)
            if agent.returncode != 0:
                raise GateError(f"OpenCode exited with status {agent.returncode}")
            events = _BASE._parse_json_events(agent.stdout)
            tool_gate = _evaluate_tool_events(events, fixture)

            final_test = _BASE._run_process(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=fixture,
                timeout=args.fixture_timeout_seconds,
            )
            (run_dir / "fixture.final-pytest.stdout.txt").write_bytes(final_test.stdout)
            (run_dir / "fixture.final-pytest.stderr.txt").write_bytes(final_test.stderr)
            if final_test.returncode != 0:
                raise GateError("independent focused pytest did not pass")
            manifest_gate, pdf_gate = _evaluate_manifest_and_pdf_cli(
                fixture, timeout=args.fixture_timeout_seconds
            )
            replay_gate = _evaluate_replay(
                fixture, temp_root / "independent-replays", timeout=args.fixture_timeout_seconds
            )
            git_gate = _evaluate_git_state(fixture, baseline)
            diff = _BASE._run_git(fixture, ["diff", "--binary", "HEAD", "--", *sorted(MUTABLE_TRACKED_PATHS)])
            (run_dir / "fixture.expected-changes.patch").write_bytes(diff)

            evidence["agent_gate"] = {
                "passed": True,
                "opencode_version": version,
                "command_arguments": command[1:-1],
                "prompt_sha256": _sha256_bytes(_agent_prompt().encode("utf-8")),
                "elapsed_seconds": elapsed,
                "json_event_count": len(events),
                "tool_audit": tool_gate,
                "isolated_environment": {
                    "same_digest_pinned_contract_installed": True,
                    "same_pinned_config_validated": True,
                    "parent_environment_reduced_to_operational_allowlist": True,
                    "credential_like_environment_inherited": False,
                    "plugins_mcp_skills_project_config_autoupdate_share_telemetry_disabled": True,
                },
            }
            evidence["fixture"] = {
                **baseline,
                "baseline_pytest_returncode": initial_test.returncode,
                "final_pytest": {
                    "passed": True,
                    "returncode": final_test.returncode,
                    "stdout_sha256": _sha256_bytes(final_test.stdout),
                    "stderr_sha256": _sha256_bytes(final_test.stderr),
                },
                "manifest_gate": manifest_gate,
                "pdf_gate": pdf_gate,
                "replay_gate": replay_gate,
                "git_gate": git_gate,
            }
        evidence["status"] = "PASSED"
        evidence["all_gates_passed"] = True
        exit_code = 0
    except (GateError, ValueError) as exc:
        evidence["status"] = "FAILED"
        evidence["all_gates_passed"] = False
        evidence["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        exit_code = 1
    except Exception as exc:
        evidence["status"] = "ERROR"
        evidence["all_gates_passed"] = False
        evidence["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        exit_code = 2
    evidence["finished_at_utc"] = _utc_now()
    _write_json(run_dir / "evidence.json", evidence)
    _write_sha256sums(run_dir)
    return exit_code, run_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the bounded literal-substrate coding gate. This is not broad "
            "intelligence or a compression proof; audits are retrospective."
        )
    )
    parser.add_argument("--server-url", required=True, help="existing loopback local server URL")
    parser.add_argument("--opencode-exe", required=True, help=f"OpenCode {PINNED_OPENCODE_VERSION} executable")
    parser.add_argument("--config", required=True, help="canonical digest-pinned local coder config")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--http-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--agent-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--fixture-timeout-seconds", type=float, default=120.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    for name in ("http_timeout_seconds", "agent_timeout_seconds", "fixture_timeout_seconds"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    exit_code, run_dir = _run_gate(args)
    print(json.dumps({"status": "PASSED" if exit_code == 0 else "FAILED", "evidence": str(run_dir)}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
