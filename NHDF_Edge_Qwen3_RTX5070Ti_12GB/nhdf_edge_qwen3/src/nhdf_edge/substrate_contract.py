"""Fail-closed contracts for the compact UGTS/UGTOMS substrate.

This module deliberately separates the substrate kernel from selectable
profiles and from applications which use it.  The bounded transition chain is
one discipline of the kernel, not a claim that the chain alone defines the
substrate.  Compact generative representation, typed mappings, resource
bounds, observable failures, and evidence remain explicit.

Only JSON-compatible values and the Python standard library are used.  File
references are repository-relative, cannot escape their declared root, and
are verified by byte length and SHA-256 before a contract is accepted.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


KERNEL_FORMAT = "ugtoms-kernel-contract-0.1"
PROFILE_REGISTRY_FORMAT = "ugtoms-profile-registry-0.1"
PROFILE_FORMAT = "ugtoms-profile-0.1"
APPLICATION_FORMAT = "ugtoms-application-manifest-0.1"
EXTENSION_PROPOSAL_FORMAT = "ugtoms-extension-proposal-0.1"

KERNEL_ID = "ugtoms-kernel-v0.1"
DEFAULT_KERNEL_PATH = Path("substrate/kernel/contract.json")
DEFAULT_PROFILE_REGISTRY_PATH = Path("substrate/profiles/registry.json")

CANONICAL_TRANSITION_CHAIN = (
    "local_support",
    "compatibility",
    "guard_crossing",
    "verified_event",
    "route_transition",
    "lineage",
    "novelty_log",
)

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

COMPOSED_EXECUTION = (
    "input_residual",
    "log_polar_address_and_metric",
    "cell_local_nondegenerate_zero_set",
    "typed_parity_jitter_control_predicates",
    "bounded_BST_L_system_or_radix_routing",
    "causal_vector_kinematics",
    "cone_sphere_SDF_sweep_relation_and_projection",
    "event_admission_discipline",
    "atomic_transition",
    "lineage_and_novelty",
    "explicit_next_generation_feedback",
)

KERNEL_MAPPING_REQUIREMENTS = {
    "typed": ("domain", "codomain", "units", "bounds", "failure_modes"),
    "vector": ("origin", "directed_displacement", "frame", "units"),
    "kinematic": (
        "linear_time",
        "position",
        "velocity",
        "acceleration",
        "integration_or_closed_form",
    ),
    "geometry": (
        "local_zero_set",
        "sphere_SDF",
        "circle_relation",
        "cone_relation",
        "distributed_apex",
    ),
    "topology": ("chart", "sheet", "orientation", "ports", "transfer_map", "winding"),
    "packing": (
        "field_widths",
        "quantizer",
        "layout",
        "inverse_or_error_contract",
        "finite_capacity",
    ),
    "predicate": ("support", "compatibility", "guard", "certificate", "indeterminate_path"),
    "operator": ("equation_or_algorithm", "dependencies", "evaluation_phase", "resource_budget"),
    "self_reference": (
        "definition_hash",
        "generative_address",
        "lineage",
        "feedback_edge",
        "generation_boundary",
    ),
}

REQUIRED_PROFILE_IDS = frozenset(
    {"nhdf-v0.1", "nhdf-v0.3-ccd", "sclp-foundational"}
)

SOURCE_ROLES = frozenset(
    {
        "FOUNDATIONAL_SOURCE",
        "FOUNDATIONAL_NORMALIZATION",
        "EXECUTABLE_REFERENCE",
        "FORMAL_PROFILE",
        "APPLICATION_EVIDENCE",
        "PROFILE_SOURCE",
    }
)
EVIDENCE_KINDS = frozenset(
    {
        "ANALYSIS",
        "BENCHMARK",
        "REFERENCE_VECTOR",
        "RENDERED_DOCUMENT",
        "REPLAY",
        "TEST",
    }
)
ADMISSION_FORMS = frozenset(
    {
        "TYPED_FIELD",
        "COORDINATE_TRANSFORM",
        "SUPPORT_PREDICATE",
        "RELATION_SURFACE",
        "COMPATIBILITY_PREDICATE",
        "TRANSITION_ROUTING_MAP",
        "LINEAGE_RULE",
        "CALIBRATED_TRANSFER_FUNCTION",
        "RESOURCE_ERROR_QUANTITY",
        "GENERATIVE_RECONSTRUCTION_RULE",
    }
)

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


class SubstrateContractError(ValueError):
    """Raised when a substrate document or one of its sealed inputs is invalid."""


@dataclass(frozen=True)
class ContractBundle:
    """A verified kernel plus its verified selectable-profile registry."""

    repository_root: Path
    kernel_path: Path
    registry_path: Path
    kernel: dict[str, Any]
    registry: dict[str, Any]
    profiles: dict[str, dict[str, Any]]
    kernel_sha256: str
    profile_sha256s: dict[str, str]


def canonical_json_bytes(value: object) -> bytes:
    """Return deterministic UTF-8 JSON bytes for content-addressed records."""

    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _fail(location: str, message: str) -> None:
    raise SubstrateContractError(f"{location}: {message}")


def _object(
    value: object,
    location: str,
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(location, "must be an object")
    result = dict(value)
    required_keys = set(required)
    allowed = required_keys | set(optional)
    missing = sorted(required_keys - result.keys())
    unknown = sorted(result.keys() - allowed)
    if missing:
        _fail(location, f"missing required keys {missing!r}")
    if unknown:
        _fail(location, f"contains unknown keys {unknown!r}; bump the format to extend it")
    return result


def _string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        _fail(location, "must be a non-empty string without surrounding whitespace")
    return value


def _identifier(value: object, location: str) -> str:
    text = _string(value, location)
    if not _IDENTIFIER.fullmatch(text):
        _fail(location, f"is not a stable lowercase identifier: {text!r}")
    return text


def _boolean(value: object, location: str) -> bool:
    if not isinstance(value, bool):
        _fail(location, "must be a boolean")
    return value


def _integer(value: object, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(location, f"must be an integer >= {minimum}")
    return value


def _number(value: object, location: str, *, minimum: float = 0.0) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(location, "must be a finite number")
    if not math.isfinite(float(value)) or float(value) < minimum:
        _fail(location, f"must be finite and >= {minimum}")
    return value


def _enum(value: object, location: str, allowed: Iterable[str]) -> str:
    text = _string(value, location)
    allowed_values = frozenset(allowed)
    if text not in allowed_values:
        _fail(location, f"must be one of {sorted(allowed_values)!r}; got {text!r}")
    return text


def _list(value: object, location: str, *, nonempty: bool = True) -> list[Any]:
    if not isinstance(value, list):
        _fail(location, "must be an array")
    if nonempty and not value:
        _fail(location, "must not be empty")
    return value


def _string_list(
    value: object,
    location: str,
    *,
    identifiers: bool = False,
    nonempty: bool = True,
) -> list[str]:
    items = _list(value, location, nonempty=nonempty)
    result = [
        (_identifier(item, f"{location}[{index}]") if identifiers else _string(item, f"{location}[{index}]"))
        for index, item in enumerate(items)
    ]
    if len(set(result)) != len(result):
        _fail(location, "must not contain duplicates")
    return result


def _digest(value: object, location: str) -> str:
    text = _string(value, location)
    if not _SHA256.fullmatch(text):
        _fail(location, "must be a 64-character hexadecimal SHA-256")
    return text.lower()


def _root(root: str | Path) -> Path:
    path = Path(root).resolve(strict=True)
    if not path.is_dir():
        raise SubstrateContractError(f"repository root is not a directory: {path}")
    return path


def _contained_file(root: Path, reference: object, location: str) -> Path:
    text = _string(reference, location)
    candidate = Path(text)
    if candidate.is_absolute() or candidate.drive or ".." in candidate.parts:
        _fail(location, "must be a repository-relative path without '..'")
    try:
        resolved = (root / candidate).resolve(strict=True)
    except OSError as exc:
        _fail(location, f"does not resolve to an existing file: {exc}")
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail(location, "resolves outside the declared repository root")
    if not resolved.is_file():
        _fail(location, "must resolve to a regular file")
    return resolved


def _parameter_file(root: Path, value: str | Path, location: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        _fail(location, f"does not resolve to an existing file: {exc}")
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail(location, "must remain inside the repository root")
    if not resolved.is_file():
        _fail(location, "must resolve to a regular file")
    return resolved


def _repo_reference(root: Path, path: Path) -> str:
    return Path(os.path.relpath(path, root)).as_posix()


def _load_json(path: Path, location: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail(location, f"is not valid UTF-8 JSON: {exc}")
    if not isinstance(value, dict):
        _fail(location, "must contain a JSON object")
    return value


def _verify_file_record(
    value: object,
    *,
    root: Path,
    location: str,
    role_values: Iterable[str] | None,
    verify_file: bool,
) -> dict[str, Any]:
    record = _object(
        value,
        location,
        required=("id", "role", "path", "bytes", "sha256"),
        optional=("title", "note"),
    )
    _identifier(record["id"], f"{location}.id")
    if role_values is None:
        _string(record["role"], f"{location}.role")
    else:
        _enum(record["role"], f"{location}.role", role_values)
    expected_bytes = _integer(record["bytes"], f"{location}.bytes", minimum=1)
    expected_digest = _digest(record["sha256"], f"{location}.sha256")
    if "title" in record:
        _string(record["title"], f"{location}.title")
    if "note" in record:
        _string(record["note"], f"{location}.note")
    if verify_file:
        path = _contained_file(root, record["path"], f"{location}.path")
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            _fail(location, f"byte length mismatch: expected {expected_bytes}, got {actual_bytes}")
        actual_digest = sha256_file(path)
        if actual_digest != expected_digest:
            _fail(location, f"SHA-256 mismatch: expected {expected_digest}, got {actual_digest}")
    else:
        text = _string(record["path"], f"{location}.path")
        candidate = Path(text)
        if candidate.is_absolute() or candidate.drive or ".." in candidate.parts:
            _fail(f"{location}.path", "must be a safe repository-relative path")
    return record


def _verify_file_records(
    value: object,
    *,
    root: Path,
    location: str,
    role_values: Iterable[str] | None,
    verify_files: bool,
    nonempty: bool = True,
) -> list[dict[str, Any]]:
    records = _list(value, location, nonempty=nonempty)
    checked = [
        _verify_file_record(
            record,
            root=root,
            location=f"{location}[{index}]",
            role_values=role_values,
            verify_file=verify_files,
        )
        for index, record in enumerate(records)
    ]
    ids = [record["id"] for record in checked]
    paths = [record["path"] for record in checked]
    if len(ids) != len(set(ids)):
        _fail(location, "contains duplicate record IDs")
    if len(paths) != len(set(paths)):
        _fail(location, "contains duplicate file paths")
    return checked


def _find_parent_repository_root(root: Path) -> Path:
    """Find the containing Git repository without invoking Git."""

    for candidate in (root, *root.parents):
        if (candidate / ".git").exists():
            return candidate
    return root


def _source_records(
    value: object,
    *,
    repository_root: Path,
    parent_repository_root: Path,
    location: str,
    verify_files: bool,
    nonempty: bool = True,
) -> dict[str, int]:
    """Validate mixed committed and external read-only provenance records."""

    records = _list(value, location, nonempty=nonempty)
    identities: set[str] = set()
    committed = 0
    external = 0
    for index, value_record in enumerate(records):
        record_location = f"{location}[{index}]"
        if not isinstance(value_record, Mapping):
            _fail(record_location, "must be an object")
        availability = _enum(
            value_record.get("availability"),
            record_location + ".availability",
            {
                "COMMITTED",
                "COMMITTED_PARENT_REPOSITORY",
                "READ_ONLY_EXTERNAL",
                "REGISTERED_NOT_INSPECTED",
            },
        )
        common_optional = {"source_id", "class", "pages", "authority"}
        if availability in {"COMMITTED", "COMMITTED_PARENT_REPOSITORY"}:
            required = {"availability", "path", "bytes", "sha256"}
            allowed = required | common_optional
        else:
            required = {"availability", "source_id", "sha256", "redistributed"}
            allowed = required | {"class", "bytes", "pages", "authority"}
        missing = sorted(required - value_record.keys())
        unknown = sorted(value_record.keys() - allowed)
        if missing:
            _fail(record_location, f"missing required keys {missing!r}")
        if unknown:
            _fail(record_location, f"contains unknown keys {unknown!r}")
        if "source_id" in value_record:
            identity = _identifier(value_record["source_id"], record_location + ".source_id")
        else:
            identity = _string(value_record["path"], record_location + ".path")
        if identity in identities:
            _fail(record_location, f"duplicates source identity {identity!r}")
        identities.add(identity)
        if "class" in value_record:
            _string(value_record["class"], record_location + ".class")
        if "authority" in value_record:
            _string(value_record["authority"], record_location + ".authority")
        if "pages" in value_record:
            _integer(value_record["pages"], record_location + ".pages", minimum=1)
        expected_digest = _digest(value_record["sha256"], record_location + ".sha256")
        if availability in {"COMMITTED", "COMMITTED_PARENT_REPOSITORY"}:
            source_root = (
                repository_root
                if availability == "COMMITTED"
                else parent_repository_root
            )
            expected_bytes = _integer(
                value_record["bytes"], record_location + ".bytes", minimum=1
            )
            if verify_files:
                path = _contained_file(source_root, value_record["path"], record_location + ".path")
                actual_bytes = path.stat().st_size
                if actual_bytes != expected_bytes:
                    _fail(
                        record_location,
                        f"byte length mismatch: expected {expected_bytes}, got {actual_bytes}",
                    )
                actual_digest = sha256_file(path)
                if actual_digest != expected_digest:
                    _fail(
                        record_location,
                        f"SHA-256 mismatch: expected {expected_digest}, got {actual_digest}",
                    )
            else:
                text = _string(value_record["path"], record_location + ".path")
                path = Path(text)
                if path.is_absolute() or path.drive or ".." in path.parts:
                    _fail(record_location + ".path", "must be a safe repository-relative path")
            committed += 1
        else:
            if _boolean(value_record["redistributed"], record_location + ".redistributed"):
                _fail(
                    record_location + ".redistributed",
                    "read-only external provenance must not claim redistribution",
                )
            if "bytes" in value_record:
                _integer(value_record["bytes"], record_location + ".bytes", minimum=1)
            external += 1
    return {"record_count": len(records), "committed_verified": committed, "external_registered": external}


def _resource_bounds(value: object, location: str) -> list[dict[str, Any]]:
    rows = _list(value, location)
    checked: list[dict[str, Any]] = []
    for index, value_row in enumerate(rows):
        row_location = f"{location}[{index}]"
        row = _object(
            value_row,
            row_location,
            required=("id", "resource", "limit", "unit", "enforcement"),
        )
        _identifier(row["id"], f"{row_location}.id")
        _string(row["resource"], f"{row_location}.resource")
        _number(row["limit"], f"{row_location}.limit")
        _string(row["unit"], f"{row_location}.unit")
        _string(row["enforcement"], f"{row_location}.enforcement")
        checked.append(row)
    _unique_ids(checked, location)
    return checked


def _failure_modes(value: object, location: str) -> list[dict[str, Any]]:
    rows = _list(value, location)
    checked: list[dict[str, Any]] = []
    for index, value_row in enumerate(rows):
        row_location = f"{location}[{index}]"
        row = _object(
            value_row,
            row_location,
            required=("id", "condition", "observable", "handling"),
        )
        _identifier(row["id"], f"{row_location}.id")
        for key in ("condition", "observable", "handling"):
            _string(row[key], f"{row_location}.{key}")
        checked.append(row)
    _unique_ids(checked, location)
    return checked


def _evidence_requirements(value: object, location: str) -> list[dict[str, Any]]:
    rows = _list(value, location)
    checked: list[dict[str, Any]] = []
    for index, value_row in enumerate(rows):
        row_location = f"{location}[{index}]"
        row = _object(
            value_row,
            row_location,
            required=("id", "claim", "method", "pass_condition"),
        )
        _identifier(row["id"], f"{row_location}.id")
        for key in ("claim", "method", "pass_condition"):
            _string(row[key], f"{row_location}.{key}")
        checked.append(row)
    _unique_ids(checked, location)
    return checked


def _unique_ids(rows: Sequence[Mapping[str, Any]], location: str) -> None:
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        _fail(location, "contains duplicate IDs")


def _mapping_shapes(
    value: object,
    location: str,
    *,
    dispositions: Iterable[str],
) -> dict[str, list[dict[str, Any]]]:
    mappings = _object(value, location, required=MAPPING_CATEGORIES)
    checked: dict[str, list[dict[str, Any]]] = {}
    all_ids: list[str] = []
    for category in MAPPING_CATEGORIES:
        rows = _list(mappings[category], f"{location}.{category}")
        checked_rows: list[dict[str, Any]] = []
        for index, value_row in enumerate(rows):
            row_location = f"{location}.{category}[{index}]"
            row = _object(
                value_row,
                row_location,
                required=(
                    "id",
                    "domain",
                    "codomain",
                    "definition",
                    "primitive_refs",
                    "bound_refs",
                    "failure_refs",
                    "evidence_refs",
                    "disposition",
                ),
            )
            row_id = _identifier(row["id"], f"{row_location}.id")
            _string_list(row["domain"], f"{row_location}.domain")
            _string_list(row["codomain"], f"{row_location}.codomain")
            _string(row["definition"], f"{row_location}.definition")
            _string_list(row["primitive_refs"], f"{row_location}.primitive_refs", identifiers=True)
            _string_list(row["bound_refs"], f"{row_location}.bound_refs", identifiers=True)
            _string_list(row["failure_refs"], f"{row_location}.failure_refs", identifiers=True)
            _string_list(row["evidence_refs"], f"{row_location}.evidence_refs", identifiers=True)
            _enum(row["disposition"], f"{row_location}.disposition", dispositions)
            checked_rows.append(row)
            all_ids.append(row_id)
        checked[category] = checked_rows
    if len(all_ids) != len(set(all_ids)):
        _fail(location, "mapping IDs must be globally unique across categories")
    return checked


def _check_mapping_references(
    mappings: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    location: str,
    primitive_ids: Iterable[str],
    bound_ids: Iterable[str],
    failure_ids: Iterable[str],
    evidence_ids: Iterable[str],
) -> None:
    universes = {
        "primitive_refs": frozenset(primitive_ids),
        "bound_refs": frozenset(bound_ids),
        "failure_refs": frozenset(failure_ids),
        "evidence_refs": frozenset(evidence_ids),
    }
    for category, rows in mappings.items():
        for index, row in enumerate(rows):
            for key, universe in universes.items():
                unknown = sorted(set(row[key]) - universe)
                if unknown:
                    _fail(
                        f"{location}.{category}[{index}].{key}",
                        f"contains unknown references {unknown!r}",
                    )


def _mapping_ids(mappings: Mapping[str, Sequence[Mapping[str, Any]]]) -> set[str]:
    return {str(row["id"]) for rows in mappings.values() for row in rows}


def _self_reference(value: object, location: str) -> dict[str, Any]:
    record = _object(
        value,
        location,
        required=(
            "enabled",
            "bounded_generations",
            "may_propose_extensions",
            "may_promote_extensions",
            "proposal_disposition",
        ),
    )
    enabled = _boolean(record["enabled"], f"{location}.enabled")
    generations = _integer(record["bounded_generations"], f"{location}.bounded_generations")
    may_propose = _boolean(record["may_propose_extensions"], f"{location}.may_propose_extensions")
    may_promote = _boolean(record["may_promote_extensions"], f"{location}.may_promote_extensions")
    if may_promote:
        _fail(f"{location}.may_promote_extensions", "self-reference may propose but cannot promote")
    if _enum(
        record["proposal_disposition"],
        f"{location}.proposal_disposition",
        {"QUARANTINED"},
    ) != "QUARANTINED":  # pragma: no cover - _enum already raises
        _fail(location, "invalid proposal disposition")
    if enabled and (generations < 1 or not may_propose):
        _fail(location, "enabled self-reference needs a positive generation bound and proposal permission")
    if not enabled and (generations != 0 or may_propose):
        _fail(location, "disabled self-reference must have zero generations and no proposal permission")
    return record


def validate_kernel_contract(
    contract: object,
    *,
    repository_root: str | Path,
    parent_repository_root: str | Path | None = None,
    verify_sources: bool = True,
) -> dict[str, Any]:
    """Validate the foundational kernel and all committed source records."""

    root = _root(repository_root)
    parent_root = _root(parent_repository_root) if parent_repository_root is not None else _find_parent_repository_root(root)
    value = _object(
        contract,
        "kernel",
        required=(
            "format",
            "kernel_id",
            "title",
            "lineage",
            "identity",
            "state_schema",
            "symbol_firewall",
            "definition_graph",
            "composed_execution",
            "canonical_chain",
            "mappings",
            "source_records",
            "policy",
        ),
    )
    if value["format"] != KERNEL_FORMAT:
        _fail("kernel.format", f"must be {KERNEL_FORMAT!r}")
    if value["kernel_id"] != KERNEL_ID:
        _fail("kernel.kernel_id", f"must be {KERNEL_ID!r}")
    _string(value["title"], "kernel.title")
    lineage = _object(
        value["lineage"],
        "kernel.lineage",
        required=(
            "semantic_base",
            "early_executable_algebra",
            "referential_layer",
            "foundational_correction",
            "execution_discipline",
            "later_profiles_do_not_replace_base",
        ),
    )
    for key in (
        "semantic_base",
        "early_executable_algebra",
        "referential_layer",
        "foundational_correction",
        "execution_discipline",
    ):
        _string(lineage[key], f"kernel.lineage.{key}")
    if not _boolean(
        lineage["later_profiles_do_not_replace_base"],
        "kernel.lineage.later_profiles_do_not_replace_base",
    ):
        _fail("kernel.lineage.later_profiles_do_not_replace_base", "must be true")

    identity = _object(
        value["identity"],
        "kernel.identity",
        required=(
            "definition",
            "not_a_renderer",
            "not_query_calculus_only",
            "not_general_ai_by_claim",
            "learned_components_are_optional_consumers",
        ),
    )
    _string(identity["definition"], "kernel.identity.definition")
    for key in (
        "not_a_renderer",
        "not_query_calculus_only",
        "not_general_ai_by_claim",
        "learned_components_are_optional_consumers",
    ):
        if not _boolean(identity[key], f"kernel.identity.{key}"):
            _fail(f"kernel.identity.{key}", "must be true")

    state_schema = _object(
        value["state_schema"],
        "kernel.state_schema",
        required=("continuous", "discrete", "quality"),
    )
    continuous = set(
        _string_list(state_schema["continuous"], "kernel.state_schema.continuous", identifiers=True)
    )
    discrete = set(
        _string_list(state_schema["discrete"], "kernel.state_schema.discrete", identifiers=True)
    )
    quality = set(
        _string_list(state_schema["quality"], "kernel.state_schema.quality", identifiers=True)
    )
    required_state = {
        "continuous": {"position", "linear_time", "velocity", "acceleration"},
        "discrete": {"generation", "orientation", "generative_address", "lineage_head"},
        "quality": {"uncertainty_interval", "resource_status", "failure_status"},
    }
    for name, actual in (("continuous", continuous), ("discrete", discrete), ("quality", quality)):
        missing = sorted(required_state[name] - actual)
        if missing:
            _fail(f"kernel.state_schema.{name}", f"is missing foundational state {missing!r}")

    firewall = _list(value["symbol_firewall"], "kernel.symbol_firewall")
    for index, item in enumerate(firewall):
        location = f"kernel.symbol_firewall[{index}]"
        row = _object(item, location, required=("symbols", "rule"))
        _string_list(row["symbols"], location + ".symbols")
        _string(row["rule"], location + ".rule")

    graph = _object(
        value["definition_graph"],
        "kernel.definition_graph",
        required=(
            "node_fields",
            "same_generation",
            "instance_rule",
            "pipeline_rule",
            "self_reference_rule",
            "unproven_claim",
        ),
    )
    node_fields = set(
        _string_list(graph["node_fields"], "kernel.definition_graph.node_fields", identifiers=True)
    )
    graph_required = {
        "id",
        "kind",
        "domain",
        "codomain",
        "dependencies",
        "equation_or_algorithm",
        "units",
        "bounds",
        "failure_modes",
        "provenance",
        "content_hash",
    }
    if not graph_required.issubset(node_fields):
        _fail(
            "kernel.definition_graph.node_fields",
            f"is missing typed graph fields {sorted(graph_required - node_fields)!r}",
        )
    for key in (
        "same_generation",
        "instance_rule",
        "pipeline_rule",
        "self_reference_rule",
        "unproven_claim",
    ):
        _string(graph[key], f"kernel.definition_graph.{key}")

    composed = _string_list(
        value["composed_execution"], "kernel.composed_execution"
    )
    if tuple(composed) != COMPOSED_EXECUTION:
        _fail("kernel.composed_execution", "must preserve the compact generative execution order")

    chain_rows = _list(value["canonical_chain"], "kernel.canonical_chain")
    chain: list[str] = []
    for index, item in enumerate(chain_rows):
        location = f"kernel.canonical_chain[{index}]"
        row = _object(item, location, required=("id", "role"))
        chain.append(_identifier(row["id"], location + ".id"))
        _string(row["role"], location + ".role")
    if tuple(chain) != CANONICAL_TRANSITION_CHAIN:
        _fail(
            "kernel.canonical_chain",
            "must exactly preserve the bounded transition/handoff discipline "
            f"{list(CANONICAL_TRANSITION_CHAIN)!r}",
        )

    mappings = _object(value["mappings"], "kernel.mappings", required=MAPPING_CATEGORIES)
    for category in MAPPING_CATEGORIES:
        location = f"kernel.mappings.{category}"
        mapping = _object(mappings[category], location, required=("required", "rule"))
        required_fields = _string_list(mapping["required"], location + ".required")
        if tuple(required_fields) != KERNEL_MAPPING_REQUIREMENTS[category]:
            _fail(
                location + ".required",
                f"must be {list(KERNEL_MAPPING_REQUIREMENTS[category])!r}",
            )
        _string(mapping["rule"], location + ".rule")

    source_summary = _source_records(
        value["source_records"],
        repository_root=root,
        parent_repository_root=parent_root,
        location="kernel.source_records",
        verify_files=verify_sources,
    )

    policy = _object(
        value["policy"],
        "kernel.policy",
        required=(
            "determinism",
            "resource_bounds_required",
            "automatic_extension_promotion",
            "same_generation_cycles",
            "exogenous_events_must_be_logged",
            "legacy_repository_mode",
            "bulk_legacy_import",
            "learned_semantics_may_only_rank_or_propose",
        ),
    )
    determinism = _object(
        policy["determinism"],
        "kernel.policy.determinism",
        required=(
            "required",
            "discrete_state",
            "geometric_numeric_backend",
            "rounding",
            "uncertainty",
            "randomness",
        ),
    )
    if not _boolean(determinism["required"], "kernel.policy.determinism.required"):
        _fail("kernel.policy.determinism.required", "must be true")
    for key in (
        "discrete_state",
        "geometric_numeric_backend",
        "rounding",
        "uncertainty",
        "randomness",
    ):
        _string(determinism[key], f"kernel.policy.determinism.{key}")
    expected_flags = {
        "resource_bounds_required": True,
        "automatic_extension_promotion": False,
        "same_generation_cycles": False,
        "exogenous_events_must_be_logged": True,
        "bulk_legacy_import": False,
        "learned_semantics_may_only_rank_or_propose": True,
    }
    for key, expected in expected_flags.items():
        actual = _boolean(policy[key], f"kernel.policy.{key}")
        if actual is not expected:
            _fail(f"kernel.policy.{key}", f"must be {str(expected).lower()}")
    if policy["legacy_repository_mode"] != "READ_ONLY_SELECTIVE_PROVENANCE":
        _fail(
            "kernel.policy.legacy_repository_mode",
            "must be 'READ_ONLY_SELECTIVE_PROVENANCE'",
        )

    return {
        "ok": True,
        "kernel_id": KERNEL_ID,
        "canonical_sha256": canonical_sha256(value),
        "source_count": source_summary["record_count"],
        "committed_sources_verified": source_summary["committed_verified"],
        "external_sources_registered": source_summary["external_registered"],
        "mapping_count": len(mappings),
        "transition_chain": list(CANONICAL_TRANSITION_CHAIN),
    }


def _validate_profile_document(
    profile: object,
    *,
    profile_id: str,
    status: str,
    kernel: Mapping[str, Any],
    kernel_sha256: str,
    repository_root: Path,
    parent_repository_root: Path,
    verify_sources: bool,
) -> dict[str, Any]:
    location = f"profile[{profile_id}]"
    value = _object(
        profile,
        location,
        required=(
            "format",
            "profile_id",
            "title",
            "classification",
            "kernel_id",
            "kernel_sha256",
            "kernel_stage_mapping",
            "mappings",
            "resource_bounds",
            "failure_modes",
            "evidence_requirements",
            "source_records",
        ),
    )
    if value["format"] != PROFILE_FORMAT:
        _fail(f"{location}.format", f"must be {PROFILE_FORMAT!r}")
    if value["profile_id"] != profile_id:
        _fail(f"{location}.profile_id", "does not match its registry entry")
    _identifier(profile_id, f"{location}.profile_id")
    _string(value["title"], f"{location}.title")
    if value["classification"] != "SELECTABLE_PROFILE":
        _fail(f"{location}.classification", "must be 'SELECTABLE_PROFILE'")
    if value["kernel_id"] != kernel["kernel_id"]:
        _fail(f"{location}.kernel_id", "does not bind the verified kernel")
    if _digest(value["kernel_sha256"], f"{location}.kernel_sha256") != kernel_sha256:
        _fail(f"{location}.kernel_sha256", "does not match the verified kernel file")

    bounds = _string_list(value["resource_bounds"], f"{location}.resource_bounds")
    failures = _string_list(value["failure_modes"], f"{location}.failure_modes")
    requirements = _string_list(
        value["evidence_requirements"], f"{location}.evidence_requirements"
    )
    _source_records(
        value["source_records"],
        repository_root=repository_root,
        parent_repository_root=parent_repository_root,
        location=f"{location}.source_records",
        verify_files=verify_sources,
        nonempty=status == "ACTIVE",
    )
    mappings = _object(
        value["mappings"], f"{location}.mappings", required=MAPPING_CATEGORIES
    )
    for category in MAPPING_CATEGORIES:
        _string(mappings[category], f"{location}.mappings.{category}")

    stages = value["kernel_stage_mapping"]
    if not isinstance(stages, Mapping) or not stages:
        _fail(f"{location}.kernel_stage_mapping", "must be a non-empty object")
    for stage_name, stage_mapping in stages.items():
        _string(stage_name, f"{location}.kernel_stage_mapping key")
        _string(stage_mapping, f"{location}.kernel_stage_mapping.{stage_name}")
    if not bounds or not failures or not requirements:  # pragma: no cover - helpers reject empties
        _fail(location, "profile bounds, failures, and evidence requirements are mandatory")
    return value


def validate_profile_registry(
    registry: object,
    *,
    kernel_contract: Mapping[str, Any],
    repository_root: str | Path,
    kernel_contract_path: str | Path = DEFAULT_KERNEL_PATH,
    verify_sources: bool = True,
) -> dict[str, Any]:
    """Validate registry bindings, referenced profile files, and profile schemas."""

    root = _root(repository_root)
    parent_root = _find_parent_repository_root(root)
    kernel_path = _parameter_file(root, kernel_contract_path, "kernel_contract_path")
    parsed_kernel = _load_json(kernel_path, "kernel_contract_path")
    if parsed_kernel != dict(kernel_contract):
        _fail("kernel_contract_path", "file content differs from the supplied kernel object")
    validate_kernel_contract(
        parsed_kernel,
        repository_root=root,
        parent_repository_root=parent_root,
        verify_sources=verify_sources,
    )
    kernel_file_digest = sha256_file(kernel_path)

    value = _object(
        registry,
        "profile_registry",
        required=("format", "kernel", "automatic_promotion", "profiles"),
    )
    if value["format"] != PROFILE_REGISTRY_FORMAT:
        _fail("profile_registry.format", f"must be {PROFILE_REGISTRY_FORMAT!r}")
    if _boolean(value["automatic_promotion"], "profile_registry.automatic_promotion"):
        _fail("profile_registry.automatic_promotion", "must be false")
    binding = _object(
        value["kernel"],
        "profile_registry.kernel",
        required=("kernel_id", "path", "sha256"),
    )
    if binding["kernel_id"] != KERNEL_ID:
        _fail("profile_registry.kernel.kernel_id", f"must be {KERNEL_ID!r}")
    declared_kernel_path = _contained_file(root, binding["path"], "profile_registry.kernel.path")
    if declared_kernel_path != kernel_path:
        _fail("profile_registry.kernel.path", "does not identify the supplied kernel path")
    if _digest(binding["sha256"], "profile_registry.kernel.sha256") != kernel_file_digest:
        _fail("profile_registry.kernel.sha256", "does not match the kernel file")

    profile_entries = _list(value["profiles"], "profile_registry.profiles")
    profiles: dict[str, dict[str, Any]] = {}
    profile_digests: dict[str, str] = {}
    paths: set[Path] = set()
    for index, value_entry in enumerate(profile_entries):
        location = f"profile_registry.profiles[{index}]"
        entry = _object(
            value_entry,
            location,
            required=("profile_id", "path", "sha256", "status", "selectable"),
        )
        profile_id = _identifier(entry["profile_id"], location + ".profile_id")
        if profile_id in profiles:
            _fail(location + ".profile_id", "is duplicated")
        status = _enum(entry["status"], location + ".status", {"ACTIVE", "PROVISIONAL"})
        selectable = _boolean(entry["selectable"], location + ".selectable")
        if selectable != (status == "ACTIVE"):
            _fail(location, "only ACTIVE profiles may be selectable")
        profile_path = _contained_file(root, entry["path"], location + ".path")
        if profile_path in paths:
            _fail(location + ".path", "is reused by another profile")
        paths.add(profile_path)
        profile_digest = sha256_file(profile_path)
        if _digest(entry["sha256"], location + ".sha256") != profile_digest:
            _fail(location + ".sha256", "does not match the referenced profile file")
        profile = _load_json(profile_path, location + ".path")
        _validate_profile_document(
            profile,
            profile_id=profile_id,
            status=status,
            kernel=parsed_kernel,
            kernel_sha256=kernel_file_digest,
            repository_root=root,
            parent_repository_root=parent_root,
            verify_sources=verify_sources,
        )
        profiles[profile_id] = profile
        profile_digests[profile_id] = profile_digest

    if set(profiles) != REQUIRED_PROFILE_IDS:
        _fail(
            "profile_registry.profiles",
            "must contain exactly the bounded foundational profile set "
            f"{sorted(REQUIRED_PROFILE_IDS)!r}; got {sorted(profiles)!r}",
        )
    return {
        "ok": True,
        "kernel_id": KERNEL_ID,
        "kernel_sha256": kernel_file_digest,
        "profiles": profiles,
        "profile_sha256s": profile_digests,
        "selectable_profiles": sorted(
            entry["profile_id"] for entry in profile_entries if entry["selectable"]
        ),
    }


def load_contract_bundle(
    repository_root: str | Path,
    *,
    kernel_contract_path: str | Path = DEFAULT_KERNEL_PATH,
    profile_registry_path: str | Path = DEFAULT_PROFILE_REGISTRY_PATH,
    verify_sources: bool = True,
) -> ContractBundle:
    """Load and fully verify the committed kernel and profile registry."""

    root = _root(repository_root)
    kernel_path = _parameter_file(root, kernel_contract_path, "kernel_contract_path")
    registry_path = _parameter_file(root, profile_registry_path, "profile_registry_path")
    kernel = _load_json(kernel_path, "kernel_contract_path")
    validate_kernel_contract(kernel, repository_root=root, verify_sources=verify_sources)
    registry = _load_json(registry_path, "profile_registry_path")
    result = validate_profile_registry(
        registry,
        kernel_contract=kernel,
        repository_root=root,
        kernel_contract_path=kernel_path,
        verify_sources=verify_sources,
    )
    return ContractBundle(
        repository_root=root,
        kernel_path=kernel_path,
        registry_path=registry_path,
        kernel=kernel,
        registry=registry,
        profiles=result["profiles"],
        kernel_sha256=result["kernel_sha256"],
        profile_sha256s=result["profile_sha256s"],
    )


def _evidence_records(
    value: object, *, evidence_root: Path, location: str, verify_files: bool
) -> list[dict[str, Any]]:
    records = _verify_file_records(
        value,
        root=evidence_root,
        location=location,
        role_values=EVIDENCE_KINDS,
        verify_files=verify_files,
    )
    for index, record in enumerate(records):
        if "title" not in record:
            _fail(f"{location}[{index}]", "evidence records require a title naming the bounded claim")
    return records


def _profile_selections(
    value: object,
    *,
    bundle: ContractBundle,
    location: str,
) -> list[dict[str, Any]]:
    selections = _list(value, location, nonempty=False)
    registry_entries = {
        entry["profile_id"]: entry for entry in bundle.registry["profiles"]
    }
    checked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value_selection in enumerate(selections):
        row_location = f"{location}[{index}]"
        selection = _object(
            value_selection, row_location, required=("profile_id", "sha256")
        )
        profile_id = _identifier(selection["profile_id"], row_location + ".profile_id")
        if profile_id in seen:
            _fail(row_location + ".profile_id", "is duplicated")
        seen.add(profile_id)
        entry = registry_entries.get(profile_id)
        if entry is None:
            _fail(row_location + ".profile_id", "is not in the verified registry")
        if entry["status"] != "ACTIVE" or entry["selectable"] is not True:
            _fail(row_location + ".profile_id", "is provisional and cannot be selected")
        if _digest(selection["sha256"], row_location + ".sha256") != bundle.profile_sha256s[profile_id]:
            _fail(row_location + ".sha256", "does not bind the selected profile file")
        checked.append(selection)
    return checked


def _kernel_binding(value: object, *, bundle: ContractBundle, location: str) -> dict[str, Any]:
    binding = _object(value, location, required=("kernel_id", "path", "sha256"))
    if binding["kernel_id"] != KERNEL_ID:
        _fail(location + ".kernel_id", f"must be {KERNEL_ID!r}")
    path = _contained_file(bundle.repository_root, binding["path"], location + ".path")
    if path != bundle.kernel_path:
        _fail(location + ".path", "does not identify the verified kernel")
    if _digest(binding["sha256"], location + ".sha256") != bundle.kernel_sha256:
        _fail(location + ".sha256", "does not bind the verified kernel file")
    return binding


def _validate_application_or_proposal_mappings(
    value: object,
    *,
    bundle: ContractBundle,
    resource_bounds: Sequence[Mapping[str, Any]],
    failure_modes: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    location: str,
    dispositions: Iterable[str],
) -> dict[str, list[dict[str, Any]]]:
    mappings = _mapping_shapes(value, location, dispositions=dispositions)
    kernel_primitives = set(bundle.kernel["mappings"])
    _check_mapping_references(
        mappings,
        location=location,
        primitive_ids=kernel_primitives,
        bound_ids={row["id"] for row in resource_bounds},
        failure_ids={row["id"] for row in failure_modes},
        evidence_ids={row["id"] for row in evidence},
    )
    return mappings


def validate_application_manifest(
    manifest: object,
    *,
    repository_root: str | Path,
    evidence_root: str | Path | None = None,
    kernel_contract_path: str | Path = DEFAULT_KERNEL_PATH,
    profile_registry_path: str | Path = DEFAULT_PROFILE_REGISTRY_PATH,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Validate an evidence-bound application manifest against the kernel bundle."""

    bundle = load_contract_bundle(
        repository_root,
        kernel_contract_path=kernel_contract_path,
        profile_registry_path=profile_registry_path,
        verify_sources=verify_files,
    )
    value = _object(
        manifest,
        "application",
        required=(
            "format",
            "application_id",
            "application_version",
            "kernel",
            "profiles",
            "profile_selection_rationale",
            "mappings",
            "resource_bounds",
            "failure_modes",
            "evidence",
            "self_reference",
        ),
    )
    if value["format"] != APPLICATION_FORMAT:
        _fail("application.format", f"must be {APPLICATION_FORMAT!r}")
    _identifier(value["application_id"], "application.application_id")
    _string(value["application_version"], "application.application_version")
    _kernel_binding(value["kernel"], bundle=bundle, location="application.kernel")
    profiles = _profile_selections(value["profiles"], bundle=bundle, location="application.profiles")
    rationale = _string(value["profile_selection_rationale"], "application.profile_selection_rationale")
    if not profiles and "kernel-only" not in rationale.lower():
        _fail(
            "application.profile_selection_rationale",
            "must explicitly say 'kernel-only' when no optional profile is selected",
        )
    evidence_base = _root(evidence_root if evidence_root is not None else repository_root)
    evidence = _evidence_records(
        value["evidence"],
        evidence_root=evidence_base,
        location="application.evidence",
        verify_files=verify_files,
    )
    bounds = _resource_bounds(value["resource_bounds"], "application.resource_bounds")
    failures = _failure_modes(value["failure_modes"], "application.failure_modes")
    mappings = _validate_application_or_proposal_mappings(
        value["mappings"],
        bundle=bundle,
        resource_bounds=bounds,
        failure_modes=failures,
        evidence=evidence,
        location="application.mappings",
        dispositions={"IMPLEMENTED", "BYPASS", "NOT_APPLICABLE"},
    )
    self_reference = _self_reference(value["self_reference"], "application.self_reference")
    return {
        "ok": True,
        "application_id": value["application_id"],
        "kernel_sha256": bundle.kernel_sha256,
        "selected_profiles": [row["profile_id"] for row in profiles],
        "mapping_count": sum(len(rows) for rows in mappings.values()),
        "evidence_count": len(evidence),
        "self_reference_enabled": self_reference["enabled"],
    }


def create_application_manifest(
    application_id: str,
    application_version: str,
    *,
    repository_root: str | Path,
    mappings: Mapping[str, Any],
    resource_bounds: Sequence[Mapping[str, Any]],
    failure_modes: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    selected_profiles: Sequence[str] = (),
    profile_selection_rationale: str,
    self_reference: Mapping[str, Any] | None = None,
    evidence_root: str | Path | None = None,
    kernel_contract_path: str | Path = DEFAULT_KERNEL_PATH,
    profile_registry_path: str | Path = DEFAULT_PROFILE_REGISTRY_PATH,
) -> dict[str, Any]:
    """Create and immediately validate an application manifest."""

    bundle = load_contract_bundle(
        repository_root,
        kernel_contract_path=kernel_contract_path,
        profile_registry_path=profile_registry_path,
    )
    if len(set(selected_profiles)) != len(selected_profiles):
        raise SubstrateContractError("selected_profiles contains duplicates")
    unknown = sorted(set(selected_profiles) - set(bundle.profiles))
    if unknown:
        raise SubstrateContractError(f"selected_profiles contains unknown IDs {unknown!r}")
    profile_rows = [
        {"profile_id": profile_id, "sha256": bundle.profile_sha256s[profile_id]}
        for profile_id in selected_profiles
    ]
    if self_reference is None:
        self_reference = {
            "enabled": False,
            "bounded_generations": 0,
            "may_propose_extensions": False,
            "may_promote_extensions": False,
            "proposal_disposition": "QUARANTINED",
        }
    manifest = {
        "format": APPLICATION_FORMAT,
        "application_id": application_id,
        "application_version": application_version,
        "kernel": {
            "kernel_id": KERNEL_ID,
            "path": _repo_reference(bundle.repository_root, bundle.kernel_path),
            "sha256": bundle.kernel_sha256,
        },
        "profiles": profile_rows,
        "profile_selection_rationale": profile_selection_rationale,
        "mappings": dict(mappings),
        "resource_bounds": list(resource_bounds),
        "failure_modes": list(failure_modes),
        "evidence": list(evidence),
        "self_reference": dict(self_reference),
    }
    validate_application_manifest(
        manifest,
        repository_root=bundle.repository_root,
        evidence_root=evidence_root,
        kernel_contract_path=bundle.kernel_path,
        profile_registry_path=bundle.registry_path,
    )
    return manifest


def validate_extension_proposal(
    proposal: object,
    *,
    repository_root: str | Path,
    evidence_root: str | Path | None = None,
    kernel_contract_path: str | Path = DEFAULT_KERNEL_PATH,
    profile_registry_path: str | Path = DEFAULT_PROFILE_REGISTRY_PATH,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Validate a quarantined proposal; this function cannot promote it."""

    bundle = load_contract_bundle(
        repository_root,
        kernel_contract_path=kernel_contract_path,
        profile_registry_path=profile_registry_path,
        verify_sources=verify_files,
    )
    value = _object(
        proposal,
        "extension",
        required=(
            "format",
            "proposal_id",
            "proposal_version",
            "status",
            "kernel",
            "target_profiles",
            "origin",
            "admission",
            "mappings",
            "resource_bounds",
            "failure_modes",
            "evidence",
            "self_reference",
            "promotion",
        ),
    )
    if value["format"] != EXTENSION_PROPOSAL_FORMAT:
        _fail("extension.format", f"must be {EXTENSION_PROPOSAL_FORMAT!r}")
    _identifier(value["proposal_id"], "extension.proposal_id")
    _string(value["proposal_version"], "extension.proposal_version")
    if value["status"] != "QUARANTINED":
        _fail("extension.status", "must remain 'QUARANTINED'")
    _kernel_binding(value["kernel"], bundle=bundle, location="extension.kernel")
    profiles = _profile_selections(
        value["target_profiles"], bundle=bundle, location="extension.target_profiles"
    )

    origin = _object(
        value["origin"],
        "extension.origin",
        required=("kind", "description", "source_refs"),
    )
    origin_kind = _enum(
        origin["kind"],
        "extension.origin.kind",
        {"HUMAN", "LEGACY_AUDIT", "SELF_REFERENCE"},
    )
    _string(origin["description"], "extension.origin.description")

    admission = _object(
        value["admission"],
        "extension.admission",
        required=("typed_substrate_forms", "canonical_chain_effect", "justification"),
    )
    forms = _string_list(
        admission["typed_substrate_forms"],
        "extension.admission.typed_substrate_forms",
    )
    unknown_forms = sorted(set(forms) - ADMISSION_FORMS)
    if unknown_forms:
        _fail("extension.admission.typed_substrate_forms", f"unknown forms {unknown_forms!r}")
    _enum(
        admission["canonical_chain_effect"],
        "extension.admission.canonical_chain_effect",
        {"UNCHANGED", "PROPOSED_CHANGE"},
    )
    _string(admission["justification"], "extension.admission.justification")

    evidence_base = _root(evidence_root if evidence_root is not None else repository_root)
    evidence = _evidence_records(
        value["evidence"],
        evidence_root=evidence_base,
        location="extension.evidence",
        verify_files=verify_files,
    )
    evidence_ids = {record["id"] for record in evidence}
    origin_refs = _string_list(
        origin["source_refs"], "extension.origin.source_refs", identifiers=True
    )
    unknown_origin = sorted(set(origin_refs) - evidence_ids)
    if unknown_origin:
        _fail("extension.origin.source_refs", f"unknown evidence references {unknown_origin!r}")

    bounds = _resource_bounds(value["resource_bounds"], "extension.resource_bounds")
    failures = _failure_modes(value["failure_modes"], "extension.failure_modes")
    mappings = _validate_application_or_proposal_mappings(
        value["mappings"],
        bundle=bundle,
        resource_bounds=bounds,
        failure_modes=failures,
        evidence=evidence,
        location="extension.mappings",
        dispositions={"PROPOSED", "BYPASS", "NOT_APPLICABLE"},
    )
    self_reference = _self_reference(value["self_reference"], "extension.self_reference")
    if origin_kind == "SELF_REFERENCE" and not self_reference["enabled"]:
        _fail("extension.self_reference", "a self-referential origin must declare bounded self-reference")

    promotion = _object(
        value["promotion"],
        "extension.promotion",
        required=("automatic", "promoted", "review_disposition", "reviewer"),
    )
    if _boolean(promotion["automatic"], "extension.promotion.automatic"):
        _fail("extension.promotion.automatic", "automatic promotion is forbidden")
    if _boolean(promotion["promoted"], "extension.promotion.promoted"):
        _fail("extension.promotion.promoted", "a quarantined proposal cannot be promoted")
    _enum(
        promotion["review_disposition"],
        "extension.promotion.review_disposition",
        {"PENDING", "REJECTED"},
    )
    if promotion["reviewer"] is not None:
        _fail("extension.promotion.reviewer", "must be null while the proposal is quarantined")
    return {
        "ok": True,
        "proposal_id": value["proposal_id"],
        "status": "QUARANTINED",
        "target_profiles": [row["profile_id"] for row in profiles],
        "mapping_count": sum(len(rows) for rows in mappings.values()),
        "evidence_count": len(evidence),
        "automatic_promotion": False,
        "promoted": False,
    }


def create_extension_proposal(
    proposal_id: str,
    proposal_version: str,
    *,
    repository_root: str | Path,
    origin_kind: str,
    origin_description: str,
    origin_source_refs: Sequence[str],
    admission_forms: Sequence[str],
    canonical_chain_effect: str,
    admission_justification: str,
    mappings: Mapping[str, Any],
    resource_bounds: Sequence[Mapping[str, Any]],
    failure_modes: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    target_profiles: Sequence[str] = (),
    self_reference: Mapping[str, Any] | None = None,
    evidence_root: str | Path | None = None,
    kernel_contract_path: str | Path = DEFAULT_KERNEL_PATH,
    profile_registry_path: str | Path = DEFAULT_PROFILE_REGISTRY_PATH,
) -> dict[str, Any]:
    """Create a proposal in quarantine and prove that it has no promotion path."""

    bundle = load_contract_bundle(
        repository_root,
        kernel_contract_path=kernel_contract_path,
        profile_registry_path=profile_registry_path,
    )
    if len(set(target_profiles)) != len(target_profiles):
        raise SubstrateContractError("target_profiles contains duplicates")
    unknown_profiles = sorted(set(target_profiles) - set(bundle.profiles))
    if unknown_profiles:
        raise SubstrateContractError(
            f"target_profiles contains unknown IDs {unknown_profiles!r}"
        )
    if self_reference is None:
        self_reference = {
            "enabled": False,
            "bounded_generations": 0,
            "may_propose_extensions": False,
            "may_promote_extensions": False,
            "proposal_disposition": "QUARANTINED",
        }
    proposal = {
        "format": EXTENSION_PROPOSAL_FORMAT,
        "proposal_id": proposal_id,
        "proposal_version": proposal_version,
        "status": "QUARANTINED",
        "kernel": {
            "kernel_id": KERNEL_ID,
            "path": _repo_reference(bundle.repository_root, bundle.kernel_path),
            "sha256": bundle.kernel_sha256,
        },
        "target_profiles": [
            {"profile_id": profile_id, "sha256": bundle.profile_sha256s[profile_id]}
            for profile_id in target_profiles
        ],
        "origin": {
            "kind": origin_kind,
            "description": origin_description,
            "source_refs": list(origin_source_refs),
        },
        "admission": {
            "typed_substrate_forms": list(admission_forms),
            "canonical_chain_effect": canonical_chain_effect,
            "justification": admission_justification,
        },
        "mappings": dict(mappings),
        "resource_bounds": list(resource_bounds),
        "failure_modes": list(failure_modes),
        "evidence": list(evidence),
        "self_reference": dict(self_reference),
        "promotion": {
            "automatic": False,
            "promoted": False,
            "review_disposition": "PENDING",
            "reviewer": None,
        },
    }
    validate_extension_proposal(
        proposal,
        repository_root=bundle.repository_root,
        evidence_root=evidence_root,
        kernel_contract_path=bundle.kernel_path,
        profile_registry_path=bundle.registry_path,
    )
    return proposal


__all__ = [
    "APPLICATION_FORMAT",
    "CANONICAL_TRANSITION_CHAIN",
    "COMPOSED_EXECUTION",
    "ContractBundle",
    "DEFAULT_KERNEL_PATH",
    "DEFAULT_PROFILE_REGISTRY_PATH",
    "EXTENSION_PROPOSAL_FORMAT",
    "KERNEL_FORMAT",
    "KERNEL_ID",
    "KERNEL_MAPPING_REQUIREMENTS",
    "MAPPING_CATEGORIES",
    "PROFILE_FORMAT",
    "PROFILE_REGISTRY_FORMAT",
    "REQUIRED_PROFILE_IDS",
    "SubstrateContractError",
    "canonical_json_bytes",
    "canonical_sha256",
    "create_application_manifest",
    "create_extension_proposal",
    "load_contract_bundle",
    "sha256_file",
    "validate_application_manifest",
    "validate_extension_proposal",
    "validate_kernel_contract",
    "validate_profile_registry",
]
