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
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


KERNEL_FORMAT = "ugtoms-kernel-contract-0.2"
PROFILE_REGISTRY_FORMAT = "ugtoms-profile-registry-0.2"
PROFILE_FORMAT = "ugtoms-profile-0.2"
APPLICATION_FORMAT = "ugtoms-application-manifest-0.2"
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
    "typed_payload_topology_parity_jitter_branch_control_predicates",
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

REQUIRED_SYMBOL_FIREWALL = (
    (
        ("T_cone", "t", "X"),
        "cone slant length, linear time, and modular tick are distinct",
    ),
    (
        ("phi_g", "phase_phi"),
        "golden ratio and phase or hinge angle are distinct",
    ),
    (
        ("rho_jitter", "rho_spatial"),
        "residual log magnitude and spatial log radius are distinct",
    ),
    (
        ("epsilon_jitter", "epsilon_guard"),
        "jitter amplitude must be strictly below a declared guard margin",
    ),
    (
        (
            "payload_parity_bit",
            "topology_parity_bit",
            "jitter_control_bit",
            "branch_control_bit",
        ),
        "the four bit roles cannot be silently aliased",
    ),
    (
        ("comparison_BST", "radix_prefix_trie"),
        "comparison ordering and packed-prefix refinement are different operators",
    ),
    (
        ("half_turn_bundle_map", "reflective_Klein_gluing"),
        "source twist and non-orientable quotient are separate profiles",
    ),
    (
        ("cone_implicit_field", "finite_cone_SDF", "sweep_interval"),
        "an implicit relation, exact finite SDF, and certified sampled sweep are different claim classes",
    ),
    (
        ("circle", "sphere", "apex"),
        "circle is a base, section, or projection; sphere is a support SDF; apex is a local or distributed anchor",
    ),
)

REQUIRED_DEFINITION_GRAPH_INVARIANTS = (
    (
        "same_generation",
        "Merkle content-addressed acyclic typed graph; child content identity binds dependency content hashes",
    ),
    (
        "instance_rule",
        "instances bind definition content hashes; coordinates and semantic IDs are not content identity",
    ),
    (
        "pipeline_rule",
        "pipelines bind ordered step content hashes, generation, domain, codomain, and typed adjacency",
    ),
    (
        "self_reference_rule",
        "feedback binds endpoint content hashes, named typed ports, and enters exactly generation n+1 through an explicit edge",
    ),
    (
        "unproven_claim",
        "an unrestricted fixed point or same-generation authority cycle is not implemented",
    ),
)

REQUIRED_DEFINITION_NODE_FIELDS = (
    "record_type",
    "id",
    "kind",
    "domain",
    "codomain",
    "dependencies",
    "dependency_hashes",
    "evaluation_phase",
    "parameters",
    "equation",
    "units",
    "bounds",
    "failures",
    "provenance",
    "input_ports",
    "output_ports",
    "content_hash",
)

REQUIRED_DEFINITION_INSTANCE_FIELDS = (
    "record_type",
    "id",
    "definition_ref",
    "definition_hash",
    "literal",
    "state",
    "content_hash",
)

REQUIRED_PIPELINE_FIELDS = (
    "record_type",
    "id",
    "description",
    "generation",
    "domain",
    "codomain",
    "steps",
    "step_hashes",
    "content_hash",
)

REQUIRED_FEEDBACK_FIELDS = (
    "record_type",
    "id",
    "source_ref",
    "source_port",
    "source_port_type",
    "source_hash",
    "source_generation",
    "target_ref",
    "target_port",
    "target_port_type",
    "target_hash",
    "target_generation",
    "semantics",
    "fixed_point_claim",
    "provenance",
    "content_hash",
)

REQUIRED_SELF_REFERENCE_MAPPING_RULE = (
    "generated output may propose the next state or an extension but cannot rewrite its own authority"
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


def _canonical_json_value(value: object, location: str = "$") -> object:
    """Normalize one JSON-compatible value for portable content addressing."""

    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SubstrateContractError(f"{location}: non-finite numbers are not valid JSON")
        return 0.0 if value == 0.0 else value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, (list, tuple)):
        return [
            _canonical_json_value(item, f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SubstrateContractError(f"{location}: object keys must be strings")
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized:
                raise SubstrateContractError(
                    f"{location}: duplicate object key after NFC normalization: {normalized_key!r}"
                )
            normalized[normalized_key] = _canonical_json_value(
                item, f"{location}.{normalized_key}"
            )
        return normalized
    raise SubstrateContractError(
        f"{location}: {type(value).__name__} is not a JSON-compatible value"
    )


def _strict_json_loads(text: str, location: str) -> object:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in result:
                raise ValueError(
                    f"duplicate object key after NFC normalization: {normalized_key!r}"
                )
            result[normalized_key] = item
        return result

    def reject_constant(token: str) -> object:
        raise ValueError(f"non-finite numeric token {token!r} is not valid JSON")

    try:
        parsed = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
        return _canonical_json_value(parsed)
    except (json.JSONDecodeError, ValueError, SubstrateContractError) as exc:
        raise SubstrateContractError(f"{location}: is not strict UTF-8 JSON: {exc}") from exc


def canonical_json_bytes(value: object) -> bytes:
    """Return NFC, finite-number, negative-zero-normalized canonical JSON."""

    normalized = _canonical_json_value(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
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
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        _fail(location, f"is not valid UTF-8 JSON: {exc}")
    value = _strict_json_loads(text, location)
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
        optional=("title", "note", "claim_coverage"),
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
    if enabled and generations < 1:
        _fail(location, "enabled next-generation feedback needs a positive generation bound")
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
    firewall_rows: list[tuple[tuple[str, ...], str]] = []
    for index, item in enumerate(firewall):
        location = f"kernel.symbol_firewall[{index}]"
        row = _object(item, location, required=("symbols", "rule"))
        symbols = tuple(_string_list(row["symbols"], location + ".symbols"))
        rule = _string(row["rule"], location + ".rule")
        firewall_rows.append((symbols, rule))
    if tuple(firewall_rows) != REQUIRED_SYMBOL_FIREWALL:
        _fail(
            "kernel.symbol_firewall",
            "must exactly preserve the required typed symbol roles and separation rules",
        )

    graph = _object(
        value["definition_graph"],
        "kernel.definition_graph",
        required=(
            "node_fields",
            "instance_fields",
            "pipeline_fields",
            "feedback_fields",
            "same_generation",
            "instance_rule",
            "pipeline_rule",
            "self_reference_rule",
            "unproven_claim",
        ),
    )
    graph_field_sets = (
        ("node_fields", REQUIRED_DEFINITION_NODE_FIELDS),
        ("instance_fields", REQUIRED_DEFINITION_INSTANCE_FIELDS),
        ("pipeline_fields", REQUIRED_PIPELINE_FIELDS),
        ("feedback_fields", REQUIRED_FEEDBACK_FIELDS),
    )
    for field_name, required_fields in graph_field_sets:
        actual_fields = tuple(
            _string_list(
                graph[field_name],
                f"kernel.definition_graph.{field_name}",
                identifiers=True,
            )
        )
        if actual_fields != required_fields:
            _fail(
                f"kernel.definition_graph.{field_name}",
                f"must exactly bind the hardened v2 fields {list(required_fields)!r}",
            )
    for key, required_value in REQUIRED_DEFINITION_GRAPH_INVARIANTS:
        actual = _string(graph[key], f"kernel.definition_graph.{key}")
        if actual != required_value:
            _fail(
                f"kernel.definition_graph.{key}",
                f"must be {required_value!r}",
            )

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
        rule = _string(mapping["rule"], location + ".rule")
        if category == "self_reference" and rule != REQUIRED_SELF_REFERENCE_MAPPING_RULE:
            _fail(
                location + ".rule",
                "must preserve the proposal-only boundary and prohibit self-authority",
            )

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
    requirements = _evidence_requirements(
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


def _json_pointer(value: object, location: str) -> str:
    pointer = _string(value, location)
    if not pointer.startswith("/"):
        _fail(location, "must be an absolute JSON Pointer beginning with '/'")
    for token in pointer.split("/")[1:]:
        index = 0
        while index < len(token):
            if token[index] == "~":
                if index + 1 >= len(token) or token[index + 1] not in "01":
                    _fail(location, "contains an invalid JSON Pointer escape")
                index += 2
            else:
                index += 1
    return pointer


def _resolve_json_pointer(document: object, pointer: str, location: str) -> object:
    current = document
    for encoded in pointer.split("/")[1:]:
        token = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            if token not in current:
                _fail(location, f"does not resolve; missing object key {token!r}")
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                _fail(location, f"does not resolve to a canonical array index: {token!r}")
            index = int(token)
            if index >= len(current):
                _fail(location, f"array index {index} is out of range")
            current = current[index]
        else:
            _fail(location, f"cannot traverse through {type(current).__name__}")
    return current


def _validate_selected_profile_claim_coverage(
    evidence: Sequence[Mapping[str, Any]],
    *,
    selected_profile_ids: Sequence[str],
    bundle: ContractBundle,
    evidence_root: Path,
    location: str,
) -> int:
    selected = frozenset(selected_profile_ids)
    declared: dict[tuple[str, str], Mapping[str, Any]] = {}
    for profile_id in selected_profile_ids:
        for requirement in bundle.profiles[profile_id]["evidence_requirements"]:
            declared[(profile_id, str(requirement["id"]))] = requirement

    observed: dict[tuple[str, str], str] = {}
    documents: dict[str, Mapping[str, Any]] = {}
    for evidence_index, record in enumerate(evidence):
        coverage_location = f"{location}[{evidence_index}].claim_coverage"
        coverage_rows = _list(record.get("claim_coverage", []), coverage_location, nonempty=False)
        if not coverage_rows:
            continue
        path = _contained_file(evidence_root, record["path"], f"{location}[{evidence_index}].path")
        document = documents.setdefault(str(path), _load_json(path, coverage_location + ".document"))
        for coverage_index, value_row in enumerate(coverage_rows):
            row_location = f"{coverage_location}[{coverage_index}]"
            row = _object(
                value_row,
                row_location,
                required=("profile_id", "requirement_id", "proof_pointer"),
            )
            profile_id = _identifier(row["profile_id"], row_location + ".profile_id")
            requirement_id = _identifier(
                row["requirement_id"], row_location + ".requirement_id"
            )
            if profile_id not in selected:
                _fail(
                    row_location + ".profile_id",
                    f"wrong-profile coverage for unselected profile {profile_id!r}",
                )
            key = (profile_id, requirement_id)
            if key not in declared:
                _fail(
                    row_location + ".requirement_id",
                    f"unknown requirement {requirement_id!r} for selected profile {profile_id!r}",
                )
            if key in observed:
                _fail(
                    row_location,
                    f"duplicate coverage for {profile_id!r}/{requirement_id!r}; first seen at {observed[key]}",
                )
            proof_pointer = _json_pointer(row["proof_pointer"], row_location + ".proof_pointer")
            proof_value = _resolve_json_pointer(
                document, proof_pointer, row_location + ".proof_pointer"
            )
            proof = _object(
                proof_value,
                row_location + ".proof",
                required=(
                    "profile_id",
                    "requirement_id",
                    "passed",
                    "evidence_paths",
                ),
            )
            if _identifier(proof["profile_id"], row_location + ".proof.profile_id") != profile_id:
                _fail(row_location + ".proof.profile_id", "contradicts the coverage profile_id")
            if (
                _identifier(
                    proof["requirement_id"], row_location + ".proof.requirement_id"
                )
                != requirement_id
            ):
                _fail(
                    row_location + ".proof.requirement_id",
                    "contradicts the coverage requirement_id",
                )
            if not _boolean(proof["passed"], row_location + ".proof.passed"):
                _fail(row_location + ".proof.passed", "must be true for declared claim coverage")
            evidence_paths = _string_list(
                proof["evidence_paths"], row_location + ".proof.evidence_paths"
            )
            for path_index, evidence_pointer_value in enumerate(evidence_paths):
                evidence_pointer_location = (
                    f"{row_location}.proof.evidence_paths[{path_index}]"
                )
                evidence_pointer = _json_pointer(
                    evidence_pointer_value, evidence_pointer_location
                )
                _resolve_json_pointer(document, evidence_pointer, evidence_pointer_location)
            observed[key] = row_location

    missing = sorted(set(declared) - set(observed))
    if missing:
        rendered = [f"{profile_id}/{requirement_id}" for profile_id, requirement_id in missing]
        _fail(location, f"missing selected-profile evidence coverage {rendered!r}")
    return len(observed)


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
    profile_requirement_count = _validate_selected_profile_claim_coverage(
        evidence,
        selected_profile_ids=[row["profile_id"] for row in profiles],
        bundle=bundle,
        evidence_root=evidence_base,
        location="application.evidence",
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
        "profile_requirement_count": profile_requirement_count,
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
    if origin_kind == "SELF_REFERENCE" and not (
        self_reference["enabled"] and self_reference["may_propose_extensions"]
    ):
        _fail(
            "extension.self_reference",
            "a self-referential extension origin needs bounded feedback and explicit extension-proposal permission",
        )

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
    "REQUIRED_DEFINITION_GRAPH_INVARIANTS",
    "REQUIRED_PROFILE_IDS",
    "REQUIRED_SELF_REFERENCE_MAPPING_RULE",
    "REQUIRED_SYMBOL_FIREWALL",
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
