"""Typed, content-addressed definitions for the NHDF/UGTS substrate.

This module describes definitions and their referential ordering.  It does not
execute arbitrary code and it does not claim that a next-generation feedback
edge converges to a fixed point.  Feedback is deliberately kept outside the
same-generation dependency DAG: it records only the source-grounded closure
``observable[n] -> input[n + 1]``.

Only Python's standard library is used so the graph can be inspected without
loading the tensor runtime.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Iterable, Mapping, Sequence


_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9:._/-]*$")
_SYMBOL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class SubstrateGraphError(ValueError):
    """Base class for malformed graph records."""


class ContentHashError(SubstrateGraphError):
    """A supplied content digest does not identify the supplied content."""


class ReferenceResolutionError(SubstrateGraphError):
    """A definition, instance, pipeline, or feedback reference is unresolved."""


class DependencyCycleError(SubstrateGraphError):
    """The same-generation definition graph contains a directed cycle."""


class PhaseOrderError(SubstrateGraphError):
    """A dependency or explicit pipeline violates evaluation-phase order."""


class SymbolFirewallError(SubstrateGraphError):
    """Two substrate concepts were assigned an ambiguous or reserved symbol."""


def _normal_text(value: str, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise SubstrateGraphError(f"{name} must be a string")
    normalized = unicodedata.normalize("NFC", value)
    if not allow_empty and not normalized.strip():
        raise SubstrateGraphError(f"{name} must be non-empty")
    return normalized


def _identifier(value: str, name: str) -> str:
    normalized = _normal_text(value, name)
    if _IDENTIFIER.fullmatch(normalized) is None:
        raise SubstrateGraphError(
            f"{name} must start with a letter and contain only identifier-safe characters"
        )
    return normalized


def _json_ready(value: Any, path: str = "$") -> Any:
    """Return the deterministic JSON representation used for all hashes."""

    if isinstance(value, Enum):
        return _json_ready(value.value, path)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_ready(getattr(value, item.name), f"{path}.{item.name}")
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SubstrateGraphError(f"{path} contains a non-string mapping key")
            canonical_key = unicodedata.normalize("NFC", key)
            if canonical_key in normalized:
                raise SubstrateGraphError(
                    f"{path} contains duplicate keys after Unicode normalization"
                )
            normalized[canonical_key] = _json_ready(item, f"{path}.{canonical_key}")
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SubstrateGraphError(f"{path} contains a non-finite float")
        # JSON distinguishes -0.0 even though the substrate does not.
        return 0.0 if value == 0.0 else value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    raise SubstrateGraphError(
        f"{path} contains non-JSON value of type {type(value).__name__}"
    )


def canonical_json(value: Any) -> bytes:
    """Serialize *value* to the UTF-8 canonical form used by this module.

    Mapping keys are normalized and sorted, insignificant whitespace is
    removed, non-finite numbers and unordered containers are rejected, and
    negative zero is normalized.  The result is suitable for hashing and
    replay comparisons; it is not presented as an implementation of an
    external canonical-JSON standard.
    """

    return json.dumps(
        _json_ready(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    """Return a tagged SHA-256 digest of :func:`canonical_json`."""

    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def _freeze_json(value: Any, path: str) -> Any:
    """Detach caller-owned containers and make a canonical value immutable."""

    ready = _json_ready(value, path)
    if isinstance(ready, dict):
        return MappingProxyType(
            {key: _freeze_json(item, f"{path}.{key}") for key, item in ready.items()}
        )
    if isinstance(ready, list):
        return tuple(
            _freeze_json(item, f"{path}[{index}]") for index, item in enumerate(ready)
        )
    return ready


def _string_tuple(
    values: Iterable[str], name: str, *, sort_values: bool = False
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise SubstrateGraphError(f"{name} must be an iterable of strings")
    normalized = tuple(_normal_text(value, name) for value in values)
    if len(set(normalized)) != len(normalized):
        raise SubstrateGraphError(f"{name} must not contain duplicates")
    return tuple(sorted(normalized)) if sort_values else normalized


@dataclass(frozen=True)
class DefinitionNode:
    """One typed operator or relation definition.

    ``dependencies`` are definition IDs and describe one generation only.
    The digest covers all semantic fields but never covers itself.  Supplying a
    digest is supported for loading manifests; it must match the recomputed
    digest.  Omitting it computes the authoritative value.
    """

    id: str
    kind: str
    domain: str
    codomain: str
    dependencies: tuple[str, ...] = ()
    evaluation_phase: int = 0
    parameters: Mapping[str, Any] = field(default_factory=dict)
    equation: str | None = None
    units: Mapping[str, str] | str = field(default_factory=dict)
    bounds: Mapping[str, Any] = field(default_factory=dict)
    failures: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _identifier(self.id, "definition id"))
        object.__setattr__(self, "kind", _normal_text(self.kind, "definition kind"))
        object.__setattr__(self, "domain", _normal_text(self.domain, "definition domain"))
        object.__setattr__(self, "codomain", _normal_text(self.codomain, "definition codomain"))

        dependencies = _string_tuple(self.dependencies, "dependencies", sort_values=True)
        dependencies = tuple(_identifier(item, "dependency") for item in dependencies)
        if self.id in dependencies:
            raise DependencyCycleError(f"definition {self.id!r} depends on itself")
        object.__setattr__(self, "dependencies", dependencies)

        if (
            isinstance(self.evaluation_phase, bool)
            or not isinstance(self.evaluation_phase, int)
            or self.evaluation_phase < 0
        ):
            raise SubstrateGraphError("evaluation_phase must be a non-negative integer")

        object.__setattr__(
            self, "parameters", _freeze_json(self.parameters, "$.parameters")
        )
        if self.equation is not None:
            object.__setattr__(
                self,
                "equation",
                _normal_text(self.equation, "equation", allow_empty=False),
            )
        if isinstance(self.units, str):
            object.__setattr__(self, "units", _normal_text(self.units, "units"))
        else:
            frozen_units = _freeze_json(self.units, "$.units")
            if not isinstance(frozen_units, Mapping):
                raise SubstrateGraphError("units must be a string or mapping")
            for key, value in frozen_units.items():
                if not isinstance(value, str) or not value.strip():
                    raise SubstrateGraphError(f"unit for {key!r} must be a non-empty string")
            object.__setattr__(self, "units", frozen_units)
        object.__setattr__(self, "bounds", _freeze_json(self.bounds, "$.bounds"))
        object.__setattr__(
            self,
            "failures",
            _string_tuple(self.failures, "failures", sort_values=True),
        )
        object.__setattr__(
            self, "provenance", _freeze_json(self.provenance, "$.provenance")
        )

        expected = canonical_hash(self.semantic_record())
        if self.content_hash:
            supplied = self.content_hash.lower()
            if _SHA256.fullmatch(supplied) is None:
                raise ContentHashError("content_hash must have form sha256:<64 lowercase hex>")
            if supplied != expected:
                raise ContentHashError(
                    f"content_hash for {self.id!r} does not match its semantic fields"
                )
        object.__setattr__(self, "content_hash", expected)

    def semantic_record(self) -> Mapping[str, Any]:
        """Return the hash envelope, excluding ``content_hash`` itself."""

        return {
            "record_type": "nhdf.definition-node.v1",
            "id": self.id,
            "kind": self.kind,
            "domain": self.domain,
            "codomain": self.codomain,
            "dependencies": self.dependencies,
            "evaluation_phase": self.evaluation_phase,
            "parameters": self.parameters,
            "equation": self.equation,
            "units": self.units,
            "bounds": self.bounds,
            "failures": self.failures,
            "provenance": self.provenance,
        }

    def verify_content_hash(self) -> bool:
        """Recompute the digest; useful when inspecting deserialized records."""

        return self.content_hash == canonical_hash(self.semantic_record())

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "DefinitionNode":
        """Load and verify a manifest-style definition record."""

        values = dict(record)
        record_type = values.pop("record_type", "nhdf.definition-node.v1")
        if record_type != "nhdf.definition-node.v1":
            raise SubstrateGraphError(f"unsupported definition record_type {record_type!r}")
        try:
            return cls(**values)
        except TypeError as error:
            raise SubstrateGraphError(f"invalid definition record: {error}") from error


@dataclass(frozen=True)
class DefinitionInstance:
    """Literal/state bindings that refer to a reusable definition by ID."""

    id: str
    definition_ref: str
    literal: Mapping[str, Any] = field(default_factory=dict)
    state: Mapping[str, Any] = field(default_factory=dict)
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _identifier(self.id, "instance id"))
        object.__setattr__(
            self, "definition_ref", _identifier(self.definition_ref, "definition_ref")
        )
        object.__setattr__(self, "literal", _freeze_json(self.literal, "$.literal"))
        object.__setattr__(self, "state", _freeze_json(self.state, "$.state"))
        object.__setattr__(self, "content_hash", canonical_hash(self.semantic_record()))

    def semantic_record(self) -> Mapping[str, Any]:
        return {
            "record_type": "nhdf.definition-instance.v1",
            "id": self.id,
            "definition_ref": self.definition_ref,
            "literal": self.literal,
            "state": self.state,
        }


@dataclass(frozen=True)
class Pipeline:
    """An explicit, user-inspectable order of definition or instance refs."""

    id: str
    steps: tuple[str, ...]
    description: str = ""
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _identifier(self.id, "pipeline id"))
        steps = _string_tuple(self.steps, "pipeline steps")
        if not steps:
            raise SubstrateGraphError("a pipeline must contain at least one step")
        object.__setattr__(
            self, "steps", tuple(_identifier(step, "pipeline step") for step in steps)
        )
        object.__setattr__(
            self, "description", _normal_text(self.description, "description", allow_empty=True)
        )
        object.__setattr__(self, "content_hash", canonical_hash(self.semantic_record()))

    def semantic_record(self) -> Mapping[str, Any]:
        return {
            "record_type": "nhdf.pipeline.v1",
            "id": self.id,
            "description": self.description,
            "steps": self.steps,
        }


@dataclass(frozen=True)
class FeedbackEdge:
    """A provenance-bearing edge from generation ``n`` to exactly ``n + 1``.

    Such an edge expresses referential closure only.  It is intentionally
    excluded from same-generation topological resolution and carries no claim
    of existence, uniqueness, stability, or convergence of a fixed point.
    """

    id: str
    source_ref: str
    target_ref: str
    source_port: str = "observable"
    target_port: str = "residual"
    source_generation: int = 0
    target_generation: int = 1
    provenance: Mapping[str, Any] = field(default_factory=dict)
    content_hash: str = field(init=False)

    semantics: ClassVar[str] = "referential-next-generation"
    fixed_point_claim: ClassVar[bool] = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _identifier(self.id, "feedback id"))
        object.__setattr__(self, "source_ref", _identifier(self.source_ref, "source_ref"))
        object.__setattr__(self, "target_ref", _identifier(self.target_ref, "target_ref"))
        object.__setattr__(self, "source_port", _identifier(self.source_port, "source_port"))
        object.__setattr__(self, "target_port", _identifier(self.target_port, "target_port"))
        for name in ("source_generation", "target_generation"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SubstrateGraphError(f"{name} must be a non-negative integer")
        if self.target_generation != self.source_generation + 1:
            raise DependencyCycleError(
                "feedback must cross exactly one generation; same-generation and "
                "unbounded recurrence edges are not permitted"
            )
        object.__setattr__(
            self, "provenance", _freeze_json(self.provenance, "$.provenance")
        )
        source_refs = self.provenance.get("source_refs")
        if (
            not isinstance(source_refs, tuple)
            or not source_refs
            or any(not isinstance(item, str) or not item.strip() for item in source_refs)
        ):
            raise SubstrateGraphError(
                "feedback provenance must contain a non-empty source_refs sequence"
            )
        object.__setattr__(self, "content_hash", canonical_hash(self.semantic_record()))

    @property
    def generation_delay(self) -> int:
        return self.target_generation - self.source_generation

    def semantic_record(self) -> Mapping[str, Any]:
        return {
            "record_type": "nhdf.feedback-edge.v1",
            "id": self.id,
            "source_ref": self.source_ref,
            "source_port": self.source_port,
            "source_generation": self.source_generation,
            "target_ref": self.target_ref,
            "target_port": self.target_port,
            "target_generation": self.target_generation,
            "semantics": self.semantics,
            "fixed_point_claim": self.fixed_point_claim,
            "provenance": self.provenance,
        }


class SymbolRole(str, Enum):
    """Protected roles that must never collapse onto one notation slot."""

    CONE_SLANT_LENGTH = "cone_slant_length"
    LINEAR_TIME = "linear_time"
    MODULAR_TICK = "modular_tick"
    GOLDEN_RATIO = "golden_ratio"
    PERIODIC_PHASE = "periodic_phase"
    JITTER_LOG_RADIUS = "jitter_log_radius"
    SPATIAL_LOG_RADIUS = "spatial_log_radius"
    PAYLOAD_PARITY_BIT = "payload_parity_bit"
    TOPOLOGY_ORIENTATION_BIT = "topology_orientation_bit"
    JITTER_CONTROL_BIT = "jitter_control_bit"
    BRANCH_PREDICATE_BIT = "branch_predicate_bit"
    COMPARISON_BST = "comparison_bst"
    RADIX_PREFIX_TRIE = "radix_prefix_trie"
    HALF_TURN_BUNDLE_MAP = "half_turn_bundle_map"
    REFLECTIVE_KLEIN_QUOTIENT = "reflective_klein_quotient"
    IMPLICIT_CONE_FIELD = "implicit_cone_field"
    FINITE_CONE_SDF = "finite_cone_sdf"
    CERTIFIED_SWEEP_INTERVAL = "certified_sweep_interval"


DEFAULT_SYMBOL_BINDINGS: Mapping[SymbolRole, str] = MappingProxyType(
    {
        SymbolRole.CONE_SLANT_LENGTH: "T_cone",
        SymbolRole.LINEAR_TIME: "time",
        SymbolRole.MODULAR_TICK: "X",
        SymbolRole.GOLDEN_RATIO: "phi_g",
        SymbolRole.PERIODIC_PHASE: "phase",
        SymbolRole.JITTER_LOG_RADIUS: "rho_jitter",
        SymbolRole.SPATIAL_LOG_RADIUS: "rho_spatial",
        SymbolRole.PAYLOAD_PARITY_BIT: "b_payload",
        SymbolRole.TOPOLOGY_ORIENTATION_BIT: "b_topology",
        SymbolRole.JITTER_CONTROL_BIT: "b_jitter",
        SymbolRole.BRANCH_PREDICATE_BIT: "b_branch",
        SymbolRole.COMPARISON_BST: "bst_order",
        SymbolRole.RADIX_PREFIX_TRIE: "radix_prefix",
        SymbolRole.HALF_TURN_BUNDLE_MAP: "half_turn",
        SymbolRole.REFLECTIVE_KLEIN_QUOTIENT: "klein_reflection",
        SymbolRole.IMPLICIT_CONE_FIELD: "cone_implicit",
        SymbolRole.FINITE_CONE_SDF: "cone_sdf_finite",
        SymbolRole.CERTIFIED_SWEEP_INTERVAL: "sweep_interval",
    }
)


SYMBOL_FIREWALL_GROUPS: Mapping[str, tuple[SymbolRole, ...]] = MappingProxyType(
    {
        "cone/time/tick": (
            SymbolRole.CONE_SLANT_LENGTH,
            SymbolRole.LINEAR_TIME,
            SymbolRole.MODULAR_TICK,
        ),
        "ratio/phase": (SymbolRole.GOLDEN_RATIO, SymbolRole.PERIODIC_PHASE),
        "jitter/spatial radius": (
            SymbolRole.JITTER_LOG_RADIUS,
            SymbolRole.SPATIAL_LOG_RADIUS,
        ),
        "one-bit roles": (
            SymbolRole.PAYLOAD_PARITY_BIT,
            SymbolRole.TOPOLOGY_ORIENTATION_BIT,
            SymbolRole.JITTER_CONTROL_BIT,
            SymbolRole.BRANCH_PREDICATE_BIT,
        ),
        "tree structures": (
            SymbolRole.COMPARISON_BST,
            SymbolRole.RADIX_PREFIX_TRIE,
        ),
        "topology maps": (
            SymbolRole.HALF_TURN_BUNDLE_MAP,
            SymbolRole.REFLECTIVE_KLEIN_QUOTIENT,
        ),
        "cone relations": (
            SymbolRole.IMPLICIT_CONE_FIELD,
            SymbolRole.FINITE_CONE_SDF,
            SymbolRole.CERTIFIED_SWEEP_INTERVAL,
        ),
    }
)


def validate_symbol_firewall(
    bindings: Mapping[SymbolRole | str, str] | None = None,
) -> Mapping[SymbolRole, str]:
    """Validate a complete, non-aliased binding for all protected roles.

    Alternate symbols are permitted, but canonical protected symbols may not
    be reassigned to another role.  Every protected role must be present, and
    no two roles may share a symbol.
    """

    source = DEFAULT_SYMBOL_BINDINGS if bindings is None else bindings
    normalized: dict[SymbolRole, str] = {}
    for raw_role, raw_symbol in source.items():
        try:
            role = raw_role if isinstance(raw_role, SymbolRole) else SymbolRole(raw_role)
        except ValueError as error:
            raise SymbolFirewallError(f"unknown protected symbol role {raw_role!r}") from error
        if role in normalized:
            raise SymbolFirewallError(f"duplicate binding for role {role.value!r}")
        symbol = _normal_text(raw_symbol, f"symbol for {role.value}")
        if _SYMBOL.fullmatch(symbol) is None:
            raise SymbolFirewallError(f"{symbol!r} is not a valid substrate symbol")
        normalized[role] = symbol

    missing = tuple(role.value for role in SymbolRole if role not in normalized)
    if missing:
        raise SymbolFirewallError("missing protected symbol roles: " + ", ".join(missing))

    owner_by_symbol: dict[str, SymbolRole] = {}
    for role, symbol in normalized.items():
        previous = owner_by_symbol.get(symbol)
        if previous is not None:
            raise SymbolFirewallError(
                f"symbol {symbol!r} aliases {previous.value!r} and {role.value!r}"
            )
        owner_by_symbol[symbol] = role

    canonical_owner = {symbol: role for role, symbol in DEFAULT_SYMBOL_BINDINGS.items()}
    for role, symbol in normalized.items():
        reserved_for = canonical_owner.get(symbol)
        if reserved_for is not None and reserved_for is not role:
            raise SymbolFirewallError(
                f"canonical symbol {symbol!r} is reserved for {reserved_for.value!r}, "
                f"not {role.value!r}"
            )

    # Each group is checked independently so future additions cannot silently
    # bypass the distinctions merely because global validation changed.
    for name, roles in SYMBOL_FIREWALL_GROUPS.items():
        symbols = [normalized[role] for role in roles]
        if len(symbols) != len(set(symbols)):
            raise SymbolFirewallError(f"symbol collision in firewall group {name!r}")
    return MappingProxyType(dict(normalized))


@dataclass(frozen=True)
class ResolvedPipelineStep:
    step_ref: str
    definition: DefinitionNode
    instance: DefinitionInstance | None = None


def _duplicates(items: Sequence[Any], attribute: str) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicate: set[str] = set()
    for item in items:
        value = getattr(item, attribute)
        if value in seen:
            duplicate.add(value)
        seen.add(value)
    return tuple(sorted(duplicate))


class SubstrateGraph:
    """Validated definitions, instances, pipelines, and delayed feedback."""

    closure_semantics: ClassVar[str] = "source-grounded-referential-closure"
    fixed_point_engine: ClassVar[bool] = False

    def __init__(
        self,
        definitions: Iterable[DefinitionNode],
        *,
        instances: Iterable[DefinitionInstance] = (),
        pipelines: Iterable[Pipeline] = (),
        feedback_edges: Iterable[FeedbackEdge] = (),
        symbol_bindings: Mapping[SymbolRole | str, str] | None = None,
    ) -> None:
        self.definitions = tuple(definitions)
        self.instances = tuple(instances)
        self.pipelines = tuple(pipelines)
        self.feedback_edges = tuple(feedback_edges)
        self.symbol_bindings = validate_symbol_firewall(symbol_bindings)

        if not self.definitions:
            raise SubstrateGraphError("a substrate graph needs at least one definition")
        for collection, label in (
            (self.definitions, "definition"),
            (self.instances, "instance"),
            (self.pipelines, "pipeline"),
            (self.feedback_edges, "feedback edge"),
        ):
            duplicate = _duplicates(collection, "id")
            if duplicate:
                raise SubstrateGraphError(
                    f"duplicate {label} ids: {', '.join(duplicate)}"
                )

        definition_ids = {item.id for item in self.definitions}
        instance_ids = {item.id for item in self.instances}
        if definition_ids & instance_ids:
            overlap = ", ".join(sorted(definition_ids & instance_ids))
            raise SubstrateGraphError(
                f"definition and instance ids share a reference namespace: {overlap}"
            )

        self._definition_by_id = MappingProxyType(
            {item.id: item for item in self.definitions}
        )
        self._definition_by_hash = MappingProxyType(
            {item.content_hash: item for item in self.definitions}
        )
        if len(self._definition_by_hash) != len(self.definitions):
            raise ContentHashError("two definition records unexpectedly share a content hash")
        self._instance_by_id = MappingProxyType({item.id: item for item in self.instances})
        self._pipeline_by_id = MappingProxyType({item.id: item for item in self.pipelines})

        for instance in self.instances:
            if instance.definition_ref not in self._definition_by_id:
                raise ReferenceResolutionError(
                    f"instance {instance.id!r} refers to unknown definition "
                    f"{instance.definition_ref!r}"
                )

        self._topological_definitions = self._resolve_topology()
        for pipeline in self.pipelines:
            self._validate_pipeline(pipeline)
        for edge in self.feedback_edges:
            self._resolve_step(edge.source_ref)
            self._resolve_step(edge.target_ref)

        self.content_hash = canonical_hash(self.manifest())

    def definition(self, reference: str) -> DefinitionNode:
        """Resolve either a stable definition ID or its tagged content hash."""

        try:
            return self._definition_by_id[reference]
        except KeyError:
            try:
                return self._definition_by_hash[reference]
            except KeyError as error:
                raise ReferenceResolutionError(
                    f"unknown definition reference {reference!r}"
                ) from error

    def instance(self, reference: str) -> DefinitionInstance:
        try:
            return self._instance_by_id[reference]
        except KeyError as error:
            raise ReferenceResolutionError(f"unknown instance {reference!r}") from error

    def pipeline(self, reference: str) -> Pipeline:
        try:
            return self._pipeline_by_id[reference]
        except KeyError as error:
            raise ReferenceResolutionError(f"unknown pipeline {reference!r}") from error

    def topological_definitions(self) -> tuple[DefinitionNode, ...]:
        """Return a deterministic same-generation dependency order."""

        return self._topological_definitions

    def resolve_pipeline(self, reference: str) -> tuple[ResolvedPipelineStep, ...]:
        """Resolve an explicit pipeline without inventing implicit steps."""

        return tuple(self._resolve_step(step) for step in self.pipeline(reference).steps)

    def _resolve_step(self, reference: str) -> ResolvedPipelineStep:
        definition = self._definition_by_id.get(reference)
        if definition is not None:
            return ResolvedPipelineStep(reference, definition)
        instance = self._instance_by_id.get(reference)
        if instance is not None:
            return ResolvedPipelineStep(
                reference,
                self._definition_by_id[instance.definition_ref],
                instance,
            )
        raise ReferenceResolutionError(
            f"step reference {reference!r} is neither a definition nor an instance"
        )

    def _resolve_topology(self) -> tuple[DefinitionNode, ...]:
        by_id = self._definition_by_id
        indegree: dict[str, int] = {node.id: 0 for node in self.definitions}
        dependents: dict[str, list[str]] = {node.id: [] for node in self.definitions}
        phase_violations: list[tuple[DefinitionNode, DefinitionNode]] = []
        for node in self.definitions:
            for dependency in node.dependencies:
                dependency_node = by_id.get(dependency)
                if dependency_node is None:
                    raise ReferenceResolutionError(
                        f"definition {node.id!r} depends on unknown definition {dependency!r}"
                    )
                if dependency_node.evaluation_phase > node.evaluation_phase:
                    phase_violations.append((node, dependency_node))
                indegree[node.id] += 1
                dependents[dependency].append(node.id)

        ready: list[tuple[int, str]] = [
            (by_id[node_id].evaluation_phase, node_id)
            for node_id, degree in indegree.items()
            if degree == 0
        ]
        heapq.heapify(ready)
        ordered: list[DefinitionNode] = []
        while ready:
            _, node_id = heapq.heappop(ready)
            ordered.append(by_id[node_id])
            for dependent in sorted(dependents[node_id]):
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    heapq.heappush(
                        ready, (by_id[dependent].evaluation_phase, dependent)
                    )
        if len(ordered) != len(self.definitions):
            cycle = self._find_cycle(tuple(node_id for node_id, value in indegree.items() if value))
            raise DependencyCycleError(
                "same-generation definition cycle: " + " -> ".join(cycle)
            )
        if phase_violations:
            node, dependency_node = phase_violations[0]
            raise PhaseOrderError(
                f"definition {node.id!r} at phase {node.evaluation_phase} depends on "
                f"later-phase {dependency_node.id!r} at phase "
                f"{dependency_node.evaluation_phase}"
            )
        return tuple(ordered)

    def _find_cycle(self, candidates: tuple[str, ...]) -> tuple[str, ...]:
        candidate_set = set(candidates)
        visited: set[str] = set()
        active: list[str] = []
        active_set: set[str] = set()

        def visit(node_id: str) -> tuple[str, ...] | None:
            visited.add(node_id)
            active.append(node_id)
            active_set.add(node_id)
            for dependency in self._definition_by_id[node_id].dependencies:
                if dependency not in candidate_set:
                    continue
                if dependency in active_set:
                    start = active.index(dependency)
                    return tuple(active[start:] + [dependency])
                if dependency not in visited:
                    found = visit(dependency)
                    if found is not None:
                        return found
            active.pop()
            active_set.remove(node_id)
            return None

        for node_id in sorted(candidates):
            if node_id not in visited:
                found = visit(node_id)
                if found is not None:
                    return found
        return tuple(sorted(candidates))

    def _validate_pipeline(self, pipeline: Pipeline) -> None:
        seen_definitions: set[str] = set()
        prior_phase = -1
        for step_ref in pipeline.steps:
            step = self._resolve_step(step_ref)
            node = step.definition
            missing = tuple(
                dependency
                for dependency in node.dependencies
                if dependency not in seen_definitions
            )
            if missing:
                raise ReferenceResolutionError(
                    f"pipeline {pipeline.id!r} places {step_ref!r} before required "
                    f"dependencies: {', '.join(missing)}"
                )
            if node.evaluation_phase < prior_phase:
                raise PhaseOrderError(
                    f"pipeline {pipeline.id!r} moves backward from phase {prior_phase} "
                    f"to phase {node.evaluation_phase} at {step_ref!r}"
                )
            seen_definitions.add(node.id)
            prior_phase = node.evaluation_phase

    def manifest(self) -> Mapping[str, Any]:
        """Return the deterministic graph envelope used for ``content_hash``."""

        return {
            "record_type": "nhdf.substrate-graph.v1",
            "closure": {
                "semantics": self.closure_semantics,
                "fixed_point_engine": self.fixed_point_engine,
            },
            "symbols": {
                role.value: symbol
                for role, symbol in sorted(
                    self.symbol_bindings.items(), key=lambda item: item[0].value
                )
            },
            "definitions": [
                {**node.semantic_record(), "content_hash": node.content_hash}
                for node in sorted(self.definitions, key=lambda item: item.id)
            ],
            "instances": [
                {**item.semantic_record(), "content_hash": item.content_hash}
                for item in sorted(self.instances, key=lambda entry: entry.id)
            ],
            "pipelines": [
                {**item.semantic_record(), "content_hash": item.content_hash}
                for item in sorted(self.pipelines, key=lambda entry: entry.id)
            ],
            "feedback_edges": [
                {**item.semantic_record(), "content_hash": item.content_hash}
                for item in sorted(self.feedback_edges, key=lambda entry: entry.id)
            ],
        }


__all__ = [
    "ContentHashError",
    "DEFAULT_SYMBOL_BINDINGS",
    "DefinitionInstance",
    "DefinitionNode",
    "DependencyCycleError",
    "FeedbackEdge",
    "PhaseOrderError",
    "Pipeline",
    "ReferenceResolutionError",
    "ResolvedPipelineStep",
    "SYMBOL_FIREWALL_GROUPS",
    "SubstrateGraph",
    "SubstrateGraphError",
    "SymbolFirewallError",
    "SymbolRole",
    "canonical_hash",
    "canonical_json",
    "validate_symbol_firewall",
]
