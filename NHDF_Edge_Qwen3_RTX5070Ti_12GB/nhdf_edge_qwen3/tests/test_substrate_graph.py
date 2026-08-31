from __future__ import annotations

import json

import pytest

from nhdf_edge.substrate_graph import (
    ContentHashError,
    DEFAULT_SYMBOL_BINDINGS,
    DefinitionInstance,
    DefinitionNode,
    DependencyCycleError,
    FeedbackEdge,
    PhaseOrderError,
    Pipeline,
    ReferenceResolutionError,
    SYMBOL_FIREWALL_GROUPS,
    SubstrateGraph,
    SubstrateGraphError,
    SymbolFirewallError,
    SymbolRole,
    canonical_hash,
    canonical_json,
    validate_symbol_firewall,
)


def _definition(
    id: str,
    *,
    dependencies: tuple[str, ...] = (),
    phase: int = 0,
    kind: str = "typed-operator",
) -> DefinitionNode:
    return DefinitionNode(
        id=id,
        kind=kind,
        domain="TypedInput",
        codomain="TypedOutput",
        dependencies=dependencies,
        evaluation_phase=phase,
        parameters={"bounded": True},
        equation="output = operator(input)",
        units={"input": "declared", "output": "declared"},
        bounds={"maximum_steps": 32},
        failures=("OUT_OF_BOUNDS",),
        provenance={
            "class": "source-derived",
            "source_refs": ("NHDF-v0.1:p10",),
        },
    )


def test_canonical_json_and_definition_hash_are_order_independent_and_detached() -> None:
    parameters = {"z": [3, 2, 1], "a": {"right": 2, "left": 1}}
    first = DefinitionNode(
        id="def:log-polar",
        kind="coordinate-transform",
        domain="Residual2",
        codomain="LogPolarAddress",
        dependencies=("def:clock", "def:input"),
        evaluation_phase=1,
        parameters=parameters,
        equation="rho_jitter = log1p(gamma * magnitude)",
        units={"rho_jitter": "dimensionless"},
        bounds={"gamma": [0.0, 100.0]},
        failures=("NON_FINITE", "OUT_OF_RANGE"),
        provenance={"source_refs": ["NHDF-v0.1:p8"]},
    )
    second = DefinitionNode(
        id="def:log-polar",
        kind="coordinate-transform",
        domain="Residual2",
        codomain="LogPolarAddress",
        dependencies=("def:input", "def:clock"),
        evaluation_phase=1,
        parameters={"a": {"left": 1, "right": 2}, "z": [3, 2, 1]},
        equation="rho_jitter = log1p(gamma * magnitude)",
        units={"rho_jitter": "dimensionless"},
        bounds={"gamma": [0.0, 100.0]},
        failures=("OUT_OF_RANGE", "NON_FINITE"),
        provenance={"source_refs": ["NHDF-v0.1:p8"]},
    )
    assert first.content_hash == second.content_hash
    assert first.content_hash.startswith("sha256:")
    assert first.verify_content_hash()
    assert json.loads(canonical_json({"b": -0.0, "a": 1})) == {"a": 1, "b": 0.0}
    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})

    parameters["z"].append(0)
    assert first.parameters["z"] == (3, 2, 1)
    with pytest.raises(TypeError):
        first.parameters["new"] = True  # type: ignore[index]


def test_supplied_definition_hash_is_verified() -> None:
    node = _definition("def:verified")
    loaded = DefinitionNode.from_record(
        {**node.semantic_record(), "content_hash": node.content_hash}
    )
    assert loaded == node
    with pytest.raises(ContentHashError, match="does not match"):
        DefinitionNode(
            id="def:verified",
            kind="typed-operator",
            domain="TypedInput",
            codomain="TypedOutput",
            content_hash="sha256:" + "0" * 64,
        )


def test_instances_and_explicit_pipeline_resolve_to_typed_definitions() -> None:
    address = _definition("def:address", phase=0, kind="log-polar-address")
    field = _definition(
        "def:field",
        dependencies=("def:address",),
        phase=1,
        kind="implicit-cone-field",
    )
    gate = _definition(
        "def:gate",
        dependencies=("def:field",),
        phase=2,
        kind="support-compatibility-guard",
    )
    instance = DefinitionInstance(
        id="instance:local-cell-7",
        definition_ref="def:field",
        literal={"cell": 7, "T_cone": 2.0},
        state={"generation": 0},
    )
    pipeline = Pipeline(
        id="pipeline:reference",
        description="Explicit in-generation referential path",
        steps=("def:address", "instance:local-cell-7", "def:gate"),
    )
    graph = SubstrateGraph(
        (gate, field, address), instances=(instance,), pipelines=(pipeline,)
    )

    assert tuple(node.id for node in graph.topological_definitions()) == (
        "def:address",
        "def:field",
        "def:gate",
    )
    resolved = graph.resolve_pipeline("pipeline:reference")
    assert tuple(step.definition.id for step in resolved) == (
        "def:address",
        "def:field",
        "def:gate",
    )
    assert resolved[1].instance is instance
    assert graph.definition(field.content_hash) is field
    assert graph.content_hash == canonical_hash(graph.manifest())


def test_unknown_refs_and_pipeline_dependency_inversion_are_rejected() -> None:
    with pytest.raises(ReferenceResolutionError, match="unknown definition"):
        SubstrateGraph(
            (_definition("def:child", dependencies=("def:missing",), phase=1),)
        )

    root = _definition("def:root", phase=0)
    child = _definition("def:child", dependencies=(root.id,), phase=1)
    with pytest.raises(ReferenceResolutionError, match="before required dependencies"):
        SubstrateGraph(
            (root, child),
            pipelines=(Pipeline("pipeline:bad", (child.id, root.id)),),
        )

    with pytest.raises(ReferenceResolutionError, match="unknown definition"):
        SubstrateGraph(
            (root,),
            instances=(DefinitionInstance("instance:bad", "def:missing"),),
        )


def test_topological_resolution_rejects_cycles_and_later_phase_dependencies() -> None:
    left = _definition("def:left", dependencies=("def:right",), phase=0)
    right = _definition("def:right", dependencies=("def:left",), phase=1)
    with pytest.raises(DependencyCycleError, match="def:left|def:right"):
        SubstrateGraph((left, right))

    early = _definition("def:early", dependencies=("def:late",), phase=0)
    late = _definition("def:late", phase=1)
    with pytest.raises(PhaseOrderError, match="later-phase"):
        SubstrateGraph((early, late))


def test_feedback_is_next_generation_only_and_not_a_fixed_point_claim() -> None:
    input_node = _definition("def:input", phase=0)
    observable = _definition(
        "def:observable", dependencies=(input_node.id,), phase=1
    )
    feedback = FeedbackEdge(
        id="feedback:U",
        source_ref=observable.id,
        target_ref=input_node.id,
        source_port="observable",
        target_port="residual",
        source_generation=8,
        target_generation=9,
        provenance={"source_refs": ("NHDF-v0.1:p10", "NHDF-v0.1:p15")},
    )
    graph = SubstrateGraph((observable, input_node), feedback_edges=(feedback,))

    # The reverse edge is legal only because it crosses a generation boundary;
    # it is not injected into the acyclic in-generation resolver.
    assert tuple(node.id for node in graph.topological_definitions()) == (
        input_node.id,
        observable.id,
    )
    assert feedback.generation_delay == 1
    assert feedback.semantics == "referential-next-generation"
    assert feedback.fixed_point_claim is False
    assert graph.fixed_point_engine is False
    assert graph.manifest()["closure"] == {
        "semantics": "source-grounded-referential-closure",
        "fixed_point_engine": False,
    }

    with pytest.raises(DependencyCycleError, match="exactly one generation"):
        FeedbackEdge(
            "feedback:same-generation",
            observable.id,
            input_node.id,
            source_generation=4,
            target_generation=4,
        )
    with pytest.raises(DependencyCycleError, match="exactly one generation"):
        FeedbackEdge(
            "feedback:skips-unboundedly",
            observable.id,
            input_node.id,
            source_generation=4,
            target_generation=6,
        )
    with pytest.raises(SubstrateGraphError, match="source_refs"):
        FeedbackEdge("feedback:ungrounded", observable.id, input_node.id)


def test_symbol_firewall_covers_every_required_distinction() -> None:
    bindings = validate_symbol_firewall()
    assert bindings[SymbolRole.CONE_SLANT_LENGTH] == "T_cone"
    assert bindings[SymbolRole.LINEAR_TIME] == "time"
    assert bindings[SymbolRole.MODULAR_TICK] == "X"
    assert bindings[SymbolRole.GOLDEN_RATIO] == "phi_g"
    assert bindings[SymbolRole.PERIODIC_PHASE] == "phase"
    assert bindings[SymbolRole.JITTER_LOG_RADIUS] != bindings[SymbolRole.SPATIAL_LOG_RADIUS]

    covered_roles = {role for group in SYMBOL_FIREWALL_GROUPS.values() for role in group}
    assert covered_roles == set(SymbolRole)
    for group in SYMBOL_FIREWALL_GROUPS.values():
        assert len({bindings[role] for role in group}) == len(group)

    bit_roles = (
        SymbolRole.PAYLOAD_PARITY_BIT,
        SymbolRole.TOPOLOGY_ORIENTATION_BIT,
        SymbolRole.JITTER_CONTROL_BIT,
        SymbolRole.BRANCH_PREDICATE_BIT,
    )
    assert len({bindings[role] for role in bit_roles}) == 4


def test_symbol_firewall_rejects_aliases_rebinding_and_incomplete_tables() -> None:
    aliased = dict(DEFAULT_SYMBOL_BINDINGS)
    aliased[SymbolRole.PERIODIC_PHASE] = aliased[SymbolRole.GOLDEN_RATIO]
    with pytest.raises(SymbolFirewallError, match="aliases"):
        validate_symbol_firewall(aliased)

    rebound = dict(DEFAULT_SYMBOL_BINDINGS)
    rebound[SymbolRole.LINEAR_TIME] = "T_cone"
    rebound[SymbolRole.CONE_SLANT_LENGTH] = "slant"
    with pytest.raises(SymbolFirewallError, match="reserved"):
        validate_symbol_firewall(rebound)

    incomplete = dict(DEFAULT_SYMBOL_BINDINGS)
    del incomplete[SymbolRole.FINITE_CONE_SDF]
    with pytest.raises(SymbolFirewallError, match="missing protected"):
        validate_symbol_firewall(incomplete)


def test_cone_tree_and_topology_concepts_remain_separate_content() -> None:
    kinds = (
        "comparison-bst",
        "radix-prefix-trie",
        "half-turn-bundle-map",
        "reflective-klein-quotient",
        "implicit-cone-field",
        "finite-cone-sdf",
        "certified-sweep-interval",
    )
    nodes = tuple(
        DefinitionNode(
            id=f"def:{kind}",
            kind=kind,
            domain="DeclaredDomain",
            codomain="DeclaredCodomain",
            evaluation_phase=index,
            parameters={"bounded": True},
            provenance={"source_refs": ("SCLP-3.6.2",)},
        )
        for index, kind in enumerate(kinds)
    )
    graph = SubstrateGraph(nodes)
    assert len({node.kind for node in graph.definitions}) == len(kinds)
    assert len({node.content_hash for node in graph.definitions}) == len(kinds)


def test_canonical_hash_rejects_unordered_or_nonfinite_content() -> None:
    with pytest.raises(SubstrateGraphError, match="non-JSON"):
        canonical_hash({"unordered": {1, 2, 3}})
    with pytest.raises(SubstrateGraphError, match="non-finite"):
        canonical_hash({"not_a_number": float("nan")})
