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
    TypeCompatibilityError,
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
    domain: str = "TypedState",
    codomain: str = "TypedState",
) -> DefinitionNode:
    return DefinitionNode(
        id=id,
        kind=kind,
        domain=domain,
        codomain=codomain,
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
    dependency_hashes = {
        "def:clock": "sha256:" + "1" * 64,
        "def:input": "sha256:" + "2" * 64,
    }
    first = DefinitionNode(
        id="def:log-polar",
        kind="coordinate-transform",
        domain="Residual2",
        codomain="LogPolarAddress",
        dependencies=("def:clock", "def:input"),
        dependency_hashes=dependency_hashes,
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
        dependency_hashes={
            "def:input": "sha256:" + "2" * 64,
            "def:clock": "sha256:" + "1" * 64,
        },
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
    with pytest.raises(ContentHashError, match="must supply"):
        DefinitionNode.from_record(node.semantic_record())
    with pytest.raises(ContentHashError, match="must supply"):
        DefinitionNode.from_record({**node.semantic_record(), "content_hash": ""})
    missing_type = {**node.semantic_record(), "content_hash": node.content_hash}
    missing_type.pop("record_type")
    with pytest.raises(SubstrateGraphError, match="record_type"):
        DefinitionNode.from_record(missing_type)
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
    assert resolved[1].instance is graph.instance(instance.id)
    bound_field = graph.definition(field.id)
    assert bound_field.content_hash
    assert bound_field.dependency_hashes == {
        "def:address": graph.definition(address.id).content_hash
    }
    assert graph.definition(bound_field.content_hash) is bound_field
    assert graph.instance(instance.id).definition_hash == bound_field.content_hash
    assert graph.pipeline(pipeline.id).step_hashes[1] == graph.instance(instance.id).content_hash
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
    bound_feedback = graph.feedback_edges[0]
    assert bound_feedback.generation_delay == 1
    assert bound_feedback.semantics == "referential-next-generation"
    assert bound_feedback.fixed_point_claim is False
    assert bound_feedback.source_port_type == "TypedState"
    assert bound_feedback.target_port_type == "TypedState"
    assert bound_feedback.source_hash == graph.definition(observable.id).content_hash
    assert bound_feedback.target_hash == graph.definition(input_node.id).content_hash
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


def test_graph_is_immutable_and_its_root_digest_remains_verifiable() -> None:
    root = _definition("def:root")
    instance = DefinitionInstance("instance:root", root.id)
    pipeline = Pipeline("pipeline:root", (instance.id,))
    feedback = FeedbackEdge(
        "feedback:root",
        root.id,
        root.id,
        provenance={"source_refs": ("first-party:test",)},
    )
    graph = SubstrateGraph(
        (root,),
        instances=(instance,),
        pipelines=(pipeline,),
        feedback_edges=(feedback,),
    )
    assert graph.verify_content_hash()
    assert not hasattr(graph, "__dict__")

    mutations = {
        "definitions": (),
        "instances": (),
        "pipelines": (),
        "feedback_edges": (),
        "symbol_bindings": {},
        "closure_semantics": "unrestricted-fixed-point",
        "fixed_point_engine": True,
        "content_hash": "sha256:" + "0" * 64,
    }
    for attribute, value in mutations.items():
        with pytest.raises(AttributeError, match="immutable"):
            setattr(graph, attribute, value)
    assert graph.fixed_point_engine is False
    assert graph.verify_content_hash()


def test_all_manifest_record_loaders_require_type_hash_and_verify_round_trip() -> None:
    root = _definition("def:root")
    instance = DefinitionInstance("instance:root", root.id, state={"generation": 0})
    pipeline = Pipeline("pipeline:root", (instance.id,))
    feedback = FeedbackEdge(
        "feedback:root",
        root.id,
        root.id,
        provenance={"source_refs": ("first-party:test",)},
    )
    graph = SubstrateGraph(
        (root,),
        instances=(instance,),
        pipelines=(pipeline,),
        feedback_edges=(feedback,),
    )
    records_and_loaders = (
        (
            {
                **graph.instance(instance.id).semantic_record(),
                "content_hash": graph.instance(instance.id).content_hash,
            },
            DefinitionInstance.from_record,
        ),
        (
            {
                **graph.pipeline(pipeline.id).semantic_record(),
                "content_hash": graph.pipeline(pipeline.id).content_hash,
            },
            Pipeline.from_record,
        ),
        (
            {
                **graph.feedback_edges[0].semantic_record(),
                "content_hash": graph.feedback_edges[0].content_hash,
            },
            FeedbackEdge.from_record,
        ),
    )
    for record, loader in records_and_loaders:
        assert loader(record).content_hash == record["content_hash"]
        without_type = dict(record)
        without_type.pop("record_type")
        with pytest.raises(SubstrateGraphError, match="record_type"):
            loader(without_type)
        without_hash = dict(record)
        without_hash.pop("content_hash")
        with pytest.raises(ContentHashError, match="must supply"):
            loader(without_hash)
        forged = dict(record)
        forged["content_hash"] = "sha256:" + "f" * 64
        with pytest.raises(ContentHashError, match="does not match"):
            loader(forged)


def test_merkle_bindings_propagate_definition_instance_pipeline_and_feedback_changes() -> None:
    def build(root_parameter: int) -> SubstrateGraph:
        root = DefinitionNode(
            "def:root",
            "input",
            "State",
            "State",
            parameters={"version": root_parameter},
        )
        child = DefinitionNode(
            "def:child",
            "operator",
            "State",
            "State",
            dependencies=(root.id,),
            evaluation_phase=1,
        )
        instance = DefinitionInstance("instance:child", child.id, state={"generation": 0})
        pipeline = Pipeline("pipeline:merkle", (root.id, instance.id))
        feedback = FeedbackEdge(
            "feedback:merkle",
            child.id,
            root.id,
            provenance={"source_refs": ("first-party:test",)},
        )
        return SubstrateGraph(
            (root, child),
            instances=(instance,),
            pipelines=(pipeline,),
            feedback_edges=(feedback,),
        )

    first = build(1)
    second = build(2)
    assert first.definition("def:root").content_hash != second.definition("def:root").content_hash
    assert first.definition("def:child").content_hash != second.definition("def:child").content_hash
    assert first.instance("instance:child").content_hash != second.instance("instance:child").content_hash
    assert first.pipeline("pipeline:merkle").content_hash != second.pipeline("pipeline:merkle").content_hash
    assert first.feedback_edges[0].content_hash != second.feedback_edges[0].content_hash
    assert first.content_hash != second.content_hash


def test_unbound_or_forged_merkle_content_is_not_accepted_as_identity() -> None:
    child = _definition("def:child", dependencies=("def:root",), phase=1)
    assert child.content_hash == ""
    assert child.verify_content_hash() is False
    with pytest.raises(ContentHashError, match="cannot be verified"):
        _definition("def:child", dependencies=("def:root",), phase=1).__class__(
            id="def:child",
            kind="typed-operator",
            domain="TypedState",
            codomain="TypedState",
            dependencies=("def:root",),
            content_hash="sha256:" + "0" * 64,
        )

    root = _definition("def:root")
    forged = DefinitionNode(
        "def:child",
        "typed-operator",
        "TypedState",
        "TypedState",
        dependencies=(root.id,),
        dependency_hashes={root.id: "sha256:" + "f" * 64},
        evaluation_phase=1,
    )
    with pytest.raises(ContentHashError, match="do not match the resolved graph"):
        SubstrateGraph((root, forged))


def test_pipeline_and_feedback_type_boundaries_fail_closed() -> None:
    source = _definition("def:source", codomain="Address")
    sink = _definition(
        "def:sink",
        dependencies=(source.id,),
        phase=1,
        domain="Metric",
        codomain="Observable",
    )
    with pytest.raises(TypeCompatibilityError, match="codomain.*domain"):
        SubstrateGraph(
            (source, sink),
            pipelines=(Pipeline("pipeline:bad-types", (source.id, sink.id)),),
        )

    observable = DefinitionNode(
        "def:observable",
        "typed-operator",
        "State",
        "Observable",
        output_ports={"observable": "Observable"},
    )
    residual = DefinitionNode(
        "def:residual",
        "typed-operator",
        "Residual",
        "State",
        input_ports={"residual": "Residual"},
    )
    feedback = FeedbackEdge(
        "feedback:bad-types",
        observable.id,
        residual.id,
        source_port_type="Observable",
        target_port_type="Residual",
        provenance={"source_refs": ("first-party:test",)},
    )
    with pytest.raises(TypeCompatibilityError, match="incompatible"):
        SubstrateGraph((observable, residual), feedback_edges=(feedback,))

    port_source = DefinitionNode(
        "def:port-source",
        "typed-operator",
        "State",
        "State",
        output_ports={"declared-output": "State"},
    )
    port_target = DefinitionNode(
        "def:port-target",
        "typed-operator",
        "State",
        "State",
        input_ports={"declared-input": "State"},
    )
    undeclared_port = FeedbackEdge(
        "feedback:undeclared-port",
        port_source.id,
        port_target.id,
        provenance={"source_refs": ("first-party:test",)},
    )
    with pytest.raises(ReferenceResolutionError, match="no declared output port"):
        SubstrateGraph(
            (port_source, port_target), feedback_edges=(undeclared_port,)
        )

    with pytest.raises(SubstrateGraphError, match="generation"):
        Pipeline("pipeline:negative-generation", (source.id,), generation=-1)
