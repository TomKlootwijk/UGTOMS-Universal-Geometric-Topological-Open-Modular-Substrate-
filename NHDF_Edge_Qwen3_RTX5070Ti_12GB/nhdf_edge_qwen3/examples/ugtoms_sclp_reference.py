"""Deterministic first-party UGTOMS/SCLP reference application.

This is a bounded executable composition, not a renderer and not a fixed-point
engine.  It commits one verified ``n -> n + 1`` transition and derives a stable
prefix of render-only instances from a fixed recipe.  Generated display records
never acquire ECS, collider, or gameplay identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from nhdf_edge.substrate_graph import (  # noqa: E402
    DefinitionInstance,
    DefinitionNode,
    FeedbackEdge,
    Pipeline,
    SubstrateGraph,
    canonical_json as graph_json,
)
from nhdf_edge.substrate_packing import (  # noqa: E402
    ComponentProfile,
    FixedRecipe,
    LogPolarProfile,
    MotionBounds,
    OperatorMeaning,
    PackedComponentRecord,
    PackedMotion64,
    PackedPose64,
    RecipePack,
    SharedLogPolarLUT,
    SparseComponentPack,
    generated_display_instances,
    semantic_address,
)
from nhdf_edge.substrate_runtime import (  # noqa: E402
    BSTTRouter,
    EventOrigin,
    FiniteConeSDF,
    GuardCrossing,
    KinematicState,
    LogPolarLUT,
    NoveltyLog,
    OneBitJitter,
    PredicateValue,
    SpatialLogPolarChart,
    SphereSDF,
    canonical_json as runtime_json,
    deterministic_bit,
    klein_normalize,
    lineage_digest,
    verify_event,
    xor_parity,
)


APPLICATION_ID = "ugtoms-sclp-reference"
APPLICATION_VERSION = "0.1.0"
EVIDENCE_FORMAT = "ugtoms-sclp-reference-evidence-0.1"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _q(value: float) -> float:
    """Use a presentation quantum without changing runtime calculations."""

    rounded = round(float(value), 12)
    return 0.0 if rounded == 0.0 else rounded


def _qv(values: tuple[float, ...]) -> list[float]:
    return [_q(value) for value in values]


def _definition_graph() -> tuple[SubstrateGraph, FeedbackEdge]:
    descriptions = (
        ("def:input", "input", (), "residual and generation input"),
        ("def:log-polar", "coordinate", ("def:input",), "residual address and spatial metric"),
        ("def:predicates", "predicate", ("def:log-polar",), "separate parity, jitter, and branch control"),
        ("def:routing", "routing", ("def:predicates",), "bounded BST-T bifurcation"),
        ("def:kinematics", "kinematic", ("def:routing",), "causal vector kinematics"),
        ("def:geometry", "geometry", ("def:kinematics",), "finite cone and sphere support"),
        ("def:event", "admission", ("def:geometry", "def:predicates"), "tri-state event admission"),
        ("def:transition", "transition", ("def:event",), "single canonical atomic commit"),
        ("def:packing", "packing", ("def:transition",), "bounded pose, motion, LUT, and recipe packing"),
        ("def:observable", "projection", ("def:packing",), "display-only observable projection"),
    )
    nodes = tuple(
        DefinitionNode(
            id=node_id,
            kind=kind,
            domain="bounded substrate state",
            codomain="bounded substrate state",
            dependencies=dependencies,
            evaluation_phase=phase,
            parameters={"implementation": description},
            equation=description,
            units={"linear_time": "s", "position": "m", "angle": "rad"},
            bounds={"generation_count": 1, "display_instances": 64},
            failures=("invalid-domain", "resource-bound", "indeterminate-event"),
            provenance={"clean_room": True, "source_refs": ("first-party:ugtoms-sclp-reference-v0.1",)},
        )
        for phase, (node_id, kind, dependencies, description) in enumerate(descriptions)
    )
    instance = DefinitionInstance(
        "instance:display-prototype",
        "def:packing",
        literal={"prototype_index": 7},
        state={"generation": 0},
    )
    pipeline = Pipeline(
        "pipeline:reference",
        tuple(node_id for node_id, _, _, _ in descriptions),
        "One bounded, inspectable generation of the reference application.",
    )
    feedback = FeedbackEdge(
        "feedback:observable-to-input",
        "def:observable",
        "def:input",
        source_generation=0,
        target_generation=1,
        provenance={"source_refs": ("first-party:ugtoms-sclp-reference-v0.1",)},
    )
    return (
        SubstrateGraph(
            nodes,
            instances=(instance,),
            pipelines=(pipeline,),
            feedback_edges=(feedback,),
        ),
        feedback,
    )


def build_reference_result() -> dict[str, Any]:
    """Execute the bounded reference chain and return JSON-safe evidence."""

    graph, feedback = _definition_graph()

    residual_lut = LogPolarLUT(
        radial_bins=8,
        angular_bins=16,
        time_bins=4,
        maximum_magnitude=16.0,
    )
    residual_address = residual_lut.encode((3.0, 4.0), forward_step=0)
    spatial_chart = SpatialLogPolarChart(reference_radius=1.0, rho_min=-4.0, rho_max=4.0)
    spatial = spatial_chart.encode(0.7, 0.0)
    rho = float(spatial["rho"])
    theta = float(spatial["theta"])
    rho_rate, theta_rate = -0.6, 0.1
    rho_acceleration, theta_acceleration = 0.05, 0.02
    planar_velocity = spatial_chart.velocity(rho, theta, rho_rate, theta_rate)
    planar_acceleration = spatial_chart.acceleration(
        rho,
        theta,
        rho_rate,
        theta_rate,
        rho_acceleration,
        theta_acceleration,
    )

    klein = klein_normalize(1.25, 0.3)
    payload_parity = xor_parity(b"UGTOMS")
    topology_parity = klein.orientation_reversals & 1
    jitter = OneBitJitter(amplitude=0.01, guard_margin=0.1, seed="ugtoms-sclp-reference")
    jitter_certificate = jitter.certificate(-0.25, residual_address.packed_index, 0)
    branch_control = deterministic_bit(5, 0, "branch-control")
    router = BSTTRouter(maximum_depth=4, maximum_active_branches=8)
    route = router.route(
        residual_address,
        phase=0.25,
        phase_acceleration=0.02,
        generation=0,
        parity_gate=branch_control,
        depth=1,
        active_branches=2,
    )

    initial = KinematicState(
        position=(0.7, 0.0, 1.0),
        velocity=(planar_velocity[0], planar_velocity[1], 0.0),
        acceleration=(planar_acceleration[0], planar_acceleration[1], 0.0),
    )
    advanced = initial.advance(0.5)
    velocity_arrow, acceleration_arrow = initial.arrows()
    cone = FiniteConeSDF(slant_length=2.0, half_angle_radians=math.pi / 6.0)
    sphere = SphereSDF(centre=(0.0, 0.0, 1.0), radius=1.0)
    cone_before = cone.evaluate(initial.position)
    cone_after = cone.evaluate(advanced.position)
    sphere_after = sphere.evaluate(advanced.position)
    event = verify_event(
        support=PredicateValue.TRUE if sphere_after <= 0.0 else PredicateValue.FALSE,
        compatibility=PredicateValue.TRUE if route.bounded else PredicateValue.FALSE,
        guard=GuardCrossing(cone_before, cone_after),
    )
    indeterminate_probe = verify_event(
        support=PredicateValue.INDETERMINATE,
        compatibility=PredicateValue.TRUE,
        guard=GuardCrossing(-1.0, 1.0),
    )
    if event.status.value != "VERIFIED":
        raise RuntimeError("reference transition was not admitted")

    previous_state = {
        "generation": 0,
        "lineage_head": "0" * 64,
        "position": _qv(initial.position),
    }
    proposed_state = {
        "generation": 1,
        "position": _qv(advanced.position),
        "residual_address": residual_address.packed_index,
        "route_paths": list(route.branch_paths),
    }
    previous_hash = _sha256(runtime_json(previous_state))
    proposed_hash = _sha256(runtime_json(proposed_state))
    novelty = NoveltyLog(capacity=4)
    lineage_event = novelty.append(
        "verified-reference-transition",
        {
            "decision": event.status.value,
            "feedback_edge": feedback.content_hash,
            "previous_state": previous_hash,
            "proposed_state": proposed_hash,
        },
        origin=EventOrigin.CLOSED_DYNAMICS,
    )
    if lineage_digest(novelty.events) != lineage_event.lineage_digest:
        raise RuntimeError("lineage replay failed")
    committed_state = {**proposed_state, "lineage_head": lineage_event.lineage_digest}
    commit_unit = {
        "state": committed_state,
        "lineage_event": {
            "sequence": lineage_event.sequence,
            "origin": lineage_event.origin.value,
            "novelty_digest": lineage_event.novelty_digest,
            "lineage_digest": lineage_event.lineage_digest,
        },
    }

    polar_profile = LogPolarProfile(
        reference_radius=1.0,
        rho_min=-4.0,
        rho_max=4.0,
        core_radius=1.0e-9,
        resolution=32,
    )
    shared_lut = SharedLogPolarLUT.generate(polar_profile)
    lut_bytes = shared_lut.to_bytes()
    if SharedLogPolarLUT.from_bytes(lut_bytes).to_bytes() != lut_bytes:
        raise RuntimeError("shared LUT round trip failed")
    motion_bounds = MotionBounds(2.0, 2.0, 2.0, 2.0)
    pose = PackedPose64.from_values(rho, theta, 0, route.geometric_angles[0], rho_min=-4.0, rho_max=4.0)
    motion = PackedMotion64.from_values(
        rho_rate,
        theta_rate,
        rho_acceleration,
        theta_acceleration,
        bounds=motion_bounds,
    )
    component_profile = ComponentProfile("sclp-reference", polar_profile, motion_bounds, shared_lut)
    component_pack = SparseComponentPack.build(
        (component_profile,),
        (PackedComponentRecord(7, component_profile.profile_id, pose, motion),),
    )
    component_bytes = component_pack.to_bytes()
    if SparseComponentPack.from_bytes(component_bytes).to_bytes() != component_bytes:
        raise RuntimeError("component pack round trip failed")

    operators = (
        OperatorMeaning(0x1001, 0, 1, "display-phase", "derive a render-only phase lane from stable lineage"),
        OperatorMeaning(0x1002, 1, 2, "logpolar-placement", "address display placement through the declared log-polar profile"),
    )
    recipe = FixedRecipe(
        prototype_index=7,
        instance_count=64,
        root_seed=0x5547544F4D53,
        recipe_seed=0x53434C50,
        operators=operators,
        profile_address=component_profile.semantics_address,
        prototype_address=semantic_address("prototype", graph.content_hash.encode("ascii")),
        parameters=(rho, theta, 0.25, 0.02, cone.slant_length, cone.half_angle_radians, 1.0, 0.5),
    )
    recipe_pack = RecipePack.build((recipe,))
    recipe_bytes = recipe_pack.to_bytes()
    if RecipePack.from_bytes(recipe_bytes, operator_registry=operators).to_bytes() != recipe_bytes:
        raise RuntimeError("recipe pack round trip failed")
    short_prefix = generated_display_instances(recipe, prototype_pose=pose, prototype_motion=motion, count=9)
    long_prefix = generated_display_instances(recipe, prototype_pose=pose, prototype_motion=motion, count=17)
    short_ids = [item.lineage_id for item in short_prefix]
    long_ids = [item.lineage_id for item in long_prefix]
    if short_ids != long_ids[: len(short_ids)]:
        raise RuntimeError("display prefix is not stable")
    forbidden_display_fields = ("entity_id", "collider", "gameplay_state", "ecs_component")
    if any(hasattr(long_prefix[0], name) for name in forbidden_display_fields):
        raise RuntimeError("render-only display instance crossed the authority boundary")

    graph_manifest = graph.manifest()
    graph_bytes = graph_json(graph_manifest)
    return {
        "application_id": APPLICATION_ID,
        "application_version": APPLICATION_VERSION,
        "display_boundary": {
            "authoritative_component_records": 1,
            "forbidden_generated_authority": list(forbidden_display_fields),
            "generated_type": "GeneratedDisplayInstance",
            "render_only": True,
            "ecs_identity": False,
            "collider_identity": False,
            "gameplay_authority": False,
        },
        "format": EVIDENCE_FORMAT,
        "geometry": {
            "cone": {
                "claim": "exact finite filled right-circular cone SDF",
                "before": _q(cone_before),
                "after": _q(cone_after),
                "height": _q(cone.height),
                "base_radius": _q(cone.base_radius),
            },
            "sphere_support_sdf": _q(sphere_after),
        },
        "graph": {
            "content_hash": graph.content_hash,
            "manifest_sha256": _sha256(graph_bytes),
            "definition_hashes": {node.id: node.content_hash for node in graph.definitions},
            "topological_order": [node.id for node in graph.topological_definitions()],
            "feedback": {
                "content_hash": feedback.content_hash,
                "source_generation": feedback.source_generation,
                "target_generation": feedback.target_generation,
                "generation_delay": feedback.generation_delay,
                "semantics": feedback.semantics,
                "fixed_point_claim": feedback.fixed_point_claim,
            },
            "fixed_point_engine": graph.fixed_point_engine,
        },
        "kinematics": {
            "linear_time_step_seconds": 0.5,
            "initial_position": _qv(initial.position),
            "advanced_position": _qv(advanced.position),
            "velocity": _qv(initial.velocity),
            "acceleration": _qv(initial.acceleration),
            "velocity_arrow": {"origin": _qv(velocity_arrow.origin), "displacement": _qv(velocity_arrow.displacement)},
            "acceleration_arrow": {"origin": _qv(acceleration_arrow.origin), "displacement": _qv(acceleration_arrow.displacement)},
        },
        "log_polar": {
            "residual_address": {
                "packed_index": residual_address.packed_index,
                "rho_jitter": _q(residual_address.rho),
                "theta": _q(residual_address.theta),
                "capacity": residual_lut.capacity,
                "saturated": residual_address.saturated,
            },
            "spatial_metric": {
                "rho_spatial": _q(rho),
                "theta": _q(theta),
                "metric_scale": _q(spatial_chart.metric_scale(rho)),
                "jacobian": [_qv(row) for row in spatial_chart.jacobian(rho, theta)],
                "core": bool(spatial["core"]),
                "saturated": bool(spatial["saturated"]),
            },
        },
        "logic": {
            "semantics": "three-valued",
            "allowed_predicates": [value.value for value in PredicateValue],
            "fuzzy_logic": False,
            "event": {
                "status": event.status.value,
                "support": event.support.value,
                "compatibility": event.compatibility.value,
                "crossing": event.crossing.value,
            },
            "indeterminate_probe": indeterminate_probe.status.value,
        },
        "numeric_policy": {
            "geometry": "IEEE-754 binary64 reference calculations; reported floats rounded to 1e-12",
            "pose": "20/18/14/12 round-half-up bounded fields",
            "motion": "four signed16 lanes, round-half-away-from-zero, code 0x8000 rejected",
            "lut": "canonical little-endian binary16 samples with binary64 interpolation",
        },
        "packing": {
            "pose_word": pose.word,
            "motion_word": motion.word,
            "pose_bytes": len(pose.to_bytes()),
            "motion_bytes": len(motion.to_bytes()),
            "shared_lut": {"optional": True, "resolution": 32, "bytes": len(lut_bytes), "sha256": _sha256(lut_bytes)},
            "component_pack": {"records": 1, "bytes": len(component_bytes), "sha256": _sha256(component_bytes)},
            "recipe": {
                "instance_count": recipe.instance_count,
                "bytes": len(recipe_bytes),
                "sha256": _sha256(recipe_bytes),
                "content_address": recipe.content_address.hex(),
                "lineage_namespace": recipe.lineage_namespace.hex(),
                "identity_domains_distinct": recipe.content_address != recipe.lineage_namespace,
            },
            "stable_display_prefix": {
                "short_count": len(short_ids),
                "long_count": len(long_ids),
                "short_ids": short_ids,
                "long_prefix_ids": long_ids[: len(short_ids)],
                "short_sha256": _sha256(runtime_json(short_ids)),
            },
        },
        "predicates": {
            "roles_are_distinct": True,
            "payload_parity_bit": payload_parity,
            "topology_parity_bit": topology_parity,
            "jitter_control_bit": int(jitter_certificate["bit"]),
            "branch_control_bit": branch_control,
            "jitter_interval": _qv(tuple(jitter_certificate["interval"])),
            "jitter_safe_under_margin": bool(jitter_certificate["safe_under_margin"]),
            "klein_orientation": klein.orientation,
            "klein_sheet": klein.sheet,
        },
        "routing": {
            "maximum_depth": router.maximum_depth,
            "maximum_active_branches": router.maximum_active_branches,
            "ordering_key": list(route.ordering_key),
            "branch_paths": list(route.branch_paths),
            "geometric_angles": _qv(route.geometric_angles),
            "bifurcated": route.bifurcated,
            "bounded": route.bounded,
        },
        "transition": {
            "admission_required": "VERIFIED",
            "commit_policy": "state and lineage event are one canonical commit unit",
            "previous_state_sha256": previous_hash,
            "proposed_state_sha256": proposed_hash,
            "committed_state_sha256": _sha256(runtime_json(committed_state)),
            "lineage_head": lineage_event.lineage_digest,
            "commit_unit_sha256": _sha256(runtime_json(commit_unit)),
            "origin": lineage_event.origin.value,
            "generation": {"source": 0, "target": 1},
        },
    }


def canonical_result_bytes(result: Mapping[str, Any]) -> bytes:
    return json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write canonical evidence JSON; otherwise use stdout")
    args = parser.parse_args(argv)
    payload = canonical_result_bytes(build_reference_result())
    if args.output is None:
        sys.stdout.buffer.write(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
