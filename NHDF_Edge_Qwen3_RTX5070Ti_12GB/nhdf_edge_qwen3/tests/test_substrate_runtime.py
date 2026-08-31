from __future__ import annotations

import math

import pytest

from nhdf_edge.substrate_runtime import (
    BSTTRouter,
    CircleSDF,
    ClosedDynamicsSeed,
    ConeField,
    DistributedApex,
    EventOrigin,
    EventStatus,
    GuardCrossing,
    KinematicState,
    FiniteConeSDF,
    LogPolarLUT,
    NoveltyLog,
    OneBitJitter,
    ParityDebounce,
    PredicateValue,
    SphereSDF,
    SCLPKeyLayout64,
    SpatialLogPolarChart,
    SubstrateError,
    field_difference,
    field_intersection,
    field_union,
    klein_normalize,
    lineage_digest,
    verify_event,
    VectorArrow3,
    xor_parity,
    zero_predicate,
)


def test_log_polar_lut_is_bounded_deterministic_and_has_apex_sentinel() -> None:
    lut = LogPolarLUT(radial_bins=8, angular_bins=16, time_bins=4, maximum_magnitude=10.0)
    address = lut.encode((3.0, 4.0), forward_step=5)
    assert address == lut.encode((3.0, 4.0), forward_step=5)
    assert address.magnitude == 5.0
    assert address.time_bin == 1
    assert 0 <= address.packed_index < lut.capacity
    assert not address.apex
    assert lut.encode((0.0, 0.0)).apex
    assert lut.encode((100.0, 0.0)).saturated


def test_one_bit_parity_keeps_its_known_even_flip_blind_spot() -> None:
    original = bytes([0b00000000])
    one_flip = bytes([0b00000001])
    two_flips = bytes([0b00000011])
    assert xor_parity(original) == 0
    assert xor_parity(one_flip) == 1
    assert xor_parity(two_flips) == 0
    assert xor_parity(original, orientation_reversals=1) == 1
    state = ParityDebounce().update(1, required_samples=2)
    assert state.stable_bit == 0
    assert state.update(1, required_samples=2).stable_bit == 1


def test_klein_gluing_reverses_once_and_restores_after_two_seams() -> None:
    once = klein_normalize(1.25, 0.2)
    assert once.u == pytest.approx(0.25)
    assert once.v == pytest.approx(0.8)
    assert once.sheet == 1
    assert once.orientation == -1
    twice = klein_normalize(2.25, 0.2)
    assert twice.u == pytest.approx(0.25)
    assert twice.v == pytest.approx(0.2)
    assert twice.sheet == 0
    assert twice.orientation == 1


def test_sphere_circle_cone_and_apex_relations_share_zero_predicate() -> None:
    sphere = SphereSDF((0.0, 0.0, 0.0), 2.0)
    circle = CircleSDF((0.0, 0.0), 2.0)
    cone = ConeField((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), math.pi / 4)
    assert zero_predicate(sphere.evaluate((2.0, 0.0, 0.0))) is PredicateValue.TRUE
    assert zero_predicate(circle.evaluate((0.0, -2.0))) is PredicateValue.TRUE
    assert zero_predicate(cone.evaluate((1.0, 0.0, 1.0))) is PredicateValue.TRUE
    assert cone.evaluate((0.0, 0.0, -1.0)) > 0
    apex = DistributedApex(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)), (1.0, 3.0))
    assert apex.observed_centroid() == pytest.approx((1.5, 0.0, 0.0))


def test_sclp_finite_cone_is_exact_on_side_base_apex_and_certificate_vector() -> None:
    cone = FiniteConeSDF(2.0, math.pi / 6)
    assert cone.height == pytest.approx(math.sqrt(3.0))
    assert cone.base_radius == pytest.approx(1.0)
    assert cone.evaluate((0.0, 0.0, 0.0)) == pytest.approx(0.0)
    assert cone.evaluate((1.0, 0.0, cone.height)) == pytest.approx(0.0)
    assert cone.evaluate((0.15, 0.0, 0.6)) == pytest.approx(-0.1700961894323342)


def test_spatial_logpolar_chart_preserves_metric_velocity_and_acceleration() -> None:
    chart = SpatialLogPolarChart(reference_radius=1.0, rho_min=-20.0, rho_max=20.0)
    encoded = chart.encode(3.0, 4.0)
    rho = float(encoded["rho"])
    theta = float(encoded["theta"])
    assert chart.decode(rho, theta) == pytest.approx((3.0, 4.0))
    assert chart.metric_scale(rho) == pytest.approx(25.0)
    assert chart.velocity(rho, theta, 0.0, 1.0) == pytest.approx((-4.0, 3.0))
    assert chart.acceleration(rho, theta, 0.0, 1.0, 0.0, 0.0) == pytest.approx((-3.0, -4.0))
    assert chart.exact_radial_increment(rho, 1e-3) == pytest.approx(5.0 * math.expm1(1e-3))


def test_sclp_key_layout_roundtrips_both_distinct_64_bit_forms() -> None:
    layout = SCLPKeyLayout64(rho_min=-20.0, rho_max=0.0)
    state = layout.quantize(-1.8971199848858813, 0.0, 1920, math.radians(20.0))
    contiguous = layout.pack_contiguous(state)
    morton = layout.pack_morton(state)
    assert layout.unpack_contiguous(contiguous) == state
    assert layout.unpack_morton(morton) == state
    assert contiguous != morton
    assert len(layout.morton_schedule()) == 64


def test_one_bit_jitter_is_bounded_metadata_not_complete_state() -> None:
    jitter = OneBitJitter(amplitude=1e-4, guard_margin=1e-3, seed="sclp-vector")
    certificate = jitter.certificate(-0.17, 1234, 1920)
    assert certificate == jitter.certificate(-0.17, 1234, 1920)
    assert certificate["safe_under_margin"] is True
    assert certificate["interval"] == pytest.approx((-0.1701, -0.1699))


def test_implicit_csg_operators_preserve_declared_sign_logic() -> None:
    assert field_union(2.0, -1.0) == -1.0
    assert field_intersection(2.0, -1.0) == 2.0
    assert field_difference(-2.0, -1.0) == 1.0


def test_vector_arrows_are_compact_directed_motion_primitives() -> None:
    arrow = VectorArrow3.between((1.0, 2.0, 3.0), (4.0, 6.0, 3.0), role="motion")
    assert arrow.displacement == pytest.approx((3.0, 4.0, 0.0))
    assert arrow.endpoint == pytest.approx((4.0, 6.0, 3.0))
    assert arrow.magnitude == pytest.approx(5.0)
    assert arrow.direction() == pytest.approx((0.6, 0.8, 0.0))
    state = KinematicState((1.0, 1.0, 1.0), (2.0, 0.0, 0.0), (0.0, -1.0, 0.0))
    velocity, acceleration = state.arrows(scale=0.5)
    assert velocity.endpoint == pytest.approx((2.0, 1.0, 1.0))
    assert acceleration.endpoint == pytest.approx((1.0, 0.5, 1.0))


def test_event_gate_exposes_each_non_success_state() -> None:
    crossing = GuardCrossing(-1.0, 1.0)
    assert verify_event(support=True, compatibility=True, guard=crossing).status is EventStatus.VERIFIED
    assert verify_event(support=False, compatibility=True, guard=crossing).status is EventStatus.NO_SUPPORT
    assert verify_event(support=True, compatibility=False, guard=crossing).status is EventStatus.INCOMPATIBLE
    assert verify_event(
        support=True, compatibility=True, guard=GuardCrossing(1.0, 2.0)
    ).status is EventStatus.NO_CROSSING
    assert verify_event(
        support=True, compatibility=True, guard=GuardCrossing(float("nan"), 2.0)
    ).status is EventStatus.INDETERMINATE


def test_bst_t_keeps_order_key_separate_from_bifurcation_geometry_and_bounds() -> None:
    address = LogPolarLUT().encode((1.0, 1.0), forward_step=3)
    router = BSTTRouter(maximum_depth=2, maximum_active_branches=4)
    split = router.route(
        address,
        phase=0.25,
        phase_acceleration=0.1,
        generation=7,
        parity_gate=1,
        depth=1,
        active_branches=2,
    )
    assert split.bifurcated
    assert split.branch_paths == ("r0", "r1")
    assert split.ordering_key[0] == address.packed_index
    bounded = router.route(
        address,
        phase=0.25,
        phase_acceleration=0.1,
        generation=7,
        parity_gate=1,
        depth=2,
        active_branches=2,
    )
    assert not bounded.bifurcated
    assert not bounded.bounded


def test_seed_replay_is_deterministic_and_exogenous_log_cannot_be_silently_lost() -> None:
    seed = ClosedDynamicsSeed(
        grammar_id="constant-acceleration-v1",
        seed=23,
        initial=KinematicState((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.0, 0.0)),
        step_seconds=0.5,
        phase_rate=0.25,
    )
    first = seed.reconstruct(8)
    second = seed.reconstruct(8)
    assert first == second
    assert first["kinematics"].position == pytest.approx((8.0, 0.0, 0.0))

    log = NoveltyLog(2)
    event_a = log.append("sensor", {"value": 4}, origin=EventOrigin.EXOGENOUS)
    event_b = log.append("user-correction", {"value": 5}, origin=EventOrigin.EXOGENOUS)
    assert lineage_digest((event_a, event_b)) == event_b.lineage_digest
    with pytest.raises(SubstrateError, match="full of exogenous"):
        log.append("sensor", {"value": 6}, origin=EventOrigin.EXOGENOUS)


def test_closed_dynamics_can_be_compacted_before_exogenous_evidence() -> None:
    log = NoveltyLog(2)
    log.append("derived", {"generation": 1}, origin=EventOrigin.CLOSED_DYNAMICS)
    kept = log.append("sensor", {"sample": 9}, origin=EventOrigin.EXOGENOUS)
    newest = log.append("sensor", {"sample": 10}, origin=EventOrigin.EXOGENOUS)
    assert [event.event_type for event in log.events] == ["sensor", "sensor"]
    assert kept in log.events and newest in log.events
