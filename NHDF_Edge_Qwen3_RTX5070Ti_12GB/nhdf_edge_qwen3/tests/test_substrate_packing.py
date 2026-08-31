from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import hashlib
import math

import pytest

from nhdf_edge.substrate_packing import (
    OPERATOR_RECORD_BYTES,
    RECIPE_RECORD_BYTES,
    SPARSE_COMPONENT_BYTES,
    ComponentProfile,
    FixedRecipe,
    GeneratedDisplayInstance,
    LogPolarProfile,
    MotionBounds,
    OperatorMeaning,
    PackedComponentRecord,
    PackedMotion64,
    PackedPose64,
    RecipePack,
    SharedLogPolarLUT,
    SparseComponentPack,
    SubstratePackingError,
    generated_display_instances,
    semantic_address,
    splitmix64,
    stable_lineage_id,
)


BOUNDS = MotionBounds(4.0, 8.0, 16.0, 32.0)
POLAR = LogPolarProfile(
    reference_radius=2.0,
    rho_min=-4.0,
    rho_max=4.0,
    core_radius=1.0e-6,
    resolution=256,
)


def _pose(index: int = 0) -> PackedPose64:
    return PackedPose64.from_values(
        -1.25 + index * 0.01,
        0.25 + index * 0.02,
        100 + index,
        1.0 + index * 0.03,
        rho_min=POLAR.rho_min,
        rho_max=POLAR.rho_max,
    )


def _motion(index: int = 0) -> PackedMotion64:
    return PackedMotion64.from_values(
        0.25 + index * 0.001,
        -0.5,
        1.0,
        -2.0,
        bounds=BOUNDS,
    )


def _operators() -> tuple[OperatorMeaning, OperatorMeaning]:
    return (
        OperatorMeaning(
            0x101,
            0,
            2,
            "radial-offset",
            "Add one bounded binary32 radial offset selected by lineage lane zero.",
        ),
        OperatorMeaning(
            0x102,
            1,
            1,
            "angular-step",
            "Add ordinal times one binary32 turn step, reduced periodically.",
        ),
    )


def _recipe(instance_count: int) -> FixedRecipe:
    return FixedRecipe(
        prototype_index=7,
        instance_count=instance_count,
        root_seed=0x5EED_0001,
        recipe_seed=0xA11C_E002,
        operators=_operators(),
        profile_address=semantic_address("profile", b"profile-v1"),
        prototype_address=semantic_address("prototype", b"mesh+material+anchor-v1"),
        parameters=(0.25, 1.5, 1.0 / 7.0, 0.0, -0.0, 2.0, 0.75, 1.25),
    )


def _component_profile(*, with_lut: bool = True) -> ComponentProfile:
    lut = SharedLogPolarLUT.generate(POLAR) if with_lut else None
    return ComponentProfile("orbit", POLAR, BOUNDS, lut)


def test_pose_is_exactly_20_18_14_12_and_roundtrips_little_endian_word() -> None:
    pose = _pose()
    codes = pose.codes
    assert 0 <= codes.rho < 1 << 20
    assert 0 <= codes.theta < 1 << 18
    assert 0 <= codes.time < 1 << 14
    assert 0 <= codes.phi < 1 << 12
    assert pose.word == (
        (codes.rho << (18 + 14 + 12))
        | (codes.theta << (14 + 12))
        | (codes.time << 12)
        | codes.phi
    )
    assert len(pose.to_bytes()) == 8
    assert PackedPose64.from_bytes(pose.to_bytes()) == pose
    decoded = pose.decode(rho_min=POLAR.rho_min, rho_max=POLAR.rho_max)
    assert decoded.rho == pytest.approx(-1.25, abs=8.0 / ((1 << 20) - 1))
    assert decoded.time_tick == 100
    assert math.remainder(decoded.theta - 0.25, 2.0 * math.pi) == pytest.approx(
        0.0, abs=2.0 * math.pi / ((1 << 18) - 1)
    )
    with pytest.raises(SubstratePackingError, match="outside"):
        PackedPose64.from_values(-5.0, 0.0, 0, 0.0, rho_min=-4.0, rho_max=4.0)
    with pytest.raises(SubstratePackingError, match="wrapping"):
        PackedPose64.from_values(0.0, 0.0, 1 << 14, 0.0)
    wrapped = PackedPose64.from_values(0.0, 0.0, 1 << 14, 0.0, wrap_time=True)
    assert wrapped.codes.time == 0


def test_motion_uses_four_bounded_symmetric_signed16_lanes() -> None:
    motion = PackedMotion64.from_values(4.0, -8.0, 8.0, -16.0, bounds=BOUNDS)
    assert motion.codes == (0x7FFF, 0x8001, 0x4000, 0xC000)
    decoded = motion.decode(bounds=BOUNDS)
    assert decoded.rho_velocity == 4.0
    assert decoded.theta_velocity == -8.0
    assert decoded.rho_acceleration == pytest.approx(8.0, abs=16.0 / 32767.0)
    assert decoded.theta_acceleration == pytest.approx(-16.0, abs=32.0 / 32767.0)
    assert PackedMotion64.from_bytes(motion.to_bytes()) == motion
    with pytest.raises(SubstratePackingError, match="outside"):
        PackedMotion64.from_values(4.0001, 0.0, 0.0, 0.0, bounds=BOUNDS)
    with pytest.raises(SubstratePackingError, match="reserved"):
        PackedMotion64(0x8000 << 48)


def test_shared_lut_is_self_describing_binary16_canonical_and_strict() -> None:
    first = SharedLogPolarLUT.generate(POLAR)
    second = SharedLogPolarLUT.generate(POLAR)
    data = first.to_bytes()
    assert data == second.to_bytes()
    assert len(data) == 60 + POLAR.resolution * 3 * 2
    assert SharedLogPolarLUT.from_bytes(data) == first
    x, y = first.direction(2.0 * math.pi - 1.0e-8)
    assert math.hypot(x, y) == pytest.approx(1.0)
    assert first.direction(0.0) == pytest.approx((1.0, 0.0))
    assert first.direction(math.pi / 2.0) == pytest.approx((0.0, 1.0))
    assert first.direction(math.pi) == pytest.approx((-1.0, 0.0))
    assert first.direction(3.0 * math.pi / 2.0) == pytest.approx((0.0, -1.0))
    assert first.radius(POLAR.rho_min) == pytest.approx(
        POLAR.reference_radius * math.exp(POLAR.rho_min), rel=2.0e-3
    )

    noncanonical = bytearray(data)
    noncanonical[-1] ^= 1
    with pytest.raises(SubstratePackingError, match="canonical"):
        SharedLogPolarLUT.from_bytes(noncanonical)
    reserved = bytearray(data)
    reserved[14] = 1
    with pytest.raises(SubstratePackingError, match="reserved"):
        SharedLogPolarLUT.from_bytes(reserved)
    with pytest.raises(SubstratePackingError, match="trailing"):
        SharedLogPolarLUT.from_bytes(data + b"\0")

    # The cosine sample at 3*pi/2 canonically stores +0.  IEEE -0 compares
    # equal numerically, so this explicitly locks byte-level rejection.
    negative_zero = bytearray(data)
    cosine_index = 3 * POLAR.resolution // 4
    cosine_offset = 60 + POLAR.resolution * 2 + cosine_index * 2
    negative_zero[cosine_offset : cosine_offset + 2] = b"\x00\x80"
    with pytest.raises(SubstratePackingError, match="canonical"):
        SharedLogPolarLUT.from_bytes(negative_zero)


def test_noncanonical_direct_lut_cannot_serve_queries_or_identity() -> None:
    noncanonical = SharedLogPolarLUT(
        POLAR,
        1.0,
        (0.0,) * POLAR.resolution,
        (1.0,) * POLAR.resolution,
        (1.0,) * POLAR.resolution,
    )
    with pytest.raises(SubstratePackingError, match="not the canonical table"):
        noncanonical.direction(math.pi / 2.0)
    with pytest.raises(SubstratePackingError, match="not the canonical table"):
        noncanonical.radius(0.0)
    with pytest.raises(SubstratePackingError, match="not the canonical table"):
        _ = noncanonical.sha256


def test_sparse_pack_sorts_records_shares_one_lut_and_grows_24_bytes_per_node() -> None:
    profile = _component_profile()
    one = SparseComponentPack.build(
        (profile,),
        (PackedComponentRecord(9, "orbit", _pose(), _motion()),),
    )
    three = SparseComponentPack.build(
        (profile,),
        (
            PackedComponentRecord(9, "orbit", _pose(), _motion()),
            PackedComponentRecord(2, "orbit", _pose(1), _motion(1)),
            PackedComponentRecord(5, "orbit", _pose(2), _motion(2)),
        ),
    )
    one_bytes = one.to_bytes()
    three_bytes = three.to_bytes()
    assert len(three_bytes) - len(one_bytes) == 2 * SPARSE_COMPONENT_BYTES
    assert three.records[0].node_index == 2
    assert three.records[-1].node_index == 9
    assert three_bytes.count(first_magic := SharedLogPolarLUT.generate(POLAR).to_bytes()[:8]) == 1
    assert first_magic == b"NHLUT001"
    assert SparseComponentPack.from_bytes(three_bytes) == three
    assert three.sha256 == hashlib.sha256(three_bytes).hexdigest()


def test_sparse_pack_rejects_reserved_unsorted_truncated_and_trailing_data() -> None:
    profile = _component_profile(with_lut=False)
    pack = SparseComponentPack.build(
        (profile,),
        (
            PackedComponentRecord(2, "orbit", _pose(), _motion()),
            PackedComponentRecord(7, "orbit", _pose(1), _motion(1)),
        ),
    )
    data = pack.to_bytes()
    records_start = len(data) - 2 * SPARSE_COMPONENT_BYTES

    reserved = bytearray(data)
    reserved[records_start + 6] = 1
    with pytest.raises(SubstratePackingError, match="reserved"):
        SparseComponentPack.from_bytes(reserved)

    unsorted = bytearray(data)
    first = bytes(unsorted[records_start : records_start + SPARSE_COMPONENT_BYTES])
    second = bytes(unsorted[records_start + SPARSE_COMPONENT_BYTES :])
    unsorted[records_start : records_start + SPARSE_COMPONENT_BYTES] = second
    unsorted[records_start + SPARSE_COMPONENT_BYTES :] = first
    with pytest.raises(SubstratePackingError, match="sorted"):
        SparseComponentPack.from_bytes(unsorted)

    with pytest.raises(SubstratePackingError, match="truncated"):
        SparseComponentPack.from_bytes(data[:-1])
    with pytest.raises(SubstratePackingError, match="trailing"):
        SparseComponentPack.from_bytes(data + b"x")


def test_operator_meaning_address_changes_with_semantics_not_recipe_count() -> None:
    operator = _operators()[0]
    changed = OperatorMeaning(
        operator.code,
        operator.slot,
        operator.arity,
        operator.name,
        operator.meaning + " Changed.",
    )
    assert len(operator.address) == 32
    assert operator.address != changed.address
    recipe_64 = _recipe(64)
    recipe_1024 = _recipe(1024)
    assert recipe_64.lineage_namespace == recipe_1024.lineage_namespace
    assert recipe_64.content_address != recipe_1024.content_address
    assert recipe_64.parameters[3] == 0.0
    assert math.copysign(1.0, recipe_64.parameters[4]) == 1.0
    composed = OperatorMeaning(0x200, 2, 1, "unicode-meaning", "Caf\u00e9 operator")
    decomposed = OperatorMeaning(0x200, 2, 1, "unicode-meaning", "Cafe\u0301 operator")
    assert composed.meaning == decomposed.meaning
    assert composed.address == decomposed.address


def test_recipe_pack_has_fixed_records_and_rejects_semantic_or_byte_drift() -> None:
    operators = _operators()
    recipe = _recipe(1024)
    pack = RecipePack.build((recipe,))
    data = pack.to_bytes()
    assert len(data) == 32 + len(operators) * OPERATOR_RECORD_BYTES + RECIPE_RECORD_BYTES
    assert RecipePack.from_bytes(data, operator_registry=operators) == pack

    altered_registry = (
        OperatorMeaning(
            operators[0].code,
            operators[0].slot,
            operators[0].arity,
            operators[0].name,
            operators[0].meaning + " Altered.",
        ),
        operators[1],
    )
    with pytest.raises(SubstratePackingError, match="meaning mismatch"):
        RecipePack.from_bytes(data, operator_registry=altered_registry)

    recipe_start = 32 + len(operators) * OPERATOR_RECORD_BYTES
    reserved = bytearray(data)
    reserved[recipe_start + RECIPE_RECORD_BYTES - 1] = 1
    with pytest.raises(SubstratePackingError, match="reserved"):
        RecipePack.from_bytes(reserved, operator_registry=operators)
    with pytest.raises(SubstratePackingError, match="trailing"):
        RecipePack.from_bytes(data + b"x", operator_registry=operators)


def test_splitmix_and_generated_display_lineage_are_random_access_prefix_stable() -> None:
    assert splitmix64(0) == 0xE220A8397B1DCDAF
    recipe_64 = _recipe(64)
    recipe_1024 = _recipe(1024)
    pose = _pose()
    motion = _motion()
    generated_64 = generated_display_instances(
        recipe_64, prototype_pose=pose, prototype_motion=motion
    )
    generated_1024 = generated_display_instances(
        recipe_1024, prototype_pose=pose, prototype_motion=motion
    )
    assert len(generated_64) == 63
    assert len(generated_1024) == 1023
    assert generated_64 == generated_1024[:63]
    assert stable_lineage_id(recipe_1024, 63) == generated_1024[62].lineage_id
    assert len({item.lineage_id for item in generated_1024}) == 1023


def test_generated_display_type_cannot_acquire_ecs_gameplay_or_collider_identity() -> None:
    item = generated_display_instances(
        _recipe(2), prototype_pose=_pose(), prototype_motion=_motion()
    )[0]
    assert isinstance(item, GeneratedDisplayInstance)
    assert item.render_only is True
    assert not hasattr(item, "__dict__")
    names = {field.name for field in fields(GeneratedDisplayInstance)}
    assert not names & {"entity_id", "ecs_id", "gameplay", "collider", "tags"}
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        item.collider = object()  # type: ignore[misc]


def test_recipe_compactness_is_count_independent_but_not_general_compression() -> None:
    profile = _component_profile()
    component = SparseComponentPack.build(
        (profile,),
        (PackedComponentRecord(7, "orbit", _pose(), _motion()),),
    )
    recipe_64 = RecipePack.build((_recipe(64),))
    recipe_1024 = RecipePack.build((_recipe(1024),))
    compact_64 = len(component.to_bytes()) + len(recipe_64.to_bytes())
    compact_1024 = len(component.to_bytes()) + len(recipe_1024.to_bytes())
    real_component_pack = SparseComponentPack.build(
        (profile,),
        tuple(
            PackedComponentRecord(index, "orbit", _pose(), _motion())
            for index in range(1024)
        ),
    )
    naive_binary32_matrices = 1024 * 16 * 4
    assert compact_64 == compact_1024
    assert len(real_component_pack.to_bytes()) == len(component.to_bytes()) + 1023 * 24
    assert compact_1024 * 10 < len(real_component_pack.to_bytes())
    assert compact_1024 < naive_binary32_matrices // 16
    # The reduction is valid only because 1,023 copies are derived display
    # state. Independent entities or exogenous geometry must still be stored.


def test_reference_bytes_and_hashes_lock_the_new_clean_room_format() -> None:
    lut = SharedLogPolarLUT.generate(POLAR)
    component = SparseComponentPack.build(
        (_component_profile(),),
        (PackedComponentRecord(7, "orbit", _pose(), _motion()),),
    )
    recipe = RecipePack.build((_recipe(1024),))
    assert _pose().word == 0x580008A2F806428C
    assert _motion().word == 0x0800F8000800F800
    assert (
        hashlib.sha256(lut.to_bytes()).hexdigest()
        == "1e62717d7858d0b91c320280e80eb2f493637a725a1006663de3639ae0f250e7"
    )
    assert component.sha256 == "4bb2e243e9b4d1a62ebcdf7dc105196d1adf9f127f40b2c37879c8503ea3cc9b"
    assert recipe.sha256 == "949b714658351100e2b250600f2f7475793cb698a1a3e684c4f6d6e0f7e36cb2"


def test_binary_parsers_reject_wrong_exact_lengths() -> None:
    with pytest.raises(SubstratePackingError, match="exactly 8"):
        PackedPose64.from_bytes(b"\0" * 7)
    with pytest.raises(SubstratePackingError, match="exactly 8"):
        PackedMotion64.from_bytes(b"\0" * 9)
    with pytest.raises(SubstratePackingError, match="truncated"):
        SharedLogPolarLUT.from_bytes(b"NHLUT001")
    with pytest.raises(SubstratePackingError, match="truncated"):
        SparseComponentPack.from_bytes(b"NHSPK001")
    with pytest.raises(SubstratePackingError, match="truncated"):
        RecipePack.from_bytes(b"NHRCP001", operator_registry=_operators())
