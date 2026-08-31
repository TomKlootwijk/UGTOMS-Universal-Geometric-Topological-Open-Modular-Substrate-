"""Clean-room, bounded binary packing for substrate-generated display state.

This module defines a new NHDF Edge wire format.  It was implemented from the
documented behaviour of earlier substrate experiments, not by copying their
source or binary layouts.  The useful separation is:

* a 64-bit SCLP pose and a separate 64-bit motion word;
* an optional, shared, self-describing binary16 log-polar lookup table;
* sparse component records that reference the shared profile;
* content-addressed operator meanings and fixed-size recipes; and
* random-access display lineage whose prefix does not depend on recipe count.

Numeric policy
--------------

Pose rho uses the closed interval declared by :class:`SCLPKeyLayout64` and is
rejected outside it.  Angles are periodic.  Time is an unsigned 14-bit tick;
wrapping must be explicitly requested.  Signed motion lanes use the symmetric
``[-32767, 32767]`` code domain with round-half-away-from-zero.  Code ``-32768``
is reserved and rejected.  Recipe parameters are rounded once to IEEE-754
binary32; profile values remain binary64.  Negative zero is normalized to
positive zero in canonical output.

Honest limitations
------------------

The formats compact endogenous state and deterministic generation rules.  They
do not compress exogenous meshes, model weights, textures, or observations.
Operator meanings are addressed, not executed by this module.  Generated
instances are frozen render records, never ECS/gameplay/collider identities.
The LUT is canonical after binary16 rounding, but its initial generation uses
the host math library; cross-language deployments still need golden vectors.
Compact recipe/profile addresses are truncated SHA-256 (128 bits), so they are
integrity names rather than an adversarial cryptographic security boundary.
No claim about GPU speed follows from a smaller byte representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
import struct
import unicodedata
from typing import Iterable, Mapping, Sequence

from .substrate_runtime import QuantizedStateKey, SCLPKeyLayout64, SubstrateError


UINT64_MASK = (1 << 64) - 1
ENDIAN_MARKER = 0x01020304

POSE_WIDTHS = (20, 18, 14, 12)
MOTION_LANE_COUNT = 4
SPARSE_COMPONENT_BYTES = 24
OPERATOR_RECORD_BYTES = 40
RECIPE_RECORD_BYTES = 160

MAX_LUT_RESOLUTION = 4096
MAX_COMPONENT_PROFILES = 64
MAX_COMPONENT_RECORDS = 65535
MAX_COMPONENT_PACK_BYTES = 8 * 1024 * 1024
MAX_RECIPE_OPERATORS = 64
MAX_RECIPES = 64
MAX_INSTANCES_PER_RECIPE = 4096
MAX_TOTAL_RECIPE_INSTANCES = 16384
MAX_RECIPE_PACK_BYTES = 128 * 1024

LUT_MAGIC = b"NHLUT001"
COMPONENT_PACK_MAGIC = b"NHSPK001"
RECIPE_PACK_MAGIC = b"NHRCP001"

_LUT_VERSION = 1
_COMPONENT_PACK_VERSION = 1
_RECIPE_PACK_VERSION = 1

_LUT_HEADER = struct.Struct("<8sIHHIddddd")
_COMPONENT_HEADER = struct.Struct("<8sIHHII")
_PROFILE_FIXED = struct.Struct("<HHddddI4dI")
_COMPONENT_RECORD = struct.Struct("<IHHQQ")
_RECIPE_HEADER = struct.Struct("<8sIHHIIII")
_OPERATOR_RECORD = struct.Struct("<HBBI32s")
_RECIPE_RECORD = struct.Struct("<IIQQQ16s16s16s16s8f32s")

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_ZERO_RECIPE_TAIL = bytes(32)

if _LUT_HEADER.size != 60:  # pragma: no cover - import-time format assertion
    raise AssertionError("NHLUT001 header must stay 60 bytes")
if _COMPONENT_HEADER.size != 24:  # pragma: no cover
    raise AssertionError("NHSPK001 header must stay 24 bytes")
if _COMPONENT_RECORD.size != SPARSE_COMPONENT_BYTES:  # pragma: no cover
    raise AssertionError("sparse component record size changed")
if _RECIPE_HEADER.size != 32:  # pragma: no cover
    raise AssertionError("NHRCP001 header must stay 32 bytes")
if _OPERATOR_RECORD.size != OPERATOR_RECORD_BYTES:  # pragma: no cover
    raise AssertionError("operator record size changed")
if _RECIPE_RECORD.size != RECIPE_RECORD_BYTES:  # pragma: no cover
    raise AssertionError("recipe record size changed")


class SubstratePackingError(ValueError):
    """A numeric value or byte stream violates the clean-room packing contract."""


def _uint(value: object, label: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise SubstratePackingError(f"{label} must be an integer from 0 to {maximum}")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SubstratePackingError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SubstratePackingError(f"{label} must be a finite number")
    return 0.0 if result == 0.0 else result


def _binary32(value: object, label: str) -> float:
    number = _finite(value, label)
    try:
        result = struct.unpack("<f", struct.pack("<f", number))[0]
    except (OverflowError, struct.error) as error:
        raise SubstratePackingError(f"{label} does not fit IEEE-754 binary32") from error
    if not math.isfinite(result):
        raise SubstratePackingError(f"{label} does not fit IEEE-754 binary32")
    return 0.0 if result == 0.0 else result


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise SubstratePackingError(f"{label} must be a stable lowercase identifier")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SubstratePackingError(f"{label} must be non-empty text without edge whitespace")
    normalized = unicodedata.normalize("NFC", value)
    encoded = normalized.encode("utf-8")
    if len(encoded) > 65535:
        raise SubstratePackingError(f"{label} exceeds 65535 UTF-8 bytes")
    return normalized


def _length_prefixed(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<H", len(encoded)) + encoded


def _address16(domain: bytes, payload: bytes) -> bytes:
    return hashlib.sha256(domain + payload).digest()[:16]


def semantic_address(label: str, payload: bytes | bytearray | memoryview) -> bytes:
    """Return a domain-separated 128-bit name for an external semantic input."""

    stable_label = _identifier(label, "semantic address label")
    raw = bytes(payload)
    preimage = _length_prefixed(stable_label) + struct.pack("<Q", len(raw)) + raw
    return _address16(b"nhdf-edge/semantic-address/v1\0", preimage)


@dataclass(frozen=True, slots=True)
class DecodedPose64:
    rho: float
    theta: float
    time_tick: int
    heading: float
    codes: QuantizedStateKey


@dataclass(frozen=True, slots=True)
class PackedPose64:
    """One contiguous SCLP pose word: rho/theta/time/heading = 20/18/14/12."""

    word: int

    def __post_init__(self) -> None:
        _uint(self.word, "packed pose", UINT64_MASK)

    @classmethod
    def from_values(
        cls,
        rho: float,
        theta: float,
        time_tick: int,
        heading: float,
        *,
        rho_min: float = -10.0,
        rho_max: float = 10.0,
        wrap_time: bool = False,
    ) -> "PackedPose64":
        low = _finite(rho_min, "rho_min")
        high = _finite(rho_max, "rho_max")
        rho_value = _finite(rho, "rho")
        if low >= high:
            raise SubstratePackingError("rho_min must be less than rho_max")
        if not low <= rho_value <= high:
            raise SubstratePackingError("rho lies outside the declared closed interval")
        if isinstance(time_tick, bool) or not isinstance(time_tick, int):
            raise SubstratePackingError("time_tick must be an integer")
        time_limit = 1 << POSE_WIDTHS[2]
        if not wrap_time and not 0 <= time_tick < time_limit:
            raise SubstratePackingError("time_tick does not fit 14 bits; request wrapping explicitly")
        layout = SCLPKeyLayout64(rho_min=low, rho_max=high)
        try:
            state = layout.quantize(
                rho_value,
                _finite(theta, "theta"),
                time_tick % time_limit,
                _finite(heading, "heading"),
            )
            return cls(layout.pack_contiguous(state))
        except SubstrateError as error:
            raise SubstratePackingError(str(error)) from error

    @classmethod
    def from_codes(cls, rho: int, theta: int, time_tick: int, heading: int) -> "PackedPose64":
        widths = dict(zip(("rho", "theta", "time", "phi"), POSE_WIDTHS))
        values = {"rho": rho, "theta": theta, "time": time_tick, "phi": heading}
        for name, width in widths.items():
            _uint(values[name], f"{name} code", (1 << width) - 1)
        state = QuantizedStateKey(rho=rho, theta=theta, time=time_tick, phi=heading)
        try:
            return cls(SCLPKeyLayout64().pack_contiguous(state))
        except SubstrateError as error:  # pragma: no cover - guarded above
            raise SubstratePackingError(str(error)) from error

    @property
    def codes(self) -> QuantizedStateKey:
        try:
            return SCLPKeyLayout64().unpack_contiguous(self.word)
        except SubstrateError as error:  # pragma: no cover - word already checked
            raise SubstratePackingError(str(error)) from error

    def decode(self, *, rho_min: float = -10.0, rho_max: float = 10.0) -> DecodedPose64:
        low = _finite(rho_min, "rho_min")
        high = _finite(rho_max, "rho_max")
        if low >= high:
            raise SubstratePackingError("rho_min must be less than rho_max")
        state = self.codes
        rho_max_code = (1 << POSE_WIDTHS[0]) - 1
        theta_max_code = (1 << POSE_WIDTHS[1]) - 1
        heading_max_code = (1 << POSE_WIDTHS[3]) - 1
        rho = low + (high - low) * (state.rho / rho_max_code)
        theta = -math.pi + 2.0 * math.pi * (state.theta / theta_max_code)
        heading = 2.0 * math.pi * (state.phi / heading_max_code)
        return DecodedPose64(rho, theta, state.time, heading, state)

    def to_bytes(self) -> bytes:
        return struct.pack("<Q", self.word)

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> "PackedPose64":
        raw = bytes(data)
        if len(raw) != 8:
            raise SubstratePackingError("packed pose bytes must contain exactly 8 bytes")
        return cls(struct.unpack("<Q", raw)[0])


@dataclass(frozen=True, slots=True)
class MotionBounds:
    rho_velocity: float
    theta_velocity: float
    rho_acceleration: float
    theta_acceleration: float

    def __post_init__(self) -> None:
        for name in (
            "rho_velocity",
            "theta_velocity",
            "rho_acceleration",
            "theta_acceleration",
        ):
            value = _finite(getattr(self, name), f"{name} bound")
            if value <= 0:
                raise SubstratePackingError(f"{name} bound must be positive")
            object.__setattr__(self, name, value)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (
            self.rho_velocity,
            self.theta_velocity,
            self.rho_acceleration,
            self.theta_acceleration,
        )


@dataclass(frozen=True, slots=True)
class MotionValues:
    rho_velocity: float
    theta_velocity: float
    rho_acceleration: float
    theta_acceleration: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (
            self.rho_velocity,
            self.theta_velocity,
            self.rho_acceleration,
            self.theta_acceleration,
        )


def _round_half_away_from_zero(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0.0 else math.ceil(value - 0.5)


def _encode_signed16(value: object, bound: float, label: str) -> int:
    number = _finite(value, label)
    if not -bound <= number <= bound:
        raise SubstratePackingError(f"{label} lies outside its declared symmetric bound")
    signed = _round_half_away_from_zero((number / bound) * 32767.0)
    if not -32767 <= signed <= 32767:  # pragma: no cover - guarded by bound
        raise SubstratePackingError(f"{label} quantized outside signed16 policy")
    return signed & 0xFFFF


def _decode_signed16(code: int, bound: float, label: str) -> float:
    _uint(code, f"{label} code", 0xFFFF)
    if code == 0x8000:
        raise SubstratePackingError(f"{label} uses reserved signed16 code 0x8000")
    signed = code - 0x10000 if code & 0x8000 else code
    return (signed / 32767.0) * bound


@dataclass(frozen=True, slots=True)
class PackedMotion64:
    """Four symmetric signed16 motion lanes in one unsigned 64-bit word."""

    word: int

    def __post_init__(self) -> None:
        _uint(self.word, "packed motion", UINT64_MASK)
        for index, code in enumerate(self.codes):
            if code == 0x8000:
                raise SubstratePackingError(f"motion lane {index} uses reserved code 0x8000")

    @classmethod
    def from_values(
        cls,
        rho_velocity: float,
        theta_velocity: float,
        rho_acceleration: float,
        theta_acceleration: float,
        *,
        bounds: MotionBounds,
    ) -> "PackedMotion64":
        values = (rho_velocity, theta_velocity, rho_acceleration, theta_acceleration)
        codes = tuple(
            _encode_signed16(value, bound, f"motion lane {index}")
            for index, (value, bound) in enumerate(zip(values, bounds.as_tuple()))
        )
        word = 0
        for code in codes:
            word = (word << 16) | code
        return cls(word)

    @property
    def codes(self) -> tuple[int, int, int, int]:
        return tuple((self.word >> shift) & 0xFFFF for shift in (48, 32, 16, 0))  # type: ignore[return-value]

    def decode(self, *, bounds: MotionBounds) -> MotionValues:
        values = tuple(
            _decode_signed16(code, bound, f"motion lane {index}")
            for index, (code, bound) in enumerate(zip(self.codes, bounds.as_tuple()))
        )
        return MotionValues(*values)

    def to_bytes(self) -> bytes:
        return struct.pack("<Q", self.word)

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> "PackedMotion64":
        raw = bytes(data)
        if len(raw) != 8:
            raise SubstratePackingError("packed motion bytes must contain exactly 8 bytes")
        return cls(struct.unpack("<Q", raw)[0])


@dataclass(frozen=True, slots=True)
class LogPolarProfile:
    reference_radius: float = 1.0
    rho_min: float = -10.0
    rho_max: float = 10.0
    core_radius: float = 1.0e-9
    resolution: int = 256

    def __post_init__(self) -> None:
        reference = _finite(self.reference_radius, "reference_radius")
        low = _finite(self.rho_min, "rho_min")
        high = _finite(self.rho_max, "rho_max")
        core = _finite(self.core_radius, "core_radius")
        if reference <= 0:
            raise SubstratePackingError("reference_radius must be positive")
        if low >= high:
            raise SubstratePackingError("rho_min must be less than rho_max")
        if core <= 0:
            raise SubstratePackingError("core_radius must be positive")
        _uint(self.resolution, "LUT resolution", MAX_LUT_RESOLUTION)
        if self.resolution < 16:
            raise SubstratePackingError("LUT resolution must be at least 16")
        try:
            endpoint = reference * math.exp(high)
        except OverflowError as error:
            raise SubstratePackingError("rho_max overflows the radius domain") from error
        if not math.isfinite(endpoint):
            raise SubstratePackingError("rho_max overflows the radius domain")
        object.__setattr__(self, "reference_radius", reference)
        object.__setattr__(self, "rho_min", low)
        object.__setattr__(self, "rho_max", high)
        object.__setattr__(self, "core_radius", core)


@dataclass(frozen=True, slots=True)
class SharedLogPolarLUT:
    """Canonical binary16 sine/cosine/radius mantissas for one profile."""

    profile: LogPolarProfile
    radius_scale: float
    sine: tuple[float, ...]
    cosine: tuple[float, ...]
    radius_mantissas: tuple[float, ...]

    @staticmethod
    def _half_round(values: Sequence[float], label: str) -> tuple[float, ...]:
        try:
            packed = struct.pack(f"<{len(values)}e", *values)
            rounded = struct.unpack(f"<{len(values)}e", packed)
            return tuple(0.0 if value == 0.0 else value for value in rounded)
        except (OverflowError, struct.error) as error:
            raise SubstratePackingError(f"{label} cannot be represented as binary16") from error

    @classmethod
    def generate(cls, profile: LogPolarProfile) -> "SharedLogPolarLUT":
        resolution = profile.resolution
        angles = tuple((2.0 * math.pi * index) / resolution for index in range(resolution))
        raw_sine = tuple(math.sin(angle) for angle in angles)
        raw_cosine = tuple(math.cos(angle) for angle in angles)
        raw_radii = tuple(
            profile.reference_radius
            * math.exp(
                profile.rho_min
                + (profile.rho_max - profile.rho_min) * index / (resolution - 1)
            )
            for index in range(resolution)
        )
        radius_scale = max(1.0, max(raw_radii) / 60000.0)
        sine = cls._half_round(raw_sine, "LUT sine")
        cosine = cls._half_round(raw_cosine, "LUT cosine")
        radius_mantissas = cls._half_round(
            tuple(radius / radius_scale for radius in raw_radii), "LUT radius"
        )
        return cls(profile, radius_scale, sine, cosine, radius_mantissas)

    def __post_init__(self) -> None:
        scale = _finite(self.radius_scale, "LUT radius scale")
        if scale <= 0:
            raise SubstratePackingError("LUT radius scale must be positive")
        resolution = self.profile.resolution
        for name in ("sine", "cosine", "radius_mantissas"):
            values = tuple(
                0.0 if value == 0.0 else value for value in getattr(self, name)
            )
            if len(values) != resolution:
                raise SubstratePackingError(f"LUT {name} length does not match resolution")
            if any(not math.isfinite(value) for value in values):
                raise SubstratePackingError(f"LUT {name} contains a non-finite value")
            object.__setattr__(self, name, values)
        if any(value <= 0 for value in self.radius_mantissas):
            raise SubstratePackingError("LUT radius mantissas must be positive")
        object.__setattr__(self, "radius_scale", scale)

    def _unchecked_bytes(self) -> bytes:
        header = _LUT_HEADER.pack(
            LUT_MAGIC,
            ENDIAN_MARKER,
            _LUT_VERSION,
            0,
            self.profile.resolution,
            self.profile.reference_radius,
            self.profile.rho_min,
            self.profile.rho_max,
            self.profile.core_radius,
            self.radius_scale,
        )
        values = self.sine + self.cosine + self.radius_mantissas
        try:
            return header + struct.pack(f"<{len(values)}e", *values)
        except (OverflowError, struct.error) as error:
            raise SubstratePackingError("LUT values do not fit canonical binary16") from error

    def _require_canonical(self) -> bytes:
        """Return bytes only when this object is the unique table for its profile."""

        result = self._unchecked_bytes()
        canonical = type(self).generate(self.profile)._unchecked_bytes()
        if result != canonical:
            raise SubstratePackingError("LUT object is not the canonical table for its profile")
        return result

    def to_bytes(self) -> bytes:
        return self._require_canonical()

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> "SharedLogPolarLUT":
        raw = bytes(data)
        if len(raw) < _LUT_HEADER.size:
            raise SubstratePackingError("LUT data is truncated")
        (
            magic,
            endian,
            version,
            reserved,
            resolution,
            reference,
            rho_min,
            rho_max,
            core,
            radius_scale,
        ) = _LUT_HEADER.unpack_from(raw)
        if magic != LUT_MAGIC:
            raise SubstratePackingError("LUT magic mismatch")
        if endian != ENDIAN_MARKER:
            raise SubstratePackingError("LUT endian marker mismatch")
        if version != _LUT_VERSION:
            raise SubstratePackingError("unsupported LUT version")
        if reserved != 0:
            raise SubstratePackingError("LUT reserved field is nonzero")
        profile = LogPolarProfile(reference, rho_min, rho_max, core, resolution)
        expected_size = _LUT_HEADER.size + resolution * 3 * 2
        if len(raw) < expected_size:
            raise SubstratePackingError("LUT payload is truncated")
        if len(raw) > expected_size:
            raise SubstratePackingError("LUT contains trailing bytes")
        # Decode once so malformed binary16 payloads still fail before the
        # bytewise canonical comparison below.
        struct.unpack(f"<{resolution * 3}e", raw[_LUT_HEADER.size :])
        canonical = cls.generate(profile)
        if raw != canonical._unchecked_bytes():
            raise SubstratePackingError("LUT bytes are valid binary16 but not canonical")
        return canonical

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def direction(self, theta: float) -> tuple[float, float]:
        self._require_canonical()
        angle = _finite(theta, "theta") % (2.0 * math.pi)
        coordinate = angle * self.profile.resolution / (2.0 * math.pi)
        low = int(math.floor(coordinate)) % self.profile.resolution
        high = (low + 1) % self.profile.resolution
        fraction = coordinate - math.floor(coordinate)
        sine = self.sine[low] + (self.sine[high] - self.sine[low]) * fraction
        cosine = self.cosine[low] + (self.cosine[high] - self.cosine[low]) * fraction
        length = math.hypot(sine, cosine)
        if length <= 1.0e-12:
            raise SubstratePackingError("interpolated LUT direction is degenerate")
        # Cartesian polar direction is (cos(theta), sin(theta)).  The wire
        # lanes remain sine then cosine; only their geometric interpretation
        # is ordered here.
        return cosine / length, sine / length

    def radius(self, rho: float) -> float:
        self._require_canonical()
        value = _finite(rho, "rho")
        if not self.profile.rho_min <= value <= self.profile.rho_max:
            raise SubstratePackingError("rho lies outside the LUT profile")
        coordinate = (
            (value - self.profile.rho_min)
            * (self.profile.resolution - 1)
            / (self.profile.rho_max - self.profile.rho_min)
        )
        low = int(math.floor(coordinate))
        high = min(self.profile.resolution - 1, low + 1)
        fraction = coordinate - low
        mantissa = self.radius_mantissas[low] + (
            self.radius_mantissas[high] - self.radius_mantissas[low]
        ) * fraction
        return mantissa * self.radius_scale


@dataclass(frozen=True, slots=True)
class ComponentProfile:
    profile_id: str
    polar: LogPolarProfile
    motion_bounds: MotionBounds
    lut: SharedLogPolarLUT | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _identifier(self.profile_id, "profile_id"))
        if self.lut is not None and self.lut.profile != self.polar:
            raise SubstratePackingError("shared LUT profile does not match component profile")

    def _body_bytes(self) -> bytes:
        lut_bytes = b"" if self.lut is None else self.lut.to_bytes()
        flags = 1 if lut_bytes else 0
        return _PROFILE_FIXED.pack(
            flags,
            0,
            self.polar.reference_radius,
            self.polar.rho_min,
            self.polar.rho_max,
            self.polar.core_radius,
            self.polar.resolution,
            *self.motion_bounds.as_tuple(),
            len(lut_bytes),
        ) + lut_bytes

    @property
    def semantics_address(self) -> bytes:
        payload = _length_prefixed(self.profile_id) + self._body_bytes()
        return _address16(b"nhdf-edge/component-profile/v1\0", payload)


@dataclass(frozen=True, slots=True)
class PackedComponentRecord:
    node_index: int
    profile_id: str
    pose: PackedPose64
    motion: PackedMotion64

    def __post_init__(self) -> None:
        _uint(self.node_index, "component node_index", 0xFFFFFFFF)
        object.__setattr__(self, "profile_id", _identifier(self.profile_id, "profile_id"))


@dataclass(frozen=True, slots=True)
class SparseComponentPack:
    """Canonical profiles plus node-index-sorted 24-byte component records."""

    profiles: tuple[ComponentProfile, ...]
    records: tuple[PackedComponentRecord, ...]

    @classmethod
    def build(
        cls,
        profiles: Iterable[ComponentProfile],
        records: Iterable[PackedComponentRecord],
    ) -> "SparseComponentPack":
        ordered_profiles = tuple(sorted(profiles, key=lambda item: item.profile_id.encode("ascii")))
        ordered_records = tuple(sorted(records, key=lambda item: item.node_index))
        return cls(ordered_profiles, ordered_records)

    def __post_init__(self) -> None:
        if not 1 <= len(self.profiles) <= MAX_COMPONENT_PROFILES:
            raise SubstratePackingError("component pack must contain 1..64 profiles")
        if not 1 <= len(self.records) <= MAX_COMPONENT_RECORDS:
            raise SubstratePackingError("component pack must contain at least one bounded record")
        ids = tuple(profile.profile_id.encode("ascii") for profile in self.profiles)
        if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
            raise SubstratePackingError("component profiles must be unique and canonically sorted")
        node_indices = tuple(record.node_index for record in self.records)
        if node_indices != tuple(sorted(node_indices)) or len(set(node_indices)) != len(node_indices):
            raise SubstratePackingError("component records must be unique and sorted by node_index")
        known = {profile.profile_id for profile in self.profiles}
        used = {record.profile_id for record in self.records}
        if not used <= known:
            raise SubstratePackingError("component record references an unknown profile")
        if used != known:
            raise SubstratePackingError("unused component profiles are noncanonical")

    def to_bytes(self) -> bytes:
        output = bytearray(
            _COMPONENT_HEADER.pack(
                COMPONENT_PACK_MAGIC,
                ENDIAN_MARKER,
                _COMPONENT_PACK_VERSION,
                0,
                len(self.profiles),
                len(self.records),
            )
        )
        for profile in self.profiles:
            output.extend(_length_prefixed(profile.profile_id))
            output.extend(profile._body_bytes())
        profile_indices = {profile.profile_id: index for index, profile in enumerate(self.profiles)}
        for record in self.records:
            output.extend(
                _COMPONENT_RECORD.pack(
                    record.node_index,
                    profile_indices[record.profile_id],
                    0,
                    record.pose.word,
                    record.motion.word,
                )
            )
        result = bytes(output)
        if len(result) > MAX_COMPONENT_PACK_BYTES:
            raise SubstratePackingError("component pack exceeds its byte limit")
        return result

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> "SparseComponentPack":
        raw = bytes(data)
        if len(raw) > MAX_COMPONENT_PACK_BYTES:
            raise SubstratePackingError("component pack exceeds its byte limit")
        if len(raw) < _COMPONENT_HEADER.size:
            raise SubstratePackingError("component pack is truncated")
        magic, endian, version, reserved, profile_count, record_count = _COMPONENT_HEADER.unpack_from(raw)
        if magic != COMPONENT_PACK_MAGIC:
            raise SubstratePackingError("component pack magic mismatch")
        if endian != ENDIAN_MARKER:
            raise SubstratePackingError("component pack endian marker mismatch")
        if version != _COMPONENT_PACK_VERSION:
            raise SubstratePackingError("unsupported component pack version")
        if reserved != 0:
            raise SubstratePackingError("component header reserved field is nonzero")
        if not 1 <= profile_count <= MAX_COMPONENT_PROFILES:
            raise SubstratePackingError("component profile count is invalid")
        if not 1 <= record_count <= MAX_COMPONENT_RECORDS:
            raise SubstratePackingError("component record count is invalid")
        offset = _COMPONENT_HEADER.size
        profiles: list[ComponentProfile] = []
        for _ in range(profile_count):
            if offset + 2 > len(raw):
                raise SubstratePackingError("component profile id is truncated")
            id_length = struct.unpack_from("<H", raw, offset)[0]
            offset += 2
            if id_length == 0 or offset + id_length + _PROFILE_FIXED.size > len(raw):
                raise SubstratePackingError("component profile is truncated")
            try:
                profile_id = raw[offset : offset + id_length].decode("ascii")
            except UnicodeDecodeError as error:
                raise SubstratePackingError("component profile id is not canonical ASCII") from error
            offset += id_length
            (
                flags,
                profile_reserved,
                reference,
                rho_min,
                rho_max,
                core,
                resolution,
                rho_velocity,
                theta_velocity,
                rho_acceleration,
                theta_acceleration,
                lut_length,
            ) = _PROFILE_FIXED.unpack_from(raw, offset)
            offset += _PROFILE_FIXED.size
            if profile_reserved != 0:
                raise SubstratePackingError("component profile reserved field is nonzero")
            if flags & ~1:
                raise SubstratePackingError("component profile flags contain unsupported bits")
            if bool(flags & 1) != bool(lut_length):
                raise SubstratePackingError("component LUT flag and length disagree")
            if offset + lut_length > len(raw):
                raise SubstratePackingError("component LUT bytes are truncated")
            polar = LogPolarProfile(reference, rho_min, rho_max, core, resolution)
            bounds = MotionBounds(
                rho_velocity, theta_velocity, rho_acceleration, theta_acceleration
            )
            lut = (
                None
                if lut_length == 0
                else SharedLogPolarLUT.from_bytes(raw[offset : offset + lut_length])
            )
            offset += lut_length
            profiles.append(ComponentProfile(profile_id, polar, bounds, lut))
        records: list[PackedComponentRecord] = []
        for _ in range(record_count):
            if offset + _COMPONENT_RECORD.size > len(raw):
                raise SubstratePackingError("component record is truncated")
            node_index, profile_index, record_reserved, pose_word, motion_word = _COMPONENT_RECORD.unpack_from(
                raw, offset
            )
            offset += _COMPONENT_RECORD.size
            if record_reserved != 0:
                raise SubstratePackingError("component record reserved field is nonzero")
            if profile_index >= len(profiles):
                raise SubstratePackingError("component record profile index is invalid")
            records.append(
                PackedComponentRecord(
                    node_index,
                    profiles[profile_index].profile_id,
                    PackedPose64(pose_word),
                    PackedMotion64(motion_word),
                )
            )
        if offset < len(raw):
            raise SubstratePackingError("component pack contains trailing bytes")
        result = cls(tuple(profiles), tuple(records))
        if result.to_bytes() != raw:
            raise SubstratePackingError("component pack is structurally valid but noncanonical")
        return result

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class OperatorMeaning:
    code: int
    slot: int
    arity: int
    name: str
    meaning: str

    def __post_init__(self) -> None:
        _uint(self.code, "operator code", 0xFFFF)
        if self.code == 0:
            raise SubstratePackingError("operator code zero is reserved")
        _uint(self.slot, "operator slot", MAX_RECIPE_OPERATORS - 1)
        _uint(self.arity, "operator arity", 0xFF)
        object.__setattr__(self, "name", _identifier(self.name, "operator name"))
        object.__setattr__(self, "meaning", _text(self.meaning, "operator meaning"))

    @property
    def address(self) -> bytes:
        payload = (
            struct.pack("<HBB", self.code, self.slot, self.arity)
            + _length_prefixed(self.name)
            + _length_prefixed(self.meaning)
        )
        return hashlib.sha256(b"nhdf-edge/operator-meaning/v1\0" + payload).digest()


@dataclass(frozen=True, slots=True)
class FixedRecipe:
    """One fixed 160-byte recipe record before its pack-level operator table."""

    prototype_index: int
    instance_count: int
    root_seed: int
    recipe_seed: int
    operators: tuple[OperatorMeaning, ...]
    profile_address: bytes
    prototype_address: bytes
    parameters: tuple[float, float, float, float, float, float, float, float]

    def __post_init__(self) -> None:
        _uint(self.prototype_index, "recipe prototype_index", 0xFFFFFFFF)
        if isinstance(self.instance_count, bool) or not isinstance(self.instance_count, int):
            raise SubstratePackingError("recipe instance_count must be an integer")
        if not 2 <= self.instance_count <= MAX_INSTANCES_PER_RECIPE:
            raise SubstratePackingError("recipe instance_count is outside its bounded domain")
        _uint(self.root_seed, "recipe root_seed", UINT64_MASK)
        _uint(self.recipe_seed, "recipe seed", UINT64_MASK)
        ordered = tuple(sorted(self.operators, key=lambda item: item.code))
        if not ordered:
            raise SubstratePackingError("recipe needs at least one operator meaning")
        if len(ordered) > MAX_RECIPE_OPERATORS:
            raise SubstratePackingError("recipe contains too many operators")
        if len({item.code for item in ordered}) != len(ordered):
            raise SubstratePackingError("recipe operator codes must be unique")
        if len({item.slot for item in ordered}) != len(ordered):
            raise SubstratePackingError("recipe operator slots must be unique")
        object.__setattr__(self, "operators", ordered)
        for name in ("profile_address", "prototype_address"):
            value = bytes(getattr(self, name))
            if len(value) != 16:
                raise SubstratePackingError(f"{name} must contain exactly 16 bytes")
            object.__setattr__(self, name, value)
        if len(self.parameters) != 8:
            raise SubstratePackingError("recipe parameters must contain exactly eight lanes")
        canonical = tuple(
            _binary32(value, f"recipe parameter {index}")
            for index, value in enumerate(self.parameters)
        )
        object.__setattr__(self, "parameters", canonical)

    @property
    def operator_mask(self) -> int:
        mask = 0
        for operator in self.operators:
            mask |= 1 << operator.slot
        return mask

    def _lineage_preimage(self) -> bytes:
        output = bytearray(
            struct.pack(
                "<IQQQ",
                self.prototype_index,
                self.root_seed,
                self.recipe_seed,
                self.operator_mask,
            )
        )
        output.extend(self.profile_address)
        output.extend(self.prototype_address)
        output.extend(struct.pack("<8f", *self.parameters))
        for operator in self.operators:
            output.extend(operator.address)
        return bytes(output)

    @property
    def lineage_namespace(self) -> bytes:
        return _address16(b"nhdf-edge/display-lineage/v1\0", self._lineage_preimage())

    @property
    def content_address(self) -> bytes:
        payload = self.lineage_namespace + struct.pack("<I", self.instance_count)
        return _address16(b"nhdf-edge/recipe-content/v1\0", payload)


def _operator_registry(
    operators: Iterable[OperatorMeaning] | Mapping[int, OperatorMeaning],
) -> dict[int, OperatorMeaning]:
    values = operators.values() if isinstance(operators, Mapping) else operators
    result: dict[int, OperatorMeaning] = {}
    occupied_slots: set[int] = set()
    for operator in values:
        if not isinstance(operator, OperatorMeaning):
            raise SubstratePackingError("operator registry contains a non-operator value")
        if operator.code in result:
            raise SubstratePackingError("operator registry contains a duplicate code")
        if operator.slot in occupied_slots:
            raise SubstratePackingError("operator registry contains a duplicate slot")
        result[operator.code] = operator
        occupied_slots.add(operator.slot)
    return result


@dataclass(frozen=True, slots=True)
class RecipePack:
    operators: tuple[OperatorMeaning, ...]
    recipes: tuple[FixedRecipe, ...]

    @classmethod
    def build(cls, recipes: Iterable[FixedRecipe]) -> "RecipePack":
        ordered_recipes = tuple(sorted(recipes, key=lambda item: item.prototype_index))
        by_code: dict[int, OperatorMeaning] = {}
        for recipe in ordered_recipes:
            for operator in recipe.operators:
                previous = by_code.get(operator.code)
                if previous is not None and previous != operator:
                    raise SubstratePackingError("one operator code has conflicting meanings")
                by_code[operator.code] = operator
        return cls(tuple(sorted(by_code.values(), key=lambda item: item.code)), ordered_recipes)

    def __post_init__(self) -> None:
        if not 1 <= len(self.operators) <= MAX_RECIPE_OPERATORS:
            raise SubstratePackingError("recipe pack operator count is invalid")
        if not 1 <= len(self.recipes) <= MAX_RECIPES:
            raise SubstratePackingError("recipe pack recipe count is invalid")
        codes = tuple(operator.code for operator in self.operators)
        if codes != tuple(sorted(codes)) or len(set(codes)) != len(codes):
            raise SubstratePackingError("recipe operators must be unique and sorted by code")
        if len({operator.slot for operator in self.operators}) != len(self.operators):
            raise SubstratePackingError("recipe operator slots must be unique")
        prototypes = tuple(recipe.prototype_index for recipe in self.recipes)
        if prototypes != tuple(sorted(prototypes)) or len(set(prototypes)) != len(prototypes):
            raise SubstratePackingError("recipes must be unique and sorted by prototype_index")
        registry = {operator.code: operator for operator in self.operators}
        used_codes: set[int] = set()
        total = 0
        for recipe in self.recipes:
            total += recipe.instance_count
            for operator in recipe.operators:
                if registry.get(operator.code) != operator:
                    raise SubstratePackingError("recipe references an absent or conflicting operator")
                used_codes.add(operator.code)
        if used_codes != set(registry):
            raise SubstratePackingError("unused operator records are noncanonical")
        if total > MAX_TOTAL_RECIPE_INSTANCES:
            raise SubstratePackingError("recipe pack exceeds its total instance bound")

    def to_bytes(self) -> bytes:
        total_instances = sum(recipe.instance_count for recipe in self.recipes)
        output = bytearray(
            _RECIPE_HEADER.pack(
                RECIPE_PACK_MAGIC,
                ENDIAN_MARKER,
                _RECIPE_PACK_VERSION,
                0,
                len(self.operators),
                len(self.recipes),
                total_instances,
                0,
            )
        )
        for operator in self.operators:
            output.extend(
                _OPERATOR_RECORD.pack(
                    operator.code,
                    operator.slot,
                    operator.arity,
                    0,
                    operator.address,
                )
            )
        for recipe in self.recipes:
            output.extend(
                _RECIPE_RECORD.pack(
                    recipe.prototype_index,
                    recipe.instance_count,
                    recipe.root_seed,
                    recipe.recipe_seed,
                    recipe.operator_mask,
                    recipe.lineage_namespace,
                    recipe.content_address,
                    recipe.profile_address,
                    recipe.prototype_address,
                    *recipe.parameters,
                    _ZERO_RECIPE_TAIL,
                )
            )
        result = bytes(output)
        if len(result) > MAX_RECIPE_PACK_BYTES:
            raise SubstratePackingError("recipe pack exceeds its byte limit")
        return result

    @classmethod
    def from_bytes(
        cls,
        data: bytes | bytearray | memoryview,
        *,
        operator_registry: Iterable[OperatorMeaning] | Mapping[int, OperatorMeaning],
    ) -> "RecipePack":
        raw = bytes(data)
        if len(raw) > MAX_RECIPE_PACK_BYTES:
            raise SubstratePackingError("recipe pack exceeds its byte limit")
        if len(raw) < _RECIPE_HEADER.size:
            raise SubstratePackingError("recipe pack is truncated")
        magic, endian, version, reserved, operator_count, recipe_count, total_instances, flags = (
            _RECIPE_HEADER.unpack_from(raw)
        )
        if magic != RECIPE_PACK_MAGIC:
            raise SubstratePackingError("recipe pack magic mismatch")
        if endian != ENDIAN_MARKER:
            raise SubstratePackingError("recipe pack endian marker mismatch")
        if version != _RECIPE_PACK_VERSION:
            raise SubstratePackingError("unsupported recipe pack version")
        if reserved != 0 or flags != 0:
            raise SubstratePackingError("recipe pack reserved fields are nonzero")
        if not 1 <= operator_count <= MAX_RECIPE_OPERATORS:
            raise SubstratePackingError("recipe operator count is invalid")
        if not 1 <= recipe_count <= MAX_RECIPES:
            raise SubstratePackingError("recipe count is invalid")
        expected_size = (
            _RECIPE_HEADER.size
            + operator_count * _OPERATOR_RECORD.size
            + recipe_count * _RECIPE_RECORD.size
        )
        if len(raw) < expected_size:
            raise SubstratePackingError("recipe pack records are truncated")
        if len(raw) > expected_size:
            raise SubstratePackingError("recipe pack contains trailing bytes")
        registry = _operator_registry(operator_registry)
        operators: list[OperatorMeaning] = []
        offset = _RECIPE_HEADER.size
        for _ in range(operator_count):
            code, slot, arity, operator_flags, address = _OPERATOR_RECORD.unpack_from(raw, offset)
            offset += _OPERATOR_RECORD.size
            if operator_flags != 0:
                raise SubstratePackingError("operator reserved flags are nonzero")
            operator = registry.get(code)
            if operator is None:
                raise SubstratePackingError(f"operator code 0x{code:04x} is not in the registry")
            if operator.slot != slot or operator.arity != arity or operator.address != address:
                raise SubstratePackingError(f"operator code 0x{code:04x} meaning mismatch")
            operators.append(operator)
        by_slot = {operator.slot: operator for operator in operators}
        recipes: list[FixedRecipe] = []
        counted_instances = 0
        for _ in range(recipe_count):
            unpacked = _RECIPE_RECORD.unpack_from(raw, offset)
            offset += _RECIPE_RECORD.size
            (
                prototype,
                instance_count,
                root_seed,
                recipe_seed,
                operator_mask,
                lineage_namespace,
                content_address,
                profile_address,
                prototype_address,
                *tail,
            ) = unpacked
            parameters = tuple(tail[:8])
            recipe_reserved = tail[8]
            if recipe_reserved != _ZERO_RECIPE_TAIL:
                raise SubstratePackingError("recipe reserved tail is nonzero")
            active = tuple(
                by_slot[slot]
                for slot in range(MAX_RECIPE_OPERATORS)
                if operator_mask & (1 << slot) and slot in by_slot
            )
            known_mask = sum(1 << slot for slot in by_slot)
            if operator_mask == 0 or operator_mask & ~known_mask:
                raise SubstratePackingError("recipe operator mask references unknown meanings")
            recipe = FixedRecipe(
                prototype,
                instance_count,
                root_seed,
                recipe_seed,
                active,
                profile_address,
                prototype_address,
                parameters,  # type: ignore[arg-type]
            )
            if recipe.lineage_namespace != lineage_namespace:
                raise SubstratePackingError("recipe lineage namespace mismatch")
            if recipe.content_address != content_address:
                raise SubstratePackingError("recipe content address mismatch")
            recipes.append(recipe)
            counted_instances += instance_count
        if counted_instances != total_instances:
            raise SubstratePackingError("recipe total instance count does not match records")
        result = cls(tuple(operators), tuple(recipes))
        if result.to_bytes() != raw:
            raise SubstratePackingError("recipe pack is structurally valid but noncanonical")
        return result

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()


def splitmix64(value: int) -> int:
    """A fixed unsigned-64 permutation used for stateless lineage lanes."""

    x = (_uint(value, "SplitMix64 input", UINT64_MASK) + 0x9E3779B97F4A7C15) & UINT64_MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
    return (x ^ (x >> 31)) & UINT64_MASK


def combine_seed(seed: int, value: int) -> int:
    """Domain-combine two uint64 values without retaining mutable RNG state."""

    left = _uint(seed, "seed", UINT64_MASK)
    right = _uint(value, "seed value", UINT64_MASK)
    mixed = (splitmix64(right) + 0x9E3779B97F4A7C15 + ((left << 6) & UINT64_MASK) + (left >> 2)) & UINT64_MASK
    return splitmix64(left ^ mixed)


def stable_lineage_id(recipe: FixedRecipe, ordinal: int) -> int:
    """Return one random-access identity; no previous ordinal is consulted."""

    if isinstance(ordinal, bool) or not isinstance(ordinal, int):
        raise SubstratePackingError("display ordinal must be an integer")
    if not 0 <= ordinal < recipe.instance_count:
        raise SubstratePackingError("display ordinal lies outside the recipe")
    session = combine_seed(recipe.root_seed, recipe.recipe_seed)
    namespace = combine_seed(
        int.from_bytes(recipe.lineage_namespace[:8], "little"),
        int.from_bytes(recipe.lineage_namespace[8:], "little"),
    )
    return combine_seed(combine_seed(session, namespace), ordinal)


@dataclass(frozen=True, slots=True)
class GeneratedDisplayInstance:
    """A frozen render-only derivative with deliberately no ECS/gameplay fields."""

    prototype_index: int
    ordinal: int
    lineage_id: int
    pose: PackedPose64
    motion: PackedMotion64
    phase_u16: int
    variation_u16: int

    def __post_init__(self) -> None:
        _uint(self.prototype_index, "display prototype_index", 0xFFFFFFFF)
        _uint(self.ordinal, "display ordinal", 0xFFFFFFFF)
        _uint(self.lineage_id, "display lineage_id", UINT64_MASK)
        _uint(self.phase_u16, "display phase", 0xFFFF)
        _uint(self.variation_u16, "display variation", 0xFFFF)

    @property
    def render_only(self) -> bool:
        return True


def generated_display_instances(
    recipe: FixedRecipe,
    *,
    prototype_pose: PackedPose64,
    prototype_motion: PackedMotion64,
    count: int | None = None,
) -> tuple[GeneratedDisplayInstance, ...]:
    """Derive copies 1..N-1 without materializing gameplay/ECS identities.

    ``count`` may request a deterministic prefix for a renderer budget, but it
    cannot exceed the authored recipe.  This function establishes lineage and
    two generic render variation lanes; it intentionally does not interpret or
    execute the addressed operator meanings.
    """

    resolved_count = recipe.instance_count if count is None else count
    if isinstance(resolved_count, bool) or not isinstance(resolved_count, int):
        raise SubstratePackingError("generated display count must be an integer")
    if not 1 <= resolved_count <= recipe.instance_count:
        raise SubstratePackingError("generated display count lies outside the recipe")
    output: list[GeneratedDisplayInstance] = []
    for ordinal in range(1, resolved_count):
        lineage = stable_lineage_id(recipe, ordinal)
        phase = splitmix64(combine_seed(lineage, 0)) >> 48
        variation = splitmix64(combine_seed(lineage, 1)) >> 48
        output.append(
            GeneratedDisplayInstance(
                prototype_index=recipe.prototype_index,
                ordinal=ordinal,
                lineage_id=lineage,
                pose=prototype_pose,
                motion=prototype_motion,
                phase_u16=phase,
                variation_u16=variation,
            )
        )
    return tuple(output)


__all__ = [
    "COMPONENT_PACK_MAGIC",
    "ComponentProfile",
    "DecodedPose64",
    "ENDIAN_MARKER",
    "FixedRecipe",
    "GeneratedDisplayInstance",
    "LUT_MAGIC",
    "LogPolarProfile",
    "MAX_INSTANCES_PER_RECIPE",
    "MotionBounds",
    "MotionValues",
    "OPERATOR_RECORD_BYTES",
    "OperatorMeaning",
    "POSE_WIDTHS",
    "PackedComponentRecord",
    "PackedMotion64",
    "PackedPose64",
    "RECIPE_PACK_MAGIC",
    "RECIPE_RECORD_BYTES",
    "RecipePack",
    "SPARSE_COMPONENT_BYTES",
    "SharedLogPolarLUT",
    "SparseComponentPack",
    "SubstratePackingError",
    "combine_seed",
    "generated_display_instances",
    "semantic_address",
    "splitmix64",
    "stable_lineage_id",
]
