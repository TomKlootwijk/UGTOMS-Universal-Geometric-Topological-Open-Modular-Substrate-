"""Small executable kernel for the geometric-topological substrate.

The module intentionally implements reusable mechanisms rather than an
application or renderer.  It keeps continuous geometry, discrete topology,
bounded routing, and provenance separate while allowing them to participate in
one deterministic state transition.

The cone primitive is named ``ConeField`` rather than ``ConeSDF`` because the
intersection of its side and axial guards is an implicit signed field and is
not, in general, an exact Euclidean distance.  Sphere and circle primitives are
exact signed-distance functions away from their conventional centre
singularity.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


Vector2 = tuple[float, float]
Vector3 = tuple[float, float, float]


class SubstrateError(ValueError):
    """A substrate invariant or declared resource bound was violated."""


class PredicateValue(str, Enum):
    FALSE = "FALSE"
    TRUE = "TRUE"
    INDETERMINATE = "INDETERMINATE"


class EventStatus(str, Enum):
    NO_SUPPORT = "NO_SUPPORT"
    INCOMPATIBLE = "INCOMPATIBLE"
    NO_CROSSING = "NO_CROSSING"
    VERIFIED = "VERIFIED"
    INDETERMINATE = "INDETERMINATE"


class EventOrigin(str, Enum):
    CLOSED_DYNAMICS = "CLOSED_DYNAMICS"
    EXOGENOUS = "EXOGENOUS"


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise SubstrateError(f"{name} must be finite")
    return result


def _vector2(value: Sequence[float], name: str) -> Vector2:
    if len(value) != 2:
        raise SubstrateError(f"{name} must contain exactly two values")
    return (_finite(value[0], f"{name}[0]"), _finite(value[1], f"{name}[1]"))


def _vector3(value: Sequence[float], name: str) -> Vector3:
    if len(value) != 3:
        raise SubstrateError(f"{name} must contain exactly three values")
    return tuple(_finite(value[i], f"{name}[{i}]") for i in range(3))  # type: ignore[return-value]


def _vsub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vadd(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vscale(a: Vector3, scalar: float) -> Vector3:
    return (a[0] * scalar, a[1] * scalar, a[2] * scalar)


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: Vector3) -> float:
    return math.sqrt(_dot(a, a))


def _unit(a: Vector3, name: str) -> Vector3:
    length = _norm(a)
    if length <= 1e-15:
        raise SubstrateError(f"{name} must be non-zero")
    return _vscale(a, 1.0 / length)


@dataclass(frozen=True)
class LogPolarAddress:
    """A bounded radial/angular/forward-time LUT address.

    ``rho`` encodes the residual magnitude with ``log1p``.  It is not spatial
    radius unless an application explicitly declares that mapping.  The apex
    has a separate sentinel rather than pretending that log(0) is finite.
    """

    magnitude: float
    rho: float
    theta: float
    radial_bin: int
    angular_bin: int
    time_bin: int
    packed_index: int
    apex: bool
    saturated: bool


@dataclass(frozen=True)
class LogPolarLUT:
    radial_bins: int = 32
    angular_bins: int = 64
    time_bins: int = 16
    gamma: float = 1.0
    maximum_magnitude: float = 1024.0
    apex_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        for name in ("radial_bins", "angular_bins", "time_bins"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise SubstrateError(f"{name} must be a positive integer")
        if _finite(self.gamma, "gamma") <= 0:
            raise SubstrateError("gamma must be positive")
        if _finite(self.maximum_magnitude, "maximum_magnitude") <= 0:
            raise SubstrateError("maximum_magnitude must be positive")
        if _finite(self.apex_epsilon, "apex_epsilon") <= 0:
            raise SubstrateError("apex_epsilon must be positive")

    @property
    def capacity(self) -> int:
        return self.radial_bins * self.angular_bins * self.time_bins

    def encode(self, residual: Sequence[float], *, forward_step: int = 0) -> LogPolarAddress:
        x, y = _vector2(residual, "residual")
        if isinstance(forward_step, bool) or not isinstance(forward_step, int):
            raise SubstrateError("forward_step must be an integer")
        magnitude = math.hypot(x, y)
        apex = magnitude <= self.apex_epsilon
        bounded = min(magnitude, self.maximum_magnitude)
        saturated = magnitude > self.maximum_magnitude
        rho = math.log1p(self.gamma * bounded)
        rho_limit = math.log1p(self.gamma * self.maximum_magnitude)
        radial_fraction = 0.0 if apex else min(rho / rho_limit, 1.0)
        radial_bin = min(int(radial_fraction * self.radial_bins), self.radial_bins - 1)
        theta = 0.0 if apex else math.atan2(y, x)
        theta_fraction = (theta + math.pi) / (2.0 * math.pi)
        angular_bin = min(int(theta_fraction * self.angular_bins), self.angular_bins - 1)
        time_bin = forward_step % self.time_bins
        packed = (
            (time_bin * self.radial_bins + radial_bin) * self.angular_bins
            + angular_bin
        )
        return LogPolarAddress(
            magnitude=magnitude,
            rho=rho,
            theta=theta,
            radial_bin=radial_bin,
            angular_bin=angular_bin,
            time_bin=time_bin,
            packed_index=packed,
            apex=apex,
            saturated=saturated,
        )

    def bin_centre(self, address: LogPolarAddress) -> Vector2:
        if not 0 <= address.radial_bin < self.radial_bins:
            raise SubstrateError("address radial bin is outside this LUT")
        if not 0 <= address.angular_bin < self.angular_bins:
            raise SubstrateError("address angular bin is outside this LUT")
        rho_limit = math.log1p(self.gamma * self.maximum_magnitude)
        rho = (address.radial_bin + 0.5) / self.radial_bins * rho_limit
        magnitude = math.expm1(rho) / self.gamma
        theta = (address.angular_bin + 0.5) / self.angular_bins * 2.0 * math.pi - math.pi
        return (magnitude * math.cos(theta), magnitude * math.sin(theta))


def xor_parity(payload: bytes | bytearray | memoryview, orientation_reversals: int = 0) -> int:
    """Return the one-bit payload/topology event.

    This deliberately remains a detector: an even number of bit flips is a
    known blind spot and the result does not identify or repair a damaged bit.
    """

    if isinstance(orientation_reversals, bool) or not isinstance(orientation_reversals, int):
        raise SubstrateError("orientation_reversals must be an integer")
    parity = orientation_reversals & 1
    for value in bytes(payload):
        parity ^= value.bit_count() & 1
    return parity


@dataclass(frozen=True)
class ParityDebounce:
    stable_bit: int = 0
    candidate_bit: int = 0
    candidate_count: int = 0

    def update(self, raw_bit: int, *, required_samples: int = 2) -> "ParityDebounce":
        if raw_bit not in (0, 1):
            raise SubstrateError("raw_bit must be 0 or 1")
        if isinstance(required_samples, bool) or not isinstance(required_samples, int) or required_samples < 1:
            raise SubstrateError("required_samples must be a positive integer")
        count = self.candidate_count + 1 if raw_bit == self.candidate_bit else 1
        stable = raw_bit if count >= required_samples else self.stable_bit
        return ParityDebounce(stable_bit=stable, candidate_bit=raw_bit, candidate_count=count)


@dataclass(frozen=True)
class KleinCoordinate:
    u: float
    v: float
    sheet: int
    orientation: int
    orientation_reversals: int


def klein_normalize(
    u: float,
    v: float,
    *,
    sheet: int = 0,
    orientation: int = 1,
) -> KleinCoordinate:
    """Normalize a point under the square Klein-bottle quotient.

    The declared gluing is ``(u + 1, v) ~ (u, 1 - v)`` and
    ``(u, v + 1) ~ (u, v)``.  Crossing the first seam reverses orientation;
    crossing it twice restores orientation.  This is an intrinsic routing rule,
    not a claim that a globally exact 3-D Klein-bottle SDF exists.
    """

    u_value = _finite(u, "u")
    v_value = _finite(v, "v")
    if sheet not in (0, 1):
        raise SubstrateError("sheet must be 0 or 1")
    if orientation not in (-1, 1):
        raise SubstrateError("orientation must be -1 or 1")
    u_wraps = math.floor(u_value)
    u_normal = u_value - u_wraps
    reversals = abs(u_wraps)
    if u_wraps & 1:
        v_value = 1.0 - v_value
        sheet ^= 1
        orientation *= -1
    v_normal = v_value - math.floor(v_value)
    return KleinCoordinate(
        u=u_normal,
        v=v_normal,
        sheet=sheet,
        orientation=orientation,
        orientation_reversals=reversals,
    )


@dataclass(frozen=True)
class SphereSDF:
    centre: Vector3
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "centre", _vector3(self.centre, "centre"))
        if _finite(self.radius, "radius") <= 0:
            raise SubstrateError("radius must be positive")

    def evaluate(self, point: Sequence[float]) -> float:
        return _norm(_vsub(_vector3(point, "point"), self.centre)) - self.radius


@dataclass(frozen=True)
class CircleSDF:
    centre: Vector2
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "centre", _vector2(self.centre, "centre"))
        if _finite(self.radius, "radius") <= 0:
            raise SubstrateError("radius must be positive")

    def evaluate(self, point: Sequence[float]) -> float:
        x, y = _vector2(point, "point")
        return math.hypot(x - self.centre[0], y - self.centre[1]) - self.radius


def _point_segment_distance_2d(point: Vector2, start: Vector2, end: Vector2) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator <= 0.0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    fraction = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / denominator
    fraction = max(0.0, min(1.0, fraction))
    closest = (start[0] + fraction * dx, start[1] + fraction * dy)
    return math.hypot(point[0] - closest[0], point[1] - closest[1])


@dataclass(frozen=True)
class FiniteConeSDF:
    """Exact SDF for a finite filled right-circular cone.

    ``T`` is represented by ``slant_length`` and is deliberately not confused
    with time.  A half-angle is also required because a side length alone does
    not determine a cone.  Rotational symmetry reduces distance to the filled
    meridian triangle ``(-R,h), (R,h), (0,0)``.
    """

    slant_length: float
    half_angle_radians: float
    apex: Vector3 = (0.0, 0.0, 0.0)
    axis: Vector3 = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if _finite(self.slant_length, "slant_length") <= 0:
            raise SubstrateError("slant_length must be positive")
        angle = _finite(self.half_angle_radians, "half_angle_radians")
        if not 0 < angle < math.pi / 2:
            raise SubstrateError("half_angle_radians must be between 0 and pi/2")
        object.__setattr__(self, "apex", _vector3(self.apex, "apex"))
        object.__setattr__(self, "axis", _unit(_vector3(self.axis, "axis"), "axis"))

    @property
    def height(self) -> float:
        return self.slant_length * math.cos(self.half_angle_radians)

    @property
    def base_radius(self) -> float:
        return self.slant_length * math.sin(self.half_angle_radians)

    def evaluate(self, point: Sequence[float]) -> float:
        relative = _vsub(_vector3(point, "point"), self.apex)
        axial = _dot(relative, self.axis)
        radial = _norm(_vsub(relative, _vscale(self.axis, axial)))
        radius = self.base_radius
        height = self.height
        meridian_point = (radial, axial)
        triangle = ((-radius, height), (radius, height), (0.0, 0.0))
        distance = min(
            _point_segment_distance_2d(meridian_point, triangle[0], triangle[1]),
            _point_segment_distance_2d(meridian_point, triangle[1], triangle[2]),
            _point_segment_distance_2d(meridian_point, triangle[2], triangle[0]),
        )
        inside = (
            0.0 <= axial <= height
            and radial <= axial * math.tan(self.half_angle_radians) + 1e-15
        )
        return -distance if inside and distance > 0.0 else distance


@dataclass(frozen=True)
class ConeField:
    """Forward cone zero-set plus explicit axial support guards.

    The returned value is a useful signed implicit field with negative interior.
    It is not advertised as an exact Euclidean distance at the apex, cap, or
    intersections of guard surfaces.
    """

    apex: Vector3
    axis: Vector3
    half_angle_radians: float
    maximum_length: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "apex", _vector3(self.apex, "apex"))
        object.__setattr__(self, "axis", _unit(_vector3(self.axis, "axis"), "axis"))
        angle = _finite(self.half_angle_radians, "half_angle_radians")
        if not 0 < angle < math.pi / 2:
            raise SubstrateError("half_angle_radians must be between 0 and pi/2")
        if self.maximum_length is not None and _finite(self.maximum_length, "maximum_length") <= 0:
            raise SubstrateError("maximum_length must be positive when present")

    def components(self, point: Sequence[float]) -> Mapping[str, float]:
        q = _vsub(_vector3(point, "point"), self.apex)
        axial = _dot(q, self.axis)
        radial_vector = _vsub(q, _vscale(self.axis, axial))
        radial = _norm(radial_vector)
        side = radial * math.cos(self.half_angle_radians) - axial * math.sin(
            self.half_angle_radians
        )
        result = {"side": side, "behind_apex": -axial, "axial": axial, "radial": radial}
        if self.maximum_length is not None:
            result["past_cap"] = axial - self.maximum_length
        return result

    def evaluate(self, point: Sequence[float]) -> float:
        values = self.components(point)
        guards = [values["side"], values["behind_apex"]]
        if "past_cap" in values:
            guards.append(values["past_cap"])
        return max(guards)


@dataclass(frozen=True)
class SpatialLogPolarChart:
    """Spatial log-polar chart with the SCLP metric and kinematic calculus.

    This type is intentionally distinct from ``LogPolarLUT``: the latter
    addresses residual/jitter state, while this chart maps physical radius.
    An application may couple them only by declaring the conversion.
    """

    reference_radius: float = 1.0
    rho_min: float = -10.0
    rho_max: float = 10.0
    core_radius: float = 1e-12

    def __post_init__(self) -> None:
        if _finite(self.reference_radius, "reference_radius") <= 0:
            raise SubstrateError("reference_radius must be positive")
        rho_min = _finite(self.rho_min, "rho_min")
        rho_max = _finite(self.rho_max, "rho_max")
        if rho_min >= rho_max:
            raise SubstrateError("rho_min must be less than rho_max")
        if _finite(self.core_radius, "core_radius") <= 0:
            raise SubstrateError("core_radius must be positive")

    def encode(self, x: float, y: float) -> Mapping[str, float | bool]:
        x_value = _finite(x, "x")
        y_value = _finite(y, "y")
        radius = math.hypot(x_value, y_value)
        core = radius <= self.core_radius
        safe_radius = max(radius, self.core_radius)
        raw_rho = math.log(safe_radius / self.reference_radius)
        rho = min(max(raw_rho, self.rho_min), self.rho_max)
        return {
            "rho": rho,
            "theta": 0.0 if core else math.atan2(y_value, x_value),
            "radius": radius,
            "core": core,
            "saturated": rho != raw_rho,
        }

    def radius(self, rho: float) -> float:
        rho_value = _finite(rho, "rho")
        if not self.rho_min <= rho_value <= self.rho_max:
            raise SubstrateError("rho lies outside the declared chart")
        return self.reference_radius * math.exp(rho_value)

    def decode(self, rho: float, theta: float) -> Vector2:
        radius = self.radius(rho)
        angle = _finite(theta, "theta")
        return (radius * math.cos(angle), radius * math.sin(angle))

    def metric_scale(self, rho: float) -> float:
        radius = self.radius(rho)
        return radius * radius

    def jacobian(self, rho: float, theta: float) -> tuple[Vector2, Vector2]:
        radius = self.radius(rho)
        angle = _finite(theta, "theta")
        cosine = math.cos(angle)
        sine = math.sin(angle)
        return ((radius * cosine, -radius * sine), (radius * sine, radius * cosine))

    def exact_radial_increment(self, rho: float, delta_rho: float) -> float:
        radius = self.radius(rho)
        return radius * math.expm1(_finite(delta_rho, "delta_rho"))

    def velocity(self, rho: float, theta: float, rho_rate: float, theta_rate: float) -> Vector2:
        radius = self.radius(rho)
        angle = _finite(theta, "theta")
        radial_rate = _finite(rho_rate, "rho_rate")
        angular_rate = _finite(theta_rate, "theta_rate")
        e_radius = (math.cos(angle), math.sin(angle))
        e_theta = (-math.sin(angle), math.cos(angle))
        return (
            radius * (radial_rate * e_radius[0] + angular_rate * e_theta[0]),
            radius * (radial_rate * e_radius[1] + angular_rate * e_theta[1]),
        )

    def acceleration(
        self,
        rho: float,
        theta: float,
        rho_rate: float,
        theta_rate: float,
        rho_acceleration: float,
        theta_acceleration: float,
    ) -> Vector2:
        radius = self.radius(rho)
        angle = _finite(theta, "theta")
        rho_dot = _finite(rho_rate, "rho_rate")
        theta_dot = _finite(theta_rate, "theta_rate")
        rho_ddot = _finite(rho_acceleration, "rho_acceleration")
        theta_ddot = _finite(theta_acceleration, "theta_acceleration")
        radial_component = rho_ddot + rho_dot * rho_dot - theta_dot * theta_dot
        angular_component = theta_ddot + 2.0 * rho_dot * theta_dot
        e_radius = (math.cos(angle), math.sin(angle))
        e_theta = (-math.sin(angle), math.cos(angle))
        return (
            radius * (radial_component * e_radius[0] + angular_component * e_theta[0]),
            radius * (radial_component * e_radius[1] + angular_component * e_theta[1]),
        )


@dataclass(frozen=True)
class OneBitJitter:
    amplitude: float
    guard_margin: float
    seed: str

    def __post_init__(self) -> None:
        if _finite(self.amplitude, "amplitude") < 0:
            raise SubstrateError("amplitude must be non-negative")
        if _finite(self.guard_margin, "guard_margin") <= 0:
            raise SubstrateError("guard_margin must be positive")
        if not self.seed:
            raise SubstrateError("seed must be non-empty")

    def bit(self, key: int, context: int) -> int:
        if isinstance(key, bool) or not isinstance(key, int) or key < 0:
            raise SubstrateError("key must be a non-negative integer")
        if isinstance(context, bool) or not isinstance(context, int):
            raise SubstrateError("context must be an integer")
        digest = hashlib.sha256(
            canonical_json({"seed": self.seed, "key": key, "context": context})
        ).digest()
        return digest[0] & 1

    def certificate(self, residual: float, key: int, context: int) -> Mapping[str, Any]:
        authoritative = _finite(residual, "residual")
        bit = self.bit(key, context)
        signed_offset = self.amplitude if bit else -self.amplitude
        interval = (authoritative - self.amplitude, authoritative + self.amplitude)
        return {
            "bit": bit,
            "signed_offset": signed_offset,
            "perturbed": authoritative + signed_offset,
            "authoritative_residual": authoritative,
            "interval": interval,
            "amplitude": self.amplitude,
            "guard_margin": self.guard_margin,
            "safe_under_margin": self.amplitude < self.guard_margin,
        }


@dataclass(frozen=True)
class QuantizedStateKey:
    rho: int
    theta: int
    time: int
    phi: int


@dataclass(frozen=True)
class SCLPKeyLayout64:
    """SCLP 20/18/14/12-bit state key with two explicit layouts."""

    rho_min: float = -10.0
    rho_max: float = 10.0

    WIDTHS = {"rho": 20, "theta": 18, "time": 14, "phi": 12}
    ORDER = ("rho", "theta", "time", "phi")

    def __post_init__(self) -> None:
        if _finite(self.rho_min, "rho_min") >= _finite(self.rho_max, "rho_max"):
            raise SubstrateError("rho_min must be less than rho_max")
        if sum(self.WIDTHS.values()) != 64:
            raise AssertionError("SCLP key widths must total 64 bits")

    @staticmethod
    def _quantize_unit(value: float, width: int) -> int:
        bounded = min(max(value, 0.0), 1.0)
        # Explicit round-half-up avoids depending on a language's implicit
        # banker-rounding convention at exact half bins.
        return int(math.floor(bounded * ((1 << width) - 1) + 0.5))

    def quantize(self, rho: float, theta: float, time_tick: int, phi: float) -> QuantizedStateKey:
        rho_value = _finite(rho, "rho")
        theta_value = _finite(theta, "theta")
        phi_value = _finite(phi, "phi")
        if isinstance(time_tick, bool) or not isinstance(time_tick, int):
            raise SubstrateError("time_tick must be an integer")
        rho_unit = (rho_value - self.rho_min) / (self.rho_max - self.rho_min)
        theta_unit = ((theta_value + math.pi) % (2.0 * math.pi)) / (2.0 * math.pi)
        phi_unit = (phi_value % (2.0 * math.pi)) / (2.0 * math.pi)
        return QuantizedStateKey(
            rho=self._quantize_unit(rho_unit, self.WIDTHS["rho"]),
            theta=self._quantize_unit(theta_unit, self.WIDTHS["theta"]),
            time=time_tick % (1 << self.WIDTHS["time"]),
            phi=self._quantize_unit(phi_unit, self.WIDTHS["phi"]),
        )

    def _validate(self, state: QuantizedStateKey) -> None:
        for name in self.ORDER:
            value = getattr(state, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < (1 << self.WIDTHS[name]):
                raise SubstrateError(f"{name} does not fit its declared key field")

    def pack_contiguous(self, state: QuantizedStateKey) -> int:
        self._validate(state)
        result = 0
        for name in self.ORDER:
            result = (result << self.WIDTHS[name]) | getattr(state, name)
        return result

    def unpack_contiguous(self, key: int) -> QuantizedStateKey:
        if isinstance(key, bool) or not isinstance(key, int) or not 0 <= key < (1 << 64):
            raise SubstrateError("key must be an unsigned 64-bit integer")
        values: dict[str, int] = {}
        remaining = key
        for name in reversed(self.ORDER):
            mask = (1 << self.WIDTHS[name]) - 1
            values[name] = remaining & mask
            remaining >>= self.WIDTHS[name]
        return QuantizedStateKey(**values)

    def morton_schedule(self) -> tuple[tuple[str, int], ...]:
        remaining = {name: self.WIDTHS[name] - 1 for name in self.ORDER}
        schedule: list[tuple[str, int]] = []
        while len(schedule) < 64:
            for name in self.ORDER:
                if remaining[name] >= 0:
                    schedule.append((name, remaining[name]))
                    remaining[name] -= 1
        return tuple(schedule)

    def pack_morton(self, state: QuantizedStateKey) -> int:
        self._validate(state)
        result = 0
        for name, bit in self.morton_schedule():
            result = (result << 1) | ((getattr(state, name) >> bit) & 1)
        return result

    def unpack_morton(self, key: int) -> QuantizedStateKey:
        if isinstance(key, bool) or not isinstance(key, int) or not 0 <= key < (1 << 64):
            raise SubstrateError("key must be an unsigned 64-bit integer")
        values = {name: 0 for name in self.ORDER}
        for index, (name, bit) in enumerate(self.morton_schedule()):
            source_bit = 63 - index
            values[name] |= ((key >> source_bit) & 1) << bit
        return QuantizedStateKey(**values)


def field_union(*values: float) -> float:
    if not values:
        raise SubstrateError("field_union requires at least one value")
    return min(_finite(value, "field value") for value in values)


def field_intersection(*values: float) -> float:
    if not values:
        raise SubstrateError("field_intersection requires at least one value")
    return max(_finite(value, "field value") for value in values)


def field_difference(left: float, right: float) -> float:
    return max(_finite(left, "left field"), -_finite(right, "right field"))


def zero_predicate(value: float, *, tolerance: float = 1e-9) -> PredicateValue:
    try:
        field = _finite(value, "field value")
        bound = _finite(tolerance, "tolerance")
    except SubstrateError:
        return PredicateValue.INDETERMINATE
    if bound < 0:
        raise SubstrateError("tolerance must be non-negative")
    return PredicateValue.TRUE if abs(field) <= bound else PredicateValue.FALSE


@dataclass(frozen=True)
class DistributedApex:
    points: tuple[Vector3, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.points or len(self.points) != len(self.weights):
            raise SubstrateError("distributed apex needs equally sized non-empty points and weights")
        normalized_points = tuple(_vector3(point, "apex point") for point in self.points)
        normalized_weights = tuple(_finite(weight, "apex weight") for weight in self.weights)
        if any(weight < 0 for weight in normalized_weights) or sum(normalized_weights) <= 0:
            raise SubstrateError("apex weights must be non-negative with a positive sum")
        object.__setattr__(self, "points", normalized_points)
        object.__setattr__(self, "weights", normalized_weights)

    def observed_centroid(self) -> Vector3:
        total = sum(self.weights)
        return tuple(
            sum(point[axis] * weight for point, weight in zip(self.points, self.weights)) / total
            for axis in range(3)
        )  # type: ignore[return-value]


@dataclass(frozen=True)
class VectorArrow3:
    """A directed geometric primitive represented by origin plus displacement."""

    origin: Vector3
    displacement: Vector3
    role: str = "vector"

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", _vector3(self.origin, "origin"))
        object.__setattr__(self, "displacement", _vector3(self.displacement, "displacement"))
        if not self.role.strip():
            raise SubstrateError("arrow role must be non-empty")

    @classmethod
    def between(cls, start: Sequence[float], end: Sequence[float], *, role: str = "vector") -> "VectorArrow3":
        origin = _vector3(start, "start")
        endpoint = _vector3(end, "end")
        return cls(origin, _vsub(endpoint, origin), role)

    @property
    def endpoint(self) -> Vector3:
        return _vadd(self.origin, self.displacement)

    @property
    def magnitude(self) -> float:
        return _norm(self.displacement)

    def direction(self) -> Vector3:
        return _unit(self.displacement, "arrow displacement")

    def translated(self, offset: Sequence[float]) -> "VectorArrow3":
        return VectorArrow3(
            _vadd(self.origin, _vector3(offset, "offset")), self.displacement, self.role
        )

    def scaled(self, scalar: float) -> "VectorArrow3":
        return VectorArrow3(
            self.origin, _vscale(self.displacement, _finite(scalar, "scalar")), self.role
        )


@dataclass(frozen=True)
class KinematicState:
    position: Vector3
    velocity: Vector3 = (0.0, 0.0, 0.0)
    acceleration: Vector3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _vector3(self.position, "position"))
        object.__setattr__(self, "velocity", _vector3(self.velocity, "velocity"))
        object.__setattr__(self, "acceleration", _vector3(self.acceleration, "acceleration"))

    def advance(self, delta_time: float) -> "KinematicState":
        dt = _finite(delta_time, "delta_time")
        if dt <= 0:
            raise SubstrateError("delta_time must be positive")
        position = _vadd(
            _vadd(self.position, _vscale(self.velocity, dt)),
            _vscale(self.acceleration, 0.5 * dt * dt),
        )
        velocity = _vadd(self.velocity, _vscale(self.acceleration, dt))
        return KinematicState(position=position, velocity=velocity, acceleration=self.acceleration)

    def arrows(self, *, scale: float = 1.0) -> tuple[VectorArrow3, VectorArrow3]:
        factor = _finite(scale, "scale")
        return (
            VectorArrow3(self.position, _vscale(self.velocity, factor), "velocity"),
            VectorArrow3(self.position, _vscale(self.acceleration, factor), "acceleration"),
        )


@dataclass(frozen=True)
class GuardCrossing:
    previous: float
    current: float
    tolerance: float = 1e-9

    def classify(self) -> PredicateValue:
        try:
            before = _finite(self.previous, "previous guard")
            after = _finite(self.current, "current guard")
            tolerance = _finite(self.tolerance, "guard tolerance")
        except SubstrateError:
            return PredicateValue.INDETERMINATE
        if tolerance < 0:
            raise SubstrateError("guard tolerance must be non-negative")
        crossed = (
            abs(before) <= tolerance
            or abs(after) <= tolerance
            or (before < -tolerance and after > tolerance)
            or (before > tolerance and after < -tolerance)
        )
        return PredicateValue.TRUE if crossed else PredicateValue.FALSE


@dataclass(frozen=True)
class EventDecision:
    status: EventStatus
    support: PredicateValue
    compatibility: PredicateValue
    crossing: PredicateValue


def verify_event(
    *,
    support: PredicateValue | bool,
    compatibility: PredicateValue | bool,
    guard: GuardCrossing,
) -> EventDecision:
    def convert(value: PredicateValue | bool) -> PredicateValue:
        if isinstance(value, PredicateValue):
            return value
        return PredicateValue.TRUE if value is True else PredicateValue.FALSE

    support_value = convert(support)
    compatibility_value = convert(compatibility)
    crossing = guard.classify()
    if PredicateValue.INDETERMINATE in (support_value, compatibility_value, crossing):
        status = EventStatus.INDETERMINATE
    elif support_value is PredicateValue.FALSE:
        status = EventStatus.NO_SUPPORT
    elif compatibility_value is PredicateValue.FALSE:
        status = EventStatus.INCOMPATIBLE
    elif crossing is PredicateValue.FALSE:
        status = EventStatus.NO_CROSSING
    else:
        status = EventStatus.VERIFIED
    return EventDecision(status, support_value, compatibility_value, crossing)


@dataclass(frozen=True)
class RouteDecision:
    ordering_key: tuple[int, int, int]
    branch_paths: tuple[str, ...]
    geometric_angles: tuple[float, ...]
    bifurcated: bool
    bounded: bool


@dataclass(frozen=True)
class BSTTRouter:
    """Bounded BST ordering plus separate L-system-style geometry."""

    maximum_depth: int = 16
    maximum_active_branches: int = 1024
    golden_angle_radians: float = math.pi * (3.0 - math.sqrt(5.0))

    def __post_init__(self) -> None:
        for name in ("maximum_depth", "maximum_active_branches"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise SubstrateError(f"{name} must be a positive integer")
        _finite(self.golden_angle_radians, "golden_angle_radians")

    def route(
        self,
        address: LogPolarAddress,
        *,
        phase: float,
        phase_acceleration: float,
        generation: int,
        parity_gate: int,
        depth: int,
        active_branches: int,
        path: str = "r",
    ) -> RouteDecision:
        if parity_gate not in (0, 1):
            raise SubstrateError("parity_gate must be 0 or 1")
        for name, value in (("generation", generation), ("depth", depth), ("active_branches", active_branches)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SubstrateError(f"{name} must be a non-negative integer")
        phase_value = _finite(phase, "phase")
        acceleration = _finite(phase_acceleration, "phase_acceleration")
        phase_bin = int(round(((phase_value + math.pi) % (2 * math.pi) - math.pi) * 1_000_000))
        ordering_key = (address.packed_index, phase_bin, generation)
        can_split = (
            parity_gate == 1
            and depth < self.maximum_depth
            and active_branches + 1 <= self.maximum_active_branches
        )
        turn = self.golden_angle_radians + acceleration
        if can_split:
            paths = (path + "0", path + "1")
            angles = (phase_value - turn, phase_value + turn)
        else:
            paths = (path,)
            angles = (phase_value,)
        return RouteDecision(
            ordering_key=ordering_key,
            branch_paths=paths,
            geometric_angles=angles,
            bifurcated=can_split,
            bounded=not (parity_gate == 1 and not can_split),
        )


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise SubstrateError(f"value of type {type(value).__name__} is not canonically serializable")


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_value(value), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


@dataclass(frozen=True)
class LineageEvent:
    sequence: int
    event_type: str
    origin: EventOrigin
    payload: Mapping[str, Any]
    novelty_digest: str
    lineage_digest: str


class NoveltyLog:
    """A bounded hash chain that never silently discards exogenous events."""

    def __init__(self, capacity: int, *, root_digest: str = "0" * 64) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise SubstrateError("capacity must be a positive integer")
        if len(root_digest) != 64 or any(char not in "0123456789abcdef" for char in root_digest.lower()):
            raise SubstrateError("root_digest must contain 64 hexadecimal characters")
        self.capacity = capacity
        self.root_digest = root_digest.lower()
        self._events: list[LineageEvent] = []
        self._next_sequence = 0

    @property
    def events(self) -> tuple[LineageEvent, ...]:
        return tuple(self._events)

    @property
    def head_digest(self) -> str:
        return self._events[-1].lineage_digest if self._events else self.root_digest

    def append(self, event_type: str, payload: Mapping[str, Any], *, origin: EventOrigin) -> LineageEvent:
        if not event_type or not event_type.strip():
            raise SubstrateError("event_type must be non-empty")
        canonical_payload = canonical_json(payload)
        novelty = hashlib.sha256(canonical_payload).hexdigest()
        envelope = {
            "previous": self.head_digest,
            "sequence": self._next_sequence,
            "event_type": event_type,
            "origin": origin.value,
            "novelty": novelty,
            "payload": payload,
        }
        lineage = hashlib.sha256(canonical_json(envelope)).hexdigest()
        event = LineageEvent(
            sequence=self._next_sequence,
            event_type=event_type,
            origin=origin,
            payload=dict(payload),
            novelty_digest=novelty,
            lineage_digest=lineage,
        )
        self._next_sequence += 1
        if len(self._events) >= self.capacity:
            removable = next(
                (index for index, prior in enumerate(self._events) if prior.origin is EventOrigin.CLOSED_DYNAMICS),
                None,
            )
            if removable is None:
                raise SubstrateError(
                    "novelty log is full of exogenous events; increase the declared capacity before appending"
                )
            del self._events[removable]
        self._events.append(event)
        return event


def deterministic_bit(seed: int, generation: int, channel: str = "jitter") -> int:
    """Return a stateless one-bit deterministic sample for seed replay."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SubstrateError("seed must be an integer")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise SubstrateError("generation must be a non-negative integer")
    if not channel:
        raise SubstrateError("channel must be non-empty")
    digest = hashlib.sha256(canonical_json({"seed": seed, "generation": generation, "channel": channel})).digest()
    return digest[0] & 1


@dataclass(frozen=True)
class ClosedDynamicsSeed:
    """A compact seed for a declared, closed constant-acceleration grammar.

    This reconstructs only endogenous state.  Sensor readings, user edits, and
    other exogenous events belong in ``NoveltyLog`` and cannot be regenerated
    from this seed.
    """

    grammar_id: str
    seed: int
    initial: KinematicState
    step_seconds: float
    phase_origin: float = 0.0
    phase_rate: float = 0.0

    def __post_init__(self) -> None:
        if not self.grammar_id.strip():
            raise SubstrateError("grammar_id must be non-empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise SubstrateError("seed must be an integer")
        if _finite(self.step_seconds, "step_seconds") <= 0:
            raise SubstrateError("step_seconds must be positive")
        _finite(self.phase_origin, "phase_origin")
        _finite(self.phase_rate, "phase_rate")

    def reconstruct(self, generation: int) -> Mapping[str, Any]:
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise SubstrateError("generation must be a non-negative integer")
        elapsed = generation * self.step_seconds
        if generation == 0:
            state = self.initial
        else:
            state = self.initial.advance(elapsed)
        phase = (self.phase_origin + self.phase_rate * elapsed + math.pi) % (2 * math.pi) - math.pi
        value = {
            "grammar_id": self.grammar_id,
            "seed": self.seed,
            "generation": generation,
            "logical_time": elapsed,
            "kinematics": state,
            "phase": phase,
            "jitter_bit": deterministic_bit(self.seed, generation),
        }
        return {**value, "state_digest": hashlib.sha256(canonical_json(value)).hexdigest()}


def lineage_digest(events: Iterable[LineageEvent], *, root_digest: str = "0" * 64) -> str:
    """Verify a retained event sequence's recorded chain where possible.

    A bounded log may have compacted older closed-dynamics events.  The stored
    lineage digests remain stable identifiers, while complete verification
    requires the un-compacted evidence stream.
    """

    head = root_digest
    for event in events:
        envelope = {
            "previous": head,
            "sequence": event.sequence,
            "event_type": event.event_type,
            "origin": event.origin.value,
            "novelty": event.novelty_digest,
            "payload": event.payload,
        }
        expected = hashlib.sha256(canonical_json(envelope)).hexdigest()
        if expected != event.lineage_digest:
            raise SubstrateError(f"lineage mismatch at sequence {event.sequence}")
        head = expected
    return head
