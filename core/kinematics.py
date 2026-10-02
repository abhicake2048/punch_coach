"""Vectorized, model-independent 2D kinematic calculations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _xy(point: ArrayLike) -> NDArray[np.float64]:
    """Return the final coordinate axis as x/y floating-point values."""
    array = np.asarray(point, dtype=np.float64)
    if array.shape == () or array.shape[-1] < 2:
        raise ValueError("points must have at least x and y coordinates")
    return array[..., :2]


def calculate_angle(
    a: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
) -> float | NDArray[np.float64]:
    """Calculate the interior angle ``a-b-c`` in degrees.

    Inputs can be individual points or arrays with a shared leading shape. The
    cosine is clipped to ``[-1, 1]`` to avoid floating-point ``arccos`` errors.
    Degenerate vectors return ``NaN``.
    """
    ba = _xy(a) - _xy(b)
    bc = _xy(c) - _xy(b)
    dot_products = np.sum(ba * bc, axis=-1)
    magnitudes = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1)

    cosine = np.full(np.shape(dot_products), np.nan, dtype=np.float64)
    np.divide(dot_products, magnitudes, out=cosine, where=magnitudes > 1e-12)
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return float(angles) if np.ndim(angles) == 0 else angles


def _valid_keypoint(point: ArrayLike | None, min_confidence: float) -> bool:
    """Check that a keypoint has finite coordinates and acceptable confidence."""
    if point is None:
        return False
    array = np.asarray(point, dtype=np.float64).reshape(-1)
    if array.size < 2 or not np.all(np.isfinite(array[:2])):
        return False
    return array.size < 3 or bool(np.isfinite(array[2]) and array[2] >= min_confidence)


def calculate_torso_scale(
    keypoints: Mapping[str, ArrayLike],
    min_confidence: float = 0.25,
) -> float:
    """Return torso length for scale-invariant motion measurements.

    Torso length is the distance from the midpoint of the shoulders to the
    midpoint of the hips. If either hip is unavailable, shoulder width is used
    as the fallback. ``NaN`` is returned when a reliable scale cannot be formed.
    """
    left_shoulder = keypoints.get("left_shoulder")
    right_shoulder = keypoints.get("right_shoulder")
    if not (
        _valid_keypoint(left_shoulder, min_confidence)
        and _valid_keypoint(right_shoulder, min_confidence)
    ):
        return float("nan")

    left_shoulder_xy = _xy(left_shoulder)
    right_shoulder_xy = _xy(right_shoulder)
    shoulder_width = float(np.linalg.norm(left_shoulder_xy - right_shoulder_xy))

    left_hip = keypoints.get("left_hip")
    right_hip = keypoints.get("right_hip")
    if _valid_keypoint(left_hip, min_confidence) and _valid_keypoint(
        right_hip, min_confidence
    ):
        mid_shoulder = (left_shoulder_xy + right_shoulder_xy) / 2.0
        mid_hip = (_xy(left_hip) + _xy(right_hip)) / 2.0
        torso_length = float(np.linalg.norm(mid_hip - mid_shoulder))
        if np.isfinite(torso_length) and torso_length > 1e-12:
            return torso_length

    if np.isfinite(shoulder_width) and shoulder_width > 1e-12:
        return shoulder_width
    return float("nan")


def compute_wrist_metrics(
    prev_wrist: ArrayLike,
    curr_wrist: ArrayLike,
    torso_scale: float,
    dt: float,
) -> tuple[NDArray[np.float64], float]:
    """Calculate normalized wrist velocity and scalar speed.

    Returns:
        A tuple ``(velocity, speed)`` where velocity is ``[vx, vy]`` and speed
        is its Euclidean magnitude. Units are torso lengths per second.

    Invalid coordinates, non-positive time deltas, or unusable torso scales
    produce ``NaN`` values rather than misleading motion estimates.
    """
    if dt <= 0.0 or not np.isfinite(dt):
        return np.full(2, np.nan, dtype=np.float64), float("nan")
    if torso_scale <= 1e-12 or not np.isfinite(torso_scale):
        return np.full(2, np.nan, dtype=np.float64), float("nan")

    previous = _xy(prev_wrist)
    current = _xy(curr_wrist)
    if previous.shape != current.shape or previous.shape != (2,):
        raise ValueError("wrist inputs must each represent one x/y point")
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(current)):
        return np.full(2, np.nan, dtype=np.float64), float("nan")

    normalized_velocity = (current - previous) / (float(torso_scale) * float(dt))
    speed = float(np.linalg.norm(normalized_velocity))
    return normalized_velocity, speed


@dataclass(frozen=True)
class ProcessedPose:
    """One pose represented in smoothed pixels and torso-normalized coordinates."""

    pixel_keypoints: dict[str, NDArray[np.float64]]
    normalized_keypoints: dict[str, NDArray[np.float64]] | None
    torso_length_px: float
    origin_name: str

    @property
    def shoulder_width_px(self) -> float:
        """Backward-compatible alias for older UI code (now torso scale)."""
        return self.torso_length_px


def normalize_keypoints_by_torso(
    keypoints: Mapping[str, ArrayLike],
    min_confidence: float = 0.25,
) -> tuple[dict[str, NDArray[np.float64]] | None, float, str]:
    """Center on the neck proxy (or hips) and divide by torso length.

    COCO does not expose a neck joint, so the shoulder midpoint is used as the
    neck proxy.  Torso length is the distance between shoulder and hip
    midpoints.  When hips are temporarily unavailable, shoulder width is a
    stable scale fallback; when shoulders are unavailable, the hip midpoint is
    used as the origin and the last usable scale must be supplied by temporal
    processing.
    """
    left_shoulder = keypoints.get("left_shoulder")
    right_shoulder = keypoints.get("right_shoulder")
    left_hip = keypoints.get("left_hip")
    right_hip = keypoints.get("right_hip")
    shoulders_valid = _valid_keypoint(left_shoulder, min_confidence) and _valid_keypoint(
        right_shoulder, min_confidence
    )
    hips_valid = _valid_keypoint(left_hip, min_confidence) and _valid_keypoint(
        right_hip, min_confidence
    )
    if not shoulders_valid and not hips_valid:
        return None, float("nan"), "unavailable"

    neck = (
        (_xy(left_shoulder) + _xy(right_shoulder)) / 2.0
        if shoulders_valid
        else None
    )
    hip_center = (
        (_xy(left_hip) + _xy(right_hip)) / 2.0 if hips_valid else None
    )
    origin = neck if neck is not None else hip_center
    origin_name = "neck" if neck is not None else "hip"
    assert origin is not None

    if neck is not None and hip_center is not None:
        torso_length = float(np.linalg.norm(hip_center - neck))
    elif shoulders_valid:
        torso_length = float(np.linalg.norm(_xy(left_shoulder) - _xy(right_shoulder)))
    else:
        torso_length = float("nan")
    if not np.isfinite(torso_length) or torso_length <= 1e-12:
        return None, float("nan"), origin_name

    normalized: dict[str, NDArray[np.float64]] = {}
    for name, raw_point in keypoints.items():
        point = np.asarray(raw_point, dtype=np.float64).reshape(-1)
        confidence = float(point[2]) if point.size >= 3 else 1.0
        if point.size < 2 or not np.all(np.isfinite(point[:2])):
            normalized[name] = np.array([np.nan, np.nan, confidence], dtype=np.float64)
            continue
        normalized_xy = (point[:2] - origin) / torso_length
        normalized[name] = np.array(
            [normalized_xy[0], normalized_xy[1], confidence], dtype=np.float64
        )
    return normalized, torso_length, origin_name


def normalize_keypoints_by_shoulders(
    keypoints: Mapping[str, ArrayLike],
    min_confidence: float = 0.25,
) -> tuple[dict[str, NDArray[np.float64]] | None, float]:
    """Center a pose on the shoulder midpoint and divide by shoulder width.

    The returned coordinates are translation- and scale-invariant: the
    shoulder midpoint is ``(0, 0)`` and the distance between shoulders is one.
    Confidence values are preserved. ``(None, NaN)`` is returned when either
    shoulder is unavailable or their separation is degenerate.
    """
    left = keypoints.get("left_shoulder")
    right = keypoints.get("right_shoulder")
    if not (
        _valid_keypoint(left, min_confidence)
        and _valid_keypoint(right, min_confidence)
    ):
        return None, float("nan")

    left_xy = _xy(left)
    right_xy = _xy(right)
    midpoint = (left_xy + right_xy) / 2.0
    shoulder_width = float(np.linalg.norm(left_xy - right_xy))
    if not np.isfinite(shoulder_width) or shoulder_width <= 1e-12:
        return None, float("nan")

    normalized: dict[str, NDArray[np.float64]] = {}
    for name, raw_point in keypoints.items():
        point = np.asarray(raw_point, dtype=np.float64).reshape(-1)
        confidence = float(point[2]) if point.size >= 3 else 1.0
        if point.size < 2 or not np.all(np.isfinite(point[:2])):
            normalized[name] = np.array(
                [np.nan, np.nan, confidence], dtype=np.float64
            )
            continue
        normalized_xy = (point[:2] - midpoint) / shoulder_width
        normalized[name] = np.array(
            [normalized_xy[0], normalized_xy[1], confidence], dtype=np.float64
        )
    return normalized, shoulder_width


class PoseSignalProcessor:
    """One-Euro low-pass keypoints, then produce torso-relative coordinates.

    The One-Euro filter raises its cutoff during rapid movement.  It strongly
    suppresses stationary network jitter while retaining punch transients with
    substantially less phase lag than a fixed moving-window filter.
    """

    def __init__(
        self,
        window_length: int = 5,
        polynomial_order: int = 2,
        min_confidence: float = 0.25,
        sample_frequency: float = 30.0,
        min_cutoff: float = 1.7,
        beta: float = 0.30,
        derivative_cutoff: float = 1.0,
    ) -> None:
        """Configure the rolling filter and shoulder confidence threshold."""
        if window_length not in {5, 7}:
            raise ValueError("window_length must be 5 or 7")
        if polynomial_order < 0 or polynomial_order >= window_length:
            raise ValueError("polynomial_order must be below window_length")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if sample_frequency <= 0.0 or min_cutoff <= 0.0 or derivative_cutoff <= 0.0:
            raise ValueError("filter frequencies must be positive")
        if beta < 0.0:
            raise ValueError("beta cannot be negative")
        self.window_length = int(window_length)
        self.polynomial_order = int(polynomial_order)
        self.min_confidence = float(min_confidence)
        self.sample_frequency = float(sample_frequency)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.derivative_cutoff = float(derivative_cutoff)
        self._states: dict[str, tuple[NDArray[np.float64], NDArray[np.float64], float]] = {}
        self._implicit_timestamp = 0.0

    def reset(self) -> None:
        """Discard temporal history after a seek or stream discontinuity."""
        self._states.clear()
        self._implicit_timestamp = 0.0

    @staticmethod
    def _copy_pose(
        keypoints: Mapping[str, ArrayLike] | None,
    ) -> dict[str, NDArray[np.float64]] | None:
        if keypoints is None:
            return None
        return {
            name: np.asarray(point, dtype=np.float64).reshape(-1).copy()
            for name, point in keypoints.items()
        }

    @staticmethod
    def _alpha(cutoff: NDArray[np.float64] | float, dt: float) -> NDArray[np.float64]:
        time_constant = 1.0 / (2.0 * np.pi * np.asarray(cutoff, dtype=np.float64))
        return 1.0 / (1.0 + time_constant / dt)

    def _smooth_point(
        self,
        name: str,
        value: NDArray[np.float64],
        timestamp: float,
    ) -> NDArray[np.float64]:
        state = self._states.get(name)
        if state is None:
            filtered = value.copy()
            derivative = np.zeros(2, dtype=np.float64)
        else:
            previous, previous_derivative, previous_timestamp = state
            dt = timestamp - previous_timestamp
            if not np.isfinite(dt) or dt <= 0.0:
                dt = 1.0 / self.sample_frequency
            raw_derivative = (value - previous) / dt
            derivative_alpha = self._alpha(self.derivative_cutoff, dt)
            derivative = derivative_alpha * raw_derivative + (1.0 - derivative_alpha) * previous_derivative
            cutoff = self.min_cutoff + self.beta * np.abs(derivative)
            signal_alpha = self._alpha(cutoff, dt)
            filtered = signal_alpha * value + (1.0 - signal_alpha) * previous
        self._states[name] = (filtered.copy(), derivative.copy(), float(timestamp))
        return filtered

    def update(
        self,
        keypoints: Mapping[str, ArrayLike] | None,
        timestamp: float | None = None,
    ) -> ProcessedPose | None:
        """Filter one raw pose and return pixel plus normalized coordinates."""
        current = self._copy_pose(keypoints)
        if current is None:
            return None
        if timestamp is None:
            timestamp = self._implicit_timestamp
            self._implicit_timestamp += 1.0 / self.sample_frequency
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        smoothed: dict[str, NDArray[np.float64]] = {}
        for name, point in current.items():
            confidence = float(point[2]) if point.size >= 3 else 1.0
            current_is_valid = bool(
                point.size >= 2
                and np.all(np.isfinite(point[:2]))
                and confidence >= self.min_confidence
            )
            if not current_is_valid:
                smoothed[name] = np.array(
                    [np.nan, np.nan, confidence], dtype=np.float64
                )
                continue
            filtered_xy = self._smooth_point(name, point[:2], float(timestamp))
            smoothed[name] = np.array(
                [
                    filtered_xy[0],
                    filtered_xy[1],
                    confidence,
                ],
                dtype=np.float64,
            )

        normalized, torso_length, origin_name = normalize_keypoints_by_torso(
            smoothed,
            min_confidence=self.min_confidence,
        )
        return ProcessedPose(
            pixel_keypoints=smoothed,
            normalized_keypoints=normalized,
            torso_length_px=torso_length,
            origin_name=origin_name,
        )


@dataclass(frozen=True)
class ArmMotion:
    """Smoothed shoulder-relative wrist motion for one frame."""

    velocity: NDArray[np.float64]
    speed: float
    reach: float
    extension_velocity: float

    @classmethod
    def missing(cls) -> "ArmMotion":
        """Return an explicitly invalid sample for an occluded arm."""
        return cls(
            velocity=np.full(2, np.nan, dtype=np.float64),
            speed=float("nan"),
            reach=float("nan"),
            extension_velocity=float("nan"),
        )


class WristMotionTracker:
    """Differentiate One-Euro-smoothed, torso-normalized wrists."""

    def __init__(self) -> None:
        """Create independent motion histories for the left and right hands."""
        self._positions: dict[str, NDArray[np.float64] | None] = {
            "left": None,
            "right": None,
        }
        self._reaches: dict[str, float | None] = {"left": None, "right": None}

    def reset(self, hand: str | None = None) -> None:
        """Reset one hand or both hands after an occlusion/discontinuity."""
        hands = (hand,) if hand is not None else ("left", "right")
        for selected_hand in hands:
            if selected_hand not in self._positions:
                raise ValueError("hand must be 'left' or 'right'")
            self._positions[selected_hand] = None
            self._reaches[selected_hand] = None

    def update(
        self,
        hand: str,
        wrist: ArrayLike | None,
        shoulder: ArrayLike | None,
        dt: float,
    ) -> ArmMotion:
        """Return velocity and reach in shoulder-widths per second."""
        if hand not in self._positions:
            raise ValueError("hand must be 'left' or 'right'")
        if (
            wrist is None
            or shoulder is None
            or not np.isfinite(dt)
            or dt <= 0.0
        ):
            self.reset(hand)
            return ArmMotion.missing()

        wrist_xy = _xy(wrist)
        shoulder_xy = _xy(shoulder)
        if (
            wrist_xy.shape != (2,)
            or shoulder_xy.shape != (2,)
            or not np.all(np.isfinite(wrist_xy))
            or not np.all(np.isfinite(shoulder_xy))
        ):
            self.reset(hand)
            return ArmMotion.missing()

        previous_position = self._positions[hand]
        reach = float(np.linalg.norm(wrist_xy - shoulder_xy))
        if previous_position is None:
            self._positions[hand] = wrist_xy.copy()
            self._reaches[hand] = reach
            return ArmMotion.missing()

        velocity = (wrist_xy - previous_position) / float(dt)
        speed = float(np.linalg.norm(velocity))
        previous_reach = self._reaches[hand]
        extension_velocity = (
            (reach - previous_reach) / float(dt)
            if previous_reach is not None
            else float("nan")
        )
        self._positions[hand] = wrist_xy.copy()
        self._reaches[hand] = reach
        return ArmMotion(
            velocity=velocity,
            speed=speed,
            reach=reach,
            extension_velocity=extension_velocity,
        )
