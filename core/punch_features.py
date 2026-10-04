"""Runtime-only pose normalization and tensor construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .kinematics import calculate_angle


COCO_KEYPOINT_NAMES: Final[tuple[str, ...]] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


def _point(point: ArrayLike | None) -> NDArray[np.float64] | None:
    if point is None:
        return None
    value = np.asarray(point, dtype=np.float64).reshape(-1)
    if value.size < 2:
        return None
    confidence = float(value[2]) if value.size >= 3 and np.isfinite(value[2]) else 0.0
    return np.array([value[0], value[1], confidence], dtype=np.float64)


def _valid(point: NDArray[np.float64] | None, threshold: float) -> bool:
    return bool(
        point is not None
        and np.all(np.isfinite(point[:2]))
        and np.isfinite(point[2])
        and point[2] >= threshold
    )


@dataclass(frozen=True)
class NormalizedPose:
    pixels: dict[str, NDArray[np.float64]]
    normalized: dict[str, NDArray[np.float64]]
    neck_px: NDArray[np.float64]
    hip_center_px: NDArray[np.float64]
    torso_length_px: float
    shoulder_width_px: float
    torso_orientation_deg: float


def normalize_pose_at_neck(
    keypoints: Mapping[str, ArrayLike] | None,
    min_confidence: float,
) -> NormalizedPose | None:
    if keypoints is None:
        return None
    pixels: dict[str, NDArray[np.float64]] = {}
    for name in COCO_KEYPOINT_NAMES:
        value = _point(keypoints.get(name))
        pixels[name] = value if value is not None else np.full(3, np.nan)
    left_shoulder = pixels["left_shoulder"]
    right_shoulder = pixels["right_shoulder"]
    if not (_valid(left_shoulder, min_confidence) and _valid(right_shoulder, min_confidence)):
        return None
    neck = (left_shoulder[:2] + right_shoulder[:2]) / 2.0
    shoulder_width = float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2]))
    if not np.isfinite(shoulder_width) or shoulder_width <= 1e-6:
        return None
    left_hip, right_hip = pixels["left_hip"], pixels["right_hip"]
    if _valid(left_hip, min_confidence) and _valid(right_hip, min_confidence):
        hip_center = (left_hip[:2] + right_hip[:2]) / 2.0
        torso_length = float(np.linalg.norm(hip_center - neck))
    else:
        hip_center = np.full(2, np.nan)
        torso_length = shoulder_width
    if not np.isfinite(torso_length) or torso_length <= 1e-6:
        return None
    normalized: dict[str, NDArray[np.float64]] = {}
    for name, point in pixels.items():
        if _valid(point, min_confidence):
            xy = (point[:2] - neck) / torso_length
            normalized[name] = np.array([xy[0], xy[1], point[2]])
        else:
            confidence = point[2] if np.isfinite(point[2]) else 0.0
            normalized[name] = np.array([np.nan, np.nan, confidence])
    orientation = (
        float(np.degrees(np.arctan2(hip_center[1] - neck[1], hip_center[0] - neck[0])))
        if np.all(np.isfinite(hip_center))
        else float("nan")
    )
    return NormalizedPose(
        pixels, normalized, neck, hip_center, torso_length, shoulder_width, orientation
    )


def _elbow_angle(pose: NormalizedPose, hand: str, threshold: float) -> float:
    points = [pose.normalized[f"{hand}_{joint}"] for joint in ("shoulder", "elbow", "wrist")]
    if not all(_valid(point, threshold) for point in points):
        return float("nan")
    return float(calculate_angle(*points))


@dataclass
class _HandState:
    timestamp: float | None = None
    wrist: NDArray[np.float64] | None = None
    velocity: NDArray[np.float64] | None = None
    reach: float | None = None
    extension_velocity: float | None = None
    elbow_angle: float | None = None


class KinematicFeatureTracker:
    _NAMES = (
        "wrist_neck_x", "wrist_neck_y", "wrist_shoulder_x", "wrist_shoulder_y",
        "wrist_reach", "wrist_extension_velocity", "wrist_extension_acceleration",
        "wrist_velocity_x", "wrist_velocity_y", "wrist_speed",
        "wrist_acceleration_x", "wrist_acceleration_y", "wrist_acceleration_magnitude",
        "elbow_angle", "elbow_angular_velocity",
    )

    def __init__(self, min_confidence: float) -> None:
        self.min_confidence = float(min_confidence)
        self.states = {"left": _HandState(), "right": _HandState()}

    @classmethod
    def _missing(cls) -> dict[str, float]:
        return {name: float("nan") for name in cls._NAMES}

    def update(self, pose: NormalizedPose | None, timestamp: float) -> dict[str, float]:
        output: dict[str, float] = {}
        for hand in ("left", "right"):
            output.update(
                {f"{hand}_{name}": value for name, value in self._update_hand(hand, pose, timestamp).items()}
            )
        return output

    def _update_hand(self, hand: str, pose: NormalizedPose | None, timestamp: float) -> dict[str, float]:
        state = self.states[hand]
        values = self._missing()
        if pose is None:
            self.states[hand] = _HandState()
            return values
        wrist = pose.normalized[f"{hand}_wrist"]
        shoulder = pose.normalized[f"{hand}_shoulder"]
        if not (_valid(wrist, self.min_confidence) and _valid(shoulder, self.min_confidence)):
            self.states[hand] = _HandState()
            return values
        wrist_xy = wrist[:2]
        shoulder_vector = wrist_xy - shoulder[:2]
        reach = float(np.linalg.norm(shoulder_vector))
        angle = _elbow_angle(pose, hand, self.min_confidence)
        values.update(
            wrist_neck_x=float(wrist_xy[0]), wrist_neck_y=float(wrist_xy[1]),
            wrist_shoulder_x=float(shoulder_vector[0]), wrist_shoulder_y=float(shoulder_vector[1]),
            wrist_reach=reach, elbow_angle=angle,
        )
        dt = timestamp - state.timestamp if state.timestamp is not None else float("nan")
        velocity = None
        extension_velocity = None
        if state.wrist is not None and np.isfinite(dt) and dt > 0.0:
            velocity = (wrist_xy - state.wrist) / dt
            values.update(
                wrist_velocity_x=float(velocity[0]),
                wrist_velocity_y=float(velocity[1]),
                wrist_speed=float(np.linalg.norm(velocity)),
            )
            if state.reach is not None:
                extension_velocity = (reach - state.reach) / dt
                values["wrist_extension_velocity"] = float(extension_velocity)
            if state.velocity is not None:
                acceleration = (velocity - state.velocity) / dt
                values.update(
                    wrist_acceleration_x=float(acceleration[0]),
                    wrist_acceleration_y=float(acceleration[1]),
                    wrist_acceleration_magnitude=float(np.linalg.norm(acceleration)),
                )
            if extension_velocity is not None and state.extension_velocity is not None:
                values["wrist_extension_acceleration"] = float(
                    (extension_velocity - state.extension_velocity) / dt
                )
            if np.isfinite(angle) and state.elbow_angle is not None and np.isfinite(state.elbow_angle):
                values["elbow_angular_velocity"] = float((angle - state.elbow_angle) / dt)
        self.states[hand] = _HandState(
            float(timestamp), wrist_xy.copy(), None if velocity is None else velocity.copy(),
            reach, extension_velocity, angle,
        )
        return values


def pose_to_record(pose: NormalizedPose | None) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in COCO_KEYPOINT_NAMES:
        if pose is None:
            pixel = normalized = np.array([np.nan, np.nan, 0.0])
        else:
            pixel, normalized = pose.pixels[name], pose.normalized[name]
        values.update(
            {
                f"{name}_x_px": float(pixel[0]),
                f"{name}_y_px": float(pixel[1]),
                f"{name}_confidence": float(pixel[2]) if np.isfinite(pixel[2]) else 0.0,
                f"{name}_x_norm": float(normalized[0]),
                f"{name}_y_norm": float(normalized[1]),
            }
        )
    values["torso_length_px"] = float(pose.torso_length_px) if pose else float("nan")
    values["torso_orientation_deg"] = float(pose.torso_orientation_deg) if pose else float("nan")
    return values


def _interpolate(values: NDArray[np.float32], max_gap: int = 3) -> NDArray[np.float32]:
    result = values.copy()
    index = 0
    while index < len(result):
        if np.isfinite(result[index]):
            index += 1
            continue
        start = index
        while index < len(result) and not np.isfinite(result[index]):
            index += 1
        gap = index - start
        if start > 0 and index < len(result) and gap <= max_gap:
            result[start:index] = np.linspace(result[start - 1], result[index], gap + 2)[1:-1]
    return result


def build_runtime_arrays(
    records: Sequence[Mapping[str, Any]],
    fps: float,
    schema: Mapping[str, Any],
    minimum_coverage: float,
) -> dict[str, NDArray[np.float32]] | None:
    length = len(records)
    torso = np.asarray([row.get("torso_length_px", np.nan) for row in records], dtype=np.float32)
    if float(np.isfinite(torso).mean()) < minimum_coverage:
        return None
    columns: dict[str, NDArray[np.float32]] = {}
    for joint in COCO_KEYPOINT_NAMES:
        for axis in ("x", "y"):
            name = f"{joint}_{axis}_norm"
            columns[name] = _interpolate(
                np.asarray([row.get(name, np.nan) for row in records], dtype=np.float32)
            )
        confidence_name = f"{joint}_confidence"
        confidence = np.asarray([row.get(confidence_name, 0.0) for row in records], dtype=np.float32)
        columns[confidence_name] = np.nan_to_num(confidence, nan=0.0)
    for name in set(schema["lstm_kinematic_features"]) | set(schema["stgcn_kinematic_features"]):
        if name not in columns:
            columns[name] = np.asarray([row.get(name, np.nan) for row in records], dtype=np.float32)
    rich = np.column_stack([columns[name] for name in schema["lstm_kinematic_features"]])
    rich = np.nan_to_num(rich, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    positions = np.zeros((2, length, 17), dtype=np.float32)
    confidence = np.zeros((length, 17), dtype=np.float32)
    for joint_index, joint in enumerate(COCO_KEYPOINT_NAMES):
        positions[0, :, joint_index] = columns[f"{joint}_x_norm"]
        positions[1, :, joint_index] = columns[f"{joint}_y_norm"]
        confidence[:, joint_index] = columns[f"{joint}_confidence"]
    positions = np.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0)
    velocity = np.zeros_like(positions)
    acceleration = np.zeros_like(positions)
    valid = confidence > 0.0
    dt = 1.0 / fps
    for timestep in range(1, length):
        joint_valid = valid[timestep] & valid[timestep - 1]
        velocity[:, timestep, joint_valid] = (
            positions[:, timestep, joint_valid] - positions[:, timestep - 1, joint_valid]
        ) / dt
    for timestep in range(2, length):
        joint_valid = valid[timestep] & valid[timestep - 1] & valid[timestep - 2]
        acceleration[:, timestep, joint_valid] = (
            velocity[:, timestep, joint_valid] - velocity[:, timestep - 1, joint_valid]
        ) / dt
    graph = np.concatenate([positions, confidence[None], velocity, acceleration], axis=0)
    graph_kinematics = np.column_stack(
        [columns[name] for name in schema["stgcn_kinematic_features"]]
    )
    return {
        "lstm_kinematic": rich,
        "stgcn": graph.astype(np.float32),
        "stgcn_kinematic": np.nan_to_num(
            graph_kinematics, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32),
    }
