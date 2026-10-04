"""Skeleton and compact boxing-metric overlays using OpenCV."""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray


LEFT_COLOR = (80, 220, 80)
RIGHT_COLOR = (60, 170, 255)


def _point(
    keypoints: Mapping[str, ArrayLike],
    name: str,
    min_confidence: float,
) -> tuple[int, int] | None:
    """Convert one valid keypoint into an OpenCV pixel tuple."""
    raw = keypoints.get(name)
    if raw is None:
        return None
    value = np.asarray(raw, dtype=np.float64).reshape(-1)
    if value.size < 2 or not np.all(np.isfinite(value[:2])):
        return None
    if value.size >= 3 and (not np.isfinite(value[2]) or value[2] < min_confidence):
        return None
    return int(round(value[0])), int(round(value[1]))


def _draw_limb(
    frame: NDArray[np.uint8],
    points: list[tuple[int, int] | None],
    color: tuple[int, int, int],
) -> None:
    """Draw available consecutive limb segments and joints."""
    for start, end in zip(points, points[1:]):
        if start is not None and end is not None:
            cv2.line(frame, start, end, color, 3, cv2.LINE_AA)
    for point in points:
        if point is not None:
            cv2.circle(frame, point, 5, color, -1, cv2.LINE_AA)


def _finite_metric(metrics: Mapping[str, float] | None, side: str) -> float | None:
    """Read a finite metric value by side."""
    if metrics is None or side not in metrics:
        return None
    value = float(metrics[side])
    return value if np.isfinite(value) else None


def draw_skeleton(
    frame: NDArray[np.uint8],
    keypoints: Mapping[str, ArrayLike] | None,
    elbow_angles: Mapping[str, float] | None = None,
    wrist_speeds: Mapping[str, float] | None = None,
    min_confidence: float = 0.25,
    punch_label: str | None = None,
    guard_warning: str | None = None,
    session_stats: Mapping[str, str | int | float] | None = None,
) -> NDArray[np.uint8]:
    """Return a frame with skeleton, local metrics, and coaching HUD labels.

    ``elbow_angles`` and ``wrist_speeds`` use ``"left"`` and ``"right"`` keys.
    The input frame is copied so the caller's image is not modified.
    """
    annotated = frame.copy()
    if keypoints is None:
        cv2.putText(
            annotated,
            "No pose detected",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        keypoints = {}

    side_colors = {"left": LEFT_COLOR, "right": RIGHT_COLOR}
    for side, color in side_colors.items():
        shoulder = _point(keypoints, f"{side}_shoulder", min_confidence)
        elbow = _point(keypoints, f"{side}_elbow", min_confidence)
        wrist = _point(keypoints, f"{side}_wrist", min_confidence)
        _draw_limb(annotated, [shoulder, elbow, wrist], color)

        angle = _finite_metric(elbow_angles, side)
        if elbow is not None and angle is not None:
            cv2.putText(
                annotated,
                f"{angle:.0f} deg",
                (elbow[0] + 8, elbow[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

        speed = _finite_metric(wrist_speeds, side)
        if wrist is not None and speed is not None:
            cv2.putText(
                annotated,
                f"{speed:.2f} TL/s",
                (wrist[0] + 8, wrist[1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

    left_shoulder = _point(keypoints, "left_shoulder", min_confidence)
    right_shoulder = _point(keypoints, "right_shoulder", min_confidence)
    if left_shoulder is not None and right_shoulder is not None:
        cv2.line(
            annotated,
            left_shoulder,
            right_shoulder,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

    if session_stats:
        entries = [f"{name}: {value}" for name, value in session_stats.items()]
        panel_width = min(
            annotated.shape[1] - 20,
            max(230, max((len(entry) for entry in entries), default=0) * 9),
        )
        panel_height = 18 + 25 * len(entries)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.58, annotated, 0.42, 0.0, annotated)
        for index, entry in enumerate(entries):
            cv2.putText(
                annotated,
                entry,
                (20, 35 + index * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    if punch_label:
        text = punch_label
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
        x = max(10, (annotated.shape[1] - text_size[0]) // 2)
        cv2.putText(
            annotated,
            text,
            (x, 55),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if guard_warning:
        text_size, _ = cv2.getTextSize(
            guard_warning, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2
        )
        x = max(10, (annotated.shape[1] - text_size[0]) // 2)
        y = max(45, annotated.shape[0] - 35)
        cv2.putText(
            annotated,
            guard_warning,
            (x, y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.85,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated
