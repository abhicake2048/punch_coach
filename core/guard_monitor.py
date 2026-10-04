"""Confidence-aware guard monitoring with timestamped drop episodes."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike


LOGGER = logging.getLogger("cornercoach.guard_monitor")
GUARD_LEVELS = ("below_chin", "below_shoulder")


@dataclass(frozen=True)
class GuardDropEvent:
    """One continuous guard-drop episode for a non-punching hand."""

    hand: str
    start_time_s: float
    end_time_s: float
    duration_s: float
    deepest_level: str
    start_wrist_y: float
    start_chin_line_y: float | None
    start_shoulder_line_y: float
    end_reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class _ActiveDrop:
    hand: str
    start_time_s: float
    deepest_level: str
    start_wrist_y: float
    start_chin_line_y: float | None
    start_shoulder_line_y: float


@dataclass(frozen=True)
class GuardStatus:
    """Guard result for one eligible video frame."""

    evaluated: bool
    safe: bool
    dropped_hands: tuple[str, ...]
    dropped_levels: tuple[tuple[str, str], ...] = ()

    @property
    def warning(self) -> str | None:
        """Return a human-readable HUD warning when guard is dropped."""
        if not self.dropped_levels:
            return None
        labels = [
            f"{hand.upper()} BELOW {level.removeprefix('below_').upper()}"
            for hand, level in self.dropped_levels
        ]
        return "GUARD DROP: " + " / ".join(labels)


def _finite_xy(
    keypoints: Mapping[str, ArrayLike] | None,
    name: str,
) -> np.ndarray | None:
    """Return a finite point or None for missing/occluded keypoints."""
    if keypoints is None or name not in keypoints:
        return None
    point = np.asarray(keypoints[name], dtype=np.float64).reshape(-1)
    if point.size < 2 or not np.all(np.isfinite(point[:2])):
        return None
    return point[:2]


class GuardMonitor:
    """Score guard position only for arms that are not currently punching.

    COCO pose has no closed-palm landmark, so the wrist keypoint is used as the
    glove-cuff/palm proxy. Image y increases downward. A wrist below the chin
    line is a guard drop; crossing the shoulder line is the more severe level.
    """

    def __init__(
        self,
        chin_fraction: float = 0.65,
        wrist_tolerance: float = 0.15,
    ) -> None:
        if not 0.0 <= chin_fraction <= 1.0:
            raise ValueError("chin_fraction must be between 0 and 1")
        if wrist_tolerance < 0.0:
            raise ValueError("wrist_tolerance cannot be negative")
        self.chin_fraction = float(chin_fraction)
        self.wrist_tolerance = float(wrist_tolerance)
        self.eligible_frames = 0
        self.safe_frames = 0
        self.arm_evaluations = {"left": 0, "right": 0}
        self.arm_safe = {"left": 0, "right": 0}
        self.drop_episodes = {"left": 0, "right": 0}
        self.events: list[GuardDropEvent] = []
        self._active_drops: dict[str, _ActiveDrop | None] = {
            "left": None,
            "right": None,
        }
        self._last_evaluated_time: dict[str, float | None] = {
            "left": None,
            "right": None,
        }

    def _close_drop(self, hand: str, timestamp: float, reason: str) -> None:
        active = self._active_drops[hand]
        if active is None:
            return
        end_time = max(float(timestamp), active.start_time_s)
        event = GuardDropEvent(
            hand=hand,
            start_time_s=active.start_time_s,
            end_time_s=end_time,
            duration_s=end_time - active.start_time_s,
            deepest_level=active.deepest_level,
            start_wrist_y=active.start_wrist_y,
            start_chin_line_y=active.start_chin_line_y,
            start_shoulder_line_y=active.start_shoulder_line_y,
            end_reason=reason,
        )
        self.events.append(event)
        self._active_drops[hand] = None
        LOGGER.info(
            "GUARD_DROP_END time=%.3fs hand=%s duration=%.3fs level=%s reason=%s",
            end_time,
            hand,
            event.duration_s,
            event.deepest_level,
            reason,
        )

    def _mark_tracking_gap(self, hand: str) -> None:
        last_time = self._last_evaluated_time[hand]
        if last_time is not None:
            self._close_drop(hand, last_time, "tracking_lost")
        self._last_evaluated_time[hand] = None

    def update(
        self,
        keypoints: Mapping[str, ArrayLike] | None,
        timestamp: float,
        punch_active: Mapping[str, bool] | None = None,
    ) -> GuardStatus:
        """Evaluate one frame and update per-hand episodes and aggregate score."""
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("timestamp must be a finite non-negative value")
        active = punch_active or {}
        nose = _finite_xy(keypoints, "nose")
        left_shoulder = _finite_xy(keypoints, "left_shoulder")
        right_shoulder = _finite_xy(keypoints, "right_shoulder")
        shoulder_width = (
            float(np.linalg.norm(left_shoulder - right_shoulder))
            if left_shoulder is not None and right_shoulder is not None
            else 0.0
        )
        tolerance = self.wrist_tolerance * shoulder_width
        dropped_levels: list[tuple[str, str]] = []
        evaluated_hands: list[str] = []

        for hand in ("left", "right"):
            if bool(active.get(hand, False)):
                self._close_drop(hand, timestamp, "punch_started")
                self._last_evaluated_time[hand] = None
                continue

            wrist = _finite_xy(keypoints, f"{hand}_wrist")
            shoulder = _finite_xy(keypoints, f"{hand}_shoulder")
            if wrist is None or shoulder is None:
                self._mark_tracking_gap(hand)
                continue

            evaluated_hands.append(hand)
            self.arm_evaluations[hand] += 1
            self._last_evaluated_time[hand] = float(timestamp)
            shoulder_line_y = float(shoulder[1] + tolerance)
            chin_line_y: float | None = None
            if nose is not None:
                candidate = float(
                    nose[1] + self.chin_fraction * (shoulder[1] - nose[1]) + tolerance
                )
                chin_line_y = min(candidate, shoulder_line_y)

            level: str | None = None
            if wrist[1] > shoulder_line_y:
                level = "below_shoulder"
            elif chin_line_y is not None and wrist[1] > chin_line_y:
                level = "below_chin"

            if level is None:
                self.arm_safe[hand] += 1
                self._close_drop(hand, timestamp, "recovered")
                continue

            dropped_levels.append((hand, level))
            active_drop = self._active_drops[hand]
            if active_drop is None:
                self.drop_episodes[hand] += 1
                self._active_drops[hand] = _ActiveDrop(
                    hand=hand,
                    start_time_s=float(timestamp),
                    deepest_level=level,
                    start_wrist_y=float(wrist[1]),
                    start_chin_line_y=chin_line_y,
                    start_shoulder_line_y=shoulder_line_y,
                )
                LOGGER.info(
                    "GUARD_DROP_START time=%.3fs hand=%s level=%s wrist_y=%.3f "
                    "chin_line_y=%s shoulder_line_y=%.3f",
                    timestamp,
                    hand,
                    level,
                    wrist[1],
                    f"{chin_line_y:.3f}" if chin_line_y is not None else "unavailable",
                    shoulder_line_y,
                )
            elif level == "below_shoulder":
                active_drop.deepest_level = "below_shoulder"

        if not evaluated_hands:
            return GuardStatus(evaluated=False, safe=False, dropped_hands=())

        self.eligible_frames += 1
        frame_safe = not dropped_levels
        if frame_safe:
            self.safe_frames += 1
        return GuardStatus(
            evaluated=True,
            safe=frame_safe,
            dropped_hands=tuple(hand for hand, _ in dropped_levels),
            dropped_levels=tuple(dropped_levels),
        )

    def finalize(self, timestamp: float) -> None:
        """Close any open drop episode at the end of the video."""
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("timestamp must be a finite non-negative value")
        for hand in ("left", "right"):
            self._close_drop(hand, float(timestamp), "video_ended")

    @property
    def discipline_score(self) -> float:
        """Return safe non-punching-hand observations as a percentage."""
        total_evaluations = sum(self.arm_evaluations.values())
        if total_evaluations == 0:
            return 0.0
        return 100.0 * sum(self.arm_safe.values()) / total_evaluations

    def summary(self) -> dict[str, object]:
        """Return score, per-arm aggregates, and the timestamped timeline."""
        per_arm_scores = {
            hand: (
                100.0 * self.arm_safe[hand] / self.arm_evaluations[hand]
                if self.arm_evaluations[hand]
                else 0.0
            )
            for hand in ("left", "right")
        }
        level_counts = {level: 0 for level in GUARD_LEVELS}
        for event in self.events:
            level_counts[event.deepest_level] += 1
        return {
            "discipline_score": self.discipline_score,
            "eligible_frames": self.eligible_frames,
            "safe_frames": self.safe_frames,
            "arm_evaluations": dict(self.arm_evaluations),
            "drop_episodes": dict(self.drop_episodes),
            "drop_levels": level_counts,
            "per_arm_scores": per_arm_scores,
            "events": [event.to_dict() for event in self.events],
        }
