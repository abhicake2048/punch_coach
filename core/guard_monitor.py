"""Confidence-aware boxing guard monitoring."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


LOGGER = logging.getLogger("cornercoach.guard_monitor")


@dataclass(frozen=True)
class GuardStatus:
    """Guard result for one eligible video frame."""

    evaluated: bool
    safe: bool
    dropped_hands: tuple[str, ...]

    @property
    def warning(self) -> str | None:
        """Return a human-readable HUD warning when guard is dropped."""
        if not self.dropped_hands:
            return None
        hands = " & ".join(hand.upper() for hand in self.dropped_hands)
        return f"GUARD DROP: {hands}"


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
    """Track guard safety on frames where the corresponding arm is idle."""

    def __init__(
        self,
        chin_fraction: float = 0.65,
        wrist_tolerance: float = 0.15,
    ) -> None:
        """Create a monitor with a chin line between nose and shoulders.

        ``chin_fraction`` interpolates from nose (0) to shoulder level (1).
        This is more tolerant of pose-model nose placement than using nose Y
        directly while still marking wrists below the upper torso as unsafe.
        """
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
        self._was_dropped = {"left": False, "right": False}

    def update(
        self,
        keypoints: Mapping[str, ArrayLike] | None,
        timestamp: float,
        punch_active: Mapping[str, bool] | None = None,
    ) -> GuardStatus:
        """Evaluate one frame and update aggregate guard statistics."""
        active = punch_active or {}
        nose = _finite_xy(keypoints, "nose")
        left_shoulder = _finite_xy(keypoints, "left_shoulder")
        right_shoulder = _finite_xy(keypoints, "right_shoulder")
        shoulder_width = (
            float(np.linalg.norm(left_shoulder - right_shoulder))
            if left_shoulder is not None and right_shoulder is not None
            else 0.0
        )
        dropped: list[str] = []
        evaluated_hands: list[str] = []

        for hand in ("left", "right"):
            if bool(active.get(hand, False)):
                continue
            wrist = _finite_xy(keypoints, f"{hand}_wrist")
            shoulder = _finite_xy(keypoints, f"{hand}_shoulder")
            if wrist is None or shoulder is None:
                continue

            evaluated_hands.append(hand)
            self.arm_evaluations[hand] += 1
            reference_nose_y = nose[1] if nose is not None else shoulder[1]
            chin_y = reference_nose_y + self.chin_fraction * (
                shoulder[1] - reference_nose_y
            )
            # The COCO wrist point is commonly placed at the glove cuff rather
            # than the glove center. A shoulder-width-relative tolerance makes
            # guard scoring usable across bare-hand and gloved footage.
            chin_y += self.wrist_tolerance * shoulder_width
            is_dropped = bool(wrist[1] > chin_y)
            if is_dropped:
                dropped.append(hand)
            else:
                self.arm_safe[hand] += 1

            if is_dropped and not self._was_dropped[hand]:
                self.drop_episodes[hand] += 1
                LOGGER.info(
                    "GUARD_DROP_START time=%.3fs hand=%s wrist_y=%.1f "
                    "chin_line_y=%.1f shoulder_y=%.1f",
                    timestamp,
                    hand,
                    wrist[1],
                    chin_y,
                    shoulder[1],
                )
            elif not is_dropped and self._was_dropped[hand]:
                LOGGER.info("GUARD_RECOVERED time=%.3fs hand=%s", timestamp, hand)
            self._was_dropped[hand] = is_dropped

        if not evaluated_hands:
            return GuardStatus(evaluated=False, safe=False, dropped_hands=())

        self.eligible_frames += 1
        frame_safe = not dropped
        if frame_safe:
            self.safe_frames += 1
        return GuardStatus(
            evaluated=True,
            safe=frame_safe,
            dropped_hands=tuple(dropped),
        )

    @property
    def discipline_score(self) -> float:
        """Return safe hand-frame percentage across all eligible observations."""
        total_evaluations = sum(self.arm_evaluations.values())
        if total_evaluations == 0:
            return 0.0
        return 100.0 * sum(self.arm_safe.values()) / total_evaluations

    def summary(self) -> dict[str, object]:
        """Return overall and per-arm guard aggregates."""
        per_arm_scores = {
            hand: (
                100.0 * self.arm_safe[hand] / self.arm_evaluations[hand]
                if self.arm_evaluations[hand]
                else 0.0
            )
            for hand in ("left", "right")
        }
        return {
            "discipline_score": self.discipline_score,
            "eligible_frames": self.eligible_frames,
            "safe_frames": self.safe_frames,
            "drop_episodes": dict(self.drop_episodes),
            "per_arm_scores": per_arm_scores,
        }
