"""Per-arm punch state machine and trajectory-based classification."""

from __future__ import annotations

import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike


LOGGER = logging.getLogger("cornercoach.punch_detector")


class PunchPhase(str, Enum):
    """Internal lifecycle for one arm."""

    READY = "ready"
    ACTIVE = "active"
    REFRACTORY = "refractory"


@dataclass(frozen=True)
class PunchEvent:
    """One classified punch at its observed peak velocity."""

    timestamp: float
    frame_index: int
    hand: str
    punch_type: str
    speed: float
    elbow_angle: float
    velocity_x: float
    velocity_y: float
    extension_gain: float = float("nan")
    peak_extension_velocity: float = float("nan")

    @property
    def label(self) -> str:
        """Return a compact HUD label."""
        return f"{self.hand.upper()} {self.punch_type.upper()}"


@dataclass
class _ArmState:
    """Mutable state maintained independently for each arm."""

    phase: PunchPhase = PunchPhase.READY
    active_frames: int = 0
    below_release_frames: int = 0
    peak: PunchEvent | None = None
    max_elbow_angle: float = float("nan")
    start_reach: float = float("nan")
    max_reach: float = float("nan")
    peak_extension_velocity: float = float("nan")
    refractory_age: int = 0
    retraction_observed: bool = False
    baseline_speeds: deque[float] = field(default_factory=lambda: deque(maxlen=90))


class _RetiredSpeedBurstDetector:
    """Deprecated pre-cycle implementation retained for migration reference.

    The detector establishes a robust rolling motion baseline while each arm is
    ready. A burst above the dynamic threshold enters ``ACTIVE``. Once motion
    decelerates, the peak sample is classified and the arm enters a refractory
    lockout, preventing retraction from being counted as a second punch.
    """

    def __init__(
        self,
        stance: str = "orthodox",
        refractory_frames: int = 2,
        max_refractory_frames: int = 8,
        min_speed_threshold: float = 0.7,
        min_extension_velocity: float = 0.30,
        retraction_velocity: float = 0.10,
        min_extension_gain: float = 0.035,
        single_frame_extension_velocity: float = 2.0,
        release_speed_ratio: float = 0.50,
        max_wrist_speed: float = 20.0,
        max_active_frames: int = 8,
        uppercut_upward_ratio: float = 0.45,
        uppercut_max_angle: float = 115.0,
        hook_horizontal_ratio: float = 0.62,
        hook_min_angle: float = 75.0,
        hook_max_angle: float = 130.0,
        straight_min_angle: float = 140.0,
    ) -> None:
        """Initialize punch heuristics and independent arm states."""
        normalized_stance = stance.strip().lower()
        if normalized_stance not in {"orthodox", "southpaw"}:
            raise ValueError("stance must be 'orthodox' or 'southpaw'")
        if (
            refractory_frames < 0
            or max_refractory_frames < 1
            or max_active_frames < 1
        ):
            raise ValueError("frame counts must be positive")
        if max_refractory_frames < refractory_frames:
            raise ValueError("max_refractory_frames cannot be less than refractory_frames")
        if min_speed_threshold <= 0.0:
            raise ValueError("min_speed_threshold must be positive")
        if min_extension_velocity < 0.0 or retraction_velocity < 0.0:
            raise ValueError("extension/retraction velocities cannot be negative")
        if min_extension_gain < 0.0:
            raise ValueError("min_extension_gain cannot be negative")
        if single_frame_extension_velocity <= min_extension_velocity:
            raise ValueError(
                "single_frame_extension_velocity must exceed min_extension_velocity"
            )
        if not 0.0 < release_speed_ratio < 1.0:
            raise ValueError("release_speed_ratio must be between 0 and 1")
        if max_wrist_speed <= min_speed_threshold:
            raise ValueError("max_wrist_speed must exceed min_speed_threshold")
        if not 0.0 <= uppercut_upward_ratio <= 1.0:
            raise ValueError("uppercut_upward_ratio must be between 0 and 1")
        if not 0.0 <= hook_horizontal_ratio <= 1.0:
            raise ValueError("hook_horizontal_ratio must be between 0 and 1")
        if not 0.0 <= hook_min_angle < hook_max_angle <= 180.0:
            raise ValueError("hook angle bounds must satisfy 0 <= min < max <= 180")
        if not 0.0 <= uppercut_max_angle <= 180.0:
            raise ValueError("uppercut_max_angle must be between 0 and 180")
        if not 0.0 <= straight_min_angle <= 180.0:
            raise ValueError("straight_min_angle must be between 0 and 180")

        self.stance = normalized_stance
        self.refractory_frames = int(refractory_frames)
        self.max_refractory_frames = int(max_refractory_frames)
        self.min_speed_threshold = float(min_speed_threshold)
        self.min_extension_velocity = float(min_extension_velocity)
        self.retraction_velocity = float(retraction_velocity)
        self.min_extension_gain = float(min_extension_gain)
        self.single_frame_extension_velocity = float(single_frame_extension_velocity)
        self.release_speed_ratio = float(release_speed_ratio)
        self.max_wrist_speed = float(max_wrist_speed)
        self.max_active_frames = int(max_active_frames)
        self.uppercut_upward_ratio = float(uppercut_upward_ratio)
        self.uppercut_max_angle = float(uppercut_max_angle)
        self.hook_horizontal_ratio = float(hook_horizontal_ratio)
        self.hook_min_angle = float(hook_min_angle)
        self.hook_max_angle = float(hook_max_angle)
        self.straight_min_angle = float(straight_min_angle)
        self._states = {"left": _ArmState(), "right": _ArmState()}
        self.events: list[PunchEvent] = []

    def _threshold(self, state: _ArmState) -> float:
        """Return a robust adaptive threshold in torso lengths per second."""
        if len(state.baseline_speeds) < 12:
            return self.min_speed_threshold
        baseline = np.asarray(state.baseline_speeds, dtype=np.float64)
        median = float(np.median(baseline))
        mad = float(np.median(np.abs(baseline - median)))
        adaptive = median + 3.5 * max(mad, 0.05)
        return float(
            np.clip(
                adaptive,
                self.min_speed_threshold,
                self.min_speed_threshold * 2.75,
            )
        )

    def dynamic_threshold(self, hand: str) -> float:
        """Expose the current per-arm threshold for diagnostics."""
        normalized_hand = hand.strip().lower()
        if normalized_hand not in self._states:
            raise ValueError("hand must be 'left' or 'right'")
        return self._threshold(self._states[normalized_hand])

    def is_punch_active(self, hand: str) -> bool:
        """Return whether an arm is punching or retracting under lockout."""
        normalized_hand = hand.strip().lower()
        if normalized_hand not in self._states:
            raise ValueError("hand must be 'left' or 'right'")
        return self._states[normalized_hand].phase is not PunchPhase.READY

    def phase(self, hand: str) -> PunchPhase:
        """Return the current public state-machine phase for one hand."""
        normalized_hand = hand.strip().lower()
        if normalized_hand not in self._states:
            raise ValueError("hand must be 'left' or 'right'")
        return self._states[normalized_hand].phase

    def _classify(self, event: PunchEvent, max_elbow_angle: float) -> str:
        """Classify a peak sample from image-plane trajectory and elbow angle."""
        if not np.isfinite(event.elbow_angle) or event.speed <= 0.0:
            return "Unclassified"

        horizontal_ratio = abs(event.velocity_x) / event.speed
        upward_ratio = -event.velocity_y / event.speed  # image y decreases upward

        if (
            upward_ratio >= self.uppercut_upward_ratio
            and event.elbow_angle <= self.uppercut_max_angle
        ):
            return "Uppercut"
        if (
            horizontal_ratio >= self.hook_horizontal_ratio
            and self.hook_min_angle <= event.elbow_angle <= self.hook_max_angle
        ):
            return "Hook"
        if max_elbow_angle >= self.straight_min_angle:
            lead_hand = "left" if self.stance == "orthodox" else "right"
            return "Jab" if event.hand == lead_hand else "Cross"
        return "Unclassified"

    def _begin_active(
        self,
        state: _ArmState,
        hand: str,
        vector: np.ndarray,
        speed: float,
        elbow_angle: float,
        timestamp: float,
        frame_index: int,
        reach: float,
        extension_velocity: float,
    ) -> None:
        """Initialize one outward-motion candidate."""
        state.phase = PunchPhase.ACTIVE
        state.active_frames = 1
        state.below_release_frames = 0
        state.max_elbow_angle = float(elbow_angle)
        state.start_reach = float(reach)
        state.max_reach = float(reach)
        state.peak_extension_velocity = float(extension_velocity)
        state.peak = PunchEvent(
            timestamp=float(timestamp),
            frame_index=int(frame_index),
            hand=hand,
            punch_type="Candidate",
            speed=float(speed),
            elbow_angle=float(elbow_angle),
            velocity_x=float(vector[0]),
            velocity_y=float(vector[1]),
        )

    def _finalize_active(
        self,
        hand: str,
        state: _ArmState,
        *,
        retraction_observed: bool,
    ) -> PunchEvent | None:
        """Validate, classify, log, and store one completed candidate."""
        candidate = state.peak
        max_elbow_angle = state.max_elbow_angle
        extension_gain = (
            state.max_reach - state.start_reach
            if np.isfinite(state.max_reach) and np.isfinite(state.start_reach)
            else float("nan")
        )
        peak_extension_velocity = state.peak_extension_velocity

        state.phase = PunchPhase.REFRACTORY
        state.refractory_age = 0
        state.retraction_observed = retraction_observed
        state.active_frames = 0
        state.below_release_frames = 0
        state.peak = None
        state.max_elbow_angle = float("nan")
        state.start_reach = float("nan")
        state.max_reach = float("nan")
        state.peak_extension_velocity = float("nan")
        if candidate is None:
            return None

        legacy_without_reach = not np.isfinite(extension_gain) and not np.isfinite(
            peak_extension_velocity
        )
        has_reach_evidence = legacy_without_reach or (
            np.isfinite(extension_gain)
            and extension_gain >= self.min_extension_gain
        ) or (
            np.isfinite(peak_extension_velocity)
            and peak_extension_velocity >= self.single_frame_extension_velocity
        )
        if not has_reach_evidence:
            return None

        punch_type = self._classify(candidate, max_elbow_angle)
        reported_angle = (
            max_elbow_angle
            if punch_type in {"Jab", "Cross"} and np.isfinite(max_elbow_angle)
            else candidate.elbow_angle
        )
        event = PunchEvent(
            timestamp=candidate.timestamp,
            frame_index=candidate.frame_index,
            hand=hand,
            punch_type=punch_type,
            speed=candidate.speed,
            elbow_angle=reported_angle,
            velocity_x=candidate.velocity_x,
            velocity_y=candidate.velocity_y,
            extension_gain=extension_gain,
            peak_extension_velocity=peak_extension_velocity,
        )
        self.events.append(event)
        LOGGER.info(
            "PUNCH time=%.3fs frame=%d hand=%s type=%s speed=%.3fL/s "
            "angle=%.1fdeg velocity=(%.3f,%.3f)L/s extension_gain=%.3fL "
            "extension_peak=%.3fL/s",
            event.timestamp,
            event.frame_index,
            event.hand,
            event.punch_type,
            event.speed,
            event.elbow_angle,
            event.velocity_x,
            event.velocity_y,
            event.extension_gain,
            event.peak_extension_velocity,
        )
        return event

    def update(
        self,
        hand: str,
        velocity: ArrayLike,
        speed: float,
        elbow_angle: float,
        timestamp: float,
        frame_index: int,
        reach: float = float("nan"),
        extension_velocity: float = float("nan"),
    ) -> PunchEvent | None:
        """Advance one arm by one frame and optionally emit a punch event."""
        normalized_hand = hand.strip().lower()
        if normalized_hand not in self._states:
            raise ValueError("hand must be 'left' or 'right'")
        vector = np.asarray(velocity, dtype=np.float64).reshape(-1)
        if vector.size < 2:
            raise ValueError("velocity must contain x and y components")

        state = self._states[normalized_hand]
        finite_sample = bool(
            np.isfinite(speed)
            and speed >= 0.0
            and speed <= self.max_wrist_speed
            and np.all(np.isfinite(vector[:2]))
        )

        if state.phase is PunchPhase.REFRACTORY:
            state.refractory_age += 1
            if (
                np.isfinite(extension_velocity)
                and extension_velocity <= -self.retraction_velocity
            ):
                state.retraction_observed = True
            can_rearm = (
                state.refractory_age >= self.refractory_frames
                and state.retraction_observed
            ) or state.refractory_age >= self.max_refractory_frames
            if not can_rearm:
                return None
            state.phase = PunchPhase.READY

        threshold = self._threshold(state)
        if state.phase is PunchPhase.READY:
            if not finite_sample:
                return None
            outward = (
                not np.isfinite(extension_velocity)
                or extension_velocity >= self.min_extension_velocity
            )
            if speed < threshold:
                state.baseline_speeds.append(float(speed))
            if speed < threshold or not outward:
                return None

            self._begin_active(
                state,
                normalized_hand,
                vector,
                float(speed),
                float(elbow_angle),
                float(timestamp),
                int(frame_index),
                float(reach),
                float(extension_velocity),
            )
            return None

        state.active_frames += 1
        if np.isfinite(elbow_angle):
            state.max_elbow_angle = float(
                np.nanmax([state.max_elbow_angle, float(elbow_angle)])
            )
        if finite_sample and (state.peak is None or speed > state.peak.speed):
            state.peak = PunchEvent(
                timestamp=float(timestamp),
                frame_index=int(frame_index),
                hand=normalized_hand,
                punch_type="Candidate",
                speed=float(speed),
                elbow_angle=float(elbow_angle),
                velocity_x=float(vector[0]),
                velocity_y=float(vector[1]),
            )
        if np.isfinite(reach):
            state.max_reach = float(np.nanmax([state.max_reach, float(reach)]))
        if np.isfinite(extension_velocity):
            state.peak_extension_velocity = float(
                np.nanmax([state.peak_extension_velocity, float(extension_velocity)])
            )

        is_retracting = bool(
            np.isfinite(extension_velocity)
            and extension_velocity <= -self.retraction_velocity
        )
        extension_settled = bool(
            np.isfinite(extension_velocity)
            and extension_velocity < self.min_extension_velocity * 0.20
        )
        if (
            not finite_sample
            or speed < threshold * self.release_speed_ratio
            or extension_settled
        ):
            state.below_release_frames += 1
        else:
            state.below_release_frames = 0

        should_finalize = (
            is_retracting
            or state.below_release_frames >= 1
            or state.active_frames >= self.max_active_frames
        )
        if not should_finalize:
            return None

        return self._finalize_active(
            normalized_hand,
            state,
            retraction_observed=is_retracting,
        )

    def flush(self) -> list[PunchEvent]:
        """Finalize valid candidates still active when a video ends."""
        emitted: list[PunchEvent] = []
        for hand, state in self._states.items():
            if state.phase is not PunchPhase.ACTIVE:
                continue
            event = self._finalize_active(
                hand,
                state,
                retraction_observed=False,
            )
            if event is not None:
                emitted.append(event)
        return emitted

    def summary(self, duration_seconds: float) -> dict[str, object]:
        """Return total, type/hand counts, and punches per minute."""
        by_type = Counter(event.punch_type for event in self.events)
        by_hand = Counter(event.hand for event in self.events)
        by_type_and_hand = Counter(
            f"{event.hand.title()} {event.punch_type}" for event in self.events
        )
        punches_per_minute = (
            len(self.events) * 60.0 / duration_seconds
            if duration_seconds > 0.0
            else 0.0
        )
        return {
            "total_punches": len(self.events),
            "punches_per_minute": punches_per_minute,
            "by_type": dict(by_type),
            "by_hand": dict(by_hand),
            "by_type_and_hand": dict(by_type_and_hand),
        }


# The cycle detector supersedes the earlier speed-burst implementation while
# preserving this module's public import path for existing callers.
from .punch_cycle_detector import PunchDetector, PunchEvent, PunchPhase  # noqa: E402,F401

__all__ = ["PunchDetector", "PunchEvent", "PunchPhase"]
