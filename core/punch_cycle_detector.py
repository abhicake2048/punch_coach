"""Cycle-based per-arm punch detection and trajectory classification."""

from __future__ import annotations

import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike


LOGGER = logging.getLogger("cornercoach.punch_detector")


class PunchPhase(str, Enum):
    """Lifecycle of one arm through a complete punch cycle."""

    READY = "ready"
    EXTENDING = "extending"
    RETRACTING = "retracting"


@dataclass(frozen=True)
class PunchEvent:
    """One validated punch recorded at its peak wrist speed."""

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
        """Return a compact label for the video HUD."""
        return f"{self.hand.upper()} {self.punch_type.upper()}"


@dataclass
class _ArmState:
    """Mutable signals for one arm's current punch cycle."""

    phase: PunchPhase = PunchPhase.READY
    baseline_speeds: deque[float] = field(default_factory=lambda: deque(maxlen=90))
    last_reach: float = float("nan")
    start_reach: float = float("nan")
    max_reach: float = float("nan")
    cycle_peak_reach: float = float("nan")
    peak_extension_velocity: float = float("nan")
    max_elbow_angle: float = float("nan")
    peak: PunchEvent | None = None
    active_frames: int = 0
    outward_frames: int = 0
    settled_frames: int = 0
    retraction_frames: int = 0
    retraction_observed: bool = False


class PunchDetector:
    """Count extension/retraction cycles rather than isolated fast frames.

    A candidate needs sustained outward motion and plausible reach gain. It is
    counted at the extension peak. The same arm becomes eligible after a real
    partial retraction plus a short minimum gap. A 12-frame safety ceiling
    prevents a lost keypoint from locking the arm indefinitely.
    """

    def __init__(
        self,
        stance: str = "orthodox",
        refractory_frames: int = 9,
        max_refractory_frames: int = 12,
        min_speed_threshold: float = 0.30,
        min_extension_velocity: float = 0.10,
        retraction_velocity: float = 0.12,
        min_extension_gain: float = 0.04,
        min_retraction_gain: float = 0.02,
        max_extension_gain: float = 1.40,
        max_extension_velocity: float = 25.0,
        min_outward_frames: int = 2,
        min_count_angle: float = 20.0,
        release_speed_ratio: float = 0.50,
        max_wrist_speed: float = 20.0,
        max_active_frames: int = 10,
        uppercut_upward_ratio: float = 0.45,
        uppercut_max_angle: float = 115.0,
        hook_horizontal_ratio: float = 0.62,
        hook_min_angle: float = 75.0,
        hook_max_angle: float = 130.0,
        straight_min_angle: float = 140.0,
    ) -> None:
        """Initialize cycle, quality, and classification thresholds."""
        normalized_stance = stance.strip().lower()
        if normalized_stance not in {"orthodox", "southpaw"}:
            raise ValueError("stance must be 'orthodox' or 'southpaw'")
        if refractory_frames < 0 or max_refractory_frames < 1:
            raise ValueError("refractory frame counts cannot be negative")
        if max_refractory_frames < refractory_frames:
            raise ValueError("max_refractory_frames cannot be less than refractory_frames")
        if min_outward_frames < 1 or max_active_frames < min_outward_frames:
            raise ValueError("active-frame limits are inconsistent")
        if min_speed_threshold <= 0.0 or max_wrist_speed <= min_speed_threshold:
            raise ValueError("wrist speed thresholds are inconsistent")
        if min_extension_velocity <= 0.0 or retraction_velocity < 0.0:
            raise ValueError("extension/retraction thresholds are inconsistent")
        if not 0.0 <= min_extension_gain < max_extension_gain:
            raise ValueError("extension gain bounds are inconsistent")
        if min_retraction_gain < 0.0:
            raise ValueError("min_retraction_gain cannot be negative")
        if max_extension_velocity <= min_extension_velocity:
            raise ValueError("max_extension_velocity must exceed its minimum")
        if not 0.0 < release_speed_ratio < 1.0:
            raise ValueError("release_speed_ratio must be between 0 and 1")
        if not 0.0 <= min_count_angle <= 180.0:
            raise ValueError("min_count_angle must be between 0 and 180")
        if not 0.0 <= hook_min_angle < hook_max_angle <= 180.0:
            raise ValueError("hook angle bounds are inconsistent")

        self.stance = normalized_stance
        self.refractory_frames = int(refractory_frames)
        self.max_refractory_frames = int(max_refractory_frames)
        self.min_speed_threshold = float(min_speed_threshold)
        self.min_extension_velocity = float(min_extension_velocity)
        self.retraction_velocity = float(retraction_velocity)
        self.min_extension_gain = float(min_extension_gain)
        self.min_retraction_gain = float(min_retraction_gain)
        self.max_extension_gain = float(max_extension_gain)
        self.max_extension_velocity = float(max_extension_velocity)
        self.min_outward_frames = int(min_outward_frames)
        self.min_count_angle = float(min_count_angle)
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

    def _state(self, hand: str) -> tuple[str, _ArmState]:
        normalized = hand.strip().lower()
        if normalized not in self._states:
            raise ValueError("hand must be 'left' or 'right'")
        return normalized, self._states[normalized]

    def _threshold(self, state: _ArmState) -> float:
        if len(state.baseline_speeds) < 12:
            return self.min_speed_threshold
        baseline = np.asarray(state.baseline_speeds, dtype=np.float64)
        median = float(np.median(baseline))
        mad = float(np.median(np.abs(baseline - median)))
        return float(
            np.clip(
                median + 3.5 * max(mad, 0.05),
                self.min_speed_threshold,
                self.min_speed_threshold * 2.75,
            )
        )

    def dynamic_threshold(self, hand: str) -> float:
        """Return the current speed threshold for diagnostics."""
        _, state = self._state(hand)
        return self._threshold(state)

    def phase(self, hand: str) -> PunchPhase:
        """Return one arm's current cycle phase."""
        _, state = self._state(hand)
        return state.phase

    def is_punch_active(self, hand: str) -> bool:
        """Return whether the arm is extending or completing retraction."""
        _, state = self._state(hand)
        return state.phase is not PunchPhase.READY

    def _classify(self, candidate: PunchEvent, max_angle: float) -> str:
        if not np.isfinite(candidate.elbow_angle) or candidate.speed <= 0.0:
            return "Unclassified"
        horizontal_ratio = abs(candidate.velocity_x) / candidate.speed
        upward_ratio = -candidate.velocity_y / candidate.speed
        if (
            upward_ratio >= self.uppercut_upward_ratio
            and candidate.elbow_angle <= self.uppercut_max_angle
        ):
            return "Uppercut"
        if (
            horizontal_ratio >= self.hook_horizontal_ratio
            and self.hook_min_angle <= candidate.elbow_angle <= self.hook_max_angle
        ):
            return "Hook"
        if max_angle >= self.straight_min_angle:
            lead_hand = "left" if self.stance == "orthodox" else "right"
            return "Jab" if candidate.hand == lead_hand else "Cross"
        return "Unclassified"

    @staticmethod
    def _reset_candidate(state: _ArmState) -> None:
        state.start_reach = float("nan")
        state.max_reach = float("nan")
        state.peak_extension_velocity = float("nan")
        state.max_elbow_angle = float("nan")
        state.peak = None
        state.active_frames = 0
        state.outward_frames = 0
        state.settled_frames = 0

    def _begin_extension(
        self,
        hand: str,
        state: _ArmState,
        vector: np.ndarray,
        speed: float,
        elbow_angle: float,
        timestamp: float,
        frame_index: int,
        reach: float,
        extension_velocity: float,
    ) -> None:
        prior_reach = state.last_reach
        self._reset_candidate(state)
        state.phase = PunchPhase.EXTENDING
        state.start_reach = prior_reach if np.isfinite(prior_reach) else reach
        state.max_reach = reach
        state.peak_extension_velocity = extension_velocity
        state.max_elbow_angle = elbow_angle
        state.active_frames = 1
        state.outward_frames = 1
        state.peak = PunchEvent(
            timestamp=timestamp,
            frame_index=frame_index,
            hand=hand,
            punch_type="Candidate",
            speed=speed,
            elbow_angle=elbow_angle,
            velocity_x=float(vector[0]),
            velocity_y=float(vector[1]),
        )

    def _candidate_event(self, hand: str, state: _ArmState) -> PunchEvent | None:
        candidate = state.peak
        extension_gain = (
            state.max_reach - state.start_reach
            if np.isfinite(state.max_reach) and np.isfinite(state.start_reach)
            else float("nan")
        )
        valid = bool(
            candidate is not None
            and state.outward_frames >= self.min_outward_frames
            and np.isfinite(extension_gain)
            and self.min_extension_gain <= extension_gain <= self.max_extension_gain
            and np.isfinite(state.peak_extension_velocity)
            and self.min_extension_velocity
            <= state.peak_extension_velocity
            <= self.max_extension_velocity
            and np.isfinite(state.max_elbow_angle)
            and state.max_elbow_angle >= self.min_count_angle
        )
        if not valid or candidate is None:
            LOGGER.debug(
                "CANDIDATE_REJECTED hand=%s outward_frames=%d gain=%.3f "
                "extension_peak=%.3f max_angle=%.1f",
                hand,
                state.outward_frames,
                extension_gain,
                state.peak_extension_velocity,
                state.max_elbow_angle,
            )
            return None

        punch_type = self._classify(candidate, state.max_elbow_angle)
        reported_angle = (
            state.max_elbow_angle
            if punch_type in {"Jab", "Cross"}
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
            peak_extension_velocity=state.peak_extension_velocity,
        )
        self.events.append(event)
        LOGGER.info(
            "PUNCH time=%.3fs frame=%d hand=%s type=%s speed=%.3fTL/s "
            "angle=%.1fdeg extension_gain=%.3fTL extension_peak=%.3fTL/s "
            "outward_frames=%d",
            event.timestamp,
            event.frame_index,
            event.hand,
            event.punch_type,
            event.speed,
            event.elbow_angle,
            event.extension_gain,
            event.peak_extension_velocity,
            state.outward_frames,
        )
        return event

    def _finish_extension(self, hand: str, state: _ArmState) -> PunchEvent | None:
        state.cycle_peak_reach = state.max_reach
        event = self._candidate_event(hand, state)
        self._reset_candidate(state)
        state.phase = PunchPhase.RETRACTING
        state.retraction_frames = 0
        state.retraction_observed = False
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
        """Advance one arm by one frame and optionally emit one punch."""
        normalized_hand, state = self._state(hand)
        vector = np.asarray(velocity, dtype=np.float64).reshape(-1)
        if vector.size < 2:
            raise ValueError("velocity must contain x and y components")
        finite_sample = bool(
            np.isfinite(speed)
            and 0.0 <= speed <= self.max_wrist_speed
            and np.all(np.isfinite(vector[:2]))
            and np.isfinite(reach)
            and np.isfinite(extension_velocity)
        )
        threshold = self._threshold(state)

        if state.phase is PunchPhase.RETRACTING:
            state.retraction_frames += 1
            if finite_sample and np.isfinite(state.cycle_peak_reach):
                retraction_gain = state.cycle_peak_reach - reach
                if (
                    retraction_gain >= self.min_retraction_gain
                    or extension_velocity <= -self.retraction_velocity
                ):
                    state.retraction_observed = True
            can_rearm = (
                state.retraction_frames >= self.refractory_frames
                and state.retraction_observed
            ) or state.retraction_frames >= self.max_refractory_frames
            if not can_rearm:
                if np.isfinite(reach):
                    state.last_reach = reach
                return None
            state.phase = PunchPhase.READY
            state.cycle_peak_reach = float("nan")

        if state.phase is PunchPhase.READY:
            outward = bool(
                finite_sample
                and speed >= threshold
                and extension_velocity >= self.min_extension_velocity
            )
            if outward:
                self._begin_extension(
                    normalized_hand,
                    state,
                    vector,
                    float(speed),
                    float(elbow_angle),
                    float(timestamp),
                    int(frame_index),
                    float(reach),
                    float(extension_velocity),
                )
            elif finite_sample and speed < threshold:
                state.baseline_speeds.append(float(speed))
            if np.isfinite(reach):
                state.last_reach = reach
            return None

        state.active_frames += 1
        if finite_sample:
            if extension_velocity >= self.min_extension_velocity * 0.50:
                state.outward_frames += 1
            if state.peak is None or speed > state.peak.speed:
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
                state.max_reach = (
                    max(state.max_reach, reach)
                    if np.isfinite(state.max_reach)
                    else float(reach)
                )
            if np.isfinite(elbow_angle):
                state.max_elbow_angle = (
                    max(state.max_elbow_angle, elbow_angle)
                    if np.isfinite(state.max_elbow_angle)
                    else float(elbow_angle)
                )
            if np.isfinite(extension_velocity):
                state.peak_extension_velocity = (
                    max(state.peak_extension_velocity, extension_velocity)
                    if np.isfinite(state.peak_extension_velocity)
                    else float(extension_velocity)
                )

        reversing = bool(
            finite_sample and extension_velocity <= -self.retraction_velocity
        )
        settled = bool(
            not finite_sample
            or speed < threshold * self.release_speed_ratio
            or extension_velocity < self.min_extension_velocity * 0.15
        )
        state.settled_frames = state.settled_frames + 1 if settled else 0
        should_finish = (
            reversing
            or state.settled_frames >= 2
            or state.active_frames >= self.max_active_frames
        )
        if np.isfinite(reach):
            state.last_reach = reach
        if not should_finish:
            return None
        return self._finish_extension(normalized_hand, state)

    def flush(self) -> list[PunchEvent]:
        """Close candidates still extending when the stream ends."""
        emitted: list[PunchEvent] = []
        for hand, state in self._states.items():
            if state.phase is not PunchPhase.EXTENDING:
                continue
            event = self._finish_extension(hand, state)
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
