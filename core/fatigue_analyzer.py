"""Post-session punch work-rate and velocity fatigue analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class PunchEventLike(Protocol):
    """Minimal punch event interface required by fatigue analysis."""

    timestamp: float
    speed: float


@dataclass(frozen=True)
class FatigueReport:
    """First-third versus last-third output comparison."""

    fatigued: bool
    first_third_count: int
    last_third_count: int
    first_work_rate_ppm: float
    last_work_rate_ppm: float
    work_rate_drop_percent: float
    first_average_speed: float
    last_average_speed: float
    speed_drop_percent: float
    message: str

class FatigueAnalyzer:
    """Collect detected punches and compare early versus late output."""

    def __init__(
        self,
        work_rate_drop_threshold: float = 20.0,
        speed_drop_threshold: float = 15.0,
    ) -> None:
        """Set percentage drops that trigger the fatigue indicator."""
        self.work_rate_drop_threshold = float(work_rate_drop_threshold)
        self.speed_drop_threshold = float(speed_drop_threshold)
        self.timeline: list[PunchEventLike] = []

    def add_punch(self, event: PunchEventLike) -> None:
        """Append one classified punch to the session timeline."""
        self.timeline.append(event)

    @staticmethod
    def _drop_percent(first: float, last: float) -> float:
        """Calculate positive output drop, guarding against empty baselines."""
        if first <= 0.0 or not np.isfinite(first):
            return 0.0
        return max(0.0, (first - last) / first * 100.0)

    def analyze(self, duration_seconds: float) -> FatigueReport:
        """Compare punch rate and mean peak speed in the first/last thirds."""
        if duration_seconds <= 0.0:
            return FatigueReport(
                fatigued=False,
                first_third_count=0,
                last_third_count=0,
                first_work_rate_ppm=0.0,
                last_work_rate_ppm=0.0,
                work_rate_drop_percent=0.0,
                first_average_speed=0.0,
                last_average_speed=0.0,
                speed_drop_percent=0.0,
                message="Insufficient video duration for fatigue analysis.",
            )

        segment_duration = duration_seconds / 3.0
        first = [event for event in self.timeline if event.timestamp < segment_duration]
        last = [
            event
            for event in self.timeline
            if event.timestamp >= 2.0 * segment_duration
        ]
        first_rate = len(first) * 60.0 / segment_duration
        last_rate = len(last) * 60.0 / segment_duration
        first_speed = float(np.mean([event.speed for event in first])) if first else 0.0
        last_speed = float(np.mean([event.speed for event in last])) if last else 0.0
        rate_drop = self._drop_percent(first_rate, last_rate)
        speed_drop = self._drop_percent(first_speed, last_speed)

        enough_data = len(first) >= 2
        fatigued = enough_data and (
            rate_drop >= self.work_rate_drop_threshold
            or speed_drop >= self.speed_drop_threshold
        )
        if not enough_data:
            message = "Not enough first/last-third punches for a reliable fatigue flag."
        elif fatigued:
            message = (
                f"Output drop detected: work rate {rate_drop:.1f}% and peak speed "
                f"{speed_drop:.1f}% lower in the final third."
            )
        else:
            message = "No material first-to-last-third output drop detected."

        return FatigueReport(
            fatigued=fatigued,
            first_third_count=len(first),
            last_third_count=len(last),
            first_work_rate_ppm=first_rate,
            last_work_rate_ppm=last_rate,
            work_rate_drop_percent=rate_drop,
            first_average_speed=first_speed,
            last_average_speed=last_speed,
            speed_drop_percent=speed_drop,
            message=message,
        )
