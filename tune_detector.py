"""Replay a diagnostics CSV across punch-detector parameter combinations."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

from core.punch_detector import PunchDetector


@dataclass(frozen=True)
class Trial:
    """One parameter combination and its resulting count."""

    count: int
    left: int
    right: int
    min_speed: float
    min_extension_speed: float
    min_extension_gain: float
    min_retraction_gain: float
    refractory_frames: int
    max_wrist_speed: float
    max_extension_gain: float
    max_extension_velocity: float
    min_outward_frames: int
    min_count_angle: float


def _float_grid(value: str) -> list[float]:
    """Parse a comma-separated floating-point search grid."""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _int_grid(value: str) -> list[int]:
    """Parse a comma-separated integer search grid."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _number(row: dict[str, str], field: str) -> float:
    """Parse a diagnostics field, preserving invalid samples as NaN."""
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _signal_number(row: dict[str, str], current: str, legacy: str) -> float:
    """Read current shoulder-width units with legacy CSV compatibility."""
    value = _number(row, current)
    return value if math.isfinite(value) else _number(row, legacy)


def _run_trial(rows: list[dict[str, str]], parameters: tuple[object, ...]) -> Trial:
    """Replay cached signals through one detector configuration."""
    (
        min_speed,
        min_extension_speed,
        min_extension_gain,
        min_retraction_gain,
        refractory,
        max_speed,
        max_extension_gain,
        max_extension_velocity,
        min_outward_frames,
        min_count_angle,
    ) = parameters
    detector = PunchDetector(
        min_speed_threshold=float(min_speed),
        min_extension_velocity=float(min_extension_speed),
        min_extension_gain=float(min_extension_gain),
        min_retraction_gain=float(min_retraction_gain),
        refractory_frames=int(refractory),
        max_refractory_frames=max(12, int(refractory)),
        max_wrist_speed=float(max_speed),
        max_extension_gain=float(max_extension_gain),
        max_extension_velocity=float(max_extension_velocity),
        min_outward_frames=int(min_outward_frames),
        min_count_angle=float(min_count_angle),
    )
    for row in rows:
        speed = _signal_number(row, "speed_swps", "speed_lps")
        detector.update(
            hand=row["hand"],
            velocity=[speed, 0.0] if math.isfinite(speed) else [float("nan")] * 2,
            speed=speed,
            elbow_angle=_number(row, "elbow_angle_deg"),
            timestamp=_number(row, "time_s"),
            frame_index=int(row["frame"]),
            reach=_signal_number(row, "reach_sw", "reach_l"),
            extension_velocity=_signal_number(
                row, "extension_swps", "extension_lps"
            ),
        )
    detector.flush()
    summary = detector.summary(max(_number(rows[-1], "time_s"), 0.001))
    hands = dict(summary["by_hand"])
    return Trial(
        count=int(summary["total_punches"]),
        left=int(hands.get("left", 0)),
        right=int(hands.get("right", 0)),
        min_speed=float(min_speed),
        min_extension_speed=float(min_extension_speed),
        min_extension_gain=float(min_extension_gain),
        min_retraction_gain=float(min_retraction_gain),
        refractory_frames=int(refractory),
        max_wrist_speed=float(max_speed),
        max_extension_gain=float(max_extension_gain),
        max_extension_velocity=float(max_extension_velocity),
        min_outward_frames=int(min_outward_frames),
        min_count_angle=float(min_count_angle),
    )


def main() -> int:
    """Search parameter grids against cached per-frame motion signals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostics_csv", type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--min-speeds", default="0.65,0.7,0.75")
    parser.add_argument("--extension-speeds", default="0.2,0.25,0.3")
    parser.add_argument("--extension-gains", default="0.05,0.06,0.07")
    parser.add_argument("--retraction-gains", default="0.03,0.04,0.05")
    parser.add_argument("--refractory-frames", default="2,3,4")
    parser.add_argument("--max-wrist-speeds", default="20")
    parser.add_argument("--max-extension-gains", default="1.3,1.4,1.5")
    parser.add_argument("--max-extension-velocities", default="20,25")
    parser.add_argument("--min-outward-frames", default="2")
    parser.add_argument("--min-count-angles", default="40,45,50")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    with args.diagnostics_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        parser.error("diagnostics CSV has no rows")

    grid = itertools.product(
        _float_grid(args.min_speeds),
        _float_grid(args.extension_speeds),
        _float_grid(args.extension_gains),
        _float_grid(args.retraction_gains),
        _int_grid(args.refractory_frames),
        _float_grid(args.max_wrist_speeds),
        _float_grid(args.max_extension_gains),
        _float_grid(args.max_extension_velocities),
        _int_grid(args.min_outward_frames),
        _float_grid(args.min_count_angles),
    )
    trials = [_run_trial(rows, parameters) for parameters in grid]
    if args.expected_count is None:
        trials.sort(key=lambda trial: (-trial.count, trial.min_speed))
    else:
        trials.sort(
            key=lambda trial: (
                abs(trial.count - args.expected_count),
                trial.refractory_frames,
                trial.min_speed,
            )
        )

    print(
        "count left right min_speed ext_speed ext_gain retract_gain gap "
        "max_speed max_gain max_ext_v outward min_angle"
    )
    for trial in trials[: max(args.top, 1)]:
        print(
            f"{trial.count:5d}  {trial.left:4d} {trial.right:5d}  "
            f"{trial.min_speed:9.3f}  {trial.min_extension_speed:9.3f}  "
            f"{trial.min_extension_gain:8.3f}  {trial.min_retraction_gain:11.3f}  "
            f"{trial.refractory_frames:3d}  {trial.max_wrist_speed:9.2f}  "
            f"{trial.max_extension_gain:8.2f}  {trial.max_extension_velocity:9.2f}  "
            f"{trial.min_outward_frames:7d}  {trial.min_count_angle:9.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
