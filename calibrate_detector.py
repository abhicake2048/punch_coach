"""Jointly calibrate punch-count parameters across multiple labeled videos."""

from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

from tune_detector import _float_grid, _int_grid, _run_trial


@dataclass(frozen=True)
class JointTrial:
    """One shared parameter set evaluated on every labeled clip."""

    normalized_error: float
    absolute_error: int
    counts: tuple[int, ...]
    parameters: tuple[object, ...]


def _dataset(value: str) -> tuple[Path, int]:
    """Parse a PATH=COUNT dataset specification."""
    try:
        path_text, count_text = value.rsplit("=", 1)
        path = Path(path_text)
        expected = int(count_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("dataset must be PATH=COUNT") from exc
    if expected < 0:
        raise argparse.ArgumentTypeError("expected count cannot be negative")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"diagnostics file does not exist: {path}")
    return path, expected


def main() -> int:
    """Search for parameters that minimize count error across all clips."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=_dataset, metavar="CSV=COUNT")
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

    datasets: list[tuple[str, int, list[dict[str, str]]]] = []
    for path, expected in args.datasets:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            parser.error(f"diagnostics CSV has no rows: {path}")
        datasets.append((path.name, expected, rows))

    parameter_grid = itertools.product(
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
    results: list[JointTrial] = []
    for parameters in parameter_grid:
        counts = tuple(
            _run_trial(rows, parameters).count for _, _, rows in datasets
        )
        errors = [
            abs(count - expected)
            for count, (_, expected, _) in zip(counts, datasets)
        ]
        normalized_error = sum(
            error / max(expected, 1)
            for error, (_, expected, _) in zip(errors, datasets)
        )
        results.append(
            JointTrial(
                normalized_error=normalized_error,
                absolute_error=sum(errors),
                counts=counts,
                parameters=parameters,
            )
        )

    results.sort(key=lambda result: (result.normalized_error, result.absolute_error))
    headings = "  ".join(f"{name} expected={expected}" for name, expected, _ in datasets)
    print(f"counts: {headings}")
    print(
        "observed_counts  total_error  min_speed  ext_speed  ext_gain  "
        "retract_gain  gap  max_speed  max_gain  max_ext_v  outward  min_angle"
    )
    for result in results[: max(args.top, 1)]:
        (
            min_speed,
            ext_speed,
            ext_gain,
            retract_gain,
            refractory,
            max_speed,
            max_gain,
            max_ext_v,
            outward,
            min_angle,
        ) = result.parameters
        print(
            f"{str(result.counts):15s}  {result.absolute_error:11d}  "
            f"{float(min_speed):9.3f}  {float(ext_speed):9.3f}  "
            f"{float(ext_gain):8.3f}  {float(retract_gain):11.3f}  "
            f"{int(refractory):3d}  {float(max_speed):9.2f}  "
            f"{float(max_gain):8.2f}  {float(max_ext_v):9.2f}  "
            f"{int(outward):7d}  {float(min_angle):9.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
