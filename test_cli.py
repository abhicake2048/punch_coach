"""Run CornerCoach analytics against up to 30 seconds of video without the UI."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import cv2
import numpy as np

from core.fatigue_analyzer import FatigueAnalyzer
from core.guard_monitor import GuardMonitor
from core.kinematics import PoseSignalProcessor, WristMotionTracker, calculate_angle
from core.logging_config import configure_logging
from core.pose_engine import PoseEngine
from core.punch_detector import PunchDetector


LOGGER = logging.getLogger("cornercoach.cli")


def _angles(keypoints: dict[str, np.ndarray] | None) -> dict[str, float]:
    """Calculate bilateral elbow angles for one frame."""
    result = {"left": float("nan"), "right": float("nan")}
    if keypoints is None:
        return result
    for side in ("left", "right"):
        result[side] = float(
            calculate_angle(
                keypoints[f"{side}_shoulder"],
                keypoints[f"{side}_elbow"],
                keypoints[f"{side}_wrist"],
            )
        )
    return result


def _wrist(keypoints: dict[str, np.ndarray] | None, hand: str) -> np.ndarray | None:
    """Return a finite wrist point for one hand."""
    if keypoints is None:
        return None
    point = np.asarray(keypoints[f"{hand}_wrist"], dtype=np.float64)[:2]
    return point.copy() if np.all(np.isfinite(point)) else None


def analyze_video(
    video_path: Path,
    stance: str,
    max_seconds: float,
    *,
    min_speed: float = 0.70,
    min_extension_speed: float = 0.25,
    min_extension_gain: float = 0.06,
    min_retraction_gain: float = 0.04,
    refractory_frames: int = 3,
    max_refractory_frames: int = 12,
    max_wrist_speed: float = 20.0,
    max_extension_gain: float = 1.40,
    max_extension_velocity: float = 25.0,
    min_outward_frames: int = 2,
    min_count_angle: float = 45.0,
    savgol_window: int = 5,
    diagnostics_csv: Path | None = None,
    expected_count: int | None = None,
    imgsz: int = 480,
    guard_chin_fraction: float = 0.65,
    guard_wrist_tolerance: float = 0.15,
    fatigue_work_rate_drop: float = 20.0,
    fatigue_speed_drop: float = 15.0,
    straight_min_angle: float = 140.0,
    hook_min_angle: float = 75.0,
    hook_max_angle: float = 130.0,
    hook_horizontal_ratio: float = 0.62,
    uppercut_max_angle: float = 115.0,
    uppercut_upward_ratio: float = 0.45,
) -> int:
    """Process a bounded video segment and print/log the final statistics."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        LOGGER.error("Could not open video: %s", video_path)
        return 2

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 30.0
    dt = 1.0 / fps
    max_frames = max(1, int(round(max_seconds * fps)))

    engine = PoseEngine(weights="yolov8n-pose.pt", confidence_threshold=0.25)
    detector = PunchDetector(
        stance=stance,
        refractory_frames=refractory_frames,
        max_refractory_frames=max_refractory_frames,
        min_speed_threshold=min_speed,
        min_extension_velocity=min_extension_speed,
        min_extension_gain=min_extension_gain,
        min_retraction_gain=min_retraction_gain,
        max_extension_gain=max_extension_gain,
        max_extension_velocity=max_extension_velocity,
        min_outward_frames=min_outward_frames,
        min_count_angle=min_count_angle,
        max_wrist_speed=max_wrist_speed,
        straight_min_angle=straight_min_angle,
        hook_min_angle=hook_min_angle,
        hook_max_angle=hook_max_angle,
        hook_horizontal_ratio=hook_horizontal_ratio,
        uppercut_max_angle=uppercut_max_angle,
        uppercut_upward_ratio=uppercut_upward_ratio,
    )
    guard = GuardMonitor(
        chin_fraction=guard_chin_fraction,
        wrist_tolerance=guard_wrist_tolerance,
    )
    fatigue = FatigueAnalyzer(
        work_rate_drop_threshold=fatigue_work_rate_drop,
        speed_drop_threshold=fatigue_speed_drop,
    )
    signal_processor = PoseSignalProcessor(
        window_length=savgol_window,
        polynomial_order=2,
    )
    motion_tracker = WristMotionTracker()
    diagnostic_rows: list[dict[str, object]] = []

    LOGGER.info(
        "CLI_START video=%s stance=%s fps=%.3f max_seconds=%.1f "
        "count_settings={min_speed=%.3f,min_outward_speed=%.3f,"
        "min_reach_gain=%.3f,min_retraction_gain=%.3f,min_gap=%d,"
        "max_rearm=%d,min_outward_frames=%d,min_angle=%.1f,savgol=%d/2,"
        "max_reach_gain=%.3f,max_outward_speed=%.3f,max_wrist_speed=%.3f}",
        video_path,
        stance,
        fps,
        max_seconds,
        min_speed,
        min_extension_speed,
        min_extension_gain,
        min_retraction_gain,
        refractory_frames,
        max_refractory_frames,
        min_outward_frames,
        min_count_angle,
        savgol_window,
        max_extension_gain,
        max_extension_velocity,
        max_wrist_speed,
    )
    frame_index = 0
    try:
        while frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            timestamp = frame_index / fps
            raw_keypoints = engine.extract_keypoints(frame, imgsz=imgsz)
            processed_pose = signal_processor.update(raw_keypoints)
            keypoints = (
                processed_pose.normalized_keypoints
                if processed_pose is not None
                else None
            )
            elbow_angles = _angles(keypoints)

            frame_events = []
            for hand in ("left", "right"):
                wrist = _wrist(keypoints, hand)
                shoulder = (
                    np.asarray(keypoints[f"{hand}_shoulder"], dtype=np.float64)[:2]
                    if keypoints is not None
                    else None
                )
                if shoulder is not None and not np.all(np.isfinite(shoulder)):
                    shoulder = None
                motion = motion_tracker.update(
                    hand,
                    wrist,
                    shoulder,
                    dt,
                )
                threshold = detector.dynamic_threshold(hand)

                event = detector.update(
                    hand,
                    motion.velocity,
                    motion.speed,
                    elbow_angles[hand],
                    timestamp,
                    frame_index,
                    reach=motion.reach,
                    extension_velocity=motion.extension_velocity,
                )
                if event is not None:
                    fatigue.add_punch(event)
                    frame_events.append(event)
                if diagnostics_csv is not None:
                    diagnostic_rows.append(
                        {
                            "frame": frame_index,
                            "time_s": round(timestamp, 4),
                            "hand": hand,
                            "speed_swps": motion.speed,
                            "extension_swps": motion.extension_velocity,
                            "reach_sw": motion.reach,
                            "elbow_angle_deg": elbow_angles[hand],
                            "dynamic_threshold_lps": threshold,
                            "phase": detector.phase(hand).value,
                            "detected": event is not None,
                            "punch_type": event.punch_type if event else "",
                        }
                    )

            guard.update(
                keypoints,
                timestamp,
                {
                    hand: detector.is_punch_active(hand)
                    for hand in ("left", "right")
                },
            )
            frame_index += 1
            if frame_index % 300 == 0:
                LOGGER.info(
                    "CLI_PROGRESS frame=%d elapsed=%.1fs punches=%d",
                    frame_index,
                    timestamp,
                    len(detector.events),
                )
    finally:
        capture.release()

    for event in detector.flush():
        fatigue.add_punch(event)

    duration = frame_index / fps
    punch_summary = detector.summary(duration)
    guard_summary = guard.summary()
    fatigue_report = fatigue.analyze(duration)
    if diagnostics_csv is not None:
        diagnostics_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(diagnostic_rows[0]) if diagnostic_rows else []
        with diagnostics_csv.open("w", newline="", encoding="utf-8") as stream:
            if fieldnames:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(diagnostic_rows)
        LOGGER.info("Diagnostics CSV: %s", diagnostics_csv)
    LOGGER.info(
        "CLI_SUMMARY duration=%.2fs total=%d ppm=%.2f by_type=%s by_hand=%s "
        "guard=%.2f%% drops=%s fatigue=%s rate_drop=%.2f%% speed_drop=%.2f%%",
        duration,
        punch_summary["total_punches"],
        punch_summary["punches_per_minute"],
        punch_summary["by_type"],
        punch_summary["by_hand"],
        guard_summary["discipline_score"],
        guard_summary["drop_episodes"],
        fatigue_report.fatigued,
        fatigue_report.work_rate_drop_percent,
        fatigue_report.speed_drop_percent,
    )
    if expected_count is not None:
        difference = int(punch_summary["total_punches"]) - expected_count
        LOGGER.info(
            "COUNT_ERROR expected=%d detected=%d difference=%+d absolute_error=%d",
            expected_count,
            punch_summary["total_punches"],
            difference,
            abs(difference),
        )
    return 0


def main() -> int:
    """Parse CLI arguments and analyze one video."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to an MP4 or MOV file")
    parser.add_argument(
        "--stance",
        choices=("orthodox", "southpaw"),
        default="orthodox",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=30.0,
        help="Maximum video duration to process (default: 30)",
    )
    parser.add_argument("--min-speed", type=float, default=0.70)
    parser.add_argument("--min-extension-speed", type=float, default=0.25)
    parser.add_argument("--min-extension-gain", type=float, default=0.06)
    parser.add_argument("--min-retraction-gain", type=float, default=0.04)
    parser.add_argument("--refractory-frames", type=int, default=3)
    parser.add_argument("--max-refractory-frames", type=int, default=12)
    parser.add_argument("--max-wrist-speed", type=float, default=20.0)
    parser.add_argument("--max-extension-gain", type=float, default=1.40)
    parser.add_argument("--max-extension-velocity", type=float, default=25.0)
    parser.add_argument("--min-outward-frames", type=int, default=2)
    parser.add_argument("--min-count-angle", type=float, default=45.0)
    parser.add_argument(
        "--savgol-window",
        type=int,
        choices=(5, 7),
        default=5,
        help="Savitzky-Golay window; polynomial order is fixed at 2",
    )
    parser.add_argument(
        "--diagnostics-csv",
        type=Path,
        help="Optional per-frame signal CSV for plotting and threshold tuning",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        help="Optional manual ground-truth count used to report count error",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=480,
        choices=(320, 480, 640, 800),
        help="YOLO inference size; larger is slower but may improve wrist tracking",
    )
    parser.add_argument("--guard-chin-fraction", type=float, default=0.65)
    parser.add_argument("--guard-wrist-tolerance", type=float, default=0.15)
    parser.add_argument("--fatigue-work-rate-drop", type=float, default=20.0)
    parser.add_argument("--fatigue-speed-drop", type=float, default=15.0)
    parser.add_argument("--straight-min-angle", type=float, default=140.0)
    parser.add_argument("--hook-min-angle", type=float, default=75.0)
    parser.add_argument("--hook-max-angle", type=float, default=130.0)
    parser.add_argument("--hook-horizontal-ratio", type=float, default=0.62)
    parser.add_argument("--uppercut-max-angle", type=float, default=115.0)
    parser.add_argument("--uppercut-upward-ratio", type=float, default=0.45)
    args = parser.parse_args()
    if args.seconds <= 0.0:
        parser.error("--seconds must be positive")
    if not args.video.is_file():
        parser.error(f"video does not exist: {args.video}")

    log_path = configure_logging(console=True)
    LOGGER.info("Detailed rotating log: %s", log_path)
    return analyze_video(
        args.video,
        args.stance,
        args.seconds,
        min_speed=args.min_speed,
        min_extension_speed=args.min_extension_speed,
        min_extension_gain=args.min_extension_gain,
        min_retraction_gain=args.min_retraction_gain,
        refractory_frames=args.refractory_frames,
        max_refractory_frames=args.max_refractory_frames,
        max_wrist_speed=args.max_wrist_speed,
        max_extension_gain=args.max_extension_gain,
        max_extension_velocity=args.max_extension_velocity,
        min_outward_frames=args.min_outward_frames,
        min_count_angle=args.min_count_angle,
        savgol_window=args.savgol_window,
        diagnostics_csv=args.diagnostics_csv,
        expected_count=args.expected_count,
        imgsz=args.imgsz,
        guard_chin_fraction=args.guard_chin_fraction,
        guard_wrist_tolerance=args.guard_wrist_tolerance,
        fatigue_work_rate_drop=args.fatigue_work_rate_drop,
        fatigue_speed_drop=args.fatigue_speed_drop,
        straight_min_angle=args.straight_min_angle,
        hook_min_angle=args.hook_min_angle,
        hook_max_angle=args.hook_max_angle,
        hook_horizontal_ratio=args.hook_horizontal_ratio,
        uppercut_max_angle=args.uppercut_max_angle,
        uppercut_upward_ratio=args.uppercut_upward_ratio,
    )


if __name__ == "__main__":
    raise SystemExit(main())
