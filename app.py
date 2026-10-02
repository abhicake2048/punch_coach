"""Streamlit interface for tracked top-down boxing analysis."""

from __future__ import annotations

import logging
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st

from core.kinematics import (
    PoseSignalProcessor,
    WristMotionTracker,
    calculate_angle,
)
from core.fatigue_analyzer import FatigueAnalyzer, FatigueReport
from core.guard_monitor import GuardMonitor
from core.logging_config import configure_logging
from core.boxer_pipeline import DEFAULT_INFERENCE_SIZE, TrackedBoxerPosePipeline
from core.pose_engine import BOXING_KEYPOINT_INDICES, PoseEngine
from core.punch_detector import PunchDetector
from visualizer.video_annotator import draw_skeleton


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".mov"}
LOG_PATH = configure_logging()
LOGGER = logging.getLogger("cornercoach.app")


@st.cache_resource(show_spinner=False)
def load_pose_engine() -> PoseEngine:
    """Create one cached YOLO11 pose engine for the Streamlit process."""
    return PoseEngine(weights="yolo11s-pose.pt", confidence_threshold=0.25)


@st.cache_resource(show_spinner=False)
def load_vision_pipeline() -> TrackedBoxerPosePipeline:
    """Create the cached detector/ByteTrack/top-down pose pipeline."""
    return TrackedBoxerPosePipeline(
        pose_engine=load_pose_engine(),
        detector_weights="yolo11s-pose.pt",
    )


def _joint_angles(keypoints: dict[str, np.ndarray] | None) -> dict[str, float]:
    """Calculate left/right elbow angles, returning NaN when unavailable."""
    angles = {"left": float("nan"), "right": float("nan")}
    if keypoints is None:
        return angles

    for side in ("left", "right"):
        try:
            angles[side] = float(
                calculate_angle(
                    keypoints[f"{side}_shoulder"],
                    keypoints[f"{side}_elbow"],
                    keypoints[f"{side}_wrist"],
                )
            )
        except (KeyError, ValueError):
            angles[side] = float("nan")
    return angles


def _metric_text(value: float, suffix: str = "") -> str:
    """Format a metric for a compact Streamlit card."""
    return f"{value:.2f}{suffix}" if np.isfinite(value) else "—"


def _keypoint_rows(keypoints: dict[str, np.ndarray] | None) -> list[dict[str, Any]]:
    """Build serializable rows for the requested boxing keypoints."""
    if keypoints is None:
        return []
    rows: list[dict[str, Any]] = []
    for name, coco_index in BOXING_KEYPOINT_INDICES.items():
        point = keypoints[name]
        rows.append(
            {
                "COCO index": coco_index,
                "Keypoint": name,
                "x": round(float(point[0]), 1) if np.isfinite(point[0]) else None,
                "y": round(float(point[1]), 1) if np.isfinite(point[1]) else None,
                "confidence": round(float(point[2]), 3),
            }
        )
    return rows


def process_image(uploaded_file: Any, pipeline: TrackedBoxerPosePipeline) -> None:
    """Decode, analyze, and render one uploaded image."""
    image_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        st.error("OpenCV could not decode this image.")
        return

    pipeline.reset()
    stages = pipeline.process(frame)
    raw_keypoints = stages.raw_keypoints_original
    processed_pose = PoseSignalProcessor(window_length=5).update(raw_keypoints)
    pixel_keypoints = (
        processed_pose.pixel_keypoints if processed_pose is not None else None
    )
    keypoints = (
        processed_pose.normalized_keypoints if processed_pose is not None else None
    )
    angles = _joint_angles(keypoints)
    LOGGER.info(
        "IMAGE_ANALYSIS file=%s pose_detected=%s left_elbow=%.1f right_elbow=%.1f",
        uploaded_file.name,
        raw_keypoints is not None,
        angles["left"],
        angles["right"],
    )
    annotated = draw_skeleton(frame, pixel_keypoints, elbow_angles=angles)

    image_column, metrics_column = st.columns([2, 1])
    with image_column:
        st.subheader("Pose inspection")
        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True,
        )

    with metrics_column:
        st.subheader("Joint metrics")
        left_column, right_column = st.columns(2)
        left_column.metric("Left elbow", _metric_text(angles["left"], "°"))
        right_column.metric("Right elbow", _metric_text(angles["right"], "°"))

        if processed_pose is None or keypoints is None:
            st.warning("No person was detected with sufficient confidence.")
        else:
            st.metric("Torso scale", _metric_text(processed_pose.torso_length_px, " px"))
            st.caption(
                "Displayed coordinates are One-Euro low-pass filtered; "
                "kinematics are neck-centered and torso-length normalized."
            )
            st.dataframe(
                _keypoint_rows(pixel_keypoints),
                hide_index=True,
                use_container_width=True,
            )


def _valid_xy(keypoints: dict[str, np.ndarray] | None, name: str) -> np.ndarray | None:
    """Return a copy of a finite x/y keypoint or None."""
    if keypoints is None or name not in keypoints:
        return None
    point = np.asarray(keypoints[name], dtype=np.float64)
    if point.size < 2 or not np.all(np.isfinite(point[:2])):
        return None
    return point[:2].copy()


def _transcode_browser_mp4(source_path: Path, output_path: Path) -> None:
    """Transcode an OpenCV intermediate into browser-compatible H.264 MP4."""
    try:
        import imageio_ffmpeg

        ffmpeg_executable = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "H.264 conversion requires imageio-ffmpeg. Reinstall requirements.txt."
        ) from exc

    command = [
        ffmpeg_executable,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "23",
        "-vf",
        "scale=in_range=full:out_range=tv,format=yuv420p",
        "-pix_fmt",
        "yuv420p",
        "-color_range",
        "tv",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        details = completed.stderr.strip() or "ffmpeg produced no playable output"
        raise RuntimeError(f"H.264 video conversion failed: {details}")


def _render_session_dashboard(
    punch_summary: dict[str, object],
    guard_summary: dict[str, object],
    fatigue_report: FatigueReport,
) -> None:
    """Render structured end-of-session boxing analytics."""
    st.subheader("Session dashboard")
    total_column, rate_column, guard_column, fatigue_column = st.columns(4)
    total_column.metric("Total punches", int(punch_summary["total_punches"]))
    rate_column.metric(
        "Punches / minute", f"{float(punch_summary['punches_per_minute']):.1f}"
    )
    guard_column.metric(
        "Guard discipline", f"{float(guard_summary['discipline_score']):.1f}%"
    )
    fatigue_column.metric(
        "Fatigue indicator", "Drop detected" if fatigue_report.fatigued else "Stable"
    )

    statistics_column, fatigue_details_column = st.columns(2)
    with statistics_column:
        st.markdown("**Punch counts by hand and type**")
        combined_counts = punch_summary["by_type_and_hand"]
        count_rows = [
            {"Punch": label, "Count": count}
            for label, count in sorted(dict(combined_counts).items())
        ]
        if count_rows:
            st.dataframe(count_rows, hide_index=True, use_container_width=True)
        else:
            st.info("No punches met the classification thresholds.")

    with fatigue_details_column:
        st.markdown("**Fatigue comparison**")
        st.write(fatigue_report.message)
        st.dataframe(
            [
                {
                    "Segment": "First third",
                    "Punches": fatigue_report.first_third_count,
                    "Work rate (PPM)": round(fatigue_report.first_work_rate_ppm, 1),
                    "Mean peak speed (TL/s)": round(
                        fatigue_report.first_average_speed, 2
                    ),
                },
                {
                    "Segment": "Last third",
                    "Punches": fatigue_report.last_third_count,
                    "Work rate (PPM)": round(fatigue_report.last_work_rate_ppm, 1),
                    "Mean peak speed (TL/s)": round(
                        fatigue_report.last_average_speed, 2
                    ),
                },
            ],
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "Punch classification and fatigue are image-plane heuristics intended "
        "for coaching review, not instrument-grade measurements."
    )


def _analysis_settings_panel() -> dict[str, float | int]:
    """Expose count, classification, guard, and fatigue tuning controls."""
    with st.sidebar.expander("Punch count tuning", expanded=False):
        inference_size = st.select_slider(
            "YOLO input size (pixels)",
            options=[480, 640],
            value=DEFAULT_INFERENCE_SIZE,
            help=(
                "Both detector and pose use a square letterbox of this size. "
                "Source pixels retain their original aspect ratio."
            ),
        )
        min_speed = st.number_input(
            "Minimum wrist speed (torso lengths/s)", 0.1, 5.0, 0.3, 0.05
        )
        min_extension_speed = st.number_input(
            "Minimum outward speed (torso lengths/s)", 0.05, 5.0, 0.10, 0.05
        )
        min_extension_gain = st.number_input(
            "Minimum reach gain (torso lengths)", 0.0, 0.5, 0.04, 0.005
        )
        refractory_frames = st.number_input(
            "Minimum same-hand cycle gap (frames)", 0, 12, 9, 1
        )
        max_refractory_frames = st.number_input(
            "Maximum re-arm wait (frames)", 4, 30, 12, 1
        )
        min_retraction_gain = st.number_input(
            "Partial retraction required (torso lengths)", 0.0, 0.5, 0.02, 0.01
        )
        min_outward_frames = st.number_input(
            "Minimum outward-motion frames", 1, 6, 2, 1
        )
        max_extension_gain = st.number_input(
            "Maximum plausible reach gain", 0.1, 3.0, 1.40, 0.05
        )
        max_extension_velocity = st.number_input(
            "Maximum plausible outward speed", 1.0, 100.0, 25.0, 1.0
        )
        min_count_angle = st.number_input(
            "Minimum cycle elbow angle", 0.0, 180.0, 20.0, 2.5
        )
        max_wrist_speed = st.number_input(
            "Pose-jump rejection speed (torso lengths/s)", 3.0, 100.0, 20.0, 1.0
        )

    with st.sidebar.expander("Punch type tuning", expanded=False):
        straight_min_angle = st.number_input(
            "Jab/Cross minimum extension angle", 90.0, 180.0, 140.0, 2.0
        )
        hook_min_angle = st.number_input(
            "Hook minimum elbow angle", 30.0, 150.0, 75.0, 2.0
        )
        hook_max_angle = st.number_input(
            "Hook maximum elbow angle", 60.0, 180.0, 130.0, 2.0
        )
        hook_horizontal_ratio = st.slider(
            "Hook horizontal-motion ratio", 0.0, 1.0, 0.62, 0.02
        )
        uppercut_max_angle = st.number_input(
            "Uppercut maximum elbow angle", 30.0, 180.0, 115.0, 2.0
        )
        uppercut_upward_ratio = st.slider(
            "Uppercut upward-motion ratio", 0.0, 1.0, 0.45, 0.02
        )

    with st.sidebar.expander("Guard and fatigue tuning", expanded=False):
        guard_chin_fraction = st.slider(
            "Guard line: nose → shoulder", 0.0, 1.0, 0.65, 0.05
        )
        guard_wrist_tolerance = st.slider(
            "Glove-cuff tolerance (torso lengths)", 0.0, 1.0, 0.15, 0.05
        )
        fatigue_work_rate_drop = st.slider(
            "Fatigue work-rate drop (%)", 0.0, 100.0, 20.0, 1.0
        )
        fatigue_speed_drop = st.slider(
            "Fatigue peak-speed drop (%)", 0.0, 100.0, 15.0, 1.0
        )

    return {
        "inference_size": int(inference_size),
        "min_speed": float(min_speed),
        "min_extension_speed": float(min_extension_speed),
        "min_extension_gain": float(min_extension_gain),
        "refractory_frames": int(refractory_frames),
        "max_refractory_frames": int(max_refractory_frames),
        "min_retraction_gain": float(min_retraction_gain),
        "min_outward_frames": int(min_outward_frames),
        "max_extension_gain": float(max_extension_gain),
        "max_extension_velocity": float(max_extension_velocity),
        "min_count_angle": float(min_count_angle),
        "max_wrist_speed": float(max_wrist_speed),
        "straight_min_angle": float(straight_min_angle),
        "hook_min_angle": float(hook_min_angle),
        "hook_max_angle": float(hook_max_angle),
        "hook_horizontal_ratio": float(hook_horizontal_ratio),
        "uppercut_max_angle": float(uppercut_max_angle),
        "uppercut_upward_ratio": float(uppercut_upward_ratio),
        "guard_chin_fraction": float(guard_chin_fraction),
        "guard_wrist_tolerance": float(guard_wrist_tolerance),
        "fatigue_work_rate_drop": float(fatigue_work_rate_drop),
        "fatigue_speed_drop": float(fatigue_speed_drop),
    }


def _completed_analysis_key(
    video_bytes: bytes,
    filename: str,
    stance: str,
    settings: dict[str, float | int],
) -> str:
    """Build a stable key that changes only when input or analysis settings change."""
    digest = hashlib.sha256()
    digest.update(video_bytes)
    digest.update(filename.encode("utf-8"))
    digest.update(stance.encode("utf-8"))
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _render_completed_video(result: dict[str, Any], reused: bool = False) -> None:
    """Render a finished analysis without running either neural network again."""
    st.success(
        "Analysis complete — showing the saved result."
        if reused
        else "Analysis complete."
    )
    _render_session_dashboard(
        result["punch_summary"],
        result["guard_summary"],
        result["fatigue_report"],
    )
    st.subheader("Final annotated video")
    st.video(result["video_bytes"], format="video/mp4")
    st.download_button(
        "Download annotated MP4",
        data=result["video_bytes"],
        file_name=result["output_filename"],
        mime="video/mp4",
        key=f"download-video-{result['analysis_key']}",
    )
    st.caption(
        "Wrist speed is normalized to torso lengths per second; missing pose "
        "measurements are represented as chart gaps."
    )
    with st.expander("Parameters used for this analysis"):
        st.json(result["settings"])
    if LOG_PATH.exists():
        st.download_button(
            "Download analysis log",
            data=LOG_PATH.read_bytes(),
            file_name="cornercoach.log",
            mime="text/plain",
            key=f"download-log-{result['analysis_key']}",
        )


def process_video(
    uploaded_file: Any,
    pipeline: TrackedBoxerPosePipeline,
    stance: str = "orthodox",
    settings: dict[str, float | int] | None = None,
) -> None:
    """Analyze an uploaded video, showing live frames and an annotated result."""
    active_settings = settings or _analysis_settings_panel()
    source_bytes = uploaded_file.getvalue()
    analysis_key = _completed_analysis_key(
        source_bytes,
        uploaded_file.name,
        stance,
        active_settings,
    )
    cached_result = st.session_state.get("completed_video_analysis")
    if cached_result is not None and cached_result.get("analysis_key") == analysis_key:
        _render_completed_video(cached_result, reused=True)
        return

    input_suffix = Path(uploaded_file.name).suffix.lower()
    input_path: Path | None = None
    intermediate_path: Path | None = None
    output_path: Path | None = None
    capture: cv2.VideoCapture | None = None
    writer: cv2.VideoWriter | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=input_suffix) as source:
            source.write(source_bytes)
            input_path = Path(source.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".avi") as intermediate:
            intermediate_path = Path(intermediate.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as destination:
            output_path = Path(destination.name)

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            st.error("OpenCV could not open this video.")
            return

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0.0:
            fps = 30.0
        dt = 1.0 / fps
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            st.error("The video has invalid frame dimensions.")
            return

        writer = cv2.VideoWriter(
            str(intermediate_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            st.error("OpenCV could not initialize the intermediate video writer.")
            return

        punch_detector = PunchDetector(
            stance=stance,
            refractory_frames=int(active_settings["refractory_frames"]),
            max_refractory_frames=int(active_settings["max_refractory_frames"]),
            min_speed_threshold=float(active_settings["min_speed"]),
            min_extension_velocity=float(active_settings["min_extension_speed"]),
            min_extension_gain=float(active_settings["min_extension_gain"]),
            min_retraction_gain=float(active_settings["min_retraction_gain"]),
            min_outward_frames=int(active_settings["min_outward_frames"]),
            max_extension_gain=float(active_settings["max_extension_gain"]),
            max_extension_velocity=float(active_settings["max_extension_velocity"]),
            min_count_angle=float(active_settings["min_count_angle"]),
            max_wrist_speed=float(active_settings["max_wrist_speed"]),
            straight_min_angle=float(active_settings["straight_min_angle"]),
            hook_min_angle=float(active_settings["hook_min_angle"]),
            hook_max_angle=float(active_settings["hook_max_angle"]),
            hook_horizontal_ratio=float(active_settings["hook_horizontal_ratio"]),
            uppercut_max_angle=float(active_settings["uppercut_max_angle"]),
            uppercut_upward_ratio=float(active_settings["uppercut_upward_ratio"]),
        )
        guard_monitor = GuardMonitor(
            chin_fraction=float(active_settings["guard_chin_fraction"]),
            wrist_tolerance=float(active_settings["guard_wrist_tolerance"]),
        )
        fatigue_analyzer = FatigueAnalyzer(
            work_rate_drop_threshold=float(active_settings["fatigue_work_rate_drop"]),
            speed_drop_threshold=float(active_settings["fatigue_speed_drop"]),
        )
        signal_processor = PoseSignalProcessor(sample_frequency=fps)
        motion_tracker = WristMotionTracker()
        pipeline.reset()
        LOGGER.info(
            "VIDEO_START file=%s fps=%.3f frames=%d dimensions=%dx%d stance=%s settings=%s",
            uploaded_file.name,
            fps,
            total_frames,
            width,
            height,
            stance,
            active_settings,
        )

        st.subheader("Live pipeline inspection")
        st.caption(
            "Preprocess → YOLO11 + ByteTrack → exact boxer crop → cropped YOLO11 pose → final analytics"
        )
        stage_columns = st.columns(4)
        preprocess_placeholder = stage_columns[0].empty()
        detection_placeholder = stage_columns[1].empty()
        crop_placeholder = stage_columns[2].empty()
        pose_placeholder = stage_columns[3].empty()
        st.subheader("Final annotated output")
        frame_placeholder = st.empty()
        left_angle_card, right_angle_card, left_speed_card, right_speed_card = st.columns(4)
        angle_chart_column, speed_chart_column = st.columns(2)
        angle_chart_placeholder = angle_chart_column.empty()
        speed_chart_placeholder = speed_chart_column.empty()
        progress = st.progress(0.0, text="Processing video on CPU…")

        angle_history: list[dict[str, float]] = []
        speed_history: list[dict[str, float]] = []
        frame_index = 0
        flash_label: str | None = None
        flash_until = -1.0

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            stages = pipeline.process(
                frame,
                inference_size=int(active_settings["inference_size"]),
            )
            raw_keypoints = stages.raw_keypoints_original
            elapsed = frame_index / fps
            processed_pose = signal_processor.update(raw_keypoints, timestamp=elapsed)
            pixel_keypoints = (
                processed_pose.pixel_keypoints
                if processed_pose is not None
                else None
            )
            keypoints = (
                processed_pose.normalized_keypoints
                if processed_pose is not None
                else None
            )
            angles = _joint_angles(keypoints)
            wrist_speeds = {"left": float("nan"), "right": float("nan")}
            wrist_velocities = {
                "left": np.full(2, np.nan, dtype=np.float64),
                "right": np.full(2, np.nan, dtype=np.float64),
            }
            wrist_reaches = {"left": float("nan"), "right": float("nan")}
            extension_velocities = {
                "left": float("nan"),
                "right": float("nan"),
            }

            for side in ("left", "right"):
                current_wrist = _valid_xy(keypoints, f"{side}_wrist")
                current_shoulder = _valid_xy(keypoints, f"{side}_shoulder")
                motion = motion_tracker.update(
                    side,
                    current_wrist,
                    current_shoulder,
                    dt,
                )
                wrist_velocities[side] = motion.velocity
                wrist_speeds[side] = motion.speed
                wrist_reaches[side] = motion.reach
                extension_velocities[side] = motion.extension_velocity

            frame_events = []
            for side in ("left", "right"):
                event = punch_detector.update(
                    hand=side,
                    velocity=wrist_velocities[side],
                    speed=wrist_speeds[side],
                    elbow_angle=angles[side],
                    timestamp=elapsed,
                    frame_index=frame_index,
                    reach=wrist_reaches[side],
                    extension_velocity=extension_velocities[side],
                )
                if event is not None:
                    frame_events.append(event)
                    fatigue_analyzer.add_punch(event)

            if frame_events:
                flash_label = " / ".join(event.label for event in frame_events)
                flash_until = elapsed + 0.45
            elif elapsed > flash_until:
                flash_label = None

            punch_active = {
                side: punch_detector.is_punch_active(side)
                for side in ("left", "right")
            }
            guard_status = guard_monitor.update(
                keypoints,
                timestamp=elapsed,
                punch_active=punch_active,
            )

            live_punch_summary = punch_detector.summary(max(elapsed, dt))

            annotated = draw_skeleton(
                stages.detection_frame,
                pixel_keypoints,
                elbow_angles=angles,
                wrist_speeds=wrist_speeds,
                punch_label=flash_label,
                guard_warning=guard_status.warning,
                session_stats={
                    "Punches": live_punch_summary["total_punches"],
                    "Left": dict(live_punch_summary["by_hand"]).get("left", 0),
                    "Right": dict(live_punch_summary["by_hand"]).get("right", 0),
                    "Guard": f"{guard_monitor.discipline_score:.0f}%",
                },
            )
            writer.write(annotated)

            angle_history.append(
                {
                    "Time (s)": elapsed,
                    "Left elbow": angles["left"],
                    "Right elbow": angles["right"],
                }
            )
            speed_history.append(
                {
                    "Time (s)": elapsed,
                    "Left wrist": wrist_speeds["left"],
                    "Right wrist": wrist_speeds["right"],
                }
            )
            frame_index += 1

            if frame_index % 300 == 0:
                LOGGER.info(
                    "VIDEO_PROGRESS file=%s frame=%d elapsed=%.1fs punches=%d guard=%.1f%%",
                    uploaded_file.name,
                    frame_index,
                    elapsed,
                    len(punch_detector.events),
                    guard_monitor.discipline_score,
                )

            if frame_index == 1 or frame_index % 5 == 0:
                preprocess_placeholder.image(
                    cv2.cvtColor(stages.preprocessed_frame, cv2.COLOR_BGR2RGB),
                    caption=(
                        f"1 · {active_settings['inference_size']}px square letterbox"
                    ),
                    channels="RGB",
                    use_container_width=True,
                )
                detection_placeholder.image(
                    cv2.cvtColor(stages.detection_frame, cv2.COLOR_BGR2RGB),
                    caption=(
                        f"2 · Boxer ID {stages.tracked_box.track_id}"
                        if stages.tracked_box is not None
                        else "2 · Awaiting tracked boxer"
                    ),
                    channels="RGB",
                    use_container_width=True,
                )
                crop_display = (
                    stages.crop_frame
                    if stages.crop_frame is not None
                    else np.zeros(
                        (
                            int(active_settings["inference_size"]),
                            int(active_settings["inference_size"]),
                            3,
                        ),
                        dtype=np.uint8,
                    )
                )
                crop_placeholder.image(
                    cv2.cvtColor(crop_display, cv2.COLOR_BGR2RGB),
                    caption="3 · Original-frame crop + 10% padding",
                    channels="RGB",
                    use_container_width=True,
                )
                pose_display = (
                    stages.pose_crop_frame
                    if stages.pose_crop_frame is not None
                    else crop_display
                )
                pose_placeholder.image(
                    cv2.cvtColor(pose_display, cv2.COLOR_BGR2RGB),
                    caption="4 · YOLO11 pose on crop only",
                    channels="RGB",
                    use_container_width=True,
                )
                frame_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True,
                )
                left_angle_card.metric(
                    "Left elbow", _metric_text(angles["left"], "°")
                )
                right_angle_card.metric(
                    "Right elbow", _metric_text(angles["right"], "°")
                )
                left_speed_card.metric(
                    "Left wrist", _metric_text(wrist_speeds["left"], " TL/s")
                )
                right_speed_card.metric(
                    "Right wrist", _metric_text(wrist_speeds["right"], " TL/s")
                )
                angle_chart_placeholder.line_chart(
                    angle_history,
                    x="Time (s)",
                    y=["Left elbow", "Right elbow"],
                    y_label="Elbow angle (degrees)",
                )
                speed_chart_placeholder.line_chart(
                    speed_history,
                    x="Time (s)",
                    y=["Left wrist", "Right wrist"],
                    y_label="Speed (torso lengths/second)",
                )

            if total_frames > 0:
                progress.progress(
                    min(frame_index / total_frames, 1.0),
                    text=f"Processed {frame_index:,} / {total_frames:,} frames",
                )
            else:
                progress.progress(
                    0.0,
                    text=f"Processed {frame_index:,} frames",
                )

        if frame_index == 0:
            st.error("The video did not contain any readable frames.")
            return

        for event in punch_detector.flush():
            fatigue_analyzer.add_punch(event)

        capture.release()
        capture = None
        writer.release()
        writer = None
        progress.progress(0.98, text="Creating browser-compatible H.264 video…")
        _transcode_browser_mp4(intermediate_path, output_path)
        progress.progress(1.0, text=f"Finished {frame_index:,} frames")

        duration_seconds = frame_index / fps
        punch_summary = punch_detector.summary(duration_seconds)
        guard_summary = guard_monitor.summary()
        fatigue_report = fatigue_analyzer.analyze(duration_seconds)
        completed_result = {
            "analysis_key": analysis_key,
            "video_bytes": output_path.read_bytes(),
            "output_filename": f"{Path(uploaded_file.name).stem}_cornercoach.mp4",
            "punch_summary": punch_summary,
            "guard_summary": guard_summary,
            "fatigue_report": fatigue_report,
            "settings": active_settings,
        }
        st.session_state["completed_video_analysis"] = completed_result
        _render_completed_video(completed_result)
        LOGGER.info(
            "VIDEO_SUMMARY file=%s duration=%.3fs total_punches=%d ppm=%.2f "
            "by_type=%s by_hand=%s guard_score=%.2f%% guard_drops=%s "
            "fatigued=%s work_rate_drop=%.2f%% speed_drop=%.2f%%",
            uploaded_file.name,
            duration_seconds,
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

    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        for temporary_path in (input_path, intermediate_path, output_path):
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def main() -> None:
    """Render the CornerCoach upload and inspection workflow."""
    st.set_page_config(page_title="CornerCoach", page_icon="🥊", layout="wide")
    st.title("🥊 CornerCoach")
    st.caption("CPU-first pose and kinematics inspection for boxing coaches")

    uploaded_file = st.file_uploader(
        "Upload a fighter image or training video",
        type=["jpg", "jpeg", "png", "mp4", "mov"],
        help="Images are inspected once; videos are processed frame by frame.",
    )
    if uploaded_file is None:
        st.info("Upload a JPG, PNG, MP4, or MOV file to begin.")
        return

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
        st.error("Unsupported file extension.")
        return

    try:
        with st.spinner(
            "Loading YOLO11 detector, ByteTrack, and YOLO11 pose on CPU…"
        ):
            pipeline = load_vision_pipeline()
        if suffix in IMAGE_EXTENSIONS:
            process_image(uploaded_file, pipeline)
        else:
            stance = st.sidebar.selectbox(
                "Fighter stance",
                options=["Orthodox", "Southpaw"],
                index=0,
                help="Used to distinguish lead-hand jabs from rear-hand crosses.",
            )
            settings = _analysis_settings_panel()
            process_video(
                uploaded_file,
                pipeline,
                stance=stance.lower(),
                settings=settings,
            )
    except Exception as exc:  # Streamlit should surface model/codec errors cleanly.
        LOGGER.exception("ANALYSIS_FAILED file=%s", uploaded_file.name)
        st.error(f"Analysis failed: {exc}")
        st.exception(exc)


if __name__ == "__main__":
    main()
