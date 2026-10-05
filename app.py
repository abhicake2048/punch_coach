"""Streamlit interface for tracked top-down boxing analysis."""

from __future__ import annotations

from collections import deque
import logging
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
import torch

from core.kinematics import (
    PoseSignalProcessor,
    calculate_angle,
)
from core.coaching_report import (
    DEFAULT_GEMINI_MODEL,
    build_coaching_metrics,
    evidence_for_point,
    generate_coaching_report,
)
from core.fatigue_analyzer import FatigueAnalyzer, FatigueReport
from core.guard_monitor import GuardMonitor
from core.logging_config import configure_logging
from core.boxer_pipeline import TrackedBoxerPosePipeline
from core.pose_engine import BOXING_KEYPOINT_INDICES, PoseEngine
from core.pdf_report import create_coaching_pdf
from core.trained_punch_recognizer import (
    MINIMUM_PUNCH_CONFIDENCE,
    TrainedPunchRecognizer,
)
from visualizer.video_annotator import draw_skeleton


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".mov"}
PROJECT_ROOT = Path(__file__).resolve().parent
YOLO_WEIGHTS = PROJECT_ROOT / "weights" / "yolo11s-pose.pt"
ANALYSIS_SCHEMA_VERSION = "top-down-cpu-pdf-v3"
LOG_PATH = configure_logging()
LOGGER = logging.getLogger("cornercoach.app")


@st.cache_resource(show_spinner=False)
def configure_cpu_runtime() -> int:
    """Avoid OpenCV/Torch thread oversubscription during CPU inference."""
    available = max(1, int(os.cpu_count() or 1))
    requested = os.getenv("CORNERCOACH_CPU_THREADS", "").strip()
    try:
        thread_count = int(requested) if requested else min(8, available)
    except ValueError:
        thread_count = min(8, available)
    thread_count = max(1, min(thread_count, available))
    cv2.setNumThreads(1)
    torch.set_num_threads(thread_count)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch allows this setting only before inter-op work starts.
        pass
    return thread_count


@st.cache_resource(show_spinner=False)
def load_pose_engine() -> PoseEngine:
    """Create one cached YOLO11 pose engine for the Streamlit process."""
    return PoseEngine(weights=YOLO_WEIGHTS, confidence_threshold=0.25)


@st.cache_resource(show_spinner=False)
def load_vision_pipeline() -> TrackedBoxerPosePipeline:
    """Create the mandatory detector/ByteTrack/top-down pose pipeline."""
    return TrackedBoxerPosePipeline(
        pose_engine=load_pose_engine(),
        detector_weights=YOLO_WEIGHTS,
    )


@st.cache_resource(show_spinner=False)
def load_trained_recognizer() -> TrainedPunchRecognizer:
    """Load the production ST-GCN punch checkpoint once."""
    return TrainedPunchRecognizer(
        checkpoint_path=PROJECT_ROOT / "weights" / "stgcn" / "best_checkpoint.pt",
        model_kind="stgcn",
        device="cpu",
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

    hand_counts = dict(punch_summary.get("by_hand", {}))
    left_column, right_column = st.columns(2)
    left_column.metric("Left-hand punches", int(hand_counts.get("left", 0)))
    right_column.metric("Right-hand punches", int(hand_counts.get("right", 0)))

    statistics_column, fatigue_details_column = st.columns(2)
    with statistics_column:
        st.markdown("**Punch counts by type**")
        type_counts = dict(punch_summary.get("by_type", {}))
        if type_counts:
            st.dataframe(
                [
                    {"Punch type": label, "Count": count}
                    for label, count in sorted(type_counts.items())
                ],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No punches met the classification thresholds.")
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

    st.markdown("**Non-punching-hand guard drops**")
    guard_events = list(guard_summary.get("events", []))
    if guard_events:
        st.dataframe(
            [
                {
                    "Hand": str(event["hand"]).title(),
                    "Start (s)": round(float(event["start_time_s"]), 2),
                    "End (s)": round(float(event["end_time_s"]), 2),
                    "Duration (s)": round(float(event["duration_s"]), 2),
                    "Deepest level": str(event["deepest_level"])
                    .removeprefix("below_")
                    .replace("_", " ")
                    .title(),
                }
                for event in guard_events
            ],
            hide_index=True,
            use_container_width=True,
        )
    elif int(guard_summary.get("eligible_frames", 0)) > 0:
        st.success("No guard-drop episodes were recorded on eligible frames.")
    else:
        st.info("Guard position could not be evaluated from the available pose data.")

    st.caption(
        f"Punch type and hand were predicted by the selected "
        f"{punch_summary.get('model_display_name', 'trained')} checkpoint using "
        f"{punch_summary.get('sequence_length', 'its saved')} frame windows."
    )


def _analysis_settings_panel() -> dict[str, Any]:
    """Expose ST-GCN recognition, CPU batching, guard, and fatigue controls."""
    with st.sidebar.expander("YOLO + ST-GCN recognition", expanded=False):
        detection_size = st.select_slider(
            "Person-detection input size (pixels)",
            options=[320, 384, 416, 480],
            value=320,
            help=(
                "The first YOLO stage only locates the boxer. 320px is the "
                "fast CPU default; the cropped pose stage remains at 480px."
            ),
        )
        inference_size = st.select_slider(
            "Cropped-pose input size (pixels)",
            options=[480, 640],
            value=480,
            help=(
                "The second YOLO stage extracts keypoints from the tracked boxer "
                "crop. 480px matches the fast production feature pipeline."
            ),
        )
        cpu_batch_size = st.select_slider(
            "CPU inference batch size",
            options=[1, 2, 4, 8, 12, 16],
            value=8,
            help=(
                "Batches both YOLO stages while preserving frame order. Reduce "
                "this only if the computer runs out of memory."
            ),
        )
        model_confidence = st.slider(
            "Punch confidence threshold",
            MINIMUM_PUNCH_CONFIDENCE,
            0.99,
            MINIMUM_PUNCH_CONFIDENCE,
            0.01,
            help=(
                f"A punch is registered only above {MINIMUM_PUNCH_CONFIDENCE * 100:.0f}%. "
                "Predictions at or below "
                "the selected threshold are treated as IDLE."
            ),
        )
        keypoint_confidence = st.slider(
            "Pose keypoint confidence",
            0.10,
            0.90,
            0.25,
            0.05,
            help="The training extraction used 0.25.",
        )
        recovery_frames = st.number_input(
            "Minimum peak spacing (frames)",
            0,
            120,
            8,
            1,
            help="Prevents adjacent samples from the same neural peak being counted twice.",
        )
        peak_prominence = st.slider(
            "Model/motion peak prominence",
            0.0,
            0.50,
            0.04,
            0.01,
            help="Required activation drop before one punch peak is counted.",
        )
        minimum_wrist_speed = st.slider(
            "Minimum wrist speed (torso lengths/s)",
            0.0,
            3.0,
            0.20,
            0.05,
            help="Motion gate that reduces idle false positives from the small dataset.",
        )
        minimum_extension_velocity = st.slider(
            "Minimum outward wrist extension (torso lengths/s)",
            0.0,
            2.0,
            0.05,
            0.05,
            help="Rejects retractions and idle wrist motion before counting a punch.",
        )
        minimum_pose_coverage = st.slider(
            "Minimum valid pose coverage",
            0.50,
            1.00,
            0.80,
            0.05,
            help="Matches the minimum sequence coverage used during dataset construction.",
        )
        live_preview = st.toggle(
            "Live diagnostic preview",
            value=False,
            help=(
                "Disabled by default for faster CPU processing. The final annotated "
                "video and complete report are still produced."
            ),
        )
        preview_interval = st.select_slider(
            "Preview refresh interval (frames)",
            options=[5, 10, 15, 30],
            value=15,
            disabled=not live_preview,
            help="A larger interval reduces Streamlit rendering overhead.",
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
        "model_kind": "stgcn",
        "detection_size": int(detection_size),
        "inference_size": int(inference_size),
        "cpu_batch_size": int(cpu_batch_size),
        "model_confidence": float(model_confidence),
        "keypoint_confidence": float(keypoint_confidence),
        "recovery_frames": int(recovery_frames),
        "peak_prominence": float(peak_prominence),
        "minimum_wrist_speed": float(minimum_wrist_speed),
        "minimum_extension_velocity": float(minimum_extension_velocity),
        "minimum_pose_coverage": float(minimum_pose_coverage),
        "live_preview": bool(live_preview),
        "preview_interval": int(preview_interval),
        "guard_chin_fraction": float(guard_chin_fraction),
        "guard_wrist_tolerance": float(guard_wrist_tolerance),
        "fatigue_work_rate_drop": float(fatigue_work_rate_drop),
        "fatigue_speed_drop": float(fatigue_speed_drop),
    }


def _completed_analysis_key(
    video_bytes: bytes,
    filename: str,
    settings: dict[str, Any],
) -> str:
    """Build a stable key that changes only when input or analysis settings change."""
    digest = hashlib.sha256()
    digest.update(ANALYSIS_SCHEMA_VERSION.encode("utf-8"))
    digest.update(video_bytes)
    digest.update(filename.encode("utf-8"))
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _configured_gemini_api_key() -> str:
    """Read a server-side Gemini key without exposing it in session settings."""
    environment_key = os.getenv("GEMINI_API_KEY", "").strip()
    if environment_key:
        return environment_key
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        return ""


def _render_coaching_report(result: dict[str, Any]) -> None:
    """Generate and render an optional metric-grounded Gemini report."""
    st.subheader("Gemini coaching report")
    st.caption(
        "Only the structured punch, guard, and fatigue metrics shown above are "
        "sent to Gemini. The video and pose frames are never sent."
    )
    configured_key = _configured_gemini_api_key()
    api_key_input = st.text_input(
        "Gemini API key",
        type="password",
        value="",
        placeholder=(
            "Using configured GEMINI_API_KEY"
            if configured_key
            else "Paste your Gemini API key"
        ),
        key=f"gemini-key-{result['analysis_key']}",
        help=(
            "The key is used only for this API request and is not stored in the "
            "analysis result or written to logs. You can instead configure the "
            "GEMINI_API_KEY environment variable or Streamlit secret."
        ),
    )
    model_name = st.text_input(
        "Gemini model",
        value=DEFAULT_GEMINI_MODEL,
        key=f"gemini-model-{result['analysis_key']}",
    ).strip()
    effective_key = api_key_input.strip() or configured_key
    metrics = build_coaching_metrics(
        result["punch_summary"],
        result["guard_summary"],
        result["fatigue_report"],
        float(result.get("duration_seconds", 0.0)),
    )
    if st.button(
        "Generate coaching report",
        type="primary",
        disabled=not bool(effective_key and model_name),
        key=f"generate-report-{result['analysis_key']}",
    ):
        try:
            with st.spinner("Generating a grounded coaching report with Gemini…"):
                report = generate_coaching_report(
                    metrics,
                    api_key=effective_key,
                    model=model_name,
                )
            st.session_state["gemini_coaching_report"] = {
                "analysis_key": result["analysis_key"],
                "model": model_name,
                "report": report,
            }
        except Exception as exc:
            LOGGER.exception(
                "COACHING_REPORT_FAILED analysis=%s model=%s",
                result["analysis_key"],
                model_name,
            )
            st.error(f"Gemini coaching report failed: {exc}")

    cached = st.session_state.get("gemini_coaching_report")
    if not isinstance(cached, dict):
        if not effective_key:
            st.info("Enter a Gemini API key to enable report generation.")
        return
    if (
        cached.get("analysis_key") != result["analysis_key"]
        or cached.get("model") != model_name
    ):
        return
    report = cached.get("report")
    if not isinstance(report, dict):
        return

    st.write(report["summary"])
    sections = (
        ("Three strengths", "strengths"),
        ("Three areas to improve", "areas_to_improve"),
        ("Two suggested drills", "suggested_drills"),
    )
    for heading, key in sections:
        st.markdown(f"**{heading}**")
        for index, point in enumerate(report[key], start=1):
            st.markdown(f"{index}. **{point['title']}** — {point['comment']}")
            evidence = evidence_for_point(point, metrics)
            if evidence:
                evidence_text = ", ".join(
                    f"`{name}` = {value}" for name, value in evidence.items()
                )
                st.caption(f"Measured evidence: {evidence_text}")
    limitations = report.get("data_limitations", [])
    if limitations:
        with st.expander("Data limitations"):
            for limitation in limitations:
                st.write(f"- {limitation}")

    pdf_digest = hashlib.sha256(
        (
            result["analysis_key"]
            + model_name
            + json.dumps(report, sort_keys=True, separators=(",", ":"))
        ).encode("utf-8")
    ).hexdigest()
    cached_pdf = st.session_state.get("coaching_report_pdf")
    if not isinstance(cached_pdf, dict) or cached_pdf.get("key") != pdf_digest:
        try:
            cached_pdf = {
                "key": pdf_digest,
                "bytes": create_coaching_pdf(
                    metrics,
                    report,
                    source_filename=str(result.get("source_filename", "Uploaded video")),
                ),
            }
            st.session_state["coaching_report_pdf"] = cached_pdf
        except Exception as exc:
            LOGGER.exception("PDF_REPORT_FAILED analysis=%s", result["analysis_key"])
            st.error(f"PDF report creation failed: {exc}")
            return

    source_stem = Path(str(result.get("source_filename", "boxing_session"))).stem
    st.download_button(
        "Download coaching report PDF",
        data=cached_pdf["bytes"],
        file_name=f"{source_stem}_cornercoach_report.pdf",
        mime="application/pdf",
        key=f"download-pdf-{pdf_digest}",
        type="primary",
    )


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
    _render_coaching_report(result)
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
    processing_fps = float(result.get("processing_fps", 0.0))
    if processing_fps > 0.0:
        st.caption(
            f"Measured end-to-end processing throughput: {processing_fps:.2f} "
            "frames/second on this computer."
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
    recognizer: TrainedPunchRecognizer,
    settings: dict[str, Any] | None = None,
) -> None:
    """Analyze an uploaded video, showing live frames and an annotated result."""
    active_settings = settings or _analysis_settings_panel()
    source_bytes = uploaded_file.getvalue()
    analysis_key = _completed_analysis_key(
        source_bytes,
        uploaded_file.name,
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

        recognizer.confidence_threshold = max(
            MINIMUM_PUNCH_CONFIDENCE,
            float(active_settings["model_confidence"]),
        )
        recognizer.keypoint_confidence = float(active_settings["keypoint_confidence"])
        recognizer.recovery_frames = int(active_settings["recovery_frames"])
        recognizer.peak_prominence = float(active_settings["peak_prominence"])
        recognizer.minimum_wrist_speed = float(active_settings["minimum_wrist_speed"])
        recognizer.minimum_extension_velocity = float(
            active_settings["minimum_extension_velocity"]
        )
        recognizer.minimum_pose_coverage = float(active_settings["minimum_pose_coverage"])
        recognizer.reset(fps=fps)
        guard_monitor = GuardMonitor(
            chin_fraction=float(active_settings["guard_chin_fraction"]),
            wrist_tolerance=float(active_settings["guard_wrist_tolerance"]),
        )
        fatigue_analyzer = FatigueAnalyzer(
            work_rate_drop_threshold=float(active_settings["fatigue_work_rate_drop"]),
            speed_drop_threshold=float(active_settings["fatigue_speed_drop"]),
        )
        signal_processor = PoseSignalProcessor(sample_frequency=fps)
        pipeline.reset()
        LOGGER.info(
            "VIDEO_START file=%s fps=%.3f frames=%d dimensions=%dx%d settings=%s",
            uploaded_file.name,
            fps,
            total_frames,
            width,
            height,
            active_settings,
        )

        live_preview = bool(active_settings.get("live_preview", False))
        preview_interval = int(active_settings.get("preview_interval", 15))
        if live_preview:
            st.subheader("Live pipeline inspection")
            st.caption(
                "Preprocess -> YOLO11 + ByteTrack -> boxer crop -> pose -> final analytics"
            )
            stage_columns = st.columns(4)
            preprocess_placeholder = stage_columns[0].empty()
            detection_placeholder = stage_columns[1].empty()
            crop_placeholder = stage_columns[2].empty()
            pose_placeholder = stage_columns[3].empty()
            st.subheader("Final annotated output")
            frame_placeholder = st.empty()
            count_card, action_card, punch_conf_card, hand_conf_card = st.columns(4)
            punch_chart_column, hand_chart_column = st.columns(2)
            punch_chart_placeholder = punch_chart_column.empty()
            hand_chart_placeholder = hand_chart_column.empty()
        else:
            st.info(
                "Every frame is being processed by the top-down pipeline without "
                "redrawing the live diagnostic dashboard. The final video and "
                "analytics will still appear."
            )
        progress = st.progress(
            0.0,
            text=f"Processing video with {recognizer.display_name}…",
        )

        punch_score_history: list[dict[str, float]] = []
        hand_score_history: list[dict[str, float]] = []
        frame_index = 0
        flash_label: str | None = None
        flash_until = -1.0
        pending_frames: deque[tuple[np.ndarray, Any]] = deque()
        cpu_batch_size = max(1, int(active_settings.get("cpu_batch_size", 8)))
        inference_started = time.perf_counter()

        while True:
            if not pending_frames:
                frame_batch: list[np.ndarray] = []
                for _ in range(cpu_batch_size):
                    ok, batch_frame = capture.read()
                    if not ok:
                        break
                    frame_batch.append(batch_frame)
                if not frame_batch:
                    break
                stage_batch = pipeline.process_batch(
                    frame_batch,
                    inference_size=int(active_settings["inference_size"]),
                    detection_size=int(active_settings["detection_size"]),
                    draw_diagnostics=True,
                )
                if len(stage_batch) != len(frame_batch):
                    raise RuntimeError(
                        "Batched pose pipeline returned a different frame count"
                    )
                pending_frames.extend(zip(frame_batch, stage_batch))

            frame, stages = pending_frames.popleft()
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
            frame_events = recognizer.update(
                pixel_keypoints,
                timestamp=elapsed,
                frame_index=frame_index,
            )
            for event in frame_events:
                if event.action_kind == "punch":
                    fatigue_analyzer.add_punch(event)

            if frame_events:
                flash_label = " / ".join(
                    f"{event.action_kind.upper()}: {event.label}"
                    for event in frame_events
                )
                flash_until = elapsed + 0.45
            elif elapsed > flash_until:
                flash_label = None

            classification_label = flash_label or recognizer.current_label

            current_punching_hand = recognizer.current_punching_hand
            punch_active = {
                side: (
                    recognizer.is_punch_active(side)
                    or current_punching_hand == side
                )
                for side in ("left", "right")
            }
            guard_status = guard_monitor.update(
                keypoints,
                timestamp=elapsed,
                punch_active=punch_active,
            )

            live_punch_summary = recognizer.summary(max(elapsed, dt))

            annotated = draw_skeleton(
                stages.detection_frame,
                pixel_keypoints,
                punch_label=classification_label,
                guard_warning=guard_status.warning,
                session_stats={
                    "Punches": live_punch_summary["total_punches"],
                    "Left": dict(live_punch_summary["by_hand"]).get("left", 0),
                    "Right": dict(live_punch_summary["by_hand"]).get("right", 0),
                    "Model": recognizer.display_name,
                    "Guard": f"{guard_monitor.discipline_score:.0f}%",
                },
            )
            writer.write(annotated)

            frame_index += 1

            if frame_index % 300 == 0:
                LOGGER.info(
                    "VIDEO_PROGRESS file=%s frame=%d elapsed=%.1fs punches=%d guard=%.1f%%",
                    uploaded_file.name,
                    frame_index,
                    elapsed,
                    len(recognizer.events),
                    guard_monitor.discipline_score,
                )

            if live_preview and (
                frame_index == 1 or frame_index % preview_interval == 0
            ):
                punch_score_history.append({
                    "Time (s)": elapsed,
                    **{
                        name.title(): score
                        for name, score in recognizer.latest_scores.punch_type.items()
                    },
                })
                hand_score_history.append({
                    "Time (s)": elapsed,
                    **{
                        name.title(): score
                        for name, score in recognizer.latest_scores.hand.items()
                    },
                })
                preprocess_placeholder.image(
                    cv2.cvtColor(stages.preprocessed_frame, cv2.COLOR_BGR2RGB),
                    caption=(
                        f"1 · {active_settings['detection_size']}px detection letterbox"
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
                pose_display = draw_skeleton(
                    crop_display,
                    stages.raw_keypoints_crop,
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
                punch_scores = recognizer.latest_scores.punch_type
                hand_scores = recognizer.latest_scores.hand
                best_punch = max(punch_scores, key=punch_scores.get)
                best_hand = max(hand_scores, key=hand_scores.get)
                count_card.metric(
                    f"{recognizer.display_name} punch count",
                    live_punch_summary["total_punches"],
                )
                action_card.metric("Latest classification", classification_label)
                if recognizer.current_label == "IDLE" and flash_label is None:
                    punch_conf_card.metric(
                        "Punch confidence",
                        f"{recognizer.current_confidence * 100:.1f}% · IDLE",
                    )
                else:
                    punch_conf_card.metric(
                        best_punch.title(),
                        f"{punch_scores[best_punch] * 100:.1f}%",
                    )
                hand_conf_card.metric(
                    f"Hand: {best_hand.title()}",
                    f"{hand_scores[best_hand] * 100:.1f}%",
                )
                punch_chart_placeholder.line_chart(
                    punch_score_history,
                    x="Time (s)",
                    y=[name.title() for name in recognizer.punch_classes],
                    y_label=f"{recognizer.display_name} punch probability",
                )
                hand_chart_placeholder.line_chart(
                    hand_score_history,
                    x="Time (s)",
                    y=[name.title() for name in recognizer.hand_classes],
                    y_label=f"{recognizer.display_name} hand probability",
                )

            should_update_progress = (
                frame_index == 1
                or frame_index % 30 == 0
                or (total_frames > 0 and frame_index >= total_frames)
            )
            if total_frames > 0 and should_update_progress:
                progress.progress(
                    min(frame_index / total_frames, 1.0),
                    text=f"Processed {frame_index:,} / {total_frames:,} frames",
                )
            elif total_frames <= 0 and should_update_progress:
                progress.progress(
                    0.0,
                    text=f"Processed {frame_index:,} frames",
                )

        if frame_index == 0:
            st.error("The video did not contain any readable frames.")
            return

        for final_event in recognizer.flush():
            fatigue_analyzer.add_punch(final_event)

        capture.release()
        capture = None
        writer.release()
        writer = None
        progress.progress(0.98, text="Creating browser-compatible H.264 video…")
        _transcode_browser_mp4(intermediate_path, output_path)
        progress.progress(1.0, text=f"Finished {frame_index:,} frames")

        duration_seconds = frame_index / fps
        inference_wall_seconds = max(time.perf_counter() - inference_started, 1e-9)
        processing_fps = frame_index / inference_wall_seconds
        punch_summary = recognizer.summary(duration_seconds)
        guard_monitor.finalize(duration_seconds)
        guard_summary = guard_monitor.summary()
        fatigue_report = fatigue_analyzer.analyze(duration_seconds)
        completed_result = {
            "analysis_key": analysis_key,
            "source_filename": uploaded_file.name,
            "video_bytes": output_path.read_bytes(),
            "output_filename": f"{Path(uploaded_file.name).stem}_cornercoach.mp4",
            "punch_summary": punch_summary,
            "guard_summary": guard_summary,
            "fatigue_report": fatigue_report,
            "duration_seconds": duration_seconds,
            "processing_fps": processing_fps,
            "settings": active_settings,
        }
        st.session_state["completed_video_analysis"] = completed_result
        _render_completed_video(completed_result)
        LOGGER.info(
            "VIDEO_SUMMARY file=%s duration=%.3fs total_punches=%d ppm=%.2f "
            "by_type=%s by_hand=%s guard_score=%.2f%% guard_drops=%s "
            "fatigued=%s work_rate_drop=%.2f%% speed_drop=%.2f%% "
            "processing_fps=%.2f wall_seconds=%.2f",
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
            processing_fps,
            inference_wall_seconds,
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
    cpu_threads = configure_cpu_runtime()
    st.title("🥊 CornerCoach")
    st.caption(
        "Batched two-stage YOLO pose analysis with trained ST-GCN punch "
        f"recognition | CPU threads: {cpu_threads}"
    )

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
        if suffix in IMAGE_EXTENSIONS:
            with st.spinner("Loading YOLO11 and ByteTrack…"):
                pipeline = load_vision_pipeline()
            process_image(uploaded_file, pipeline)
        else:
            with st.spinner("Loading the trained ST-GCN checkpoint…"):
                recognizer = load_trained_recognizer()
            st.caption(
                f"Loaded ST-GCN: {recognizer.sequence_length}-frame windows · "
                f"device {recognizer.device}."
            )
            background_recall = float(
                recognizer.validation_metrics.get("background_detection_recall", 0.0)
            )
            if background_recall < 0.50:
                st.warning(
                    "Experimental checkpoint: held-out background recall is "
                    f"{background_recall * 100:.1f}%. Punch counts may contain false "
                    "positives; review the annotated video before using the report."
                )
            settings = _analysis_settings_panel()
            with st.spinner("Loading YOLO11 top-down pose pipeline…"):
                pipeline = load_vision_pipeline()
            process_video(
                uploaded_file,
                pipeline,
                recognizer,
                settings=settings,
            )
    except Exception as exc:  # Streamlit should surface model/codec errors cleanly.
        LOGGER.exception("ANALYSIS_FAILED file=%s", uploaded_file.name)
        st.error(f"Analysis failed: {exc}")
        st.exception(exc)


if __name__ == "__main__":
    main()
