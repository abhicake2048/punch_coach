"""Tracked top-down boxer detection and cropped YOLO11 pose inference.

The detector operates on a selectable 480px or 640px square letterbox,
ByteTrack keeps the selected person's identity stable, and pose inference is
restricted to an equally sized aspect-preserving crop of that tracked boxer.
All public coordinates
are mapped back to the untouched original video frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from ultralytics import YOLO

from .pose_engine import COCO_KEYPOINT_NAMES, PoseEngine


ALLOWED_INFERENCE_SIZES: Final[tuple[int, int]] = (480, 640)
DEFAULT_INFERENCE_SIZE: Final[int] = 640


@dataclass(frozen=True)
class LetterboxTransform:
    """Geometry needed to move points between an image and its canvas."""

    source_height: int
    source_width: int
    target_height: int
    target_width: int
    scale: float
    pad_x: int
    pad_y: int
    resized_width: int
    resized_height: int

    def to_canvas_points(self, points: NDArray[np.floating]) -> NDArray[np.float32]:
        """Map source-image x/y coordinates into the letterboxed canvas."""
        output = np.asarray(points, dtype=np.float32).copy()
        output[..., 0] = output[..., 0] * self.scale + self.pad_x
        output[..., 1] = output[..., 1] * self.scale + self.pad_y
        return output

    def to_source_points(self, points: NDArray[np.floating]) -> NDArray[np.float32]:
        """Map canvas x/y coordinates back to the source image."""
        output = np.asarray(points, dtype=np.float32).copy()
        output[..., 0] = (output[..., 0] - self.pad_x) / self.scale
        output[..., 1] = (output[..., 1] - self.pad_y) / self.scale
        output[..., 0] = np.clip(output[..., 0], 0, max(self.source_width - 1, 0))
        output[..., 1] = np.clip(output[..., 1], 0, max(self.source_height - 1, 0))
        return output

    def to_source_box(self, box: NDArray[np.floating]) -> NDArray[np.float32]:
        """Map one canvas ``xyxy`` box back to clipped source coordinates."""
        points = np.asarray(box, dtype=np.float32).reshape(2, 2)
        return self.to_source_points(points).reshape(4)


def letterbox(
    image: NDArray[np.uint8],
    target_size: int | tuple[int, int] = DEFAULT_INFERENCE_SIZE,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[NDArray[np.uint8], LetterboxTransform]:
    """Resize with unchanged aspect ratio and pad to exactly height x width."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.size == 0:
        raise ValueError("image must be a non-empty HxWxC NumPy array")
    source_height, source_width = image.shape[:2]
    if isinstance(target_size, int):
        target_height = target_width = int(target_size)
    else:
        target_height, target_width = (int(target_size[0]), int(target_size[1]))
    if source_height <= 0 or source_width <= 0 or target_height <= 0 or target_width <= 0:
        raise ValueError("source and target dimensions must be positive")

    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    pad_x = (target_width - resized_width) // 2
    pad_y = (target_height - resized_height) // 2
    canvas = np.full((target_height, target_width, 3), color, dtype=np.uint8)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    return canvas, LetterboxTransform(
        source_height=source_height,
        source_width=source_width,
        target_height=target_height,
        target_width=target_width,
        scale=float(scale),
        pad_x=pad_x,
        pad_y=pad_y,
        resized_width=resized_width,
        resized_height=resized_height,
    )


@dataclass(frozen=True)
class TrackedBox:
    """Selected ByteTrack identity and its box in original-frame pixels."""

    track_id: int
    xyxy: NDArray[np.float32]
    confidence: float
    predicted: bool = False


@dataclass(frozen=True)
class PipelineFrame:
    """All observable stages produced for one input frame."""

    preprocessed_frame: NDArray[np.uint8]
    detection_frame: NDArray[np.uint8]
    crop_frame: NDArray[np.uint8] | None
    pose_crop_frame: NDArray[np.uint8] | None
    tracked_box: TrackedBox | None
    crop_box_xyxy: NDArray[np.int32] | None
    raw_keypoints_original: dict[str, NDArray[np.float32]] | None
    raw_keypoints_crop: dict[str, NDArray[np.float32]] | None


class TrackedBoxerPosePipeline:
    """YOLO11 person detection + ByteTrack + top-down YOLO11 pose."""

    def __init__(
        self,
        pose_engine: PoseEngine,
        detector_weights: str | Path = "yolo11s-pose.pt",
        detection_confidence: float = 0.25,
        tracking_iou: float = 0.5,
        max_track_gap: int = 12,
        crop_padding: float = 0.10,
        device: str = "cpu",
        half: bool = False,
        single_pass_pose: bool = True,
    ) -> None:
        if not 0.0 <= detection_confidence <= 1.0:
            raise ValueError("detection_confidence must be between 0 and 1")
        if not 0.0 <= tracking_iou <= 1.0:
            raise ValueError("tracking_iou must be between 0 and 1")
        if max_track_gap < 0:
            raise ValueError("max_track_gap cannot be negative")
        if not 0.0 <= crop_padding <= 1.0:
            raise ValueError("crop_padding must be between 0 and 1")
        self.pose_engine = pose_engine
        self.device = str(device)
        self.half = bool(half)
        self.single_pass_pose = bool(single_pass_pose)
        # Fast CPU mode shares one YOLO11-pose model and reuses the tracked
        # full-frame keypoints. Compatibility mode retains the original second
        # cropped-pose call and therefore needs an independent tracker model.
        if self.single_pass_pose and str(detector_weights) == pose_engine.weights:
            self.detector = pose_engine.model
        else:
            self.detector = YOLO(str(detector_weights))
        self.detector.to(self.device)
        self.detection_confidence = float(detection_confidence)
        self.tracking_iou = float(tracking_iou)
        self.max_track_gap = int(max_track_gap)
        self.crop_padding = float(crop_padding)
        self.target_track_id: int | None = None
        self._last_box: NDArray[np.float32] | None = None
        self._missing_frames = 0

    def reset(self) -> None:
        """Forget the selected identity and reset Ultralytics tracker state."""
        self.target_track_id = None
        self._last_box = None
        self._missing_frames = 0
        predictor = getattr(self.detector, "predictor", None)
        if predictor is not None and hasattr(predictor, "trackers"):
            for tracker in predictor.trackers:
                tracker.reset()

    @staticmethod
    def _iou(first: NDArray[np.float32], second: NDArray[np.float32]) -> float:
        left = max(float(first[0]), float(second[0]))
        top = max(float(first[1]), float(second[1]))
        right = min(float(first[2]), float(second[2]))
        bottom = min(float(first[3]), float(second[3]))
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
        second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
        union = first_area + second_area - intersection
        return intersection / union if union > 0.0 else 0.0

    def _select_track(
        self,
        boxes: NDArray[np.float32],
        ids: NDArray[np.int64],
        confidences: NDArray[np.float32],
        canvas_size: int,
    ) -> tuple[int, int, NDArray[np.float32], float] | None:
        """Keep the locked ID; reacquire conservatively after a real loss."""
        if boxes.size == 0:
            return None
        if self.target_track_id is not None:
            matches = np.flatnonzero(ids == self.target_track_id)
            if matches.size:
                index = int(matches[0])
                return index, int(ids[index]), boxes[index].copy(), float(confidences[index])
            if self._missing_frames <= self.max_track_gap:
                return None

        if self._last_box is not None:
            overlaps = np.asarray([self._iou(box, self._last_box) for box in boxes])
            index = int(np.argmax(overlaps))
            if overlaps[index] >= self.tracking_iou:
                return index, int(ids[index]), boxes[index].copy(), float(confidences[index])

        # First acquisition: prioritize a large person near the frame center.
        areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        canvas_center = np.array([canvas_size / 2, canvas_size / 2], dtype=np.float32)
        distances = np.linalg.norm((centers - canvas_center) / canvas_center, axis=1)
        scores = areas * confidences / (1.0 + distances)
        index = int(np.argmax(scores))
        return index, int(ids[index]), boxes[index].copy(), float(confidences[index])

    def _result_keypoints(
        self,
        result: object,
        person_index: int,
        transform: LetterboxTransform,
    ) -> dict[str, NDArray[np.float32]] | None:
        """Read one tracked person's pose and map it to original-frame pixels."""
        result_keypoints = getattr(result, "keypoints", None)
        if result_keypoints is None or result_keypoints.xy is None:
            return None
        xy = result_keypoints.xy.cpu().numpy()
        if xy.ndim != 3 or person_index >= xy.shape[0] or xy.shape[1] < 17:
            return None
        confidence_tensor = result_keypoints.conf
        if confidence_tensor is None:
            confidence = np.ones(17, dtype=np.float32)
        else:
            confidence = (
                confidence_tensor[person_index].cpu().numpy()[:17].astype(np.float32)
            )
        mapped_xy = transform.to_source_points(
            xy[person_index, :17, :2].astype(np.float32)
        )
        points = np.column_stack((mapped_xy, confidence)).astype(np.float32)
        points[points[:, 2] < self.pose_engine.confidence_threshold, :2] = np.nan
        return {
            name: points[index].copy()
            for index, name in enumerate(COCO_KEYPOINT_NAMES)
        }

    @staticmethod
    def _keypoints_to_crop_canvas(
        keypoints: dict[str, NDArray[np.float32]],
        crop_box: NDArray[np.int32],
        transform: LetterboxTransform,
    ) -> dict[str, NDArray[np.float32]]:
        """Map original-frame keypoints into a diagnostic crop canvas."""
        x1, y1, _, _ = crop_box
        output: dict[str, NDArray[np.float32]] = {}
        for name, point in keypoints.items():
            mapped = np.asarray(point, dtype=np.float32).copy()
            if np.all(np.isfinite(mapped[:2])):
                crop_point = mapped[:2] - np.array([x1, y1], dtype=np.float32)
                mapped[:2] = transform.to_canvas_points(crop_point)
            output[name] = mapped
        return output

    @staticmethod
    def _integer_crop_box(box: NDArray[np.float32], frame: NDArray[np.uint8]) -> NDArray[np.int32] | None:
        height, width = frame.shape[:2]
        x1 = int(np.clip(np.floor(box[0]), 0, width - 1))
        y1 = int(np.clip(np.floor(box[1]), 0, height - 1))
        x2 = int(np.clip(np.ceil(box[2]), x1 + 1, width))
        y2 = int(np.clip(np.ceil(box[3]), y1 + 1, height))
        if x2 <= x1 or y2 <= y1:
            return None
        return np.array([x1, y1, x2, y2], dtype=np.int32)

    def _padded_box(
        self,
        box: NDArray[np.float32],
        frame: NDArray[np.uint8],
    ) -> NDArray[np.float32]:
        """Expand a detection by the configured fraction on every side."""
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
        pad_x = max(0.0, float(x2 - x1)) * self.crop_padding
        pad_y = max(0.0, float(y2 - y1)) * self.crop_padding
        return np.array(
            [
                np.clip(x1 - pad_x, 0, width - 1),
                np.clip(y1 - pad_y, 0, height - 1),
                np.clip(x2 + pad_x, 1, width),
                np.clip(y2 + pad_y, 1, height),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _draw_tracking(
        frame: NDArray[np.uint8],
        tracked: TrackedBox | None,
    ) -> NDArray[np.uint8]:
        output = frame.copy()
        if tracked is None:
            cv2.putText(output, "Tracked boxer unavailable", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            return output
        x1, y1, x2, y2 = np.rint(tracked.xyxy).astype(int)
        color = (0, 180, 255) if tracked.predicted else (0, 255, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        suffix = " (held)" if tracked.predicted else ""
        cv2.putText(output, f"Boxer ID {tracked.track_id}{suffix}", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        return output

    def process(
        self,
        frame: NDArray[np.uint8],
        inference_size: int = DEFAULT_INFERENCE_SIZE,
    ) -> PipelineFrame:
        """Process one original frame and return every diagnostic stage."""
        inference_size = int(inference_size)
        if inference_size not in ALLOWED_INFERENCE_SIZES:
            raise ValueError(
                f"inference_size must be one of {ALLOWED_INFERENCE_SIZES}"
            )
        preprocessed, frame_transform = letterbox(frame, inference_size)
        track_options = {
            "source": preprocessed,
            "persist": True,
            "tracker": "bytetrack.yaml",
            "classes": [0],
            "conf": self.detection_confidence,
            "iou": 0.7,
            "imgsz": inference_size,
            "device": self.device,
            "verbose": False,
        }
        if self.half:
            track_options["half"] = True
        results = self.detector.track(**track_options)

        selected = None
        tracked_result = results[0] if results else None
        if results and results[0].boxes is not None and len(results[0].boxes):
            result_boxes = results[0].boxes
            boxes = result_boxes.xyxy.cpu().numpy().astype(np.float32)
            confidences = result_boxes.conf.cpu().numpy().astype(np.float32)
            if result_boxes.id is None:
                ids = np.arange(len(boxes), dtype=np.int64)
            else:
                ids = result_boxes.id.cpu().numpy().astype(np.int64)
            selected = self._select_track(boxes, ids, confidences, inference_size)

        tracked: TrackedBox | None = None
        selected_index: int | None = None
        if selected is not None:
            selected_index, track_id, canvas_box, confidence = selected
            original_box = frame_transform.to_source_box(canvas_box)
            tracked = TrackedBox(track_id, original_box, confidence, predicted=False)
            self.target_track_id = track_id
            self._last_box = canvas_box.copy()
            self._missing_frames = 0
        elif self.target_track_id is not None and self._last_box is not None and self._missing_frames < self.max_track_gap:
            self._missing_frames += 1
            tracked = TrackedBox(
                self.target_track_id,
                frame_transform.to_source_box(self._last_box),
                0.0,
                predicted=True,
            )
        else:
            self._missing_frames += 1

        detection_frame = self._draw_tracking(frame, tracked)
        if tracked is None:
            return PipelineFrame(preprocessed, detection_frame, None, None, None, None, None, None)

        crop_box = self._integer_crop_box(self._padded_box(tracked.xyxy, frame), frame)
        if crop_box is None:
            return PipelineFrame(preprocessed, detection_frame, None, None, tracked, None, None, None)
        x1, y1, x2, y2 = crop_box
        cv2.rectangle(
            detection_frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (255, 200, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            detection_frame,
            "10% padded pose crop",
            (int(x1), min(frame.shape[0] - 8, int(y2) + 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 200, 0),
            2,
            cv2.LINE_AA,
        )
        exact_crop = frame[y1:y2, x1:x2]
        crop_canvas, crop_transform = letterbox(exact_crop, inference_size)
        original_keypoints: dict[str, NDArray[np.float32]] | None
        crop_keypoints: dict[str, NDArray[np.float32]] | None
        if (
            self.single_pass_pose
            and tracked_result is not None
            and selected_index is not None
        ):
            original_keypoints = self._result_keypoints(
                tracked_result,
                selected_index,
                frame_transform,
            )
            crop_keypoints = (
                self._keypoints_to_crop_canvas(
                    original_keypoints,
                    crop_box,
                    crop_transform,
                )
                if original_keypoints is not None
                else None
            )
        elif self.single_pass_pose:
            original_keypoints = None
            crop_keypoints = None
        else:
            crop_keypoints = self.pose_engine.extract_keypoints(
                crop_canvas,
                imgsz=inference_size,
            )
            original_keypoints = None
            if crop_keypoints is not None:
                original_keypoints = {}
                for name, point in crop_keypoints.items():
                    mapped = np.asarray(point, dtype=np.float32).copy()
                    if np.all(np.isfinite(mapped[:2])):
                        source_point = crop_transform.to_source_points(mapped[:2])
                        mapped[0] = source_point[0] + x1
                        mapped[1] = source_point[1] + y1
                    original_keypoints[name] = mapped

        # Local import avoids a core/visualizer import cycle during unit tests.
        from visualizer.video_annotator import draw_skeleton

        pose_crop = draw_skeleton(crop_canvas, crop_keypoints)
        return PipelineFrame(
            preprocessed_frame=preprocessed,
            detection_frame=detection_frame,
            crop_frame=crop_canvas,
            pose_crop_frame=pose_crop,
            tracked_box=tracked,
            crop_box_xyxy=crop_box,
            raw_keypoints_original=original_keypoints,
            raw_keypoints_crop=crop_keypoints,
        )
