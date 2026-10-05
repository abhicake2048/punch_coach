"""Tracked top-down boxer detection and cropped YOLO11 pose inference.

The full-frame detector can run at a smaller CPU-friendly resolution than the
cropped pose stage. Frame-order association keeps the selected person stable,
and all public coordinates are mapped back to the untouched original frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from ultralytics import YOLO

from .pose_engine import PoseEngine


ALLOWED_INFERENCE_SIZES: Final[tuple[int, int]] = (480, 640)
ALLOWED_DETECTION_SIZES: Final[tuple[int, ...]] = (320, 384, 416, 480, 640)
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
        # Keep independent detector and cropped-pose model instances. This is
        # the same top-down preprocessing used to create the training data and
        # prevents tracking state from being disturbed by crop inference.
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
    ) -> tuple[int, NDArray[np.float32], float] | None:
        """Keep the locked ID; reacquire conservatively after a real loss."""
        if boxes.size == 0:
            return None
        if self.target_track_id is not None:
            matches = np.flatnonzero(ids == self.target_track_id)
            if matches.size:
                index = int(matches[0])
                return int(ids[index]), boxes[index].copy(), float(confidences[index])
            if self._missing_frames <= self.max_track_gap:
                return None

        if self._last_box is not None:
            overlaps = np.asarray([self._iou(box, self._last_box) for box in boxes])
            index = int(np.argmax(overlaps))
            if overlaps[index] >= self.tracking_iou:
                return int(ids[index]), boxes[index].copy(), float(confidences[index])

        # First acquisition: prioritize a large person near the frame center.
        areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        canvas_center = np.array([canvas_size / 2, canvas_size / 2], dtype=np.float32)
        distances = np.linalg.norm((centers - canvas_center) / canvas_center, axis=1)
        scores = areas * confidences / (1.0 + distances)
        index = int(np.argmax(scores))
        return int(ids[index]), boxes[index].copy(), float(confidences[index])

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
        detection_size: int | None = None,
    ) -> PipelineFrame:
        """Process one original frame and return every diagnostic stage."""
        inference_size = int(inference_size)
        if inference_size not in ALLOWED_INFERENCE_SIZES:
            raise ValueError(
                f"inference_size must be one of {ALLOWED_INFERENCE_SIZES}"
            )
        detection_size = inference_size if detection_size is None else int(detection_size)
        if detection_size not in ALLOWED_DETECTION_SIZES:
            raise ValueError(
                f"detection_size must be one of {ALLOWED_DETECTION_SIZES}"
            )
        preprocessed, frame_transform = letterbox(frame, detection_size)
        track_options = {
            "source": preprocessed,
            "persist": True,
            "tracker": "bytetrack.yaml",
            "classes": [0],
            "conf": self.detection_confidence,
            "iou": 0.7,
            "imgsz": detection_size,
            "device": self.device,
            "verbose": False,
        }
        if self.half:
            track_options["half"] = True
        results = self.detector.track(**track_options)

        selected = None
        if results and results[0].boxes is not None and len(results[0].boxes):
            result_boxes = results[0].boxes
            boxes = result_boxes.xyxy.cpu().numpy().astype(np.float32)
            confidences = result_boxes.conf.cpu().numpy().astype(np.float32)
            if result_boxes.id is None:
                ids = np.arange(len(boxes), dtype=np.int64)
            else:
                ids = result_boxes.id.cpu().numpy().astype(np.int64)
            selected = self._select_track(boxes, ids, confidences, detection_size)

        tracked: TrackedBox | None = None
        if selected is not None:
            track_id, canvas_box, confidence = selected
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
        crop_keypoints = self.pose_engine.extract_keypoints(
            crop_canvas,
            imgsz=inference_size,
        )
        original_keypoints: dict[str, NDArray[np.float32]] | None = None
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

    def process_batch(
        self,
        frames: Sequence[NDArray[np.uint8]],
        inference_size: int = DEFAULT_INFERENCE_SIZE,
        detection_size: int | None = None,
        draw_diagnostics: bool = False,
    ) -> list[PipelineFrame]:
        """Batch both YOLO stages while associating the boxer in frame order."""
        if not frames:
            return []
        inference_size = int(inference_size)
        if inference_size not in ALLOWED_INFERENCE_SIZES:
            raise ValueError(f"inference_size must be one of {ALLOWED_INFERENCE_SIZES}")
        detection_size = inference_size if detection_size is None else int(detection_size)
        if detection_size not in ALLOWED_DETECTION_SIZES:
            raise ValueError(f"detection_size must be one of {ALLOWED_DETECTION_SIZES}")
        preprocessed, transforms = [], []
        for frame in frames:
            canvas, transform = letterbox(frame, detection_size)
            preprocessed.append(canvas)
            transforms.append(transform)
        options = {
            "source": preprocessed, "classes": [0], "conf": self.detection_confidence,
            "iou": 0.7, "imgsz": detection_size, "device": self.device,
            "verbose": False, "rect": False,
        }
        if self.half:
            options["half"] = True
        detection_results = self.detector.predict(**options)
        if len(detection_results) != len(frames):
            raise RuntimeError(
                "YOLO detector returned "
                f"{len(detection_results)} results for {len(frames)} frames"
            )
        tracked_rows: list[TrackedBox | None] = []
        crop_boxes: list[NDArray[np.int32] | None] = []
        crop_canvases: list[NDArray[np.uint8]] = []
        crop_transforms: list[LetterboxTransform] = []
        crop_indices: list[int] = []
        for index, (frame, transform, result) in enumerate(zip(frames, transforms, detection_results)):
            selected = None
            if result.boxes is not None and len(result.boxes):
                boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
                confidences = result.boxes.conf.cpu().numpy().astype(np.float32)
                if self._last_box is not None:
                    overlaps = np.asarray([self._iou(box, self._last_box) for box in boxes])
                    candidate = int(np.argmax(overlaps))
                    if overlaps[candidate] >= self.tracking_iou or self._missing_frames <= self.max_track_gap:
                        selected = (boxes[candidate], float(confidences[candidate]))
                if selected is None:
                    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
                    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
                    center = np.array([detection_size / 2, detection_size / 2], dtype=np.float32)
                    scores = areas * confidences / (1 + np.linalg.norm((centers - center) / center, axis=1))
                    candidate = int(np.argmax(scores))
                    selected = (boxes[candidate], float(confidences[candidate]))
            tracked = None
            if selected is not None:
                canvas_box, confidence = selected
                self._last_box = canvas_box.copy()
                self.target_track_id = 1
                self._missing_frames = 0
                tracked = TrackedBox(1, transform.to_source_box(canvas_box), confidence, False)
            elif self._last_box is not None and self._missing_frames < self.max_track_gap:
                self._missing_frames += 1
                tracked = TrackedBox(1, transform.to_source_box(self._last_box), 0.0, True)
            else:
                self._missing_frames += 1
            tracked_rows.append(tracked)
            crop_box = None if tracked is None else self._integer_crop_box(self._padded_box(tracked.xyxy, frame), frame)
            crop_boxes.append(crop_box)
            if crop_box is not None:
                x1, y1, x2, y2 = crop_box
                canvas, crop_transform = letterbox(frame[y1:y2, x1:x2], inference_size)
                crop_canvases.append(canvas)
                crop_transforms.append(crop_transform)
                crop_indices.append(index)
        crop_poses = self.pose_engine.extract_keypoints_batch(crop_canvases, imgsz=inference_size)
        poses_by_index = {index: pose for index, pose in zip(crop_indices, crop_poses)}
        transform_by_index = {index: transform for index, transform in zip(crop_indices, crop_transforms)}
        crop_by_index = {index: crop for index, crop in zip(crop_indices, crop_canvases)}
        outputs: list[PipelineFrame] = []
        for index, frame in enumerate(frames):
            crop_box = crop_boxes[index]
            crop_pose = poses_by_index.get(index)
            original_pose = None
            if crop_box is not None and crop_pose is not None:
                x1, y1, _, _ = crop_box
                crop_transform = transform_by_index[index]
                original_pose = {}
                for name, point in crop_pose.items():
                    mapped = np.asarray(point, dtype=np.float32).copy()
                    if np.all(np.isfinite(mapped[:2])):
                        source = crop_transform.to_source_points(mapped[:2])
                        mapped[0], mapped[1] = source[0] + x1, source[1] + y1
                    original_pose[name] = mapped
            detection_frame = self._draw_tracking(frame, tracked_rows[index]) if draw_diagnostics else frame
            outputs.append(PipelineFrame(
                preprocessed_frame=preprocessed[index], detection_frame=detection_frame,
                crop_frame=crop_by_index.get(index),
                pose_crop_frame=None, tracked_box=tracked_rows[index], crop_box_xyxy=crop_box,
                raw_keypoints_original=original_pose, raw_keypoints_crop=crop_pose,
            ))
        return outputs
