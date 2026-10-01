"""CPU-only YOLOv8 pose inference and COCO keypoint parsing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import numpy as np

# Keep Ultralytics/Torch on the CPU even on machines with a partially configured GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from ultralytics import YOLO  # noqa: E402  (environment must be set first)


COCO_KEYPOINT_NAMES: Final[tuple[str, ...]] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

BOXING_KEYPOINT_INDICES: Final[dict[str, int]] = {
    "nose": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
}


class PoseEngine:
    """Run lightweight YOLOv8-pose inference on CPU.

    The model file is downloaded automatically by Ultralytics when it is not
    already present. Model construction is deliberately kept in ``__init__``
    so callers such as Streamlit can cache one engine for the process lifetime.
    """

    def __init__(
        self,
        weights: str | Path = "yolov8n-pose.pt",
        confidence_threshold: float = 0.25,
    ) -> None:
        """Load the requested pose weights and pin all inference to CPU."""
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")

        self.weights = str(weights)
        self.confidence_threshold = float(confidence_threshold)
        self.model = YOLO(self.weights)
        self.model.to("cpu")

    def extract_keypoints(
        self,
        frame: np.ndarray,
        imgsz: int = 480,
    ) -> dict[str, np.ndarray] | None:
        """Extract the primary person's 17 COCO keypoints from a BGR frame.

        The primary person is the detection with the largest bounding-box area,
        which is a practical default for a single-fighter coaching view. Each
        dictionary value is ``[x, y, confidence]``. For low-confidence points,
        ``x`` and ``y`` are replaced by ``NaN`` while the measured confidence is
        retained. ``None`` is returned when no person/keypoints are detected.

        Args:
            frame: OpenCV-style BGR image.
            imgsz: YOLO inference size. A default of 480 limits CPU cost.
        """
        if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
            raise ValueError("frame must be a 2D or 3D NumPy image array")
        if frame.size == 0:
            raise ValueError("frame cannot be empty")
        if imgsz <= 0:
            raise ValueError("imgsz must be positive")

        results = self.model.predict(
            source=frame,
            imgsz=imgsz,
            device="cpu",
            verbose=False,
        )
        if not results:
            return None

        result = results[0]
        if result.keypoints is None or result.keypoints.xy is None:
            return None

        xy = result.keypoints.xy.cpu().numpy()
        if xy.ndim != 3 or xy.shape[0] == 0 or xy.shape[1] < 17:
            return None

        if result.boxes is not None and len(result.boxes) == xy.shape[0]:
            boxes = result.boxes.xyxy.cpu().numpy()
            areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(
                boxes[:, 3] - boxes[:, 1], 0.0
            )
            primary_index = int(np.argmax(areas))
        else:
            primary_index = 0

        confidences = result.keypoints.conf
        if confidences is None:
            conf = np.ones(17, dtype=np.float32)
        else:
            conf = confidences[primary_index].cpu().numpy()[:17].astype(np.float32)

        points = np.column_stack(
            (xy[primary_index, :17, :2].astype(np.float32), conf)
        )
        low_confidence = points[:, 2] < self.confidence_threshold
        points[low_confidence, :2] = np.nan

        return {
            name: points[index].copy()
            for index, name in enumerate(COCO_KEYPOINT_NAMES)
        }
