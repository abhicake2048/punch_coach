"""Production inference for the trained multi-task LSTM and ST-GCN models."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import ArrayLike

from .punch_features import (
    KinematicFeatureTracker,
    build_runtime_arrays,
    normalize_pose_at_neck,
    pose_to_record,
)
from .punch_models import MultiTaskLSTM, MultiTaskSTGCN


SUPPORTED_MODELS: Final[tuple[str, ...]] = ("lstm", "stgcn")
MINIMUM_PUNCH_CONFIDENCE: Final[float] = 0.91


@dataclass(frozen=True)
class MultiTaskScores:
    """Latest hand and punch-type probabilities for the live UI."""

    hand: dict[str, float]
    punch_type: dict[str, float]

    @classmethod
    def empty(cls) -> "MultiTaskScores":
        return cls(
            hand={"none": 0.0, "left": 0.0, "right": 0.0},
            punch_type={
                "background": 0.0,
                "cross": 0.0,
                "jab": 0.0,
                "hook": 0.0,
                "uppercut": 0.0,
            },
        )


@dataclass(frozen=True)
class PunchEvent:
    """One registered production punch."""

    timestamp: float
    frame_index: int
    action_kind: str
    action_name: str
    confidence: float
    hand: str
    speed: float

    @property
    def punch_type(self) -> str:
        return self.action_name

    @property
    def label(self) -> str:
        return f"{self.hand.title()} {self.action_name.title()}"


@dataclass(frozen=True)
class _Prediction:
    timestamp: float
    frame_index: int
    hand: str
    punch_type: str
    confidence: float
    activation: float
    speed: float
    extension_velocity: float


class TrainedPunchRecognizer:
    """Sliding-window inference with model/motion peak de-duplication.

    Preprocessing deliberately calls the same feature builders used to create
    the training tensors. The checkpoint is authoritative for sequence length,
    model dimensions, class maps, feature order, and normalization statistics.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        model_kind: str | None = None,
        device: str = "auto",
        confidence_threshold: float = MINIMUM_PUNCH_CONFIDENCE,
        keypoint_confidence: float = 0.25,
        recovery_frames: int = 8,
        peak_prominence: float = 0.04,
        minimum_wrist_speed: float = 0.20,
        minimum_extension_velocity: float = 0.05,
        minimum_pose_coverage: float = 0.80,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Trained checkpoint not found: {self.checkpoint_path}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        checkpoint_kind = str(checkpoint.get("model_kind", "")).lower()
        requested_kind = (model_kind or checkpoint_kind).lower()
        if requested_kind not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model kind: {requested_kind!r}")
        if checkpoint_kind and checkpoint_kind != requested_kind:
            raise ValueError(
                f"Checkpoint contains {checkpoint_kind!r}, not {requested_kind!r}"
            )
        self.model_kind = requested_kind
        self.schema = dict(checkpoint["feature_schema"])
        self.validation_metrics = dict(checkpoint.get("validation_metrics", {}))
        self.sequence_length = int(self.schema["sequence_length"])
        if self.sequence_length <= 0:
            raise ValueError("Checkpoint sequence length must be positive")
        self.hand_classes = self._ordered_classes(checkpoint["hand_classes"])
        self.punch_classes = self._ordered_classes(checkpoint["punch_type_classes"])
        self.model = self._build_model(checkpoint)
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.model.eval()

        stats = checkpoint["feature_stats"]
        stat_key = "lstm_kinematic" if self.model_kind == "lstm" else "stgcn_kinematic"
        self.feature_mean = np.asarray(stats[stat_key]["mean"], dtype=np.float32)
        self.feature_std = np.asarray(stats[stat_key]["std"], dtype=np.float32)
        if self.feature_mean.shape != self.feature_std.shape:
            raise ValueError("Checkpoint feature mean/std shapes do not match")
        self.feature_std = self.feature_std.copy()
        self.feature_std[self.feature_std < 1e-6] = 1.0

        self.confidence_threshold = float(confidence_threshold)
        self.keypoint_confidence = float(keypoint_confidence)
        self.recovery_frames = int(recovery_frames)
        self.peak_prominence = float(peak_prominence)
        self.minimum_wrist_speed = float(minimum_wrist_speed)
        self.minimum_extension_velocity = float(minimum_extension_velocity)
        self.minimum_pose_coverage = float(minimum_pose_coverage)
        self.fps = 30.0
        self._records: deque[dict[str, Any]] = deque(maxlen=self.sequence_length)
        self._kinematics = KinematicFeatureTracker(self.keypoint_confidence)
        self._peak: _Prediction | None = None
        self._last_punch_frame = -10**9
        self._cooldown = 0
        self._active_hand: str | None = None
        self.latest_prediction: _Prediction | None = None
        self.events: list[PunchEvent] = []
        self.latest_scores = MultiTaskScores.empty()
        self.latest_wrist_speeds = {"left": float("nan"), "right": float("nan")}
        self.latest_extension_velocities = {
            "left": float("nan"),
            "right": float("nan"),
        }

    @staticmethod
    def _ordered_classes(class_map: Mapping[str, int]) -> tuple[str, ...]:
        ordered = sorted(class_map.items(), key=lambda item: int(item[1]))
        indices = [int(index) for _, index in ordered]
        if indices != list(range(len(indices))):
            raise ValueError(f"Class IDs must be contiguous from zero: {dict(class_map)}")
        return tuple(name for name, _ in ordered)

    def _build_model(self, checkpoint: Mapping[str, Any]) -> torch.nn.Module:
        config = dict(checkpoint.get("config", {}))
        if self.model_kind == "lstm":
            model = MultiTaskLSTM(
                input_size=len(self.schema["lstm_kinematic_features"]),
                hidden_size=int(config.get("hidden_size", 128)),
                num_layers=int(config.get("num_layers", 2)),
                dropout=float(config.get("dropout", 0.5)),
                hand_classes=len(self.hand_classes),
                punch_classes=len(self.punch_classes),
            )
        else:
            model = MultiTaskSTGCN(
                input_channels=len(self.schema["stgcn_channels"]),
                kinematic_features=len(self.schema["stgcn_kinematic_features"]),
                dropout=float(config.get("dropout", 0.3)),
                hand_classes=len(self.hand_classes),
                punch_classes=len(self.punch_classes),
            )
        return model.to(self.device)

    @property
    def display_name(self) -> str:
        return "LSTM" if self.model_kind == "lstm" else "ST-GCN"

    @property
    def effective_confidence_threshold(self) -> float:
        """Never allow production punch registration below 78 percent."""
        return max(MINIMUM_PUNCH_CONFIDENCE, float(self.confidence_threshold))

    @property
    def current_confidence(self) -> float:
        return self.latest_prediction.confidence if self.latest_prediction else 0.0

    @property
    def current_label(self) -> str:
        """Return a punch label only when its joint confidence exceeds 78%."""
        prediction = self.latest_prediction
        if (
            prediction is None
            or prediction.confidence <= self.effective_confidence_threshold
        ):
            return "IDLE"
        return f"{prediction.hand.title()} {prediction.punch_type.title()}"

    @property
    def current_punching_hand(self) -> str | None:
        """Return the confidently active punching hand for guard exclusion."""
        prediction = self.latest_prediction
        if prediction is None or prediction.hand not in {"left", "right"}:
            return None
        if prediction.punch_type == "background":
            return None
        if prediction.confidence <= self.effective_confidence_threshold:
            return None
        if prediction.speed < self.minimum_wrist_speed:
            return None
        if prediction.extension_velocity < self.minimum_extension_velocity:
            return None
        return prediction.hand

    def reset(self, fps: float = 30.0) -> None:
        self.fps = float(fps) if np.isfinite(fps) and fps > 0.0 else 30.0
        self._records.clear()
        self._kinematics = KinematicFeatureTracker(self.keypoint_confidence)
        self._peak = None
        self._last_punch_frame = -10**9
        self._cooldown = 0
        self._active_hand = None
        self.latest_prediction = None
        self.events.clear()
        self.latest_scores = MultiTaskScores.empty()
        self.latest_wrist_speeds = {"left": float("nan"), "right": float("nan")}
        self.latest_extension_velocities = {
            "left": float("nan"),
            "right": float("nan"),
        }

    def _frame_row(
        self,
        keypoints: Mapping[str, ArrayLike] | None,
        timestamp: float,
        frame_index: int,
    ) -> dict[str, Any]:
        pose = normalize_pose_at_neck(keypoints, self.keypoint_confidence)
        kinematics = self._kinematics.update(pose, float(timestamp))
        row: dict[str, Any] = {
            "frame_index": int(frame_index),
            "timestamp_s": float(timestamp),
        }
        row.update(pose_to_record(pose))
        row.update(kinematics)
        self.latest_wrist_speeds = {
            side: float(kinematics.get(f"{side}_wrist_speed", float("nan")))
            for side in ("left", "right")
        }
        self.latest_extension_velocities = {
            side: float(
                kinematics.get(f"{side}_wrist_extension_velocity", float("nan"))
            )
            for side in ("left", "right")
        }
        return row

    def _model_inputs(self) -> tuple[torch.Tensor, ...] | None:
        if len(self._records) < self.sequence_length:
            return None
        arrays = build_runtime_arrays(
            list(self._records),
            fps=self.fps,
            schema=self.schema,
            minimum_coverage=self.minimum_pose_coverage,
        )
        if arrays is None:
            return None
        if self.model_kind == "lstm":
            features = arrays["lstm_kinematic"].astype(np.float32)
            if features.shape[1] != self.feature_mean.size:
                raise ValueError(
                    f"LSTM feature mismatch: {features.shape[1]} != {self.feature_mean.size}"
                )
            features = (features - self.feature_mean) / self.feature_std
            return (torch.from_numpy(features).unsqueeze(0).to(self.device),)
        graph = torch.from_numpy(arrays["stgcn"]).unsqueeze(0).to(self.device)
        kinematics = arrays["stgcn_kinematic"].astype(np.float32)
        if kinematics.shape[1] != self.feature_mean.size:
            raise ValueError(
                f"ST-GCN kinematic mismatch: {kinematics.shape[1]} != {self.feature_mean.size}"
            )
        kinematics = (kinematics - self.feature_mean) / self.feature_std
        return graph, torch.from_numpy(kinematics).unsqueeze(0).to(self.device)

    @staticmethod
    def _probabilities(
        logits: torch.Tensor,
        classes: tuple[str, ...],
    ) -> tuple[dict[str, float], str, float]:
        values = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
        index = int(np.argmax(values))
        scores = {name: float(values[position]) for position, name in enumerate(classes)}
        return scores, classes[index], float(values[index])

    def _predict(self, inputs: tuple[torch.Tensor, ...]) -> _Prediction:
        with torch.inference_mode():
            outputs = self.model(*inputs)
        hand_scores, hand, hand_confidence = self._probabilities(
            outputs["hand"], self.hand_classes
        )
        punch_scores, punch_type, punch_confidence = self._probabilities(
            outputs["punch_type"], self.punch_classes
        )
        self.latest_scores = MultiTaskScores(hand_scores, punch_scores)
        positive = hand != "none" and punch_type != "background"
        confidence = min(hand_confidence, punch_confidence) if positive else 0.0
        speed = self.latest_wrist_speeds.get(hand, float("nan")) if positive else 0.0
        extension_velocity = (
            self.latest_extension_velocities.get(hand, float("nan"))
            if positive
            else 0.0
        )
        finite_speed = float(speed) if np.isfinite(speed) else 0.0
        finite_extension = (
            float(extension_velocity) if np.isfinite(extension_velocity) else 0.0
        )
        # A motion-weighted activation creates a local peak even when a weak
        # background head stays overconfident throughout an idle interval.
        outward = finite_extension >= self.minimum_extension_velocity
        motion_factor = float(np.tanh(max(finite_speed, 0.0))) if outward else 0.0
        activation = confidence * motion_factor
        return _Prediction(
            timestamp=float(self._records[-1]["timestamp_s"]),
            frame_index=int(self._records[-1]["frame_index"]),
            hand=hand,
            punch_type=punch_type,
            confidence=confidence,
            activation=activation,
            speed=finite_speed,
            extension_velocity=finite_extension,
        )

    def _emit_peak(self) -> PunchEvent | None:
        peak = self._peak
        self._peak = None
        if peak is None:
            return None
        if peak.frame_index - self._last_punch_frame < self.recovery_frames:
            return None
        if peak.confidence <= self.effective_confidence_threshold:
            return None
        if peak.speed < self.minimum_wrist_speed:
            return None
        if peak.extension_velocity < self.minimum_extension_velocity:
            return None
        event = PunchEvent(
            timestamp=peak.timestamp,
            frame_index=peak.frame_index,
            action_kind="punch",
            action_name=peak.punch_type,
            confidence=peak.confidence,
            hand=peak.hand,
            speed=peak.speed,
        )
        self.events.append(event)
        self._last_punch_frame = peak.frame_index
        self._active_hand = peak.hand
        self._cooldown = self.recovery_frames
        return event

    def _advance_peak(self, current: _Prediction | None) -> list[PunchEvent]:
        if current is None or current.confidence <= 0.0:
            event = self._emit_peak()
            return [event] if event is not None else []
        if self._peak is None:
            self._peak = current
            return []
        if current.activation >= self._peak.activation:
            self._peak = current
            return []
        fell = self._peak.activation - current.activation >= self.peak_prominence
        if fell:
            event = self._emit_peak()
            self._peak = current
            return [event] if event is not None else []
        return []

    def update(
        self,
        keypoints: Mapping[str, ArrayLike] | None,
        timestamp: float,
        frame_index: int,
    ) -> list[PunchEvent]:
        """Add one smoothed pose and emit zero or one de-duplicated punch."""
        if self._cooldown > 0:
            self._cooldown -= 1
            if self._cooldown == 0:
                self._active_hand = None
        self._records.append(self._frame_row(keypoints, timestamp, frame_index))
        inputs = self._model_inputs()
        if inputs is None:
            self.latest_scores = MultiTaskScores.empty()
            self.latest_prediction = None
            return self._advance_peak(None)
        self.latest_prediction = self._predict(inputs)
        return self._advance_peak(self.latest_prediction)

    def flush(self) -> list[PunchEvent]:
        """Emit a final valid peak when the video ends before its falling edge."""
        event = self._emit_peak()
        return [event] if event is not None else []

    def is_punch_active(self, hand: str) -> bool:
        return self._cooldown > 0 and self._active_hand == hand

    def summary(self, duration_seconds: float) -> dict[str, object]:
        by_type = Counter(event.punch_type for event in self.events)
        by_hand = Counter(event.hand for event in self.events)
        by_type_and_hand = Counter(
            f"{event.hand.title()} {event.punch_type}" for event in self.events
        )
        return {
            "model_kind": self.model_kind,
            "model_display_name": self.display_name,
            "sequence_length": self.sequence_length,
            "total_punches": len(self.events),
            "punches_per_minute": (
                len(self.events) * 60.0 / duration_seconds
                if duration_seconds > 0.0
                else 0.0
            ),
            "by_type": dict(by_type),
            "by_hand": dict(by_hand),
            "by_type_and_hand": dict(by_type_and_hand),
        }
