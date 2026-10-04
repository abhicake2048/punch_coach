"""Grounded Gemini coaching reports from structured CornerCoach metrics."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Any, Final


DEFAULT_GEMINI_MODEL: Final[str] = "gemini-2.5-flash"

_POINT_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {"type": "string"},
        "comment": {"type": "string"},
        "metric_keys": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "comment", "metric_keys"],
}

REPORT_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "strengths": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": _POINT_SCHEMA,
        },
        "areas_to_improve": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": _POINT_SCHEMA,
        },
        "suggested_drills": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": _POINT_SCHEMA,
        },
        "data_limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "summary",
        "strengths",
        "areas_to_improve",
        "suggested_drills",
        "data_limitations",
    ],
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return round(value, 4) if math.isfinite(value) else None
    return str(value)


def _attribute(source: Any, name: str, default: Any = 0) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def build_coaching_metrics(
    punch_summary: Mapping[str, object],
    guard_summary: Mapping[str, object],
    fatigue_report: Any,
    duration_seconds: float,
) -> dict[str, Any]:
    """Build the only structured data that is permitted to reach Gemini."""
    metrics = {
        "session": {
            "duration_seconds": duration_seconds,
            "model": punch_summary.get("model_display_name"),
            "sequence_length_frames": punch_summary.get("sequence_length"),
        },
        "punches": {
            "total": punch_summary.get("total_punches", 0),
            "per_minute": punch_summary.get("punches_per_minute", 0.0),
            "by_hand": punch_summary.get("by_hand", {}),
            "by_type": punch_summary.get("by_type", {}),
            "by_hand_and_type": punch_summary.get("by_type_and_hand", {}),
        },
        "guard": {
            "discipline_score_percent": guard_summary.get("discipline_score", 0.0),
            "eligible_frames": guard_summary.get("eligible_frames", 0),
            "safe_frames": guard_summary.get("safe_frames", 0),
            "arm_evaluations": guard_summary.get("arm_evaluations", {}),
            "per_arm_scores_percent": guard_summary.get("per_arm_scores", {}),
            "drop_episodes_by_hand": guard_summary.get("drop_episodes", {}),
            "drop_episodes_by_level": guard_summary.get("drop_levels", {}),
            "events": guard_summary.get("events", []),
        },
        "fatigue": {
            "detected": _attribute(fatigue_report, "fatigued", False),
            "first_third_punches": _attribute(fatigue_report, "first_third_count"),
            "last_third_punches": _attribute(fatigue_report, "last_third_count"),
            "first_work_rate_ppm": _attribute(
                fatigue_report, "first_work_rate_ppm", 0.0
            ),
            "last_work_rate_ppm": _attribute(
                fatigue_report, "last_work_rate_ppm", 0.0
            ),
            "work_rate_drop_percent": _attribute(
                fatigue_report, "work_rate_drop_percent", 0.0
            ),
            "first_average_speed_torso_lengths_per_second": _attribute(
                fatigue_report, "first_average_speed", 0.0
            ),
            "last_average_speed_torso_lengths_per_second": _attribute(
                fatigue_report, "last_average_speed", 0.0
            ),
            "speed_drop_percent": _attribute(
                fatigue_report, "speed_drop_percent", 0.0
            ),
        },
    }
    return _json_safe(metrics)


def flatten_metrics(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten metric paths so every LLM evidence reference can be verified."""
    flattened: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_metrics(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            flattened.update(flatten_metrics(item, child))
    else:
        flattened[prefix] = value
    return flattened


def _validate_report(report: Any, metrics: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise ValueError("Gemini did not return a JSON object")
    expected_lengths = {
        "strengths": 3,
        "areas_to_improve": 3,
        "suggested_drills": 2,
    }
    valid_metric_keys = flatten_metrics(metrics)
    for section, expected_length in expected_lengths.items():
        points = report.get(section)
        if not isinstance(points, list) or len(points) != expected_length:
            raise ValueError(f"Gemini report must contain {expected_length} {section}")
        for point in points:
            if not isinstance(point, dict):
                raise ValueError(f"Every {section} entry must be an object")
            title = point.get("title")
            comment = point.get("comment")
            metric_keys = point.get("metric_keys")
            if not isinstance(title, str) or not title.strip():
                raise ValueError(f"Every {section} entry needs a title")
            if not isinstance(comment, str) or not comment.strip():
                raise ValueError(f"Every {section} entry needs a comment")
            if not isinstance(metric_keys, list) or not all(
                isinstance(key, str) for key in metric_keys
            ):
                raise ValueError(f"Every {section} entry needs metric_keys")
            unknown = [key for key in metric_keys if key not in valid_metric_keys]
            if unknown:
                raise ValueError(
                    "Gemini cited unknown metrics: " + ", ".join(sorted(unknown))
                )
            if not metric_keys and "insufficient evidence" not in comment.lower():
                raise ValueError(
                    f"Ungrounded {section} entry {title!r} has no metric evidence"
                )
            cited_numbers = [
                float(match)
                for match in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", comment)
            ]
            evidence_numbers = [
                float(valid_metric_keys[key])
                for key in metric_keys
                if isinstance(valid_metric_keys[key], (int, float))
                and not isinstance(valid_metric_keys[key], bool)
            ]
            unsupported_numbers = [
                number
                for number in cited_numbers
                if not any(
                    math.isclose(number, evidence, rel_tol=1e-3, abs_tol=1e-3)
                    for evidence in evidence_numbers
                )
            ]
            if unsupported_numbers:
                raise ValueError(
                    f"Gemini cited unsupported numeric values in {title!r}: "
                    + ", ".join(str(value) for value in unsupported_numbers)
                )
    if not isinstance(report.get("summary"), str):
        raise ValueError("Gemini report is missing its summary")
    limitations = report.get("data_limitations")
    if not isinstance(limitations, list) or not all(
        isinstance(item, str) for item in limitations
    ):
        raise ValueError("Gemini report has invalid data limitations")
    return report


def evidence_for_point(
    point: Mapping[str, Any], metrics: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve validated evidence paths to application-owned metric values."""
    flattened = flatten_metrics(metrics)
    return {key: flattened[key] for key in point.get("metric_keys", [])}


def generate_coaching_report(
    metrics: Mapping[str, Any],
    api_key: str,
    model: str = DEFAULT_GEMINI_MODEL,
    *,
    client: Any | None = None,
) -> dict[str, Any]:
    """Generate and validate a short report grounded only in supplied metrics."""
    if not api_key.strip():
        raise ValueError("A Gemini API key is required")
    if not model.strip():
        raise ValueError("A Gemini model name is required")
    safe_metrics = _json_safe(metrics)
    if client is None:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "Gemini reporting requires the google-genai package"
            ) from exc
        client = genai.Client(api_key=api_key.strip())

    allowed_metric_keys = sorted(flatten_metrics(safe_metrics))
    prompt = (
        "You are a concise boxing coach. Use ONLY the JSON metrics below. "
        "Do not claim to have watched the video and do not infer stance, accuracy, "
        "power, footwork, defense, or technique unless a supplied metric directly "
        "supports it. Produce exactly three strengths, three areas to improve, and "
        "two drills. For every point, cite only exact leaf paths from the JSON in "
        "metric_keys. If the data cannot support a requested point, write "
        "'Insufficient evidence' in its comment and leave metric_keys empty. Drills "
        "must address a measured weakness. Keep comments short and practical.\n\n"
        "VALID_METRIC_KEYS:\n"
        + json.dumps(allowed_metric_keys, separators=(",", ":"))
        + "\n\n"
        "SESSION_METRICS_JSON:\n"
        + json.dumps(safe_metrics, sort_keys=True, separators=(",", ":"))
    )
    response = client.models.generate_content(
        model=model.strip(),
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": REPORT_SCHEMA,
            "temperature": 0.2,
        },
    )
    response_text = getattr(response, "text", None)
    if not isinstance(response_text, str) or not response_text.strip():
        raise RuntimeError("Gemini returned an empty coaching report")
    try:
        report = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Gemini returned invalid JSON") from exc
    return _validate_report(report, safe_metrics)
