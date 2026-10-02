"""Tests for low-lag smoothing and torso-relative normalization."""

from __future__ import annotations

import unittest

import numpy as np

from core.kinematics import (
    PoseSignalProcessor,
    normalize_keypoints_by_shoulders,
    normalize_keypoints_by_torso,
)


def _point(x: float, y: float, confidence: float = 0.95) -> np.ndarray:
    return np.array([x, y, confidence], dtype=np.float64)


class PoseSignalProcessorTests(unittest.TestCase):
    def test_torso_normalization_uses_neck_origin_and_torso_scale(self) -> None:
        pose = {
            "left_shoulder": _point(80.0, 100.0),
            "right_shoulder": _point(120.0, 100.0),
            "left_hip": _point(85.0, 200.0),
            "right_hip": _point(115.0, 200.0),
            "left_wrist": _point(100.0, 0.0),
        }

        normalized, torso_length, origin_name = normalize_keypoints_by_torso(pose)

        assert normalized is not None
        self.assertEqual(origin_name, "neck")
        self.assertAlmostEqual(torso_length, 100.0)
        np.testing.assert_allclose(normalized["left_wrist"][:2], [0.0, -1.0])

    def test_shoulder_midpoint_is_origin_and_width_is_one(self) -> None:
        pose = {
            "left_shoulder": _point(100.0, 100.0),
            "right_shoulder": _point(200.0, 100.0),
            "left_wrist": _point(250.0, 50.0),
        }

        normalized, width = normalize_keypoints_by_shoulders(pose)

        self.assertIsNotNone(normalized)
        assert normalized is not None
        np.testing.assert_allclose(normalized["left_shoulder"][:2], [-0.5, 0.0])
        np.testing.assert_allclose(normalized["right_shoulder"][:2], [0.5, 0.0])
        np.testing.assert_allclose(normalized["left_wrist"][:2], [1.0, -0.5])
        self.assertAlmostEqual(width, 100.0)

    def test_normalization_is_translation_and_scale_invariant(self) -> None:
        original = {
            "left_shoulder": _point(100.0, 100.0),
            "right_shoulder": _point(200.0, 100.0),
            "left_wrist": _point(250.0, 50.0),
        }
        transformed = {
            name: _point(point[0] * 2.5 + 33.0, point[1] * 2.5 - 17.0)
            for name, point in original.items()
        }

        first, _ = normalize_keypoints_by_shoulders(original)
        second, _ = normalize_keypoints_by_shoulders(transformed)

        assert first is not None and second is not None
        for name in original:
            np.testing.assert_allclose(first[name][:2], second[name][:2], atol=1e-12)

    def test_low_pass_reduces_high_frequency_coordinate_error(self) -> None:
        processor = PoseSignalProcessor(window_length=5, polynomial_order=2)
        noise = [0.0, 4.0, -4.0, 4.0, -4.0, 4.0, -4.0]
        result = None
        for frame, error in enumerate(noise):
            result = processor.update(
                {
                    "left_shoulder": _point(100.0, 100.0),
                    "right_shoulder": _point(200.0, 100.0),
                    "left_wrist": _point(200.0 + 2.0 * frame + error, 80.0),
                }
            )

        assert result is not None
        true_final_x = 200.0 + 2.0 * (len(noise) - 1)
        raw_error = abs(noise[-1])
        filtered_error = abs(result.pixel_keypoints["left_wrist"][0] - true_final_x)
        self.assertLess(filtered_error, raw_error)

    def test_fully_missing_joint_stays_missing_without_filter_error(self) -> None:
        processor = PoseSignalProcessor(window_length=5, polynomial_order=2)
        result = processor.update(
            {
                "left_shoulder": _point(100.0, 100.0),
                "right_shoulder": _point(200.0, 100.0),
                "left_wrist": _point(np.nan, np.nan, confidence=0.1),
            }
        )

        assert result is not None
        self.assertTrue(np.all(np.isnan(result.pixel_keypoints["left_wrist"][:2])))


if __name__ == "__main__":
    unittest.main()
