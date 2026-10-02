"""Geometry tests for aspect-preserving tracked top-down processing."""

from __future__ import annotations

import unittest

import numpy as np

from core.boxer_pipeline import (
    DEFAULT_INFERENCE_SIZE,
    TrackedBoxerPosePipeline,
    letterbox,
)


class LetterboxTests(unittest.TestCase):
    def test_wide_frame_is_not_stretched(self) -> None:
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        canvas, transform = letterbox(image)

        self.assertEqual(canvas.shape[:2], (DEFAULT_INFERENCE_SIZE,) * 2)
        self.assertAlmostEqual(transform.scale, 1.0 / 3.0)
        self.assertEqual(transform.resized_width, 640)
        self.assertEqual(transform.resized_height, 360)
        self.assertEqual(transform.pad_y, 140)

    def test_portrait_frame_is_not_stretched(self) -> None:
        image = np.zeros((1920, 1080, 3), dtype=np.uint8)
        canvas, transform = letterbox(image)

        self.assertEqual(canvas.shape[:2], (DEFAULT_INFERENCE_SIZE,) * 2)
        self.assertEqual(transform.resized_height, 640)
        self.assertEqual(transform.resized_width, 360)
        self.assertEqual(transform.pad_x, 140)

    def test_box_round_trip_maps_to_original_pixels(self) -> None:
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        _, transform = letterbox(image)
        canvas_box = np.array([100.0, 200.0, 500.0, 400.0], dtype=np.float32)

        source_box = transform.to_source_box(canvas_box)

        np.testing.assert_allclose(source_box, [300.0, 180.0, 1500.0, 780.0])

    def test_crop_box_adds_ten_percent_on_every_side(self) -> None:
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        pipeline = TrackedBoxerPosePipeline.__new__(TrackedBoxerPosePipeline)
        pipeline.crop_padding = 0.10

        padded = pipeline._padded_box(
            np.array([100.0, 100.0, 300.0, 300.0], dtype=np.float32),
            frame,
        )

        np.testing.assert_allclose(padded, [80.0, 80.0, 320.0, 320.0])


if __name__ == "__main__":
    unittest.main()
