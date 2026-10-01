"""Deterministic tests for cycle-based punch counting."""

from __future__ import annotations

import unittest

from core.punch_detector import PunchDetector, PunchPhase


class PunchCycleDetectorTests(unittest.TestCase):
    """Exercise the count lifecycle without running pose inference."""

    @staticmethod
    def _sample(
        detector: PunchDetector,
        frame: int,
        *,
        reach: float,
        extension_velocity: float,
        speed: float = 1.2,
        angle: float = 150.0,
    ):
        return detector.update(
            "left",
            (speed, 0.0),
            speed,
            angle,
            frame / 25.0,
            frame,
            reach,
            extension_velocity,
        )

    def test_one_frame_pose_jump_is_not_a_punch(self) -> None:
        detector = PunchDetector()
        self._sample(detector, 0, reach=1.00, extension_velocity=0.0, speed=0.0)
        self._sample(detector, 1, reach=1.10, extension_velocity=2.5)
        event = self._sample(
            detector, 2, reach=1.04, extension_velocity=-1.5, speed=0.4
        )

        self.assertIsNone(event)
        self.assertEqual(detector.summary(1.0)["total_punches"], 0)

    def test_partial_retraction_rearms_before_twelve_frames(self) -> None:
        detector = PunchDetector()
        self._sample(detector, 0, reach=1.00, extension_velocity=0.0, speed=0.0)
        self._sample(detector, 1, reach=1.05, extension_velocity=0.6)
        self._sample(detector, 2, reach=1.12, extension_velocity=0.5)
        first = self._sample(
            detector, 3, reach=1.08, extension_velocity=-0.3, speed=0.4
        )
        self._sample(detector, 4, reach=1.00, extension_velocity=-0.3, speed=0.3)
        self._sample(detector, 5, reach=0.98, extension_velocity=-0.2, speed=0.2)
        self._sample(detector, 6, reach=1.04, extension_velocity=0.6)
        self._sample(detector, 7, reach=1.12, extension_velocity=0.5)
        second = self._sample(
            detector, 8, reach=1.07, extension_velocity=-0.3, speed=0.4
        )

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(detector.summary(1.0)["total_punches"], 2)
        self.assertEqual(detector.phase("left"), PunchPhase.RETRACTING)

    def test_fast_motion_without_outward_reach_is_ignored(self) -> None:
        detector = PunchDetector()
        for frame in range(8):
            self._sample(
                detector,
                frame,
                reach=1.0,
                extension_velocity=0.0,
                speed=4.0,
            )

        self.assertEqual(detector.summary(1.0)["total_punches"], 0)


if __name__ == "__main__":
    unittest.main()
