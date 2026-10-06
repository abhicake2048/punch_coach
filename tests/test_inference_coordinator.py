from __future__ import annotations

import unittest

from core.inference_coordinator import InferenceCoordinator


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class InferenceCoordinatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.coordinator = InferenceCoordinator(
            waiting_timeout_seconds=20.0,
            clock=self.clock,
        )

    def test_jobs_start_one_at_a_time_in_fifo_order(self) -> None:
        first = self.coordinator.register("first", "session-1")
        second = self.coordinator.register("second", "session-2")
        third = self.coordinator.register("third", "session-3")

        self.assertEqual(first.jobs_ahead, 0)
        self.assertEqual(second.jobs_ahead, 1)
        self.assertEqual(third.jobs_ahead, 2)
        self.assertIsNone(self.coordinator.try_start("second", "session-2"))

        first_lease = self.coordinator.try_start("first", "session-1")
        self.assertIsNotNone(first_lease)
        self.assertIsNone(self.coordinator.try_start("first", "session-1"))
        self.assertIsNone(self.coordinator.try_start("second", "session-2"))
        self.assertEqual(self.coordinator.status("second").jobs_ahead, 1)

        self.assertTrue(self.coordinator.finish(first_lease))
        second_lease = self.coordinator.try_start("second", "session-2")
        self.assertIsNotNone(second_lease)
        self.assertTrue(self.coordinator.finish(second_lease))
        third_lease = self.coordinator.try_start("third", "session-3")
        self.assertIsNotNone(third_lease)
        self.assertTrue(self.coordinator.finish(third_lease))
        self.assertEqual(self.coordinator.waiting_jobs, 0)
        self.assertIsNone(self.coordinator.active_job_id)

    def test_browser_rerun_does_not_duplicate_its_queue_entry(self) -> None:
        first = self.coordinator.register("same-job", "same-session")
        self.clock.advance(1.0)
        refreshed = self.coordinator.register("same-job", "same-session")

        self.assertEqual(self.coordinator.waiting_jobs, 1)
        self.assertEqual(refreshed.job_id, first.job_id)
        self.assertEqual(refreshed.initial_position, first.initial_position)
        self.assertEqual(refreshed.waited_seconds, 1.0)

    def test_stale_waiter_is_removed_and_position_shrinks(self) -> None:
        self.coordinator.register("active", "session-a")
        active_lease = self.coordinator.try_start("active", "session-a")
        self.coordinator.register("abandoned", "session-b")
        self.coordinator.register("connected", "session-c")

        self.clock.advance(21.0)
        self.coordinator.touch("connected", "session-c")
        removed = self.coordinator.prune()

        self.assertEqual(removed, ["abandoned"])
        self.assertIsNone(self.coordinator.status("abandoned"))
        self.assertEqual(self.coordinator.status("connected").jobs_ahead, 1)
        self.assertTrue(self.coordinator.finish(active_lease))

    def test_disconnected_waiter_is_removed_immediately(self) -> None:
        self.coordinator.register("active", "session-a")
        active_lease = self.coordinator.try_start("active", "session-a")
        self.coordinator.register("closed", "session-b")
        self.coordinator.register("open", "session-c")

        removed = self.coordinator.prune(
            lambda session_id: session_id in {"session-a", "session-c"}
        )

        self.assertEqual(removed, ["closed"])
        self.assertEqual(self.coordinator.status("open").jobs_ahead, 1)
        self.assertTrue(self.coordinator.finish(active_lease))

    def test_active_disconnect_waits_for_safe_release(self) -> None:
        self.coordinator.register("active", "session-a")
        active_lease = self.coordinator.try_start("active", "session-a")
        self.coordinator.register("next", "session-b")

        self.coordinator.prune(lambda session_id: session_id == "session-b")

        self.assertTrue(self.coordinator.cancellation_requested("active"))
        self.assertEqual(self.coordinator.active_job_id, "active")
        self.assertIsNone(self.coordinator.try_start("next", "session-b"))
        self.assertTrue(self.coordinator.finish(active_lease))
        self.assertIsNotNone(self.coordinator.try_start("next", "session-b"))

    def test_cancelled_waiter_cannot_block_the_queue(self) -> None:
        self.coordinator.register("active", "session-a")
        active_lease = self.coordinator.try_start("active", "session-a")
        self.coordinator.register("cancelled", "session-b")
        self.coordinator.register("next", "session-c")

        self.assertTrue(self.coordinator.request_cancel("cancelled", "session-b"))
        self.assertIsNone(self.coordinator.status("cancelled"))
        self.assertEqual(self.coordinator.status("next").jobs_ahead, 1)
        self.assertTrue(self.coordinator.finish(active_lease))
        self.assertIsNotNone(self.coordinator.try_start("next", "session-c"))

    def test_wrong_session_cannot_touch_or_cancel_a_job(self) -> None:
        self.coordinator.register("job", "owner")
        with self.assertRaisesRegex(RuntimeError, "another session"):
            self.coordinator.touch("job", "intruder")
        self.assertFalse(self.coordinator.request_cancel("job", "intruder"))
        self.assertIsNotNone(self.coordinator.status("job"))


if __name__ == "__main__":
    unittest.main()
