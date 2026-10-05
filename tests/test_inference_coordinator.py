from __future__ import annotations

import threading
import time
import unittest

from core.inference_coordinator import InferenceCoordinator


class InferenceCoordinatorTests(unittest.TestCase):
    def test_jobs_run_one_at_a_time_in_fifo_order(self) -> None:
        coordinator = InferenceCoordinator()
        barrier = threading.Barrier(4)
        state_lock = threading.Lock()
        started: list[int] = []
        active = 0
        maximum_active = 0

        def worker(index: int) -> None:
            nonlocal active, maximum_active
            barrier.wait()
            # Stagger ticket requests while still starting the threads together.
            time.sleep(index * 0.01)
            ticket = coordinator.acquire(f"job-{index}")
            try:
                with state_lock:
                    started.append(index)
                    active += 1
                    maximum_active = max(maximum_active, active)
                time.sleep(0.03)
                with state_lock:
                    active -= 1
            finally:
                coordinator.release(ticket)

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(3)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=2.0)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(started, [0, 1, 2])
        self.assertEqual(maximum_active, 1)
        self.assertEqual(coordinator.waiting_jobs, 0)
        self.assertIsNone(coordinator.active_job_id)

    def test_release_after_failure_allows_next_job(self) -> None:
        coordinator = InferenceCoordinator()
        first = coordinator.acquire("first")
        result: list[str] = []

        def second_worker() -> None:
            second = coordinator.acquire("second")
            try:
                result.append("second-started")
            finally:
                coordinator.release(second)

        thread = threading.Thread(target=second_worker)
        thread.start()
        time.sleep(0.03)
        self.assertEqual(coordinator.waiting_jobs, 1)

        # This mirrors the app's finally block after a failed analysis.
        coordinator.release(first)
        thread.join(timeout=2.0)

        self.assertEqual(result, ["second-started"])
        self.assertFalse(thread.is_alive())
        self.assertIsNone(coordinator.active_job_id)

    def test_cancelled_waiter_does_not_block_the_queue(self) -> None:
        coordinator = InferenceCoordinator()
        first = coordinator.acquire("first")

        with self.assertRaisesRegex(RuntimeError, "session disconnected"):
            coordinator.acquire(
                "cancelled",
                on_wait=lambda _position: (_ for _ in ()).throw(
                    RuntimeError("session disconnected")
                ),
            )

        coordinator.release(first)
        third = coordinator.acquire("third")
        coordinator.release(third)

        self.assertEqual(coordinator.waiting_jobs, 0)
        self.assertIsNone(coordinator.active_job_id)


if __name__ == "__main__":
    unittest.main()
