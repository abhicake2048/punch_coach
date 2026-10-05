"""Fair, process-local coordination for memory-heavy inference jobs."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceTicket:
    """Ownership token returned when a queued job reaches the worker."""

    number: int
    job_id: str
    initial_position: int

    @property
    def waited(self) -> bool:
        return self.initial_position > 0


class InferenceCoordinator:
    """Run one inference job at a time in first-in, first-out order.

    CornerCoach's YOLO predictors, tracking state, and temporal recognizer are
    deliberately shared to stay within small cloud-memory limits. This class
    prevents separate Streamlit sessions from mutating those objects together.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._next_ticket = 0
        self._serving_ticket = 0
        self._active_job_id: str | None = None
        self._cancelled: set[int] = set()

    def _advance_cancelled_locked(self) -> None:
        while self._serving_ticket in self._cancelled:
            self._cancelled.remove(self._serving_ticket)
            self._serving_ticket += 1

    def acquire(
        self,
        job_id: str,
        on_wait: Callable[[int], None] | None = None,
    ) -> InferenceTicket:
        """Wait for the shared worker and return a ticket that must be released."""
        with self._condition:
            number = self._next_ticket
            self._next_ticket += 1
            position = number - self._serving_ticket

        if position > 0 and on_wait is not None:
            try:
                on_wait(position)
            except BaseException:
                self._cancel(number)
                raise

        with self._condition:
            try:
                while number != self._serving_ticket:
                    self._condition.wait()
            except BaseException:
                self._cancelled.add(number)
                self._advance_cancelled_locked()
                self._condition.notify_all()
                raise
            self._active_job_id = str(job_id)

        return InferenceTicket(number, str(job_id), position)

    def _cancel(self, number: int) -> None:
        with self._condition:
            self._cancelled.add(number)
            self._advance_cancelled_locked()
            self._condition.notify_all()

    def release(self, ticket: InferenceTicket) -> None:
        """Release the active worker and wake the next queued session."""
        with self._condition:
            if ticket.number != self._serving_ticket:
                raise RuntimeError(
                    f"Cannot release inference ticket {ticket.number}; "
                    f"currently serving {self._serving_ticket}"
                )
            if self._active_job_id != ticket.job_id:
                raise RuntimeError("Inference ticket does not own the active worker")
            self._active_job_id = None
            self._serving_ticket += 1
            self._advance_cancelled_locked()
            self._condition.notify_all()

    @property
    def active_job_id(self) -> str | None:
        with self._condition:
            return self._active_job_id

    @property
    def waiting_jobs(self) -> int:
        with self._condition:
            pending = self._next_ticket - self._serving_ticket
            if self._active_job_id is not None:
                pending -= 1
            return max(0, pending - len(self._cancelled))
