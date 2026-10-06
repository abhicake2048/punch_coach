"""Lease-based coordination for memory-heavy inference jobs.

The coordinator is deliberately non-blocking. Streamlit sessions poll their
queue status and rerun, instead of occupying a Python thread in a condition
variable. Queue entries are tied to browser session IDs so disconnected
visitors cannot leave a permanent ticket at the front.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True)
class InferenceLease:
    """Ownership token returned when a queued job becomes active."""

    job_id: str
    session_id: str
    initial_position: int
    queued_at: float
    started_at: float

    @property
    def waited(self) -> bool:
        return self.initial_position > 0


@dataclass(frozen=True)
class QueueStatus:
    """Immutable snapshot used to render a session's queue progress."""

    job_id: str
    state: str
    jobs_ahead: int
    total_jobs: int
    initial_position: int
    waited_seconds: float
    cancel_requested: bool = False


@dataclass
class _Job:
    job_id: str
    session_id: str
    queued_at: float
    last_seen_at: float
    initial_position: int
    started_at: float | None = None
    cancel_requested: bool = False


class InferenceCoordinator:
    """Run one inference at a time while evicting abandoned queue entries.

    The mutable YOLO tracker and temporal recognizer are process-wide resources,
    so allowing simultaneous calls corrupts results and can exhaust memory. A
    waiting session never blocks in this class: it calls :meth:`touch`, reads a
    :class:`QueueStatus`, and tries again on its next Streamlit rerun.
    """

    def __init__(
        self,
        waiting_timeout_seconds: float = 20.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if waiting_timeout_seconds <= 0:
            raise ValueError("waiting_timeout_seconds must be positive")
        self._waiting_timeout_seconds = float(waiting_timeout_seconds)
        self._clock = clock
        self._lock = threading.RLock()
        self._waiting: deque[str] = deque()
        self._jobs: dict[str, _Job] = {}
        self._active_job_id: str | None = None

    def register(self, job_id: str, session_id: str) -> QueueStatus:
        """Register a job once, or refresh and return its existing state."""
        normalized_job_id = str(job_id)
        normalized_session_id = str(session_id)
        now = self._clock()
        with self._lock:
            existing = self._jobs.get(normalized_job_id)
            if existing is not None:
                if existing.session_id != normalized_session_id:
                    raise RuntimeError("Inference job belongs to another session")
                existing.last_seen_at = now
                return self._status_locked(existing, now)

            initial_position = len(self._waiting) + (
                1 if self._active_job_id is not None else 0
            )
            job = _Job(
                job_id=normalized_job_id,
                session_id=normalized_session_id,
                queued_at=now,
                last_seen_at=now,
                initial_position=initial_position,
            )
            self._jobs[normalized_job_id] = job
            self._waiting.append(normalized_job_id)
            return self._status_locked(job, now)

    def touch(self, job_id: str, session_id: str) -> QueueStatus | None:
        """Renew a waiting job's browser lease and return its latest status."""
        now = self._clock()
        with self._lock:
            job = self._jobs.get(str(job_id))
            if job is None:
                return None
            if job.session_id != str(session_id):
                raise RuntimeError("Inference job belongs to another session")
            job.last_seen_at = now
            return self._status_locked(job, now)

    def prune(
        self,
        session_is_active: Callable[[str], bool | None] | None = None,
    ) -> list[str]:
        """Remove disconnected/stale waiters and flag a disconnected active job.

        Active work is only *flagged* here. It continues owning the worker until
        its inference loop observes the flag and calls :meth:`finish`, preventing
        the next job from touching shared models during cleanup.
        """
        now = self._clock()
        removed: list[str] = []
        with self._lock:
            for job_id in list(self._waiting):
                job = self._jobs.get(job_id)
                if job is None:
                    self._waiting.remove(job_id)
                    continue
                activity = (
                    session_is_active(job.session_id)
                    if session_is_active is not None
                    else None
                )
                disconnected = activity is False
                stale = (
                    activity is None
                    and now - job.last_seen_at > self._waiting_timeout_seconds
                )
                if stale or disconnected:
                    self._waiting.remove(job_id)
                    self._jobs.pop(job_id, None)
                    removed.append(job_id)

            if self._active_job_id is not None and session_is_active is not None:
                active = self._jobs.get(self._active_job_id)
                if (
                    active is not None
                    and session_is_active(active.session_id) is False
                ):
                    active.cancel_requested = True

        return removed

    def try_start(self, job_id: str, session_id: str) -> InferenceLease | None:
        """Atomically start the FIFO head when the worker is available."""
        now = self._clock()
        with self._lock:
            job = self._jobs.get(str(job_id))
            if job is None:
                return None
            if job.session_id != str(session_id):
                raise RuntimeError("Inference job belongs to another session")
            if job.cancel_requested:
                return None
            if self._active_job_id is not None:
                return None
            if not self._waiting or self._waiting[0] != job.job_id:
                return None

            self._waiting.popleft()
            self._active_job_id = job.job_id
            job.started_at = now
            return InferenceLease(
                job_id=job.job_id,
                session_id=job.session_id,
                initial_position=job.initial_position,
                queued_at=job.queued_at,
                started_at=now,
            )

    def request_cancel(self, job_id: str, session_id: str | None = None) -> bool:
        """Cancel a waiter immediately or ask active work to stop safely."""
        normalized_job_id = str(job_id)
        with self._lock:
            job = self._jobs.get(normalized_job_id)
            if job is None:
                return False
            if session_id is not None and job.session_id != str(session_id):
                return False
            if self._active_job_id == normalized_job_id:
                job.cancel_requested = True
                return True
            try:
                self._waiting.remove(normalized_job_id)
            except ValueError:
                pass
            self._jobs.pop(normalized_job_id, None)
            return True

    def cancellation_requested(self, job_id: str) -> bool:
        """Return whether active work should stop at its next safe boundary."""
        with self._lock:
            job = self._jobs.get(str(job_id))
            return job is None or job.cancel_requested

    def finish(self, lease: InferenceLease) -> bool:
        """Release an active lease; idempotency keeps cleanup failure-safe."""
        with self._lock:
            if self._active_job_id != lease.job_id:
                return False
            job = self._jobs.get(lease.job_id)
            if job is None or job.session_id != lease.session_id:
                return False
            self._active_job_id = None
            self._jobs.pop(lease.job_id, None)
            return True

    def status(self, job_id: str) -> QueueStatus | None:
        """Return a point-in-time status for one known job."""
        with self._lock:
            job = self._jobs.get(str(job_id))
            if job is None:
                return None
            return self._status_locked(job, self._clock())

    def _status_locked(self, job: _Job, now: float) -> QueueStatus:
        if self._active_job_id == job.job_id:
            state = "active"
            jobs_ahead = 0
        else:
            state = "waiting"
            try:
                waiting_index = self._waiting.index(job.job_id)
            except ValueError:
                waiting_index = 0
            jobs_ahead = waiting_index + (1 if self._active_job_id else 0)
        total_jobs = len(self._waiting) + (1 if self._active_job_id else 0)
        return QueueStatus(
            job_id=job.job_id,
            state=state,
            jobs_ahead=jobs_ahead,
            total_jobs=total_jobs,
            initial_position=job.initial_position,
            waited_seconds=max(0.0, now - job.queued_at),
            cancel_requested=job.cancel_requested,
        )

    @property
    def active_job_id(self) -> str | None:
        with self._lock:
            return self._active_job_id

    @property
    def waiting_jobs(self) -> int:
        with self._lock:
            return len(self._waiting)
