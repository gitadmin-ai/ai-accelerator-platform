"""Job lifecycle state machine (Section 4 / Section G of the architecture plan).

    SUBMITTED -> QUEUED -> RUNNING -> COMPLETED

with RUNNING <-> CHECKPOINTING and RUNNING -> EVALUATING -> RUNNING as
repeating sub-loops (once per checkpoint/eval interval, not a single linear
pass) and FAILED reachable from any non-terminal state. COMPLETED is only
reached after the final checkpoint (and final evaluation, if enabled).

This module is deliberately pure (no I/O, no persistence) so it's trivial
to unit test on its own; app/job_manager.py is the only caller.
"""
from __future__ import annotations

import enum
from typing import Dict, Set


class JobStatus(str, enum.Enum):
    SUBMITTED = "SUBMITTED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    CHECKPOINTING = "CHECKPOINTING"
    EVALUATING = "EVALUATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


TERMINAL_STATES = frozenset({JobStatus.COMPLETED, JobStatus.FAILED})

_ALLOWED_TRANSITIONS: Dict[JobStatus, Set[JobStatus]] = {
    JobStatus.SUBMITTED: {JobStatus.QUEUED, JobStatus.FAILED},
    JobStatus.QUEUED: {JobStatus.RUNNING, JobStatus.FAILED},
    JobStatus.RUNNING: {
        JobStatus.CHECKPOINTING,
        JobStatus.EVALUATING,
        JobStatus.COMPLETED,
        JobStatus.FAILED,
    },
    JobStatus.CHECKPOINTING: {
        JobStatus.RUNNING,
        JobStatus.EVALUATING,
        JobStatus.COMPLETED,
        JobStatus.FAILED,
    },
    JobStatus.EVALUATING: {JobStatus.RUNNING, JobStatus.COMPLETED, JobStatus.FAILED},
    JobStatus.COMPLETED: set(),
    JobStatus.FAILED: set(),
}


class IllegalTransitionError(ValueError):
    """Raised when a transition isn't in _ALLOWED_TRANSITIONS."""


def transition(current: JobStatus, new: JobStatus) -> JobStatus:
    """Validates current -> new and returns `new`. Raises
    IllegalTransitionError otherwise -- never silently allows an
    out-of-order state (e.g. jumping straight from SUBMITTED to COMPLETED).
    """
    if new not in _ALLOWED_TRANSITIONS.get(current, set()):
        raise IllegalTransitionError(f"cannot transition {current.value} -> {new.value}")
    return new


def advance(current: JobStatus, stage: str) -> JobStatus:
    """Applies a training-worker event's `stage` field to the current
    status. A no-op (returns `current` unchanged) if the event's stage
    matches the current status already (e.g. a second "step" event while
    already RUNNING) -- only an actual stage change is a transition.
    """
    new = JobStatus(stage)
    if new == current:
        return current
    return transition(current, new)
