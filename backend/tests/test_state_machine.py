import pytest

from backend.state_machine import IllegalTransitionError, JobStatus, TERMINAL_STATES, advance, transition


def test_happy_path_no_checkpoint_eval():
    s = JobStatus.SUBMITTED
    s = transition(s, JobStatus.QUEUED)
    s = transition(s, JobStatus.RUNNING)
    s = transition(s, JobStatus.COMPLETED)
    assert s == JobStatus.COMPLETED


def test_happy_path_with_checkpoint_and_eval_loop():
    s = JobStatus.SUBMITTED
    s = transition(s, JobStatus.QUEUED)
    s = transition(s, JobStatus.RUNNING)
    s = transition(s, JobStatus.CHECKPOINTING)
    s = transition(s, JobStatus.EVALUATING)
    s = transition(s, JobStatus.RUNNING)
    s = transition(s, JobStatus.CHECKPOINTING)
    s = transition(s, JobStatus.EVALUATING)
    s = transition(s, JobStatus.COMPLETED)
    assert s == JobStatus.COMPLETED


@pytest.mark.parametrize(
    "start", [JobStatus.SUBMITTED, JobStatus.QUEUED, JobStatus.RUNNING,
              JobStatus.CHECKPOINTING, JobStatus.EVALUATING]
)
def test_any_non_terminal_state_can_fail(start):
    assert transition(start, JobStatus.FAILED) == JobStatus.FAILED


def test_illegal_transition_rejected():
    with pytest.raises(IllegalTransitionError):
        transition(JobStatus.SUBMITTED, JobStatus.COMPLETED)


def test_terminal_states_have_no_outgoing_transitions():
    for terminal in TERMINAL_STATES:
        with pytest.raises(IllegalTransitionError):
            transition(terminal, JobStatus.RUNNING)


def test_advance_is_noop_for_same_stage():
    assert advance(JobStatus.RUNNING, "RUNNING") == JobStatus.RUNNING


def test_advance_applies_real_transition():
    assert advance(JobStatus.RUNNING, "CHECKPOINTING") == JobStatus.CHECKPOINTING


def test_advance_rejects_illegal_stage():
    with pytest.raises(IllegalTransitionError):
        advance(JobStatus.SUBMITTED, "COMPLETED")
