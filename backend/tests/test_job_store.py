import time

from backend.config import JobConfig
from backend.job_store import JobStore
from backend.state_machine import JobStatus


def test_create_and_get():
    store = JobStore(":memory:")
    config = JobConfig(name="job-a")
    record = store.create("job-1", config)
    assert record["job_id"] == "job-1"
    assert record["name"] == "job-a"
    assert record["status"] == JobStatus.SUBMITTED.value
    assert record["config"]["name"] == "job-a"
    assert record["duration_s"] is None

    fetched = store.get("job-1")
    assert fetched == record


def test_get_missing_returns_none():
    store = JobStore(":memory:")
    assert store.get("nope") is None


def test_list_orders_newest_first():
    store = JobStore(":memory:")
    store.create("job-1", JobConfig())
    time.sleep(0.01)
    store.create("job-2", JobConfig())
    jobs = store.list()
    assert [j["job_id"] for j in jobs] == ["job-2", "job-1"]


def test_update_status_and_duration():
    store = JobStore(":memory:")
    store.create("job-1", JobConfig())
    t0 = time.time()
    store.update_status("job-1", JobStatus.RUNNING, started_at=t0)
    record = store.get("job-1")
    assert record["status"] == JobStatus.RUNNING.value
    assert record["duration_s"] is not None and record["duration_s"] >= 0

    store.update_status("job-1", JobStatus.COMPLETED, completed_at=t0 + 5)
    record = store.get("job-1")
    assert record["status"] == JobStatus.COMPLETED.value
    assert abs(record["duration_s"] - 5) < 0.01


def test_set_last_event():
    store = JobStore(":memory:")
    store.create("job-1", JobConfig())
    store.set_last_event("job-1", {"stage": "RUNNING", "event": "step", "loss": 1.23})
    record = store.get("job-1")
    assert record["last_event"]["loss"] == 1.23


def test_update_status_records_error():
    store = JobStore(":memory:")
    store.create("job-1", JobConfig())
    store.update_status("job-1", JobStatus.FAILED, completed_at=time.time(), error="boom")
    record = store.get("job-1")
    assert record["error"] == "boom"
