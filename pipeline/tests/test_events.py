import json

from pipeline.utils.events import JsonlEventEmitter, read_events


def test_emit_writes_one_json_line_per_call(tmp_path):
    path = tmp_path / "events.jsonl"
    emitter = JsonlEventEmitter(path, job_id="job-1")
    emitter.emit("RUNNING", "worker_started", model="Qwen/Qwen2.5-0.5B")
    emitter.emit("RUNNING", "step", global_step=1, loss=1.23)
    emitter.close()

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["job_id"] == "job-1"
    assert first["stage"] == "RUNNING"
    assert first["event"] == "worker_started"
    assert first["model"] == "Qwen/Qwen2.5-0.5B"
    assert "ts" in first


def test_none_path_is_a_noop():
    emitter = JsonlEventEmitter(None, job_id="job-1")
    emitter.emit("RUNNING", "worker_started")  # must not raise
    emitter.close()


def test_read_events_round_trips(tmp_path):
    path = tmp_path / "events.jsonl"
    emitter = JsonlEventEmitter(path, job_id="job-2")
    emitter.emit("RUNNING", "a")
    emitter.emit("COMPLETED", "b")
    emitter.close()

    events = read_events(path)
    assert [e["event"] for e in events] == ["a", "b"]


def test_read_events_missing_file_returns_empty_list(tmp_path):
    assert read_events(tmp_path / "does-not-exist.jsonl") == []


def test_context_manager_closes_file(tmp_path):
    path = tmp_path / "events.jsonl"
    with JsonlEventEmitter(path, job_id="job-3") as emitter:
        emitter.emit("RUNNING", "a")
    assert len(read_events(path)) == 1
