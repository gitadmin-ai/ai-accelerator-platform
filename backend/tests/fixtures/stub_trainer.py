"""Fast fake training entrypoint for JobManager/API integration tests.

Mimics just enough of train_lora_with_gpu_stats.py's event contract
(--job-id, --events-file, a sequence of stage-tagged JSONL events ending in
job_complete or job_failed) to exercise JobManager's subprocess-launch,
event-tailing, and state-machine-transition logic in milliseconds -- no
real model/dataset/torch involved. Real end-to-end behavior against the
actual training worker is covered separately (see
app/tests/test_e2e_real_training.py).

Set --dataset STUB_FAIL to make the stub raise partway through, exercising
the FAILED path (with a job_failed event the worker itself emits).

Set --dataset STUB_CRASH_SILENT to make the stub exit nonzero *without*
emitting anything first, simulating a hard crash (OOM kill, segfault) --
exercises JobManager's own synthetic-FAILED fallback in _tail_events().
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from nebula_events import JsonlEventEmitter


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--job-id", default=None)
    p.add_argument("--events-file", default=None)
    p.add_argument("--dataset", default="")
    p.add_argument("--output-dir", default=None)
    args, _unknown = p.parse_known_args()  # ignore every other flag JobManager passes

    if args.dataset == "STUB_CRASH_SILENT":
        sys.exit(7)

    emitter = JsonlEventEmitter(args.events_file, job_id=args.job_id)
    try:
        emitter.emit("RUNNING", "worker_started")
        time.sleep(0.05)
        if args.dataset == "STUB_FAIL":
            raise RuntimeError("stub trainer: simulated failure")

        emitter.emit("RUNNING", "dataset_loaded", num_train_examples=4, num_eval_examples=1)
        emitter.emit("RUNNING", "epoch_start", epoch=1, total_epochs=1)
        emitter.emit("RUNNING", "step", epoch=1, global_step=1, loss=1.0, learning_rate=1e-4)
        emitter.emit("RUNNING", "epoch_complete", epoch=1, mean_loss=1.0, elapsed_s=0.01)

        emitter.emit("CHECKPOINTING", "checkpoint_start", checkpoint_id="epoch-0-step-1")
        emitter.emit(
            "CHECKPOINTING", "checkpoint_saved", checkpoint_id="epoch-0-step-1",
            epoch=0, global_step=1, size_bytes=1024, num_tensors=2, num_chunks=2,
            num_workers=1, write_s=0.01, chunking_s=0.001, commit_s=0.001,
            total_s=0.02, throughput_mb_s=1.0,
        )

        emitter.emit("EVALUATING", "eval_start", epoch=1)
        emitter.emit("EVALUATING", "eval_complete", epoch=1, eval_loss=0.5, num_eval_batches=1)

        artifact_dir = Path(args.output_dir) / "artifacts" / "final_adapter"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "adapter_model.bin").write_bytes(b"stub-adapter-weights")
        (artifact_dir / "adapter_config.json").write_text('{"stub": true}')

        emitter.emit(
            "COMPLETED", "job_complete", total_epochs=1, total_steps=1,
            final_checkpoint_id="epoch-0-step-1", artifact_path=str(artifact_dir),
        )
    except Exception as exc:
        emitter.emit("FAILED", "job_failed", error=str(exc), error_type=type(exc).__name__)
        emitter.close()
        raise
    emitter.close()


if __name__ == "__main__":
    main()
