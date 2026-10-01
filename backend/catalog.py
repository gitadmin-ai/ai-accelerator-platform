"""Static catalog of selectable options for the Create Job form.

Kept in one server-side place instead of hardcoded strings in the React
app, so "the model must come from configuration rather than being
hardcoded into the UI" is actually true. Phase 2 intentionally keeps this
a small static list -- a real model/dataset registry (scanning a
directory, a database) is future scope, not required for a convincing
demo. Adding a future storage backend here (once one exists) is a config
change, not a UI code change.
"""
from __future__ import annotations

from typing import Dict, List

MODELS: List[Dict[str, str]] = [
    {"id": "Qwen/Qwen2.5-0.5B", "label": "Qwen2.5 0.5B (fast demo)"},
    {"id": "Qwen/Qwen2.5-Coder-3B-Instruct", "label": "Qwen2.5-Coder 3B Instruct (production target)"},
]

DATASETS: List[Dict[str, str]] = [
    {"id": "data/demo_onboarding_v1.jsonl", "label": "demo_onboarding_v1"},
]

CHECKPOINT_FREQUENCIES: List[str] = ["epoch"]

# See pipeline/checkpoint/storage_backend.py and backend/settings_store.py --
# "ddl" is only usable once Settings has a ddl_server configured and the
# ddl_client native extension is built into the backend/pipeline images.
STORAGE_BACKENDS: List[Dict[str, str]] = [
    {"id": "local", "label": "Local Storage"},
    {"id": "ddl", "label": "Nebula (DDL) — parallel object storage"},
]


def get_catalog() -> Dict[str, object]:
    return {
        "models": MODELS,
        "datasets": DATASETS,
        "checkpoint_frequencies": CHECKPOINT_FREQUENCIES,
        "storage_backends": STORAGE_BACKENDS,
    }
