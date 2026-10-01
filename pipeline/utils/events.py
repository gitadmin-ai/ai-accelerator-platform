"""Re-export shim: the actual implementation moved to the standalone
`nebula-events` package (shared/nebula-events/) so both `backend` and
`pipeline` depend on one shared, zero-dependency source of truth for the
training-event wire format without `backend` needing to import anything
from `pipeline`. Kept here so existing `from pipeline.utils.events import
...` call sites (and the CLI invocation's expectations) don't need to
change.
"""
from __future__ import annotations

from nebula_events import JsonlEventEmitter, read_events  # noqa: F401

__all__ = ["JsonlEventEmitter", "read_events"]
