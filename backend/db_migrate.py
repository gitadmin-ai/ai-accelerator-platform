"""Tiny lightweight migration helper for the sqlite-backed resource stores
(ModelStore, DatasetStore).

`CREATE TABLE IF NOT EXISTS` (what each store's _SCHEMA already does) only
creates a table if it's entirely missing -- it's a no-op against an
existing on-disk database whose table predates a newly added column (e.g.
storage_backend, added after data/nebula.db already existed with the old
models/datasets schema). ensure_column() covers exactly that one case:
"add this column with this default if the table doesn't already have it."
Nothing fancier (renames, drops, data backfills) is needed here -- if that
ever changes, reach for a real migration tool instead of growing this.
"""
from __future__ import annotations

import sqlite3


def ensure_column(conn: sqlite3.Connection, table: str, column: str, column_ddl: str) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_ddl}")
