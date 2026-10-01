"""Response shapes for the API (Section F of the architecture plan).

JobConfig (app/config.py) is reused directly as the POST /jobs request
body -- no separate request schema. These are read models only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class JobSummary(BaseModel):
    job_id: str
    name: str
    model: str
    dataset: str
    status: str
    storage_backend: str
    created_at: float
    duration_s: Optional[float] = None


class JobDetail(JobSummary):
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    config: Dict[str, Any]
    last_event: Optional[Dict[str, Any]] = None


class CatalogOption(BaseModel):
    id: str
    label: str


class CatalogResponse(BaseModel):
    models: List[CatalogOption]
    datasets: List[CatalogOption]
    checkpoint_frequencies: List[str]
    storage_backends: List[CatalogOption]


class SettingsResponse(BaseModel):
    model_storage_backend: str
    dataset_storage_backend: str
    checkpoint_storage_backend: str
    ddl_server: Optional[str] = None
    ddl_port: int
    ddl_tenant: int
    ddl_cpu_base: int


class SettingsUpdate(BaseModel):
    # All optional -- PUT /settings is a partial update (only fields the
    # caller sets are changed), same convention as JobStore.update_status.
    model_storage_backend: Optional[str] = None
    dataset_storage_backend: Optional[str] = None
    checkpoint_storage_backend: Optional[str] = None
    ddl_server: Optional[str] = None
    ddl_port: Optional[int] = None
    ddl_tenant: Optional[int] = None
    ddl_cpu_base: Optional[int] = None


class CheckpointSummary(BaseModel):
    checkpoint_id: str
    epoch: int
    global_step: int
    size_bytes: int
    num_tensors: int
    num_chunks: int
    num_workers: int
    write_s: float
    total_s: float
    throughput_mb_s: float
