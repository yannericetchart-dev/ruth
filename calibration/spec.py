"""Separate model placement from generation and context resources.

These are candidate descriptions, not validated or published profiles.
The placement payload is intentionally adapter-specific.
"""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelMapping:
    model_id: str
    engine: str
    method: str
    backend: str
    placement: dict[str, Any]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GenerationResources:
    offload_kqv: bool = True
    flash_attn: bool = False
    type_k: str | None = None
    type_v: str | None = None
    swa_full: bool | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ContextResources:
    # Used only to run a short calibration. Not a production context limit.
    probe_n_ctx: int
    probe_n_batch: int
    probe_n_ubatch: int
    selected_n_ctx: int | None = None

    def as_dict(self) -> dict:
        return asdict(self)
