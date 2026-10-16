"""Read-only model catalog implementation."""
from .contracts import ModelCatalog, ModelRecord
from .inspection import catalog, resolve

__all__ = ["ModelCatalog", "ModelRecord", "catalog", "resolve"]
