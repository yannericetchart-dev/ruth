"""Model subsystem: only the explicitly approved local catalog is active."""
from .catalog import ModelCatalog, ModelRecord, catalog, resolve

__all__ = ["ModelCatalog", "ModelRecord", "catalog", "resolve"]
