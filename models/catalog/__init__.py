"""Local-only manifest catalog, no model acquisition or GGUF inspection."""
from .runtime import ModelCatalog, ModelRecord, catalog, resolve

__all__ = ["ModelCatalog", "ModelRecord", "catalog", "resolve"]
