"""Read-only GGUF header contracts and catalog-bound reader."""
from .contracts import GGUFHeader, GGUFReport
from .reader import inspect_header, inspect_headers

__all__ = ["GGUFHeader", "GGUFReport", "inspect_header", "inspect_headers"]
