"""Safe header-only inspection for explicitly declared GGUF models."""
from .runtime import GGUFHeader, GGUFReport, inspect_header, inspect_headers

__all__ = ["GGUFHeader", "GGUFReport", "inspect_header", "inspect_headers"]
