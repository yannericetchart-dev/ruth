"""Contratti pubblici preliminari di Ruth, indipendenti dall'hardware.

Fase 00: nessun backend è ancora implementato o verificato.
"""

from .capabilities import CATALOG, INITIAL_STACKS, components, inspect_stack
from .contracts import (
    Capability, ComponentKind, DevelopmentState, DeviceInfo,
    HardwareProvider, ExecutionBackend, InferenceEngine,
    InitializationAdapter, InferenceSession,
    StackSelection, StackAssessment, UnsupportedStackError,
)

__all__ = [
    "CATALOG", "INITIAL_STACKS", "components", "inspect_stack",
    "Capability", "ComponentKind", "DevelopmentState", "DeviceInfo",
    "HardwareProvider", "ExecutionBackend", "InferenceEngine",
    "InitializationAdapter", "InferenceSession", "StackSelection",
    "StackAssessment", "UnsupportedStackError",
]
