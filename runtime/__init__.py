"""Common Ruth API; only environment inspection is available in this slice."""
from pathlib import Path

from environment import PythonEnvironment


def inspect_environment(project_root: Path) -> PythonEnvironment:
    """Inspect the project Python environment without modifying the host."""
    from environment import inspect
    return inspect(project_root)


__all__ = ['PythonEnvironment', 'inspect_environment']
