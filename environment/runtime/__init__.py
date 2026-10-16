"""Common environment API. Package planning is not implemented yet."""
from pathlib import Path

from .contracts import PythonEnvironment


def inspect(project_root: Path) -> PythonEnvironment:
    """Return the Python isolation report through the parent contract."""
    from environment.python import inspect as inspect_python
    return inspect_python(project_root)


__all__ = ['PythonEnvironment', 'inspect']
