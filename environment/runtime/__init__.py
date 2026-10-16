"""Public environment inspection and read-only dependency planning."""
from pathlib import Path
from collections.abc import Iterable

from .contracts import DependencyPlan, PackageState, PythonEnvironment


def inspect(project_root: Path) -> PythonEnvironment:
    """Return the Python isolation report through the parent contract."""
    from environment.python import inspect as inspect_python
    return inspect_python(project_root)


def plan(
    requirements: Iterable[str], *, project_root: Path | None = None
) -> DependencyPlan:
    """Inspect only explicitly named distributions in the isolated project venv."""
    from environment.dependencies import plan as plan_dependencies
    root = Path(__file__).resolve().parents[2] if project_root is None else Path(project_root)
    python_report = inspect(root)
    return plan_dependencies(requirements, python_report)


__all__ = ['DependencyPlan', 'PackageState', 'PythonEnvironment', 'inspect', 'plan']
