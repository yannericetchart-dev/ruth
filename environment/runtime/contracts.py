"""Portable environment contract; values from a real host stay local."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PythonEnvironment:
    """Interpreter isolation report, not proof of package or GPU readiness."""

    project_root: str
    interpreter: str
    python_version: tuple[int, ...]
    interpreter_in_project_venv: bool
    venv_status: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        """Return transient data; callers must obey DATA-001 when saving it."""
        return asdict(self)
