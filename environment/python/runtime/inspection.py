"""Read-only virtualenv isolation check extracted from legacy venv_packages.

Source: commit 4ceea508025650ed64b33190834d0b5c85f69db2,
ruth_runtime/venv_packages.py. Package inspection and native locks are excluded.
"""
from __future__ import annotations

from pathlib import Path
import sys

from environment.runtime.contracts import PythonEnvironment


def inspect(
    project_root: Path,
    *,
    prefix: Path | None = None,
    base_prefix: Path | None = None,
    interpreter: str | None = None,
    python_version: tuple[int, ...] | None = None,
) -> PythonEnvironment:
    """Inspect without installing packages, probing hardware or writing files.

    Optional observations allow deterministic tests. Normal callers omit them.
    A missing, unreadable or ambiguous configuration fails the isolation check.
    """
    root = Path(project_root)
    venv = root / '.venv'
    actual = Path(sys.prefix) if prefix is None else Path(prefix)
    base = Path(sys.base_prefix) if base_prefix is None else Path(base_prefix)
    version = tuple(sys.version_info[:3]) if python_version is None else tuple(python_version)
    reasons = []
    configuration = venv / 'pyvenv.cfg'
    try:
        if venv.is_symlink():
            reasons.append('venv_symlink')
        if not configuration.is_file():
            reasons.append('configuration_missing')
        else:
            contents = configuration.read_text(encoding='utf-8')
            settings = []
            for line in contents.splitlines():
                key, separator, value = line.partition('=')
                if key.strip().lower() == 'include-system-site-packages':
                    # Keep malformed declarations too: ignoring one would hide
                    # a duplicate after an otherwise valid isolation setting.
                    settings.append(value.strip().lower() if separator else '')
            if settings == ['true']:
                reasons.append('shared_site_packages')
            elif settings != ['false']:
                reasons.append('configuration_invalid')
        if actual.resolve() != venv.resolve():
            reasons.append('prefix_mismatch')
        if actual.resolve() == base.resolve():
            reasons.append('not_virtualenv')
    except (OSError, UnicodeError, RuntimeError):
        reasons.append('inspection_failed')
    if len(version) < 2 or version[:2] < (3, 10):
        reasons.append('python_too_old')
    correct = not reasons
    return PythonEnvironment(
        project_root=str(root),
        interpreter=sys.executable if interpreter is None else interpreter,
        python_version=version,
        interpreter_in_project_venv=correct,
        venv_status='ok' if correct else 'invalid_or_inactive',
        reasons=tuple(reasons),
    )
