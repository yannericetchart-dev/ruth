"""Read-only inspection of explicitly requested distributions.

Provenance: metadata-only subset of historic ruth_runtime/venv_packages.py
(commit 4ceea508025650ed64b33190834d0b5c85f69db2).
No CUDA requirements, install plans, native checks, or host-specific locks.
"""
from __future__ import annotations

from importlib import metadata
import re
from typing import Callable, Iterable

from environment.runtime.contracts import DependencyPlan, PackageState, PythonEnvironment

_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?\Z")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+!-]*\Z")


def _parse(specification: str) -> tuple[str, str, str | None]:
    """Accept a distribution name or exactly name==version, not PEP 508 ranges."""
    if not isinstance(specification, str):
        raise ValueError("Il requisito deve essere una stringa.")
    spec = specification.strip()
    name, separator, version = spec.partition("==")
    name = name.strip()
    version = version.strip()
    if not _NAME.fullmatch(name) or (separator and not _VERSION.fullmatch(version)):
        raise ValueError(f"Requisito non valido: {specification!r}")
    if not separator and any(c in name for c in "<>!=~;@[] "):
        raise ValueError(f"Requisito non supportato: {specification!r}")
    canonical = re.sub(r"[-_.]+", "-", name).lower()
    return name, canonical, version if separator else None


def plan(
    requirements: Iterable[str],
    python_report: PythonEnvironment,
    *,
    version_lookup: Callable[[str], str] = metadata.version,
) -> DependencyPlan:
    """Inspect installed metadata only when the project virtualenv is isolated.

    version_lookup is injectable for deterministic tests; no package is imported.
    An invalid environment blocks *all* queries before reading package metadata.
    """
    if isinstance(requirements, str):
        raise ValueError("Fornire un elenco di pacchetti, non una stringa singola.")
    parsed = []
    seen = set()
    try:
        iterator = iter(requirements)
    except TypeError as error:
        raise ValueError("L'elenco dei pacchetti non e' valido.") from error
    for spec in iterator:
        name, canonical, version = _parse(spec)
        if canonical in seen:
            raise ValueError(f"Pacchetto ripetuto: {name}")
        seen.add(canonical)
        parsed.append((name, spec.strip(), version))
    if not parsed:
        raise ValueError("Specificare almeno un pacchetto.")
    states = []
    isolated = python_report.interpreter_in_project_venv
    for name, spec, requested_version in parsed:
        if not isolated:
            states.append(PackageState(name, spec, None, "blocked"))
            continue
        try:
            installed = version_lookup(name)
        except metadata.PackageNotFoundError:
            states.append(PackageState(name, spec, None, "missing"))
        except Exception:
            states.append(PackageState(name, spec, None, "inspection_failed"))
        else:
            status = (
                "incompatible" if requested_version is not None
                and installed != requested_version else "present_unverified"
            )
            states.append(PackageState(name, spec, installed, status))
    return DependencyPlan(
        python=python_report,
        packages=tuple(states),
        all_declared_present=isolated and all(
            item.status == "present_unverified" for item in states
        ),
    )
