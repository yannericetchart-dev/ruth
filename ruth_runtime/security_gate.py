"""Phase 02: fail-closed diskcache risk gate for pinned CUDA Ruth runtime.

This is an application-specific mitigation audit, NOT a patch to diskcache,
proof of secure third-party code, permission to install, or LLM inference.
Read-only; no external network and no pip activity.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import subprocess
import sys
from zipfile import BadZipFile
from typing import Any, Callable

from .dependencies import PROJECT_ROOT
from .dependency_resolution import CANDIDATE_NAME, WHEELHOUSE, inspect_wheel
from .install_plan import (
    DISKCACHE_ADVISORY, LLAMA_CPP_SUBMODULE_COMMIT, SOURCE_COMMIT,
)
from .source_build_plan import SourceBuildPlan, create_source_build_plan
from .venv_packages import EnvironmentReport, inspect_python_environment

EXPECTED_DISKCACHE = "5.6.3"
EXPECTED_ADAPTER_TEST_COUNT = 12
TEST_PATTERN = "test_phase02_cuda_adapter.py"


@dataclass(frozen=True)
class SecurityGate:
    status: str
    verified_pinned_checkout: bool
    python_venv_verified: bool
    upstream_default_cache_disabled: bool
    disk_cache_attack_surface_identified: bool
    pinned_diskcache_wheel_verified: bool
    adapter_contract_tests_passed: bool
    adapter_tests_run: int
    mitigations_validated: bool
    diskcache_advisory: str
    residual_risk: str
    security_review_passed: bool
    lock_approved: bool
    installation_authorized: bool
    installation_started: bool
    native_build_started: bool
    fallback_automatically_selected: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _cache_assignments(method: ast.FunctionDef) -> list[ast.expr]:
    values: list[ast.expr] = []
    for node in ast.walk(method):
        if isinstance(node, ast.Assign):
            if any(
                isinstance(t, ast.Attribute)
                and isinstance(t.value, ast.Name)
                and t.value.id == "self" and t.attr == "cache"
                for t in node.targets
            ):
                values.append(node.value)
        elif isinstance(node, ast.AnnAssign):
            t = node.target
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) \
                    and t.value.id == "self" and t.attr == "cache":
                values.append(node.value)
    return values


def inspect_upstream_cache_policy(
    main_source: Path,
    cache_source: Path,
) -> tuple[bool, bool, str | None]:
    """AST checks for the exact expected upstream 0.3.36 cache interface."""
    try:
        main = ast.parse(main_source.read_text(encoding="utf-8"))
        cache = ast.parse(cache_source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError) as exc:
        return False, False, type(exc).__name__
    cls = next((x for x in main.body
                if isinstance(x, ast.ClassDef) and x.name == "Llama"), None)
    disk = next((x for x in cache.body
                 if isinstance(x, ast.ClassDef)
                 and x.name == "LlamaDiskCache"), None)
    if cls is None or disk is None:
        return False, False, "Expected pinned llama_cpp classes missing"
    ctor = next((x for x in cls.body
                 if isinstance(x, ast.FunctionDef)
                 and x.name == "__init__"), None)
    set_cache = next((x for x in cls.body
                      if isinstance(x, ast.FunctionDef)
                      and x.name == "set_cache"), None)
    disk_ctor = next((x for x in disk.body
                      if isinstance(x, ast.FunctionDef)
                      and x.name == "__init__"), None)
    if ctor is None or set_cache is None or disk_ctor is None:
        return False, False, "Expected cache constructor/setter interface missing"
    assignments = _cache_assignments(ctor)
    default_none = (
        len(assignments) == 1 and isinstance(assignments[0], ast.Constant)
        and assignments[0].value is None
    )
    disk_constructor = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "diskcache"
        and node.func.attr == "Cache"
        for node in ast.walk(disk_ctor)
    )
    return default_none, disk_constructor, None


def _candidate_diskcache_ok(root: Path) -> bool:
    candidate = root / WHEELHOUSE / CANDIDATE_NAME
    manifest = candidate / "candidate-lock.json"
    wheels = candidate / "wheels"
    if candidate.parent.is_symlink() or candidate.is_symlink() or manifest.is_symlink() or wheels.is_symlink():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("schema") != "ruth-python-candidate-v1" \
                or data.get("approved") is not False \
                or data.get("source_commit") != SOURCE_COMMIT \
                or data.get("llama_cpp_commit") != LLAMA_CPP_SUBMODULE_COMMIT:
            return False
        matches = [
            r for r in data["files"]
            if isinstance(r, dict) and r.get("name", "").lower() == "diskcache"
        ]
        if len(matches) != 1 or matches[0].get("version") != EXPECTED_DISKCACHE:
            return False
        row = matches[0]
        file = wheels / row["filename"]
        if file.is_symlink() or file.parent != wheels:
            return False
        return asdict(inspect_wheel(file)) == row
    except (OSError, ValueError, KeyError, TypeError, AttributeError, BadZipFile):
        return False


def review_diskcache_risk(
    *,
    project_root: Path = PROJECT_ROOT,
    environment: EnvironmentReport | None = None,
    source_plan: SourceBuildPlan | None = None,
    run: Callable[..., Any] = subprocess.run,
) -> SecurityGate:
    """Run local adapter tests and pinned source AST checks, no wheel imports."""
    root = Path(project_root)
    env = environment if environment is not None else inspect_python_environment()
    plan = source_plan if source_plan is not None else create_source_build_plan()
    venv_ok = bool(env.interpreter_in_project_venv
                   and Path(env.project_root).resolve() == root.resolve()
                   and Path(env.interpreter).resolve()
                   == (root / ".venv/bin/python").resolve())
    pinned = bool(
        plan.source_verified and plan.source_status == "verified"
        and plan.source_commit_actual == SOURCE_COMMIT
        and plan.llama_cpp_commit_actual == LLAMA_CPP_SUBMODULE_COMMIT
        and Path(plan.source_directory).resolve()
        == (root / ".ruth-src/llama-cpp-python").resolve()
    )
    errors: list[str] = []
    defaults_ok = disk_surface = wheel_ok = tests_ok = False
    test_count = 0
    if not venv_ok:
        errors.append("Ruth project-local Python venv is not verified")
    if not pinned:
        errors.append("Pinned source checkout is not verified")
    if pinned and venv_ok:
        source = Path(plan.source_directory) / "llama_cpp"
        defaults_ok, disk_surface, source_error = inspect_upstream_cache_policy(
            source / "llama.py", source / "llama_cache.py"
        )
        if source_error or not defaults_ok or not disk_surface:
            errors.append("Upstream cache interface deviates from pinned contract"
                          + (": " + source_error if source_error else ""))
        wheel_ok = _candidate_diskcache_ok(root)
        if not wheel_ok:
            errors.append("diskcache 5.6.3 candidate file/pin not verified")
        test_file = root / "tests" / TEST_PATTERN
        if test_file.is_file() and not test_file.is_symlink():
            cmd = [
                str(root / ".venv/bin/python"), "-m", "unittest",
                "discover", "-s", str(root / "tests"),
                "-p", TEST_PATTERN,
            ]
            try:
                result = run(cmd, cwd=str(root), capture_output=True,
                             text=True, check=False, timeout=60)
                match = re.search(r"\bRan (\d+) tests?\b", result.stderr)
                test_count = int(match.group(1)) if match else 0
                tests_ok = bool(result.returncode == 0
                                and test_count >= EXPECTED_ADAPTER_TEST_COUNT
                                and re.search(r"^OK$", result.stderr, re.M))
            except (OSError, subprocess.SubprocessError):
                tests_ok = False
        if not tests_ok:
            errors.append("CUDA adapter security tests failed, missing, or incomplete")
    controlled = bool(venv_ok and pinned and defaults_ok and disk_surface
                      and wheel_ok and tests_ok)
    # A mitigated attack surface does not make the vulnerable dependency safe:
    # this gate does not accept risk, approve a lock or start installation.
    return SecurityGate(
        status="controls_verified_residual_risk" if controlled else "blocked",
        verified_pinned_checkout=pinned,
        python_venv_verified=venv_ok,
        upstream_default_cache_disabled=defaults_ok,
        disk_cache_attack_surface_identified=disk_surface,
        pinned_diskcache_wheel_verified=wheel_ok,
        adapter_contract_tests_passed=tests_ok, adapter_tests_run=test_count,
        mitigations_validated=controlled,
        diskcache_advisory=DISKCACHE_ADVISORY,
        residual_risk=(
            "diskcache 5.6.3 still includes unsafe pickle deserialization. "
            "Ruth's adapter does not activate disk cache, but code in the "
            "same Python process could import diskcache directly. This "
            "is not an isolation boundary against malicious Python code."
        ),
        security_review_passed=False, lock_approved=False,
        installation_authorized=False, installation_started=False,
        native_build_started=False, fallback_automatically_selected=False,
        errors=tuple(errors),
    )
