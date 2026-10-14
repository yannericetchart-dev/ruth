"""Ruth CLI lifecycle: doctor, plan, setup, run.

No hidden software installations. A complete hash-locked build/install stage
will be integrated here after the security gate; until then stop explicitly.
"""
from __future__ import annotations

import argparse
import json
from typing import Callable

from .venv_packages import EnvironmentReport, inspect_python_environment


def build_launch_plan(
    *,
    environment: EnvironmentReport | None = None,
    source_plan_factory: Callable | None = None,
) -> dict:
    env = environment if environment is not None else inspect_python_environment()
    result = {
        "application": "Ruth",
        "mode": "cuda-toolkit-native-build",
        "venv": env.to_dict(),
        "source": None,
        "setup_ready": False,
        "install_ready": False,
        "runtime_ready": False,
        "source_download_auto": True,
        "wheel_fallback_auto": False,
        "cpu_fallback": False,
        "next_gate": "complete-hashed-python-dependency-lock",
    }
    if not env.interpreter_in_project_venv:
        result["next_gate"] = "activate-or-create-project-venv"
        return result
    if source_plan_factory is None:
        from .source_build_plan import create_source_build_plan
        source_plan_factory = create_source_build_plan
    source = source_plan_factory()
    result["source"] = source.to_dict()
    result["setup_ready"] = bool(source.build_prerequisites_met)
    if not source.build_prerequisites_met:
        result["next_gate"] = "fix-native-build-prerequisites"
    elif not source.source_verified:
        result["next_gate"] = "acquire-pinned-sources"
    return result


def _print(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ruth", description="Ruth CUDA environment and runtime lifecycle"
    )
    sub = parser.add_subparsers(dest="action")
    sub.add_parser("doctor", help="Read-only Python venv and package inventory")
    sub.add_parser("plan", help="Read-only CUDA source and Python dependency plan")
    setup = sub.add_parser("setup", help="Prepare native CUDA sources and dependencies")
    setup.add_argument("--allow-downloads", action="store_true",
                       help="Authorize GitHub source downloads for this invocation")
    sub.add_parser("run", help="Start the CUDA runtime only after verification")
    args = parser.parse_args(argv)

    action = args.action or "doctor"
    env = inspect_python_environment()
    if action == "doctor":
        _print({
            "action": "doctor", "environment": env.to_dict(),
            "runtime_ready": False, "backend_gpu_verified": False,
        })
        return 0 if env.interpreter_in_project_venv else 2
    report = build_launch_plan(environment=env)
    if action == "plan":
        _print({"action": "plan", **report})
        return 0 if env.interpreter_in_project_venv else 2
    if action == "run":
        # Never misreport a placeholder executable as an operational LLM.
        _print({
            "action": "run", "status": "blocked_unverified_cuda_runtime",
            "plan": report,
            "error": "Runtime inference CUDA non ancora installato e verificato; "
                     "il launcher rifiuta l'avvio del motore non implementato.",
        })
        return 2
    if action == "setup":
        if not env.interpreter_in_project_venv or not report["setup_ready"]:
            _print({"action": "setup", "status": "blocked_preflight", "plan": report})
            return 2
        if not args.allow_downloads:
            _print({
                "action": "setup", "status": "download_not_authorized",
                "plan": report,
                "remediation": "./ruth setup --allow-downloads",
            })
            return 2
        # Explicit authorization covers only the pinned source acquisition.
        # Build/install cannot start until every Python package SHA is vetted.
        from .source_acquisition import acquire_pinned_sources
        from .source_build_plan import create_source_build_plan
        result = acquire_pinned_sources(create_source_build_plan())
        updated = build_launch_plan()
        _print({
            "action": "setup", "status": (
                "sources_verified_dependency_lock_pending"
                if result.source_verified else "source_acquisition_failed"
            ),
            "acquisition": result.to_dict(),
            "plan": updated,
            "pip_started": False, "build_started": False,
        })
        return 0 if result.source_verified else 2
    parser.error("Unsupported operation")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
