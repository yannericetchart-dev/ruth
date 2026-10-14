"""Ruth CLI lifecycle: doctor, plan, setup, run.

No hidden software installations. A complete hash-locked build/install stage
will be integrated here after the security gate; until then stop explicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    setup.add_argument("--allow-python-dependency-downloads", action="store_true",
                       help="Authorize pip DOWNLOAD of binary Python dependencies only (not install)")
    setup.add_argument("--verify-candidate", action="store_true",
                       help="Audit staged wheel hashes and PyPI metadata; pip OFFLINE dry-run only")
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
        if not args.allow_downloads and not args.allow_python_dependency_downloads \
                and not args.verify_candidate:
            _print({
                "action": "setup", "status": "download_not_authorized",
                "plan": report,
                "remediation": "./ruth setup --allow-downloads "
                               "--allow-python-dependency-downloads",
            })
            return 2
        from .source_build_plan import create_source_build_plan
        source_plan = create_source_build_plan()
        acquisition = None
        if args.allow_downloads:
            from .source_acquisition import acquire_pinned_sources
            acquisition = acquire_pinned_sources(source_plan)
            if not acquisition.source_verified:
                _print({"action": "setup", "status": "source_acquisition_failed",
                        "acquisition": acquisition.to_dict(),
                        "pip_started": False, "build_started": False})
                return 2
            source_plan = create_source_build_plan()
        if args.verify_candidate:
            if not source_plan.source_verified:
                _print({"action": "setup", "status": "source_checkout_unverified",
                        "source": source_plan.to_dict(),
                        "installation_started": False, "build_started": False})
                return 2
            from .dependency_audit import audit_candidate
            audit = audit_candidate(
                project_root=Path(env.project_root),
                environment=env, network_authorized=True,
            )
            _print({
                "action": "setup", "status": audit.status,
                "candidate_audit": audit.to_dict(),
                "installation_started": False, "build_started": False,
            })
            return 0 if audit.status == "security_review_required" else 2
        if args.allow_python_dependency_downloads:
            if not source_plan.source_verified:
                _print({"action": "setup", "status": "source_checkout_unverified",
                        "source": source_plan.to_dict(),
                        "pip_started": False, "build_started": False})
                return 2
            from .dependency_resolution import stage_dependency_candidate
            candidate = stage_dependency_candidate(
                authorize_network=True, environment=env,
                source_verified=source_plan.source_verified,
            )
            _print({
                "action": "setup",
                "status": candidate.status,
                "acquisition": acquisition.to_dict() if acquisition else None,
                "dependency_candidate": candidate.to_dict(),
                "installation_started": False, "build_started": False,
            })
            return 0 if candidate.status == "candidate_review_required" else 2
        updated = build_launch_plan()
        _print({
            "action": "setup", "status": "sources_verified_dependency_lock_pending",
            "acquisition": acquisition.to_dict() if acquisition else None,
            "plan": updated,
            "pip_started": False, "build_started": False,
        })
        return 0
    parser.error("Unsupported operation")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
