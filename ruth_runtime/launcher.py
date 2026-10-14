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
    setup.add_argument("--review-security", action="store_true",
                       help="Offline upstream diskcache and CUDA adapter policy check; no install")
    setup.add_argument("--build-native", action="store_true",
                       help="Authorize offline CUDA sm_89 wheel compilation in isolated staging; no .venv install")
    setup.add_argument("--plan-native-install", action="store_true",
                       help="Read-only audit of the existing native CUDA wheel and runtime inputs")
    setup.add_argument("--verify-native-library", action="store_true",
                       help="Read-only installed llama.cpp shared-library integrity and direct/Python C ABI probe")
    setup.add_argument("--install-native", action="store_true",
                       help="Explicitly authorize offline staged native CUDA binding install into Ruth .venv")
    setup.add_argument("--accept-diskcache-risk", action="store_true",
                       help="Explicitly acknowledge CVE-2025-69872 for installed diskcache 5.6.3; TODO stays open")
    models = sub.add_parser("models", help="Read-only hierarchical model catalog and HF provenance")
    model_sub = models.add_subparsers(dest="model_action")
    model_sub.add_parser("list", help="List families, GGUF models and defaults (offline)")
    for operation in ("select", "resolve", "download", "smoke", "profile", "quantizations", "recommend"):
        mp = model_sub.add_parser(operation, help=(
            "Offline model selection" if operation == "select" else
            "Resolve HF immutable revision and GGUF checksum (metadata only)"
            if operation == "resolve" else
            "Explicit, resumable SHA256-verified HF GGUF download"
            if operation == "download" else
            "Isolated native GGUF test with real CUDA offload/token evidence"
            if operation == "smoke" else
            "Repeated GPU GGUF profile: warmup, prefill/decode, VRAM, token/s"
            if operation == "profile" else
            "Discover all GGUF quantization metadata (no model files)"
            if operation == "quantizations" else
            "Quality heuristic and GPU free-memory provisional shortlist"
        ))
        mp.add_argument("--purpose", choices=("production", "test"), default="test")
        mp.add_argument("--family")
        mp.add_argument("--model")
        mp.add_argument("--version")
        mp.add_argument("--format")
        mp.add_argument("--quantization")
        if operation in ("resolve", "quantizations", "recommend"):
            mp.add_argument("--allow-metadata-network", action="store_true",
                            help="Authorize public HF JSON metadata requests; never download weights")
        if operation == "recommend":
            mp.add_argument("--reserve-mib", type=int, default=None,
                            help="Provisional VRAM reserve MiB (unvalidated KV/compute budget)")
            mp.add_argument("--top-k", type=int, default=None,
                            help="Maximum number of shortlisted candidates")
        if operation == "download":
            mp.add_argument("--allow-model-download", action="store_true",
                            help="Authorize network transfer of this pinned GGUF file")
            mp.add_argument("--accept-model-license", action="store_true",
                            help="Confirm you reviewed and accept the original model license terms")
        if operation in ("smoke", "profile"):
            mp.add_argument("--allow-gpu-inference", action="store_true",
                            help="Authorize model loading/inference in isolated CUDA worker; no download")
    sub.add_parser("run", help="Start the CUDA runtime only after verification")
    args = parser.parse_args(argv)

    action = args.action or "doctor"
    if action == "models":
        from .model_catalog import (
            CatalogError, list_models, select_model, resolve_model_metadata,
        )
        try:
            if args.model_action == "list":
                _print({"action": "models", **list_models()})
                return 0
            if args.model_action not in {"select", "resolve", "download", "smoke", "profile", "quantizations", "recommend"}:
                raise CatalogError("Choose models list, select, resolve, download, smoke, profile, quantizations or recommend")
            choice = select_model(
                purpose=args.purpose, family=args.family, model=args.model,
                version=args.version, format=args.format,
                quantization=args.quantization,
            )
            if args.model_action in {"quantizations", "recommend"}:
                from .candidate_selection import (
                    discover_quantizations, recommend_quantization,
                )
                if args.model_action == "quantizations":
                    result = discover_quantizations(
                        purpose=args.purpose,
                        authorize_metadata_network=args.allow_metadata_network,
                    )
                    _print({"action": "models", **result})
                    return 0 if result["status"] == "quantizations_discovered" else 2
                result = recommend_quantization(
                    purpose=args.purpose,
                    authorize_metadata_network=args.allow_metadata_network,
                    reserve_mib=args.reserve_mib,
                    max_candidates=args.top_k,
                )
                _print({"action": "models", **result})
                return 0 if result["status"] == "provisional_shortlist_ready" else 2
            if args.model_action == "select":
                _print({"action": "models", "status": "selected_unverified",
                        "selection": choice.to_dict(),
                        "download_started": False})
                return 0
            if args.model_action == "profile":
                from .gpu_profile import run_gpu_profile
                result = run_gpu_profile(
                    purpose=args.purpose,
                    authorize_gpu_inference=args.allow_gpu_inference,
                )
                _print({"action": "models", **result.to_dict()})
                return 0 if result.gpu_profile_verified else 2
            if args.model_action == "smoke":
                from .inference_probe import probe_inference
                result = probe_inference(
                    purpose=args.purpose,
                    authorize_gpu_inference=args.allow_gpu_inference,
                )
                _print({"action": "models", **result.to_dict()})
                return 0 if result.gpu_inference_verified else 2
            if args.model_action == "download":
                from .model_download import download_pinned_model
                report = download_pinned_model(
                    choice, allow_download=args.allow_model_download,
                    acknowledge_model_license=args.accept_model_license,
                )
                _print({"action": "models", **report.to_dict()})
                return 0 if report.model_verified else 2
            resolved = resolve_model_metadata(
                choice, authorize_metadata_network=args.allow_metadata_network,
            )
            _print({"action": "models", **resolved})
            return 0 if resolved["status"] == "source_metadata_verified_license_pending" else 2
        except CatalogError as exc:
            _print({"action": "models", "status": "catalog_selection_blocked",
                    "error": str(exc), "download_started": False})
            return 2
    # Reject incompatible permissions BEFORE any preflight, Git checkout,
    # network access or build. Flag order must not change this boundary.
    standalone = (
        args.build_native, args.plan_native_install,
        args.install_native, args.verify_native_library,
    ) if action == "setup" else (False, False, False, False)
    conflicting = (
        sum(bool(x) for x in standalone) > 1 or
        (any(standalone) and (
            args.allow_downloads or args.allow_python_dependency_downloads
            or args.verify_candidate or args.review_security
        )) or
        (action == "setup" and args.accept_diskcache_risk and not args.install_native)
    )
    if conflicting:
        _print({
            "action": "setup", "status": "conflicting_build_options",
            "error": "Native build/install actions must be separate; "
                     "no implicit downloads, build, or risk acceptance",
            "installation_started": False,
            "build_started": False,
        })
        return 2
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
        if not any((
            args.allow_downloads,
            args.allow_python_dependency_downloads,
            args.verify_candidate,
            args.review_security,
            args.build_native,
            args.plan_native_install,
            args.install_native,
            args.verify_native_library,
        )):
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
        if args.verify_native_library:
            from .native_library import verify_native_library
            inspected = verify_native_library(
                project_root=Path(env.project_root), environment=env,
            )
            _print({
                "action": "setup", "status": inspected.status,
                "native_library": inspected.to_dict(),
                "installation_started": False, "build_started": False,
            })
            return 0 if inspected.status == "native_library_dual_abi_verified" else 2
        if args.plan_native_install:
            from .native_installer import inspect_native_install_inputs
            verified, _wheels = inspect_native_install_inputs(
                project_root=Path(env.project_root), environment=env,
            )
            _print({
                "action": "setup", "status": verified.status,
                "install_preflight": verified.to_dict(),
                "installation_started": False, "build_started": False,
            })
            return 0 if verified.status == "ready_for_explicit_install_authorization" else 2
        if args.install_native:
            from .native_installer import install_native_cuda
            outcome = install_native_cuda(
                project_root=Path(env.project_root), environment=env,
                authorize_install=True,
                accept_diskcache_risk=args.accept_diskcache_risk,
            )
            _print({
                "action": "setup", "status": outcome.status,
                "native_install": outcome.to_dict(),
                "build_started": False,
            })
            return 0 if outcome.status == "native_cuda_binding_installed" else 2
        if args.build_native:
            from .native_builder import build_native_cuda_wheel
            built = build_native_cuda_wheel(
                project_root=Path(env.project_root),
                environment=env, source_plan=source_plan,
            )
            _print({
                "action": "setup", "status": built.status,
                "native_wheel": built.to_dict(),
                "installation_started": False,
                "risk_todo_open": True,
            })
            return 0 if built.status == "native_cuda_wheel_built" else 2
        if args.review_security:
            from .security_gate import review_diskcache_risk
            gate = review_diskcache_risk(
                project_root=Path(env.project_root),
                environment=env, source_plan=source_plan,
            )
            _print({
                "action": "setup",
                "status": gate.status,
                "security_gate": gate.to_dict(),
                "installation_started": False,
                "build_started": False,
            })
            return 0 if gate.mitigations_validated else 2
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
