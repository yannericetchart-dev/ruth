"""Non-invasive Phase 1 audit of all registered Ruth model/source pairs.

Reports actual pipeline evidence from the current machine. It NEVER runs
inference, starts a CUDA worker, downloads weights, or promotes a profile.
"""

import argparse
import json
from pathlib import Path

from hardware import GPUManager
from model import ModelManager
from model.inspect import inspect_gguf
from .profile_identity import resolve_identity
from .profile_pipeline import Journal, atomic_json


def model_source_pairs(source: str = "all", model: str = "all") -> list[tuple[str, str]]:
    if source not in {"all", "bartowski", "kaggle"}:
        raise ValueError(f"Fonte audit non supportata: {source}")
    if model != "all" and model not in ModelManager.CATALOG:
        raise ValueError(f"Modello audit non supportato: {model}")
    pairs = [
        (name, s)
        for name in sorted(ModelManager.CATALOG)
        if model in {"all", name}
        for s in ModelManager.available_sources(name)
        if source in {"all", s}
    ]
    if not pairs:
        raise ValueError("Nessuna combinazione modello/fonte trovata.")
    return pairs


def _confirmed_gpu_runtime_proof(context: dict, n_ctx: int,
                                 rounds: int) -> bool:
    """Require repeatable native full-GPU and CUDA KV+compute evidence."""
    if (not isinstance(n_ctx, int) or isinstance(n_ctx, bool)
        or n_ctx < 64 or not isinstance(rounds, int) or rounds < 1):
        return False
    confirmations = context.get("confirmation") or []
    if (
        len(confirmations) < rounds
        or not all(c.get("success") is True for c in confirmations[-rounds:])
    ):
        return False
    probes = [
        p for p in context.get("attempts") or []
        if p.get("requested_n_ctx") == n_ctx
        and p.get("success") is True
        and (p.get("full_gpu_native_verification") or {}).get("verified") is True
        and (p.get("gpu_runtime_validation") or {}).get("verified") is True
        and (p.get("gpu_runtime_validation") or {}).get("kv", {}).get(
            "verified_gpu_resident"
        ) is True
    ]
    return len(probes) >= rounds


def audit_pair(identity: dict, *, output_root: Path | None = None,
               inspect_local: bool = True) -> dict:
    """Inspect phase evidence, existing bytes and native CUDA validation."""
    choices = identity["choices"]
    manager = ModelManager()
    root = (Path(output_root).expanduser().resolve() if output_root
            else manager.root / "calibration_runs")
    folder = root / identity["profile_key"]
    result = {
        "model_id": choices["model_id"], "source": choices["source"],
        "repository": choices["repository"],
        "profile_key": identity["profile_key"],
        "run_directory": str(folder), "quantization": None,
        "filename": None, "n_ctx_gpu_resident": None,
        "confirmed_lower_bound_gpu_resident_n_ctx": None,
        "context_maximality_proven": False,
        "inventory_verified": False,
        "downloaded_local_gguf_verified": False,
        "full_gpu_baseline_verified": False,
        "context_gpu_resident_verified": False,
        "phase1_complete": False,
        "errors": [], "checks": {},
    }
    errors = result["errors"]

    if not (folder / "state.json").is_file():
        errors.append("Nessun run per questo profilo hardware/config.")
        return result
    try:
        state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        if state.get("identity") != identity:
            raise ValueError("identity mismatch")
        # Reuse Journal's cache verification without constructing it:
        # its normal constructor updates the roadmap on disk.
        journal = object.__new__(Journal)
        journal.folder = folder
        journal.identity = identity
        journal.state = state
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        errors.append(f"Stato profilo illeggibile/incompatibile: {type(exc).__name__}")
        return result
    inventory = journal.cached("inventory")
    if inventory is None:
        errors.append("Inventario mancante, non completato o non classificato.")
    else:
        result["inventory_verified"] = True
        result["checks"]["candidate_count"] = len(inventory.get("inventory") or [])

    baseline = journal.cached("baseline")
    if baseline is None:
        errors.append("Baseline full-CUDA mancante/non completata o GGUF assente.")
        return result
    selected = baseline.get("baseline") or {}
    result["quantization"] = selected.get("quantization")
    result["filename"] = selected.get("filename")
    pinned = selected.get("kaggle_handle")
    requested = choices.get("kaggle_handle")
    handle_ok = (
        choices["source"] != "kaggle"
        or (
            isinstance(pinned, str) and isinstance(requested, str)
            and (
                pinned == requested
                or (
                    len(requested.split("/")) == 4
                    and pinned.startswith(requested + "/")
                    and pinned.rsplit("/", 1)[-1].isdigit()
                )
            )
        )
    )
    if (
        selected.get("model_id") != choices["model_id"]
        or selected.get("source") != choices["source"]
        or selected.get("repository") != choices["repository"]
        or selected.get("native_verified_all_gpu") is not True
        or not handle_ok
    ):
        errors.append("Baseline di altro modello/fonte o prova CUDA insufficiente.")
        return result

    expected_root = manager.model_directory(
        choices["model_id"], choices["source"], choices["repository"]
    ).resolve()
    path = Path(selected.get("local_path") or "/NONEXISTENT").expanduser().resolve()
    shards = selected.get("shard_filenames") or [selected.get("filename")]
    if (
        path.parent != expected_root or path.name != selected.get("filename")
        or not isinstance(shards, list) or not shards or shards[0] != path.name
        or any(not isinstance(n, str) or Path(n).name != n
               or not (expected_root / n).is_file()
               or (expected_root / n).stat().st_size <= 0 for n in shards)
    ):
        errors.append("GGUF locale/insieme shard assente o directory inattesa.")
        return result
    result["checks"]["local_path"] = str(path)
    result["checks"]["shard_count"] = len(shards)
    result["checks"]["local_file_bytes"] = sum(
        (expected_root / name).stat().st_size for name in shards
    )
    if inspect_local:
        try:
            layout = inspect_gguf(
                path, n_ctx=selected.get("baseline_probe_n_ctx", 64)
            )
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(f"Lettura metadati GGUF locale fallita: {type(exc).__name__}")
            return result
        if (
            layout["metadata"].get("general.architecture")
            != selected.get("architecture")
            or layout["shard_count"] != len(shards)
            or layout["tensor_count"] <= 0
        ):
            errors.append("GGUF locale non coerente con la baseline.")
            return result
        result["checks"]["tensor_count"] = layout["tensor_count"]
        result["checks"]["weight_payload_mib"] = round(
            layout["weights_bytes"] / 1048576, 3
        )

    result["downloaded_local_gguf_verified"] = True
    proof = [
        a for a in baseline.get("attempts") or []
        if a.get("status") == "selected_full_gpu_baseline"
        and a.get("filename") == path.name
    ]
    if len(proof) != 1:
        errors.append("Record di selezione della baseline non verificabile.")
        return result
    probe = proof[0].get("test") or {}
    if (
        probe.get("success") is not True
        or (probe.get("full_gpu_native_verification") or {}).get("verified") is not True
        or (probe.get("measurements") or {}).get("minimal_inference_completed") is not True
    ):
        errors.append("Log nativi della baseline non confermano tutti i pesi GPU.")
        return result
    result["full_gpu_baseline_verified"] = True
    result["checks"]["baseline_peak_vram_mib"] = (
        probe.get("measurements") or {}
    ).get("vram_peak_sampled_mib")

    context = journal.cached("context_gpu")
    rounds = identity["reference_runtime"]["context_confirm_rounds"]
    if context is None:
        # A blocked context run may still contain rigorously confirmed
        # lower-bound CUDA probes. Report them but do not certify a maximum.
        state_item = journal.state["phases"].get("context_gpu") or {}
        context_path = folder / "context_gpu.json"
        partial = None
        if state_item.get("status") == "blocked" and context_path.is_file():
            try:
                partial = json.loads(context_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                pass
        if isinstance(partial, dict):
            lower = partial.get("confirmed_gpu_resident_n_ctx")
            if (
                partial.get("success") is False
                and partial.get("confirmation_completed") is True
                and partial.get("maximality_proven") is False
                and partial.get("uncertain_failure_boundaries")
                and partial.get("model_id") == choices["model_id"]
                and partial.get("filename") == path.name
                and partial.get("quantization") == selected.get("quantization")
                and _confirmed_gpu_runtime_proof(partial, lower, rounds)
            ):
                result["confirmed_lower_bound_gpu_resident_n_ctx"] = lower
                result["checks"]["uncertain_context_boundaries"] = (
                    partial["uncertain_failure_boundaries"]
                )
                result["checks"]["context_min_sampled_headroom_mib"] = (
                    partial.get("confirmation_min_sampled_headroom_mib")
                )
                errors.append(
                    f"Contesto GPU-resident {lower} confermato, ma "
                    "massimo inconclusivo per SIGABRT senza causa certa."
                )
                return result
        errors.append("Contesto GPU-resident non ancora completato.")
        return result
    n_ctx = context["confirmed_max_gpu_resident_n_ctx"]
    result["n_ctx_gpu_resident"] = n_ctx
    if (
        context.get("success") is not True
        or context.get("measurement_completed") is not True
        or context.get("model_id") != choices["model_id"]
        or context.get("filename") != path.name
        or context.get("quantization") != selected.get("quantization")
    ):
        errors.append("Contesto confermato non coerente con la baseline.")
        return result
    if not _confirmed_gpu_runtime_proof(context, n_ctx, rounds):
        errors.append("Conferme contesto insufficienti: pesi/KV/compute non provati.")
        return result
    result["confirmed_lower_bound_gpu_resident_n_ctx"] = n_ctx
    result["context_maximality_proven"] = True
    result["checks"]["confirmation_rounds"] = rounds
    result["checks"]["context_min_sampled_headroom_mib"] = (
        context.get("confirmation_min_sampled_headroom_mib")
    )
    result["checks"]["search_ceiling"] = context.get("search_ceiling")
    result["checks"]["declared_context_limit_reached"] = (
        context.get("search_reached_declared_limit")
    )
    result["context_gpu_resident_verified"] = True
    result["phase1_complete"] = result["inventory_verified"]
    return result


def audit_all(*, source: str = "all", model: str = "all",
              output_root: Path | None = None, inspect_local: bool = True,
              status: dict | None = None) -> list[dict]:
    pairs = model_source_pairs(source, model)
    if status is None:
        status = GPUManager.calibration_status()
    if not status.get("ready"):
        raise RuntimeError(
            "Audit: GPU non pronta per l'identita' macchina corrente. "
            "Nessuna prova CUDA avviata."
        )
    results = []
    for name, publisher in pairs:
        identity = resolve_identity(name, source=publisher, status=status)
        results.append(
            audit_pair(identity, output_root=output_root,
                       inspect_local=inspect_local)
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit non invasivo di inventario, GGUF locale, CUDA e n_ctx."
    )
    parser.add_argument("--model", default="all",
                        choices=["all", *sorted(ModelManager.CATALOG)])
    parser.add_argument("--source", default="all",
                        choices=["all", "bartowski", "kaggle"])
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--no-local-inspection", action="store_true")
    args = parser.parse_args(argv)
    reports = audit_all(
        model=args.model, source=args.source, output_root=args.output_root,
        inspect_local=not args.no_local_inspection
    )
    print(
        "Modello/fonte | GGUF selezionato | Locale | Full CUDA | "
        "n_ctx GPU-resident | Fase 1"
    )
    for r in reports:
        ctx = r["n_ctx_gpu_resident"]
        print(
            f"{r['model_id']} / {r['source']} | "
            f"{r['quantization'] or '-'} | "
            f"{'SI' if r['downloaded_local_gguf_verified'] else 'NO'} | "
            f"{'SI' if r['full_gpu_baseline_verified'] else 'NO'} | "
            f"{ctx if r['context_gpu_resident_verified'] else 'NON VALIDATO'} | "
            f"{'COMPLETA' if r['phase1_complete'] else 'INCOMPLETA'}"
        )
        if r["errors"]:
            print("  Da verificare: " + "; ".join(r["errors"]))
    root = (Path(args.output_root).expanduser().resolve()
            if args.output_root else ModelManager().root / "calibration_runs")
    dest = root / "phase1_audit.json"
    atomic_json(dest, {"schema_version": 1, "pairs": reports,
                       "complete": all(r["phase1_complete"] for r in reports)})
    print(f"Report: {dest} | nessun download, modifica GPU o profilo operativo.")
    return 0 if all(r["phase1_complete"] for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
