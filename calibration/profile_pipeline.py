"""Single incremental Ruth calibration workflow, with live progress and resume.

Usage:
  python -m calibration.profile_pipeline --model llama-3.1-8b-instruct
  python -m calibration.profile_pipeline --model all --until inventory
  python -m calibration.profile_pipeline --model qwen-2.5-7b-instruct --rerun

Every result is bound to explicit user choices AND automatically detected
hardware. Only inventory, full-GPU baseline and GPU context are implemented
now. Later stages have reserved IDs but cannot silently run or be marked done.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys

from model import ModelManager

from .profile_identity import resolve_identity


ROADMAP = (
    ("inventory", True),
    ("baseline", True),
    ("context_gpu", True),
    ("mapping", False),
    ("runtime", False),
    ("context_tuned", False),
    ("capacity", False),
)
IMPLEMENTED = [name for name, active in ROADMAP if active]
MESSAGES = {
    "inventory": "Inventario GGUF e priorita' qualitativa",
    "baseline": "Caricamento del miglior GGUF interamente su GPU",
    "context_gpu": "Ricerca del massimo contesto GPU-resident",
    "mapping": "Ricerca placement layer/tensori",
    "runtime": "Ottimizzazione parametri llama.cpp",
    "context_tuned": "Contesto GPU con mapping ottimizzato",
    "capacity": "Contesto esteso con RAM di sistema",
}


def kaggle_access_notice(model_id: str, source: str,
                         failure: str) -> str | None:
    """Warn only when Kaggle explicitly reports unaccepted license terms."""
    if source != "kaggle":
        return None
    if "categoria=license_required" not in failure:
        return None
    return (
        "KAGGLE | LICENZA DA ACCETTARE: l'API indica che l'account "
        "non ha ancora accettato i termini della variante richiesta. "
        "Apri la pagina del modello su Kaggle, verifica con lo stesso "
        "account del token locale e accetta le condizioni prima di riprovare."
    )


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def brief_event(event: dict) -> str:
    """A human-readable, secret-free decision for every candidate/probe."""
    kind = event.get("event")
    if kind == "inventory_ready":
        return f"Inventario: {event['count']} GGUF; sconosciuti={event['unranked']}"
    if kind == "candidate_start":
        return f"GGUF candidato: {event['quantization']} ({event['filename']})"
    if kind == "candidate_download":
        return f"Download richiesto: {event['filename']}"
    if kind == "candidate_gpu_probe":
        return (f"Test full-GPU: {event['quantization']} "
                f"n_ctx={event['n_ctx']} batch={event['batch']}")
    if kind == "candidate_decision":
        info = [
            "Decisione: " + str(event["quantization"]),
            str(event["status"]),
        ]
        if event.get("weight_mib") is not None:
            info.append(f"pesi={event['weight_mib']} MiB")
        if event.get("peak_mib") is not None:
            info.append(f"picco={event['peak_mib']} MiB")
        if event.get("failure_hint"):
            info.append("causa=" + str(event["failure_hint"]))
        if event.get("error"):
            info.append("errore=" + str(event["error"])[:160])
        return " | ".join(info)
    if kind == "context_probe_start":
        return f"Test n_ctx={event['n_ctx']} (generazione={event['token_count']} token)"
    if kind == "context_probe_result":
        return (
            f"Risultato n_ctx={event['n_ctx']}: "
            f"{'OK' if event['success'] else 'NON RIUSCITO'} "
            f"| causa={event.get('failure_class')} "
            f"| VRAM picco={event.get('peak_mib')} MiB "
            f"| pesi CUDA={event.get('native_gpu_weights')} "
            f"| KV CUDA={event.get('native_gpu_kv')}"
        )
    if kind == "context_candidate_limit":
        return (f"Massimo provvisorio n_ctx={event['n_ctx']} "
                f"| primo limite={event.get('failed_boundary')}")
    if kind == "context_confirmation":
        return (f"Conferma {event['round']}/{event['total']} "
                f"di n_ctx={event['n_ctx']}")
    return "Evento: " + str(kind)


class Journal:
    def __init__(self, folder: Path, identity: dict):
        self.folder = folder
        self.identity = identity
        self.state_path = folder / "state.json"
        self.events_path = folder / "events.jsonl"
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state.get("identity") != identity:
                raise RuntimeError("Identita' hardware/configurazione del run diversa.")
        else:
            self.state = {
                "identity": identity, "created_at": now_utc(),
                "roadmap": [
                    {"phase": name, "implemented": supported}
                    for name, supported in ROADMAP
                ],
                "phases": {},
                "profile_generated": False,
                "operationally_selected": False,
            }
            atomic_json(self.state_path, self.state)

    def event(self, stage: str, payload: dict) -> None:
        rec = {"time": now_utc(), "phase": stage, **payload}
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            f.flush()
        stamp = rec["time"].split("T")[-1].replace("+00:00", "Z")
        print(f"[{stamp}][{stage}] {brief_event(payload)}", flush=True)

    def phase_state(self, phase: str, status: str,
                    result_file: str | None = None,
                    detail: dict | None = None) -> None:
        self.state["phases"][phase] = {
            "status": status, "updated_at": now_utc(),
            "result_file": result_file,
            "detail": detail or {},
        }
        atomic_json(self.state_path, self.state)

    def invalidate_after(self, phase: str) -> None:
        """Never reuse downstream results after rerunning an upstream phase.

        Keep earlier result files for forensic inspection; changing their
        status to stale prevents them from passing the resume check.
        """
        if phase not in IMPLEMENTED:
            raise ValueError(f"Fase non implementata: {phase}")
        changed = False
        for later in IMPLEMENTED[IMPLEMENTED.index(phase) + 1:]:
            item = self.state["phases"].get(later)
            if item and item.get("status") != "stale":
                item["status"] = "stale"
                item["stale_due_to"] = phase
                item["updated_at"] = now_utc()
                changed = True
        if changed:
            atomic_json(self.state_path, self.state)
            print(
                f"[{phase}] Risultati delle fasi successive invalidati "
                "(file precedenti conservati).",
                flush=True,
            )

    def cached(self, phase: str) -> dict | None:
        entry = self.state["phases"].get(phase) or {}
        if entry.get("status") != "completed":
            return None
        name = entry.get("result_file")
        if name != phase + ".json":
            return None
        path = self.folder / name
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if phase == "inventory":
            return data if data.get("inventory_only") and not data.get("unranked_quantizations") else None
        if phase == "baseline":
            selected = data.get("baseline") or {}
            model_path = Path(selected.get("local_path") or "/NONEXISTENT")
            shard_files = selected.get("shard_filenames") or [selected.get("filename")]
            shards_present = (
                isinstance(shard_files, list)
                and bool(shard_files)
                and shard_files[0] == selected.get("filename")
                and all(
                    isinstance(name, str)
                    and Path(name).name == name
                    and (model_path.parent / name).is_file()
                    for name in shard_files
                )
            )
            return data if (
                data.get("success") is True
                and selected.get("native_verified_all_gpu") is True
                and model_path.is_file()
                and model_path.name == selected.get("filename")
                and shards_present
            ) else None
        if phase == "context_gpu":
            return data if (
                data.get("success") is True
                and isinstance(data.get("confirmed_max_gpu_resident_n_ctx"), int)
                and data["confirmed_max_gpu_resident_n_ctx"] >= 64
            ) else None
        return None


def _phase_call(name: str, identity: dict, previous: dict,
                reporter) -> tuple[dict, bool, dict]:
    choices = identity["choices"]
    model = choices["model_id"]
    model_root = ModelManager().model_directory(
        model, choices["source"], choices["repository"]
    )
    opts = identity["reference_runtime"]
    if name == "inventory":
        from .baseline_discovery import choose_baseline

        report = choose_baseline(
            root=model_root, model_id=model, repo_id=choices["repository"],
            source=choices["source"],
            probe_n_ctx=opts["baseline_n_ctx"], n_batch=opts["n_batch"],
            inventory_only=True, persist_selection=False, progress=reporter,
            kaggle_handle=choices.get("kaggle_handle"),
        )
        good = bool(report.get("inventory_only")) and not report.get("unranked_quantizations")
        summary = {
            "count": len(report.get("inventory") or []),
            "unranked": report.get("unranked_quantizations") or [],
        }
        return report, good, summary

    if name == "baseline":
        from .baseline_discovery import choose_baseline

        report = choose_baseline(
            root=model_root, model_id=model, repo_id=choices["repository"],
            source=choices["source"],
            probe_n_ctx=opts["baseline_n_ctx"], n_batch=opts["n_batch"],
            inventory_only=False, persist_selection=False, progress=reporter,
            kaggle_handle=choices.get("kaggle_handle"),
        )
        selected = report.get("baseline") or {}
        good = (
            report.get("success") is True
            and selected.get("native_verified_all_gpu") is True
            and selected.get("repository") == choices["repository"]
            and selected.get("model_id") == model
        )
        summary = {"quantization": selected.get("quantization"),
                   "file": selected.get("filename"),
                   "attempts": len(report.get("attempts") or [])}
        return report, good, summary

    if name == "context_gpu":
        from .gpu_context import search_gpu_context

        baseline = previous["baseline"].get("baseline")
        if not baseline:
            raise RuntimeError("Impossibile avviare contesto senza baseline verificata.")
        report = search_gpu_context(
            model, baseline_selection=baseline, persist_reference=False,
            source=choices["source"], kaggle_handle=choices.get("kaggle_handle"),
            min_ctx=opts["baseline_n_ctx"],
            tokens=opts["context_generated_tokens"],
            batch=opts["context_n_batch"],
            confirm_rounds=opts["context_confirm_rounds"],
            progress=reporter,
        )
        result = report.get("confirmed_max_gpu_resident_n_ctx")
        good = report.get("success") is True and isinstance(result, int) and result > 0
        summary = {
            "n_ctx": result,
            "peak_headroom_mib": report.get("confirmation_min_sampled_headroom_mib"),
            "attempts": len(report.get("attempts") or []),
            "failure": report.get("error"),
        }
        return report, good, summary

    raise NotImplementedError(f"Fase non implementata: {name}")


def run_pipeline(identity: dict, *, until: str = "context_gpu",
                 rerun: bool = False, output_root: Path | None = None) -> dict:
    if until not in IMPLEMENTED:
        raise ValueError(
            f"Fase '{until}' non implementata. Disponibili: {IMPLEMENTED}"
        )
    root = (
        Path(output_root).expanduser().resolve() if output_root is not None
        else ModelManager().root / "calibration_runs"
    )
    folder = root / identity["profile_key"]
    folder.mkdir(parents=True, exist_ok=True)

    # A separate process cannot concurrently rewrite this machine/config run.
    with (folder / "run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal = Journal(folder, identity)
        choices, hardware = identity["choices"], identity["hardware"]
        print("\nRUTH | Calibrazione automatica", flush=True)
        print(
            f"Configurazione: {choices['source']} / {choices['model_id']} / "
            f"{choices['engine']} / {choices['method']}", flush=True
        )
        print(
            f"Hardware: {hardware['backend']} / {hardware['gpu_name']} "
            f"/ {hardware['vram_total_mib']} MiB VRAM "
            f"/ {hardware['ram_total_mib']} MiB RAM", flush=True
        )
        print(
            f"Identita': {identity['profile_key']} | "
            f"risultati: {folder}", flush=True
        )
        previous = {}
        for stage in IMPLEMENTED[:IMPLEMENTED.index(until) + 1]:
            cached = None if rerun else journal.cached(stage)
            if cached is not None:
                previous[stage] = cached
                print(f"[{stage}] RIUTILIZZO risultato gia' verificato.", flush=True)
                continue
            print(f"\n[{stage}] AVVIO: {MESSAGES[stage]}", flush=True)
            journal.invalidate_after(stage)
            journal.phase_state(stage, "running")
            try:
                report, good, summary = _phase_call(
                    stage, identity, previous,
                    lambda event, phase=stage: journal.event(phase, event),
                )
            except Exception as exc:
                # Never lose the last known completed stages when a worker crashes.
                detail = {"failure": f"{type(exc).__name__}: {exc}"}
                notice = kaggle_access_notice(
                    choices["model_id"], choices["source"], str(exc)
                )
                if notice:
                    detail["access_notice"] = notice
                journal.phase_state(stage, "blocked", detail=detail)
                print(f"[{stage}] BLOCCATO: {detail['failure']}", flush=True)
                if notice:
                    print(f"[{stage}] {notice}", flush=True)
                return {
                    "success": False, "blocked_phase": stage, "detail": detail,
                    "run_directory": str(folder), "profile_key": identity["profile_key"],
                }
            atomic_json(folder / (stage + ".json"), report)
            journal.phase_state(
                stage, "completed" if good else "blocked",
                stage + ".json", summary,
            )
            if not good:
                print(f"[{stage}] NON COMPLETATO: {summary}", flush=True)
                return {
                    "success": False, "blocked_phase": stage, "detail": summary,
                    "run_directory": str(folder), "profile_key": identity["profile_key"],
                }
            previous[stage] = report
            print(f"[{stage}] COMPLETATO: {summary}", flush=True)
        print("\nConclusione: fasi richieste completate. "
              "Nessun profilo operativo generato.", flush=True)
        return {
            "success": True,
            "completed_through": until,
            "run_directory": str(folder),
            "profile_key": identity["profile_key"],
            "profile_generated": False,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Orchestratore incrementale e riprendibile Ruth (GPU CUDA)"
    )
    parser.add_argument("--model", default="gemma-2-9b-it",
                        choices=["all", *sorted(ModelManager.CATALOG)])
    parser.add_argument("--source", default=None,
                        help="Fonte: bartowski (Hugging Face) o kaggle; default bartowski")
    parser.add_argument("--kaggle-handle", default=None,
                        help="Variante Kaggle, formato owner/model/framework/variation[/version]")
    parser.add_argument("--engine", default="llama.cpp",
                        help="Motore; oggi disponibile solo llama.cpp")
    parser.add_argument("--method", default="llama_cpp_python",
                        help="Metodo; oggi disponibile solo llama_cpp_python")
    parser.add_argument("--until", default="context_gpu", choices=IMPLEMENTED,
                        help="Ultima fase da completare in questo run")
    parser.add_argument("--rerun", action="store_true",
                        help="Ripete anche fasi gia' completate")
    parser.add_argument("--output-root", type=Path,
                        help="Cartella per risultati per hardware/config")
    parser.add_argument("--list-phases", action="store_true")
    args = parser.parse_args(argv)
    if args.list_phases:
        for phase, implemented in ROADMAP:
            print(f"{phase:15} {'implementato' if implemented else 'previsto'}"
                  f" | {MESSAGES[phase]}")
        return 0
    if args.kaggle_handle and args.model == "all":
        parser.error("--kaggle-handle richiede un singolo --model.")
    models = (
        sorted(ModelManager.CATALOG) if args.model == "all" else [args.model]
    )
    if args.model == "all" and args.source is not None:
        # In a multi-source catalog, --source with --model all acts as a
        # filter; individual model invocations still validate the source.
        models = [
            name for name in models
            if args.source in ModelManager.available_sources(name)
        ]
        if not models:
            parser.error(f"Nessun modello registrato per la fonte {args.source}.")
    failed = False
    for model in models:
        try:
            identity = resolve_identity(
                model, source=args.source, engine=args.engine,
                method=args.method,
                kaggle_handle=args.kaggle_handle,
            )
            outcome = run_pipeline(
                identity, until=args.until, rerun=args.rerun,
                output_root=args.output_root,
            )
        except (Exception) as exc:
            # Never display token secrets. Exceptions in these code paths
            # contain only model/hardware diagnostics.
            print(f"RUTH: ERRORE [{model}]: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            failed = True
            continue
        if not outcome["success"]:
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
