"""Incremental n_ctx sweep for SINGLE-tensor offloading (Ruth phase 2).

Runs the existing load-only -> inference test at 512, 768, 1024, ...
until the first FULLY MEASURED context with zero valid inference mappings,
or until GGUF architecture maximum context is reached.

Each context has an independent, atomically updated measurement checkpoint.
A separate sweep manifest is rebuilt/resumed safely; partial test budgets
NEVER count as evidence that no mapping works.
"""
import argparse
import hashlib
import json
from pathlib import Path

from model.inspect import inspect_gguf

from .gpu_context import context_limit
from .kv_single_to_512 import run_single_to_512, CENSUS_SCHEMA
from .profile_pipeline import atomic_json


SWEEP_SCHEMA = "ruth-phase2-single-context-sweep-v1"


def run_single_context_sweep(
    *, census_path: Path, output_dir: Path,
    start_ctx: int = 512, step: int = 256,
    max_ctx: int | None = None, benchmark_repeats: int = 2,
    timeout: int = 600,
    max_new_loads: int | None = None,
    max_new_benchmarks: int | None = None,
    progress=None, stage_runner=run_single_to_512,
) -> dict:
    if (type(start_ctx) is not int or type(step) is not int
        or step < 1 or start_ctx < 64
        or max_ctx is not None and (
            type(max_ctx) is not int or max_ctx < start_ctx)):
        raise ValueError("Intervallo n_ctx non valido.")
    census_path = Path(census_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    source = census_path.read_bytes()
    census = json.loads(source)
    if (census.get("schema") != CENSUS_SCHEMA
        or census.get("completed") is not True):
        raise ValueError("Richiesto censimento singoli COMPLETO.")
    params = census["parameters"]
    if start_ctx <= params["n_ctx"]:
        raise ValueError("Il primo contesto deve superare quello profilato.")
    model_path = Path(params["model_path"]).expanduser().resolve()
    layout = inspect_gguf(model_path, n_ctx=start_ctx)
    arch_limit = context_limit(layout)
    if start_ctx > arch_limit:
        raise ValueError("Il primo target supera il limite nativo GGUF.")
    if max_ctx is not None and max_ctx > arch_limit:
        raise ValueError("--max-ctx supera il limite nativo GGUF.")
    effective_limit = arch_limit if max_ctx is None else max_ctx
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "phase2_single_context_sweep.json"
    if census_path == manifest:
        raise ValueError("Non sovrascrivere il report census.")
    identity = {
        "census_sha256": hashlib.sha256(source).hexdigest(),
        "model_id": params["model_id"],
        "model_path": str(model_path),
        "start_ctx": start_ctx, "step": step,
        "max_ctx": max_ctx,
        "native_architecture_limit": arch_limit,
        "benchmark_repeats": benchmark_repeats,
        "timeout": timeout,
    }
    if manifest.exists():
        old = json.loads(manifest.read_text("utf-8"))
        if old.get("schema") != SWEEP_SCHEMA or old.get("parameters") != identity:
            raise ValueError("Checkpoint sweep incompatibile.")
    summary = {
        "schema": SWEEP_SCHEMA,
        "parameters": identity,
        "status": "running",
        "completed": False,
        "last_valid_target_ctx": None,
        "first_target_without_valid_inference": None,
        "ready_for_phase3": False,
        "contexts": [],
        "next_ctx": start_ctx,
        "note": (
            "Nessun risparmio misurato a n_ctx piu piccolo viene "
            "usato per eliminare con certezza mapping a contesto "
            "superiore. Inferenza valida = GPU KV e native placement "
            "verificati; un contesto parziale non determina arresto."
        ),
    }

    def save():
        atomic_json(manifest, summary)

    save()
    for ctx in range(start_ctx, effective_limit + 1, step):
        report_name = f"phase2_single_to_{ctx}.json"
        stage = stage_runner(
            census_path=census_path,
            output=output_dir / report_name,
            target_ctx=ctx, benchmark_repeats=benchmark_repeats,
            timeout=timeout, max_new_loads=max_new_loads,
            max_new_benchmarks=max_new_benchmarks,
            progress=progress,
        )
        observed = stage.get("summary") or {}
        if (not isinstance(observed, dict)
            or type(observed.get("inference_success")) is not int
            or observed["inference_success"] < 0):
            raise ValueError(f"Report stage {ctx} privo di contatori validi.")
        done = stage.get("completed") is True and (
            observed.get("ready_for_phase3") is True
        )
        record = {
            "n_ctx": ctx,
            "report_path": report_name,
            "completed": done,
            "load_success": observed.get("load_success"),
            "inference_success": observed["inference_success"],
            "remaining_loads": observed.get("remaining_loads"),
            "remaining_inference": observed.get("remaining_inference"),
        }
        summary["contexts"].append(record)
        summary["next_ctx"] = ctx if not done else ctx + step
        if not done:
            summary["status"] = "partial_context_checkpoint"
            save()
            break
        if observed["inference_success"] == 0:
            summary["status"] = "stopped_no_valid_singletons"
            summary["first_target_without_valid_inference"] = ctx
            summary["completed"] = True
            summary["ready_for_phase3"] = True
            save()
            break
        summary["last_valid_target_ctx"] = ctx
        if progress:
            progress({
                "event": "context_complete",
                "n_ctx": ctx,
                "valid_singletons": observed["inference_success"],
                "next_ctx": ctx+step if ctx+step <= effective_limit else None,
            })
        save()
    else:
        summary["status"] = (
            "architecture_context_limit_reached"
            if max_ctx is None or effective_limit == arch_limit
            else "user_context_limit_reached"
        )
        summary["completed"] = True
        summary["ready_for_phase3"] = True
        summary["next_ctx"] = None
        save()
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fase 2 progressiva: singoli tensori a 512, 768, ..."
    )
    parser.add_argument("--census", type=Path,
                        default=Path("phase2_single_tensor_census_256.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("phase2_single_contexts"))
    parser.add_argument("--start-ctx", type=int, default=512)
    parser.add_argument("--step", type=int, default=256)
    parser.add_argument("--max-ctx", type=int)
    parser.add_argument("--benchmark-repeats", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-new-loads", type=int)
    parser.add_argument("--max-new-benchmarks", type=int)
    args = parser.parse_args(argv)
    try:
        result = run_single_context_sweep(
            census_path=args.census, output_dir=args.output_dir,
            start_ctx=args.start_ctx, step=args.step,
            max_ctx=args.max_ctx,
            benchmark_repeats=args.benchmark_repeats,
            timeout=args.timeout, max_new_loads=args.max_new_loads,
            max_new_benchmarks=args.max_new_benchmarks,
            progress=lambda row: print(
                json.dumps(row, ensure_ascii=False), flush=True
            ),
        )
        print(json.dumps({
            "status": result["status"],
            "last_valid_target_ctx": result["last_valid_target_ctx"],
            "first_target_without_valid_inference":
                result["first_target_without_valid_inference"],
            "next_ctx": result["next_ctx"],
            "contexts": result["contexts"],
        }, ensure_ascii=False, indent=2))
        return 0 if result["completed"] else 2
    except Exception as exc:
        print(f"Sweep fase 2: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
