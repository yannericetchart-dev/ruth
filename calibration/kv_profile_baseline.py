"""Native all-GPU KV baseline at 256, with optional gated promotion to 512.

No tensor is moved to CPU. The ordinary mapping-phase BASELINE_MODE is NOT
used because that mode places KV on host! An empty native_tensor_trial maps
all weights to GPU AND retains the GPU KV via offload_kqv=True.

The f16/f16 census is used only to identify the model and, after a successful
alternative KV baseline, as a heuristic candidate order. It never proves
VRAM savings on another KV profile.
"""
import argparse
import hashlib
import json
from pathlib import Path

from model.inspect import inspect_gguf

from .gpu_context import context_limit, validate_kv_config
from .mapping_grouping import native_tensor_trial
from .mapping_phase import probe_native_mapping
from .kv_adaptive_tensor_offload import (
    _probe_summary, summarize_probe_repeats,
)
from .kv_single_tensor_census import _native_verified
from .kv_single_to_512 import CENSUS_SCHEMA, run_single_to_512
from .profile_pipeline import atomic_json


SCHEMA = "ruth-phase2-kv-baseline-native-v1"


def run_kv_profile_baseline(
    *, census_path: Path, output: Path,
    n_ctx: int = 256, flash_attn: bool = False,
    type_k: str = "q8_0", type_v: str = "f16",
    benchmark_repeats: int = 2, timeout: int = 600,
    probe=probe_native_mapping, progress=None,
) -> dict:
    validate_kv_config(flash_attn, type_k, type_v)
    if (type(n_ctx) is not int or n_ctx < 64
        or type(benchmark_repeats) is not int
        or not 2 <= benchmark_repeats <= 10
        or type(timeout) is not int or timeout < 1):
        raise ValueError("Parametri baseline KV non validi.")
    census_path = Path(census_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if census_path == output:
        raise ValueError("Baseline e census richiedono report distinti.")
    raw = census_path.read_bytes()
    census = json.loads(raw)
    params = census.get("parameters") or {}
    if (census.get("schema") != CENSUS_SCHEMA
        or census.get("completed") is not True
        or type(params.get("n_ctx")) is not int
        or n_ctx < params["n_ctx"]
        or params.get("flash_attn") is not False
        or params.get("type_k") != "f16"
        or params.get("type_v") != "f16"):
        raise ValueError("Richiesto censimento completo f16/f16 FA OFF a 256.")
    model_path = Path(params["model_path"]).expanduser().resolve()
    model_id = params["model_id"]
    generated_tokens = params["generated_tokens"]
    layout = inspect_gguf(model_path, n_ctx=n_ctx)
    if n_ctx > context_limit(layout):
        raise ValueError("Contesto oltre il limite GGUF.")
    trial = native_tensor_trial(layout, [])
    if trial["host_tensor_names"]:
        raise RuntimeError("La baseline non puo usare tensori CPU.")
    identity = {
        "census_sha256": hashlib.sha256(raw).hexdigest(),
        "model_id": model_id,
        "model_path": str(model_path),
        "n_ctx": n_ctx,
        "generated_tokens": generated_tokens,
        "flash_attn": flash_attn,
        "type_k": type_k,
        "type_v": type_v,
        "benchmark_repeats": benchmark_repeats,
        "timeout": timeout,
    }
    previous = {}
    if output.exists():
        previous = json.loads(output.read_text("utf-8"))
        if (previous.get("schema") != SCHEMA
            or previous.get("parameters") != identity
            or previous.get("mapping_id") != trial["id"]):
            raise ValueError("Checkpoint baseline KV incompatibile.")
    records = previous.get("samples") or []
    if (not isinstance(records, list)
        or len(records) > benchmark_repeats
        or any(not isinstance(row, dict) for row in records)):
        raise ValueError("Campioni baseline KV malformati.")
    report = {
        "schema": SCHEMA,
        "parameters": identity,
        "mapping_id": trial["id"],
        "host_tensor_names": [],
        "kv_target": "gpu",
        "samples": records,
        "completed": False,
        "success": False,
        "status": "partial_checkpoint",
        "inference": None,
        "note": (
            "Probe all-GPU nativo a 256 con offload_kqv=True; "
            "il profiling f16/f16 NON e una misura del KV alternativo. "
            "Una mancata conferma non dimostra impossibilita in generale."
        ),
    }

    def save():
        aggregate = summarize_probe_repeats(records, benchmark_repeats)
        aggregate.update({
            "n_ctx": n_ctx, "host_tensor_names": [],
            "mapping_id": trial["id"],
        })
        verified = aggregate["success"] and _native_verified(
            aggregate, n_ctx=n_ctx, names=[], trial_id=trial["id"],
            repeats=benchmark_repeats, tokens=generated_tokens,
        )
        if aggregate["success"] and not verified:
            aggregate["success"] = False
            aggregate["failure_class"] = "unverified_native_gpu_kv_baseline"
            aggregate.pop("metrics", None)
        report["inference"] = aggregate
        report["success"] = bool(verified)
        report["completed"] = (
            bool(records and not records[-1]["success"])
            or len(records) == benchmark_repeats
        )
        report["status"] = (
            "verified_gpu_kv_baseline" if report["success"]
            else "native_baseline_not_verified" if report["completed"]
            else "partial_checkpoint"
        )
        atomic_json(output, report)

    save()
    while not report["completed"] and len(records) < benchmark_repeats:
        try:
            native = probe(
                model_path, model_id, trial,
                n_ctx=n_ctx, tokens=generated_tokens, timeout=timeout,
                flash_attn=flash_attn, type_k=type_k, type_v=type_v,
            )
            row = _probe_summary(native)
        except Exception as exc:
            row = {
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "failure_class": "baseline_probe_exception",
            }
        records.append(row)
        save()
        if progress:
            progress({
                "event": "baseline_probe", "sample": len(records),
                "success": row["success"],
                "failure_class": row.get("failure_class"),
            })
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Verifica all-GPU KV q8_0/f16 FA OFF a 256 e, opzionalmente, promuove a 512."
    )
    parser.add_argument("--census", type=Path,
                        default=Path("phase2_single_tensor_census_256.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("phase2_q8_0_f16_faoff_baseline_256.json"))
    parser.add_argument("--n-ctx", type=int, default=256)
    parser.add_argument("--flash-attn", choices=("on", "off"), default="off")
    parser.add_argument("--type-k", choices=("f16", "q8_0", "q4_0"),
                        default="q8_0")
    parser.add_argument("--type-v", choices=("f16", "q8_0", "q4_0"),
                        default="f16")
    parser.add_argument("--benchmark-repeats", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--promote-if-valid", action="store_true")
    parser.add_argument("--target-ctx", type=int, default=512)
    parser.add_argument("--promotion-output", type=Path,
                        default=Path("phase2_single_to_512_q8_0_f16_faoff.json"))
    parser.add_argument("--target-control-output", type=Path,
                        default=Path("phase2_q8_0_f16_faoff_allgpu_512.json"))
    parser.add_argument("--max-new-loads", type=int)
    parser.add_argument("--max-new-benchmarks", type=int)
    args = parser.parse_args(argv)
    progress = lambda row: print(json.dumps(row, ensure_ascii=False), flush=True)
    try:
        outcome = run_kv_profile_baseline(
            census_path=args.census, output=args.output,
            n_ctx=args.n_ctx, flash_attn=args.flash_attn == "on",
            type_k=args.type_k, type_v=args.type_v,
            benchmark_repeats=args.benchmark_repeats,
            timeout=args.timeout, progress=progress,
        )
        print(json.dumps({
            "status": outcome["status"],
            "success": outcome["success"],
            "output": str(args.output),
            "failure_class": (
                (outcome.get("inference") or {}).get("failure_class")
            ),
        }, ensure_ascii=False, indent=2), flush=True)
        if not outcome["success"]:
            # Exit without promotion; this is not evidence that no future
            # alternative or change of execution environment can work.
            return 2
        if not args.promote_if_valid:
            return 0
        # First check whether full-GPU weights + GPU KV ALREADY reaches
        # the target; that would require no tensor offloading at all.
        # A target failure is not a rejection of CPU-offload promotion.
        target_control = run_kv_profile_baseline(
            census_path=args.census, output=args.target_control_output,
            n_ctx=args.target_ctx, flash_attn=args.flash_attn == "on",
            type_k=args.type_k, type_v=args.type_v,
            benchmark_repeats=args.benchmark_repeats,
            timeout=args.timeout, progress=progress,
        )
        print(json.dumps({
            "target_all_gpu_status": target_control["status"],
            "target_all_gpu_success": target_control["success"],
            "target_all_gpu_output": str(args.target_control_output),
        }, ensure_ascii=False, indent=2), flush=True)
        if target_control["success"]:
            print(
                "Contesto target gia promosso senza tensori CPU: "
                "nessun offloading necessario per questo n_ctx.",
                flush=True,
            )
            return 0
        promoted = run_single_to_512(
            census_path=args.census, output=args.promotion_output,
            target_ctx=args.target_ctx,
            flash_attn=args.flash_attn == "on",
            type_k=args.type_k, type_v=args.type_v,
            baseline_profile_path=args.output,
            benchmark_repeats=args.benchmark_repeats,
            timeout=args.timeout,
            max_new_loads=args.max_new_loads,
            max_new_benchmarks=args.max_new_benchmarks,
            progress=progress,
        )
        print(json.dumps({
            "promotion_status": promoted["status"],
            "promotion_summary": promoted["summary"],
            "promotion_output": str(args.promotion_output),
        }, ensure_ascii=False, indent=2))
        return 0 if promoted["completed"] else 2
    except Exception as exc:
        print(f"KV baseline/promozione: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
