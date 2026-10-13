"""Complete the empirical SINGLE-tensor GPU-KV offload census.

Reuses every independently verified n_ctx=256 single-tensor observation from
an immutable greedy-v3 report. All missing block tensors are probed with the
EXACT same model, generated-token count, repetitions and KV configuration.
No memory-headroom threshold, tensor-family filter or predicted fit is used.

The original greedy report is never changed. Checkpoint saves atomically after
every attempted tensor and can be resumed with the same CLI.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from model.inspect import inspect_gguf

from .gpu_context import context_limit
from .kv_adaptive_tensor_offload import (
    _probe_summary, discovery_groups, marginal_value,
    summarize_probe_repeats,
)
from .mapping_grouping import native_tensor_trial
from .mapping_phase import probe_native_mapping
from .profile_pipeline import atomic_json


SCHEMA = "ruth-phase2-single-tensor-census-v1"
SOURCE_SCHEMA = "ruth-phase2-adaptive-tensor-greedy-v3"
METRIC_FIELDS = ("prefill_tok_s", "decode_tok_s",
                 "vram_peak_mib", "vram_total_mib")


def _finite_metrics(row: dict) -> bool:
    metric = row.get("metrics")
    return isinstance(metric, dict) and all(
        isinstance(metric.get(field), (int, float))
        and not isinstance(metric[field], bool)
        and math.isfinite(metric[field]) and metric[field] > 0
        for field in METRIC_FIELDS
    )


def _native_verified(row: dict, *, n_ctx: int, names: list[str],
                     trial_id: str, repeats: int, tokens: int) -> bool:
    """Check cached evidence, not merely the 'success' boolean."""
    if (row.get("success") is not True
        or row.get("n_ctx") != n_ctx
        or row.get("host_tensor_names") != sorted(names)
        or row.get("mapping_id") != trial_id
        or row.get("repetitions") != repeats
        or not _finite_metrics(row)):
        return False
    samples = row.get("samples")
    return isinstance(samples, list) and len(samples) == repeats and all(
        sample.get("success") is True
        and sample.get("native_placement_verified") is True
        and sample.get("gpu_kv_verified") is True
        and sample.get("native_kv_type_verified") is True
        and sample.get("prompt_tokens") == n_ctx - tokens
        and sample.get("generated_tokens") == tokens
        and sample.get("total_tokens") == n_ctx
        and not sample.get("nvml_errors")
        for sample in samples
    )


def _tensor_inventory(layout: dict) -> list[str]:
    """Unique physical block tensors, no output/embedding weights."""
    groups = discovery_groups(layout)
    inventory = sorted({
        name for group in groups for name in group["tensor_names"]
    })
    if not inventory:
        raise ValueError("GGUF non contiene tensori di blocco idonei.")
    return inventory


def _from_existing(source: dict, *, control_ctx: int,
                   generated_tokens: int, repeats: int,
                   layout: dict, available: list[str]) -> tuple[dict, dict]:
    allowed = set(available)
    all_gpu_id = native_tensor_trial(layout, [])["id"]
    prior = {}
    for row in (source.get("measured_configs") or {}).values():
        if not isinstance(row, dict) or row.get("n_ctx") != control_ctx:
            continue
        names = row.get("host_tensor_names")
        if not isinstance(names, list):
            continue
        if len(names) == 0:
            if _native_verified(row, n_ctx=control_ctx, names=[],
                                trial_id=all_gpu_id, repeats=repeats,
                                tokens=generated_tokens):
                prior["__all_gpu__"] = row
            continue
        if len(names) != 1 or names[0] not in allowed:
            continue
        name = names[0]
        trial = native_tensor_trial(layout, [name])
        if _native_verified(row, n_ctx=control_ctx, names=[name],
                            trial_id=trial["id"], repeats=repeats,
                            tokens=generated_tokens):
            if name in prior and prior[name] != row:
                raise ValueError("Misure sorgente duplicate incoerenti: " + name)
            prior[name] = row
    if "__all_gpu__" not in prior:
        raise ValueError(
            "Baseline all-GPU a n_ctx scelto priva di due prove native "
            "valide: impossibile confrontare i singoli tensori."
        )
    control = prior.pop("__all_gpu__")
    return control, prior


def run_single_tensor_census(
    *, source_report: Path, output: Path,
    max_new_tests: int | None = None, retry_failures: bool = False,
    timeout: int = 600, progress=None,
    probe=probe_native_mapping,
) -> dict:
    """Measure all as-yet-unmeasured tensors; resume without repeat CUDA."""
    if (max_new_tests is not None
        and (type(max_new_tests) is not int or max_new_tests < 1)):
        raise ValueError("--max-new-tests deve essere positivo.")
    if type(timeout) is not int or timeout < 1:
        raise ValueError("--timeout non valido.")

    source_path = Path(source_report).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if source_path == output:
        raise ValueError("Il census non puo sovrascrivere il report greedy.")
    raw = source_path.read_bytes()
    source = json.loads(raw)
    if source.get("schema") != SOURCE_SCHEMA:
        raise ValueError("Richiesto report greedy v3.")
    params = source.get("parameters") or {}
    ctx = source.get("actual_profiling_start_ctx")
    tokens = params.get("generated_tokens")
    repeats = params.get("scouting_repeats")
    if (type(ctx) is not int or ctx < 64
        or type(tokens) is not int or tokens < 1 or tokens >= ctx
        or type(repeats) is not int or not 1 <= repeats <= 10):
        raise ValueError("Contesto di profiling o ripetizioni non verificabili.")
    if not (params.get("baseline_placement_protocol")
            == "exact_phase1_with_native_kv_step_fallback_v3"):
        raise ValueError("Protocollo native GPU-KV incompatibile.")
    if source.get("status") != "confirmed_all_requested_contexts":
        raise ValueError("Report greedy non confermato.")
    stages = source.get("stages") or []
    if not any(s.get("successful") is True
               and s.get("from_n_ctx") == ctx for s in stages):
        raise ValueError("Manca una fase confermata dall'anchor del census.")

    model_path = Path(params["model_path"]).expanduser().resolve()
    model_id = params["model_id"]
    layout = inspect_gguf(model_path, n_ctx=ctx)
    if ctx > context_limit(layout):
        raise ValueError("Contesto superiore al limite GGUF.")
    names = _tensor_inventory(layout)
    control, reused = _from_existing(
        source, control_ctx=ctx, generated_tokens=tokens,
        repeats=repeats, layout=layout, available=names,
    )
    identity = {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "model_path": str(model_path), "model_id": model_id,
        "n_ctx": ctx, "generated_tokens": tokens,
        "repetitions": repeats,
        "flash_attn": False, "type_k": "f16", "type_v": "f16",
        "timeout": timeout,
        "inventory_sha256": hashlib.sha256(
            json.dumps(names, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    }

    measured = {}
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("schema") != SCHEMA or previous.get("parameters") != identity:
            raise ValueError(
                "Checkpoint incompatibile: usare un altro --output per "
                "un nuovo esperimento. Non sovrascrivere quello esistente."
            )
        measured = previous.get("new_measurements")
        if not isinstance(measured, dict):
            raise ValueError("Checkpoint misure non valido.")
        for name, row in measured.items():
            if name not in names or not isinstance(row, dict):
                raise ValueError("Checkpoint con nome tensore inatteso.")
            expected = native_tensor_trial(layout, [name])["id"]
            if (row.get("n_ctx") != ctx or row.get("mapping_id") != expected
                or row.get("host_tensor_names") != [name]
                or (row.get("success") is True and not _native_verified(
                    row, n_ctx=ctx, names=[name], trial_id=expected,
                    repeats=repeats, tokens=tokens
                ))):
                raise ValueError("Checkpoint con evidenza non valida: " + name)

    report = {
        "schema": SCHEMA, "parameters": identity,
        "status": "in_progress", "completed": False,
        "baseline_control": control,
        "eligible_tensor_count": len(names),
        "reused_from_greedy": {
            name: reused[name] for name in sorted(reused)
        },
        "new_measurements": measured,
        "summary": {},
        "tensor_results": {},
        "note": (
            "Profilazione singola a contesto identico, FA OFF e KV FP16 GPU; "
            "nessun vincolo di VRAM libera. I tentativi falliti restano nel "
            "checkpoint e richiedono --retry-failures per una nuova prova."
        ),
    }

    def rebuild():
        results = {}
        for name in names:
            row = reused.get(name) or measured.get(name)
            if row is None:
                continue
            origin = "greedy_v3" if name in reused else "new_probe"
            trial_id = native_tensor_trial(layout, [name])["id"]
            valid = _native_verified(
                row, n_ctx=ctx, names=[name],
                trial_id=trial_id, repeats=repeats, tokens=tokens,
            )
            results[name] = {
                "source": origin, "success": valid,
                "failure_class": None if valid else row.get("failure_class"),
                "measurement": row,
                "marginal": (marginal_value(
                    control, row, min_gain_mib=0.0
                ) if valid else None),
            }
        good = sum(r["success"] for r in results.values())
        failures = sorted(n for n, r in results.items() if not r["success"])
        missing = sorted(set(names) - set(results))
        report["summary"] = {
            "total_eligible": len(names),
            "source_reused_success": len(reused),
            "new_success": sum(
                r["success"] and r["source"] == "new_probe"
                for r in results.values()
            ),
            "verified_total": good,
            "failed_count": len(failures),
            "missing_count": len(missing),
            "new_tests_attempted": len(measured),
            "failed_tensors": failures,
            "missing_tensors": missing,
        }
        report["tensor_results"] = results
        report["completed"] = good == len(names)
        report["status"] = (
            "confirmed_all_single_tensors" if report["completed"]
            else "partial_failures" if failures and not missing
            else "partial_incomplete"
        )
        atomic_json(output, report)

    rebuild()
    fresh_count = 0
    for idx, name in enumerate(names, start=1):
        if name in reused:
            continue
        if name in measured and (
            _native_verified(measured[name], n_ctx=ctx, names=[name],
                             trial_id=native_tensor_trial(layout, [name])["id"],
                             repeats=repeats, tokens=tokens)
            or not retry_failures
        ):
            continue
        if max_new_tests is not None and fresh_count >= max_new_tests:
            break
        trial = native_tensor_trial(layout, [name])
        samples = []
        for _ in range(repeats):
            try:
                result = probe(
                    model_path, model_id, trial, n_ctx=ctx,
                    tokens=tokens, timeout=timeout, flash_attn=False,
                    type_k="f16", type_v="f16",
                )
                sample = _probe_summary(result)
            except Exception as exc:
                sample = {
                    "success": False, "failure_class": "probe_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            samples.append(sample)
            if not sample["success"]:
                break
        summary = summarize_probe_repeats(samples, repeats)
        summary["n_ctx"] = ctx
        summary["host_tensor_names"] = [name]
        summary["mapping_id"] = trial["id"]
        if summary["success"]:
            speed = summary["metrics"]
            speed["combined_time_s"] = round(
                (ctx - tokens) / speed["prefill_tok_s"]
                + tokens / speed["decode_tok_s"], 6
            )
        measured[name] = summary
        fresh_count += 1
        rebuild()
        if progress:
            progress({
                "event": "single_tensor_measured", "tensor": name,
                "index": idx, "total_tensors": len(names),
                "success": report["tensor_results"][name]["success"],
                "failure_class": summary.get("failure_class"),
                "new_tests_this_run": fresh_count,
                "verified_total": report["summary"]["verified_total"],
                "remaining": report["summary"]["missing_count"],
            })
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Profiling di tutti i tensori di blocco, uno per volta, KV GPU."
    )
    parser.add_argument("--source-report", type=Path,
                        default=Path("phase2_adaptive_tensor_groups_v3.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("phase2_single_tensor_census_256.json"))
    parser.add_argument("--max-new-tests", type=int, default=None,
                        help="Opzionale: massimo numero di nuovi tensori per esecuzione.")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    try:
        report = run_single_tensor_census(
            source_report=args.source_report, output=args.output,
            max_new_tests=args.max_new_tests,
            retry_failures=args.retry_failures, timeout=args.timeout,
            progress=lambda item: print(
                json.dumps(item, ensure_ascii=False), flush=True
            ),
        )
        print(json.dumps({
            "status": report["status"],
            "summary": report["summary"],
            "output": str(args.output),
        }, ensure_ascii=False, indent=2))
        return 0 if report["completed"] else 2
    except Exception as exc:
        print("Census: " + type(exc).__name__ + ": " + str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
