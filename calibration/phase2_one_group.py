"""One-candidate exploratory Phase 2, with measured GPU-KV context frontier.

No operational profile is generated. Every trial runs in an isolated worker.
"""
import argparse
import json
from pathlib import Path

from model import ModelManager
from model.inspect import inspect_gguf
from .mapping_grouping import group_inventory, native_group_trial
from .mapping_phase import (
    BASELINE_MODE, PROBE_GENERATED_TOKENS, probe_native_mapping,
)


def run_one(model_path: Path, *, model_id: str = "gemma-2-9b-it",
            reference_ctx: int = 257, group_id: str | None = None,
            tokens: int = PROBE_GENERATED_TOKENS,
            timeout: int = 300) -> dict:
    """Find maximum empirically executable n_ctx, then compare host KV."""
    if reference_ctx < 64 or tokens < 1 or tokens > 128:
        raise ValueError("Parametri di contesto/generazione non validi")
    model_path = model_path.expanduser().resolve()
    layout = inspect_gguf(model_path, n_ctx=reference_ctx)
    arch = layout["metadata"]["general.architecture"]
    limit = layout["metadata"][f"{arch}.context_length"]
    groups = [g for g in group_inventory(layout) if g["eligible_for_trial"]]
    group = next((g for g in groups if g["id"] == group_id), None) if group_id else groups[0]
    if group is None:
        raise ValueError("Gruppo non disponibile o non idoneo")
    candidate = native_group_trial(layout, group)
    baseline = native_group_trial(layout, None)
    baseline["id"] = BASELINE_MODE
    report = {
        "phase": "single_group_extended_context", "model_id": model_id,
        "reference_n_ctx": reference_ctx, "model_limit": limit,
        "group_id": group["id"], "group_gguf_mib": group["gguf_unique_weight_mib"],
        "host_tensor_count": candidate["host_tensor_count"],
        "tokens_to_generate": tokens, "candidate_probes": [],
        "extended_n_ctx": None, "candidate": None, "baseline": None,
        "comparison": None, "success": False,
        "note": "Massimo empirico della configurazione fissa, non massimo teorico.",
    }

    def probe(ctx):
        row = probe_native_mapping(model_path, model_id, candidate,
                                   n_ctx=ctx, tokens=tokens, timeout=timeout)
        report["candidate_probes"].append({
            "n_ctx": ctx, "success": row.get("success"),
            "failure_class": row.get("failure_class"),
            "worker_signal": row.get("worker_signal"),
            "error": row.get("error"),
            "measurements": row.get("measurements"),
        })
        return row

    def boundary(row):
        return (row.get("failure_class") in (
            "native_cuda_oom", "native_cuda_resource_allocation_failure",
        ) or row.get("worker_signal") == "SIGABRT")

    # 257 is both our measured Phase-1 reference and the first candidate.
    low = reference_ctx
    first = probe(low)
    if not first.get("success"):
        report["error"] = "Il candidato non esegue il contesto iniziale."
        report["candidate"] = first
        return report
    best = first
    high = None
    while low < limit:
        ctx = min(limit, max(low + 1, low * 2))
        trial = probe(ctx)
        if trial.get("success"):
            best, low = trial, ctx
        elif boundary(trial):
            high = ctx
            break
        else:
            report["error"] = ("Frontiera inconclusiva: " +
                               str(trial.get("failure_class")))
            return report
    if high is not None:
        while high - low > 1:
            mid = (low + high) // 2
            trial = probe(mid)
            if trial.get("success"):
                best, low = trial, mid
            elif boundary(trial):
                high = mid
            else:
                report["error"] = ("Ricerca inconclusiva: " +
                                   str(trial.get("failure_class")))
                return report
    report["extended_n_ctx"] = low
    # Re-run in a fresh worker to confirm native placement and stable execution.
    confirmation = probe(low)
    if not confirmation.get("success"):
        report["error"] = "Conferma del contesto massimo fallita."
        return report
    report["candidate"] = confirmation
    if low <= reference_ctx:
        report["error"] = "Nessun aumento di contesto verificato."
        return report

    control = probe_native_mapping(
        model_path, model_id, baseline,
        n_ctx=low, tokens=tokens, timeout=timeout,
    )
    report["baseline"] = control
    if not control.get("success"):
        report["error"] = ("Baseline con KV RAM non valida: " +
                           str(control.get("failure_class")))
        return report
    a = confirmation["measurements"]
    b = control["measurements"]
    report["comparison"] = {
        "same_n_ctx": low, "prompt_tokens": low - tokens,
        "generated_tokens": tokens,
        "prefill_tps_cpu_group_gpu_kv": a["prefill_tokens_per_s"],
        "prefill_tps_all_gpu_weights_cpu_kv": b["prefill_tokens_per_s"],
        "decode_tps_cpu_group_gpu_kv": a["decode_tokens_per_s"],
        "decode_tps_all_gpu_weights_cpu_kv": b["decode_tokens_per_s"],
        "prefill_change_percent": round(
            100 * (a["prefill_tokens_per_s"] / b["prefill_tokens_per_s"] - 1), 3),
        "decode_change_percent": round(
            100 * (a["decode_tokens_per_s"] / b["decode_tokens_per_s"] - 1), 3),
        "candidate_vram_peak_mib": a["vram_peak_sampled_mib"],
        "baseline_vram_peak_mib": b["vram_peak_sampled_mib"],
    }
    report["success"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description="Ruth: singolo gruppo Phase 2")
    parser.add_argument("--model", default="gemma-2-9b-it")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--group-id", default=None)
    parser.add_argument("--reference-ctx", type=int, default=257)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("phase2_one_group.json"))
    args = parser.parse_args()
    try:
        model_path = args.model_path
        if model_path is None:
            selection = ModelManager().selected_baseline(args.model)
            if not selection:
                raise RuntimeError("Baseline GGUF mancante: passare --model-path")
            model_path = Path(selection["local_path"])
        result = run_one(
            model_path, model_id=args.model, reference_ctx=args.reference_ctx,
            group_id=args.group_id, tokens=args.tokens, timeout=args.timeout,
        )
    except Exception as exc:
        result = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
