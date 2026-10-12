"""Bounded CUDA context frontier measured with native-verified tensor mappings.

For each explicitly requested diagnostic context, try decreasing GPU offload
until both native baseline and dual tensor promotion complete repeated
ABBA/BAAB trials. A CUDA OOM *suspected from native evidence* allows the
next lower offload attempt. Unrelated worker/verification failures abort:
we do not silently mistake broken tests for insufficient VRAM.

This finds the highest *measured feasible* offload in the probed range,
NOT an operational n_ctx, a thermal guarantee, or a safe VRAM margin.
"""

import argparse
import json
import math

from hardware import GPUManager
from model import ModelManager

from .benchmark import benchmark, validate_probe
from .mapping_candidates import propose_mapping_candidates
from .mapping_composition import compose_candidates
from .mapping_stability import schedule, summarize_stability
from .measured_combinations import choose_compound_trials


def parse_contexts(values: list[str], tokens: int = 256) -> list[dict]:
    if not values or len(values) > 5:
        raise ValueError("Specificare 1..5 contesti nel formato n_ctx:prompt_repeat.")
    contexts = []
    used = set()
    for value in values:
        parts = value.split(":")
        if len(parts) != 2 or not all(part.isdecimal() for part in parts):
            raise ValueError(f"Contesto diagnostico non valido: {value!r}")
        ctx, repeat = map(int, parts)
        validate_probe(ctx, repeat, tokens)
        if ctx in used:
            raise ValueError(f"Contesto ripetuto: {ctx}")
        used.add(ctx)
        contexts.append({
            "probe_n_ctx": ctx,
            "prompt_repeat": repeat,
        })
    return contexts


def classify_trial(out: dict, candidate: bool, expected_ctx: int,
                   tokens: int) -> tuple[bool, str | None]:
    """Strictly distinguish allocation evidence from every other failure."""
    if out.get("success") is not True:
        if out.get("suspected_cuda_oom") is True:
            return False, "suspected_cuda_oom"
        # This exception is intentionally separate from an OOM: llama.cpp
        # context initialization can fail for more than one reason.
        if (out.get("failure_hint") == "ambiguous_context_initialization"
            or (out.get("worker_returncode") == 0
                and "failed to create llama_context" in
                str(out.get("error") or "").lower())):
            return False, "ambiguous_context_initialization"
        return False, "worker_failure"
    if candidate and (out.get("mapping_verification") or {}).get("verified") is not True:
        return False, "native_mapping_mismatch"
    mem = out.get("measurements") or {}
    windows = mem.get("decode_windows") or {}
    if (out.get("probe_n_ctx") != expected_ctx
        or mem.get("generated_tokens") != tokens
        or not isinstance(mem.get("prompt_tokens"), int)
        or not isinstance(windows.get("windows"), list)
        or len(windows["windows"]) != 4):
        return False, "incomplete_diagnostic_measurement"
    return True, None


def summarize_frontier(cases: list[dict], total_mib: float) -> dict:
    if not isinstance(total_mib, (int, float)) or not math.isfinite(total_mib) or total_mib <= 0:
        raise ValueError("Capacita' GPU invalida.")
    selected = []
    for context in cases:
        successful = [a for a in context["attempts"] if a["complete"]]
        if len(successful) > 1:
            raise ValueError("Una sola configurazione puo' essere confermata per contesto.")
        result = successful[0] if successful else None
        selected.append({
            "probe_n_ctx": context["probe_n_ctx"],
            "prompt_repeat": context["prompt_repeat"],
            "highest_tested_feasible_gpu_layers": result["gpu_layers"] if result else None,
            "full_abba_baab_comparison": result["comparison"] if result else None,
            "sampled_candidate_headroom_mib": (
                result["comparison"]["sampled_candidate_headroom_mib"] if result else None
            ),
            "attempts": [{
                "gpu_layers": a["gpu_layers"],
                "complete": a["complete"],
                "failure_class": a["failure_class"],
                "trial_count": len(a["trials"]),
            } for a in context["attempts"]],
        })
    return {
        "contexts": selected,
        "all_contexts_measured_feasible": all(
            s["highest_tested_feasible_gpu_layers"] is not None for s in selected
        ),
        "descriptive_only": True,
        "profile_generated": False,
        "selected_n_ctx": None,
        "note": (
            "La fattibilita' rilevata in worker isolati non e' una garanzia "
            "operativa e non stabilisce un margine di sicurezza. Lo sweep "
            "non interpola la memoria KV e non misura una conversazione reale."
        ),
    }


def explore_frontier(contexts: list[dict], initial_gpu_layers: int = 39,
                     minimum_gpu_layers: int = 36, rounds: int = 2,
                     tokens: int = 256, timeout: int = 180,
                     retry_ambiguous_context: bool = False) -> dict:
    contexts = parse_contexts([
        f"{item['probe_n_ctx']}:{item['prompt_repeat']}" for item in contexts
    ], tokens)
    if (not isinstance(initial_gpu_layers, int)
        or not isinstance(minimum_gpu_layers, int)
        or not 1 <= minimum_gpu_layers <= initial_gpu_layers <= 256
        or not isinstance(timeout, int) or timeout < 1
        or not isinstance(retry_ambiguous_context, bool)):
        raise ValueError("Offload iniziale/finale o timeout non valido.")
    plan = schedule(rounds)
    gpu = GPUManager.calibration_status()
    if not gpu["ready"]:
        raise RuntimeError(f"GPU non pronta: {gpu['errors']}; {gpu['foreign_processes']}")
    total_mib = (gpu.get("memory") or {}).get("total_mib")
    if not isinstance(total_mib, (int, float)) or total_mib <= 0:
        raise RuntimeError("Totale VRAM NVML non disponibile.")

    report = {
        "phase": "adaptive_diagnostic_cuda_context_frontier",
        "model": "gemma-2-9b-it",
        "backend": "cuda",
        "initial_gpu_layers": initial_gpu_layers,
        "minimum_gpu_layers": minimum_gpu_layers,
        "rounds": rounds,
        "schedule": plan,
        "tokens": tokens,
        "gpu_total_mib": total_mib,
        "contexts": [],
        "summary": None,
        "success": False,
        "measurement_completed": False,
        "retry_ambiguous_context": retry_ambiguous_context,
        "profile_generated": False,
        "selected_n_ctx": None,
        "note": (
            "Prova fino a trovare un offload misurato fattibile per ogni "
            "contesto diagnostico. Riduzione dell'offload dopo indizi "
            "espliciti di OOM; facoltativamente, con flag dedicato, anche "
            "dopo un errore ambiguo di creazione del contesto, senza "
            "considerarlo OOM confermato. Nessuna soglia VRAM inventata. "
            "Nessuna selezione operativa."
        ),
    }
    for spec in contexts:
        ctx, repeat = spec["probe_n_ctx"], spec["prompt_repeat"]
        context_report = {**spec, "attempts": [], "status": "not_feasible_in_range"}
        report["contexts"].append(context_report)
        for layers in range(initial_gpu_layers, minimum_gpu_layers - 1, -1):
            proposal = propose_mapping_candidates(
                ModelManager().inspect("gemma-2-9b-it", n_ctx=ctx), layers
            )
            seed = next(
                (c for c in choose_compound_trials(proposal)
                 if c["id"] == "dual_promotion"), None
            )
            if seed is None:
                report["error"] = f"Mapping seed non disponibile a gpu_layers={layers}"
                return report
            compose_candidates(proposal, seed["component_ids"])

            attempt = {
                "gpu_layers": layers,
                "component_ids": seed["component_ids"],
                "trials": [],
                "complete": False,
                "failure_class": None,
                "comparison": None,
            }
            context_report["attempts"].append(attempt)
            can_retry_lower = False
            for run, order in enumerate(plan, start=1):
                for position, variant in enumerate(order, start=1):
                    status = GPUManager.calibration_status()
                    if not status["ready"]:
                        attempt["failure_class"] = "gpu_status_changed"
                        report["error"] = (
                            f"GPU non pronta: {status['errors']}; {status['foreign_processes']}"
                        )
                        return report
                    is_candidate = variant == "candidate"
                    out = benchmark(
                        gpu_layers=layers, max_tokens=tokens, timeout=timeout,
                        probe_n_ctx=ctx, prompt_repeat=repeat,
                        combination_ids=seed["component_ids"] if is_candidate else None,
                        collect_gpu_telemetry=True,
                    )
                    valid, failure = classify_trial(out, is_candidate, ctx, tokens)
                    mem = out.get("measurements") or {}
                    windows = mem.get("decode_windows") or {}
                    telem = (out.get("gpu_telemetry_by_phase") or {}).get("decode") or {}
                    row = {
                        "round": run, "position": position, "variant": variant,
                        "success": valid, "failure_class": failure,
                        "mapping_verified": (
                            (out.get("mapping_verification") or {}).get("verified")
                            if is_candidate else None
                        ),
                        "requested_tokens": tokens,
                        "generated_tokens": mem.get("generated_tokens"),
                        "prompt_tokens": mem.get("prompt_tokens"),
                        "actual_probe_n_ctx": out.get("probe_n_ctx"),
                        "decode_tps": mem.get("decode_tokens_per_second"),
                        "prefill_tps": mem.get("prefill_tokens_per_second"),
                        "vram_peak_mib": mem.get("vram_peak_sampled_mib"),
                        "decode_window_details": windows.get("windows"),
                        "final_context_token_fraction": windows.get("end_context_fraction"),
                        "gpu_decode_temperature_median_c": (
                            (telem.get("gpu_temperature_c") or {}).get("median")
                        ),
                        "gpu_decode_sm_clock_median_mhz": (
                            (telem.get("sm_clock_mhz") or {}).get("median")
                        ),
                        "worker_returncode": out.get("worker_returncode"),
                        "failure_type": out.get("failure_type"),
                        "suspected_cuda_oom": out.get("suspected_cuda_oom", False),
                        "failure_hint": out.get("failure_hint"),
                        "native_evidence_available": out.get("native_evidence_available"),
                        "cuda_diagnostics": out.get("cuda_diagnostics"),
                        "stderr_tail": out.get("stderr_tail"),
                        "error": out.get("error"),
                    }
                    attempt["trials"].append(row)
                    if not valid:
                        attempt["failure_class"] = failure
                        if (failure == "suspected_cuda_oom"
                            or (failure == "ambiguous_context_initialization"
                                and retry_ambiguous_context)):
                            # Ambiguous failures stay explicitly marked as
                            # such in attempt/trial evidence and summaries.
                            can_retry_lower = True
                            break
                        report["error"] = (
                            f"Errore non-OOM a ctx={ctx}, gpu_layers={layers}, "
                            f"round={run}, posizione={position}, causa={failure}: "
                            f"{out.get('error') or out.get('failure_type')}. "
                            "La causa non e' attribuibile alla VRAM senza "
                            "evidenze CUDA native."
                        )
                        report["summary"] = summarize_frontier(
                            report["contexts"], total_mib
                        )
                        return report
                if attempt["failure_class"] is not None:
                    break
            if can_retry_lower:
                continue
            attempt["comparison"] = summarize_stability(
                attempt["trials"], rounds, total_mib
            )
            attempt["complete"] = True
            context_report["status"] = "diagnostically_feasible"
            break
    report["summary"] = summarize_frontier(report["contexts"], total_mib)
    report["measurement_completed"] = True
    report["success"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ricerca adattiva bounded context/offload basata su CUDA NVML"
    )
    parser.add_argument("--context", action="append", default=None,
                        help="Ripeti n_ctx:prompt_repeat per contesto diagnostico")
    parser.add_argument("--initial-gpu-layers", type=int, default=39)
    parser.add_argument("--minimum-gpu-layers", type=int, default=36)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retry-ambiguous-context", action="store_true",
                        help="Prova offload inferiori anche se llama_context fallisce "
                             "senza prova esplicita di OOM; conserva il risultato ambiguo")
    args = parser.parse_args()
    try:
        contexts = parse_contexts(
            args.context or ["1536:24", "2048:34"], args.tokens
        )
        result = explore_frontier(
            contexts, args.initial_gpu_layers, args.minimum_gpu_layers,
            args.rounds, args.tokens, args.timeout,
            args.retry_ambiguous_context,
        )
    except Exception as exc:
        result = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
