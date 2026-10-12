"""Observed VRAM and performance envelope across diagnostic CUDA configurations.

Each case is a (GPU offload count, *probe* context, prompt repetition).
Tests baseline before its native-verified dual promotion, with paired rounds.
The sweep does not infer missing VRAM from GGUF byte counts, select n_ctx, or
change production settings. Failed allocations stay recorded as observations.
"""

import argparse
import json
import math
import statistics

from hardware import GPUManager
from model import ModelManager

from .benchmark import benchmark, validate_probe
from .mapping_candidates import propose_mapping_candidates
from .mapping_composition import compose_candidates
from .measured_combinations import choose_compound_trials


def parse_cases(values: list[str], tokens: int = 256) -> list[dict]:
    if not values or len(values) > 8:
        raise ValueError("Specificare da 1 a 8 casi: gpu_layers:n_ctx:prompt_repeat.")
    cases = []
    used = set()
    for value in values:
        fields = value.split(":")
        if len(fields) != 3 or not all(f.isdecimal() for f in fields):
            raise ValueError(f"Caso non valido {value!r}: gpu_layers:n_ctx:prompt_repeat.")
        layers, ctx, repeat = map(int, fields)
        validate_probe(ctx, repeat, tokens)
        if layers < 1 or layers > 256 or (layers, ctx, repeat) in used:
            raise ValueError(f"Caso duplicato o gpu_layers non valido: {value!r}.")
        used.add((layers, ctx, repeat))
        cases.append({
            "id": f"gpu{layers}:ctx{ctx}:repeat{repeat}",
            "gpu_layers": layers,
            "probe_n_ctx": ctx,
            "prompt_repeat": repeat,
        })
    return cases


def summarize_envelope(trials: list[dict], cases: list[dict],
                       rounds: int, gpu_total_mib: float) -> list[dict]:
    """All complete cases get paired deltas; incomplete ones keep failure data."""
    if not isinstance(rounds, int) or rounds < 1 or gpu_total_mib <= 0:
        raise ValueError("Dati per confronto non validi.")
    known = {c["id"] for c in cases}
    if len(known) != len(cases):
        raise ValueError("Casi duplicati.")
    indexed = {}
    for t in trials:
        key = (t.get("case_id"), t.get("round"), t.get("variant"))
        if (key[0] not in known or not isinstance(key[1], int)
            or key[1] not in range(1, rounds + 1)
            or key[2] not in ("baseline", "candidate")
            or key in indexed):
            raise ValueError("Prova duplicata, sconosciuta o round errato.")
        if t.get("success") is True:
            required = ("decode_tps", "prefill_tps", "vram_peak_mib")
            if any(
                not isinstance(t.get(name), (int, float))
                or isinstance(t[name], bool)
                or not math.isfinite(t[name]) or t[name] <= 0
                for name in required
            ):
                raise ValueError("Misura positiva mancante in una prova riuscita.")
            if t.get("generated_tokens") != t.get("requested_tokens"):
                raise ValueError("Campione decode non completo.")
            if t["variant"] == "candidate" and t.get("mapping_verified") is not True:
                raise ValueError("Mapping candidato non verificato nativamente.")
        indexed[key] = t

    summaries = []
    for case in cases:
        case_id = case["id"]
        tested = [t for t in trials if t["case_id"] == case_id]
        good_pairs = []
        for cycle in range(1, rounds + 1):
            base = indexed.get((case_id, cycle, "baseline"))
            candidate = indexed.get((case_id, cycle, "candidate"))
            if base is None or candidate is None:
                continue
            if not (base["success"] and candidate["success"]):
                continue
            first_base = base.get("decode_first_quarter_tps")
            first_candidate = candidate.get("decode_first_quarter_tps")
            last_base = base.get("decode_last_quarter_tps")
            last_candidate = candidate.get("decode_last_quarter_tps")
            good_pairs.append({
                "round": cycle,
                "first_quarter_delta_tps": (
                    round(first_candidate - first_base, 3)
                    if isinstance(first_base, (int, float))
                    and isinstance(first_candidate, (int, float)) else None
                ),
                "last_quarter_delta_tps": (
                    round(last_candidate - last_base, 3)
                    if isinstance(last_base, (int, float))
                    and isinstance(last_candidate, (int, float)) else None
                ),
                "baseline_decode_tps": base["decode_tps"],
                "candidate_decode_tps": candidate["decode_tps"],
                "decode_delta_tps": round(
                    candidate["decode_tps"] - base["decode_tps"], 3
                ),
                "prefill_delta_tps": round(
                    candidate["prefill_tps"] - base["prefill_tps"], 3
                ),
                "vram_peak_delta_mib": round(
                    candidate["vram_peak_mib"] - base["vram_peak_mib"], 1
                ),
                "candidate_sampled_vram_headroom_mib": round(
                    gpu_total_mib - candidate["vram_peak_mib"], 1
                ),
            })
        successful = [t for t in tested if t["success"]]
        failures = [{
            "round": t["round"], "variant": t["variant"],
            "error": t.get("error"), "failure_type": t.get("failure_type"),
            "suspected_cuda_oom": t.get("suspected_cuda_oom", False),
        } for t in tested if not t["success"]]
        summaries.append({
            **case,
            "paired_rounds": len(good_pairs),
            "complete": len(good_pairs) == rounds,
            "successful_trials": len(successful),
            "failures": failures,
            "pairs": good_pairs,
            "median_decode_delta_tps": (
                round(statistics.median(p["decode_delta_tps"] for p in good_pairs), 3)
                if good_pairs else None
            ),
            "decode_faster_pairs": sum(p["decode_delta_tps"] > 0 for p in good_pairs),
            "last_quarter_positive_pairs": sum(
                p["last_quarter_delta_tps"] > 0 for p in good_pairs
                if p["last_quarter_delta_tps"] is not None
            ),
            "median_last_quarter_delta_tps": (
                round(statistics.median(p["last_quarter_delta_tps"]
                                        for p in good_pairs
                                        if p["last_quarter_delta_tps"] is not None), 3)
                if any(p["last_quarter_delta_tps"] is not None for p in good_pairs)
                else None
            ),
            "min_sampled_headroom_mib": (
                round(gpu_total_mib - max(t["vram_peak_mib"] for t in successful), 1)
                if successful else None
            ),
            "median_baseline_decode_tps": (
                round(statistics.median(
                    t["decode_tps"] for t in successful if t["variant"] == "baseline"
                ), 2) if any(t["variant"] == "baseline" for t in successful) else None
            ),
            "median_candidate_decode_tps": (
                round(statistics.median(
                    t["decode_tps"] for t in successful if t["variant"] == "candidate"
                ), 2) if any(t["variant"] == "candidate" for t in successful) else None
            ),
        })
    return summaries


def envelope(cases: list[dict], rounds: int = 2, tokens: int = 256,
             timeout: int = 180) -> dict:
    if not 1 <= rounds <= 4 or not 64 <= tokens <= 256 or timeout < 1:
        raise ValueError("rounds=1..4, tokens=64..256, timeout>=1.")
    raw = [f"{c['gpu_layers']}:{c['probe_n_ctx']}:{c['prompt_repeat']}" for c in cases]
    cases = parse_cases(raw, tokens)

    state = GPUManager.calibration_status()
    if not state["ready"]:
        raise RuntimeError(f"GPU non pronta: {state['errors']}; {state['foreign_processes']}")
    gpu_total = (state.get("memory") or {}).get("total_mib")
    if not isinstance(gpu_total, (int, float)) or gpu_total <= 0:
        raise RuntimeError("Totale VRAM NVML non disponibile.")

    definitions = {}
    for case in cases:
        proposals = propose_mapping_candidates(
            ModelManager().inspect("gemma-2-9b-it", n_ctx=case["probe_n_ctx"]),
            case["gpu_layers"],
        )
        seed = next(c for c in choose_compound_trials(proposals)
                    if c["id"] == "dual_promotion")
        compose_candidates(proposals, seed["component_ids"])
        definitions[case["id"]] = seed["component_ids"]

    result = {
        "phase": "mapping_diagnostic_context_memory_envelope",
        "model": "gemma-2-9b-it", "backend": "cuda",
        "cases": cases,
        "seed_by_case": definitions,
        "rounds": rounds, "tokens": tokens,
        "gpu_total_mib": gpu_total,
        "trials": [],
        "summary": [],
        "success": False,
        "measurement_completed": False,
        "all_cases_complete": False,
        "profile_generated": False,
        "selected_n_ctx": None,
        "note": (
            "I casi sono punti diagnostici espliciti, non profili selezionati. "
            "Con prompt_repeat uguale, il testo del prompt resta invariato "
            "tra contesti: si misura soprattutto la capacita' KV allocata. "
            "Si eseguono prove isolate baseline e doppia promozione; "
            "i fallimenti vengono riportati senza interpolare allocazioni. "
            "I quartili decode descrivono la velocita' in posizioni di contesto "
            "crescenti e non sono una misura isolata del costo KV. "
            "Un picco NVML inferiore alla capacita' non dimostra stabilita' "
            "a contesti operativi o con carichi concorrenti."
        ),
    }

    blocked = set()
    for cycle in range(1, rounds + 1):
        ordered = cases if cycle % 2 else list(reversed(cases))
        for case in ordered:
            key = case["id"]
            if key in blocked:
                continue
            # Always evaluate baseline first for an unverified case, even
            # when the pair direction is reversed in subsequent rounds.
            order = ("baseline", "candidate") if cycle % 2 else ("candidate", "baseline")
            for variant in order:
                current = GPUManager.calibration_status()
                if not current["ready"]:
                    result["error"] = (
                        f"Stato GPU non valido: {current['errors']}; "
                        f"{current['foreign_processes']}"
                    )
                    result["summary"] = summarize_envelope(
                        result["trials"], cases, rounds, gpu_total
                    )
                    return result
                out = benchmark(
                    gpu_layers=case["gpu_layers"],
                    max_tokens=tokens, timeout=timeout,
                    probe_n_ctx=case["probe_n_ctx"],
                    prompt_repeat=case["prompt_repeat"],
                    combination_ids=definitions[key] if variant == "candidate" else None,
                    collect_gpu_telemetry=True,
                )
                metrics = out.get("measurements") or {}
                windows = metrics.get("decode_windows") or {}
                verification = out.get("mapping_verification") or {}
                verified = verification.get("verified") is True if variant == "candidate" else None
                good = (
                    out.get("success") is True
                    and (variant == "baseline" or verified is True)
                    and out.get("probe_n_ctx") == case["probe_n_ctx"]
                    and metrics.get("generated_tokens") == tokens
                )
                row = {
                    "round": cycle, "case_id": key, "variant": variant,
                    "success": good, "mapping_verified": verified,
                    "actual_probe_n_ctx": out.get("probe_n_ctx"),
                    "prompt_tokens": metrics.get("prompt_tokens"),
                    "generated_tokens": metrics.get("generated_tokens"),
                    "requested_tokens": tokens,
                    "decode_first_quarter_tps": windows.get("first_window_tps"),
                    "decode_last_quarter_tps": windows.get("last_window_tps"),
                    "final_context_token_fraction": windows.get("end_context_fraction"),
                    "decode_window_details": windows.get("windows"),
                    "decode_tps": metrics.get("decode_tokens_per_second"),
                    "prefill_tps": metrics.get("prefill_tokens_per_second"),
                    "vram_peak_mib": metrics.get("vram_peak_sampled_mib"),
                    "gpu_decode_telemetry": (
                        (out.get("gpu_telemetry_by_phase") or {}).get("decode")
                    ),
                    "worker_returncode": out.get("worker_returncode"),
                    "native_signal": out.get("native_signal"),
                    "failure_type": out.get("failure_type"),
                    "suspected_cuda_oom": out.get("suspected_cuda_oom", False),
                    "error": (
                        out.get("error") or
                        ("Override nativo non verificato" if variant == "candidate" and not verified
                         else "Contesto diagnostico o numero di token non confermato")
                    ) if not good else None,
                }
                result["trials"].append(row)
                if not good:
                    blocked.add(key)
                    break

    result["summary"] = summarize_envelope(result["trials"], cases, rounds, gpu_total)
    result["measurement_completed"] = True
    result["all_cases_complete"] = all(c["complete"] for c in result["summary"])
    # success indicates that the diagnostic sweep completed and accounted
    # for failures, not that every mapping is feasible.
    result["success"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Matrice diagnostica CUDA: layer/context/VRAM")
    parser.add_argument(
        "--case", action="append", default=None,
        help="Ripetere: gpu_layers:probe_n_ctx:prompt_repeat",
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    try:
        cases = parse_cases(
            args.case or ["40:768:6", "39:768:6", "39:1024:6"], args.tokens
        )
        result = envelope(cases, args.rounds, args.tokens, args.timeout)
    except Exception as exc:
        result = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
