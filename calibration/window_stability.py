"""Counterbalanced per-quartile decode comparisons, independent of GPU runtime.

This module summarizes complete ABBA/BAAB trials of an identical model,
input length, context and generation token count. Comparing the averaged
samples within each batch reduces idealized linear time drift; it does
not remove intra-run drift, thermal effects or the dependence on prompt.
"""

import math
import statistics


def compare_decode_windows(trials: list[dict], rounds: int) -> dict | None:
    """Return quartile A/B differences or None for older, unsegmented data.

    Partial segmentation is an error, not an excuse to silently report
    misleading results. Four quartiles are required for every worker.
    """
    present = [row.get("decode_window_details") is not None for row in trials]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("Segmentazione decode parziale: manca un worker.")
    if not isinstance(rounds, int) or rounds < 2:
        raise ValueError("Numero di cicli ABBA/BAAB non valido.")
    if len(trials) != 4 * rounds:
        raise ValueError("Segmentazione incompleta.")

    indexed = {}
    for trial in trials:
        ident = (trial.get("round"), trial.get("position"))
        if (not isinstance(ident[0], int) or ident[0] not in range(1, rounds + 1)
            or not isinstance(ident[1], int) or ident[1] not in range(1, 5)
            or ident in indexed or trial.get("variant") not in ("baseline", "candidate")):
            raise ValueError("Worker duplicato o sconosciuto.")
        windows = trial["decode_window_details"]
        if not isinstance(windows, list) or len(windows) != 4:
            raise ValueError("Il report richiede quattro quartili per worker.")
        if not isinstance(trial.get("prompt_tokens"), int) or trial["prompt_tokens"] < 1:
            raise ValueError("Prompt reale mancante.")
        occupancy = trial.get("final_context_token_fraction")
        if not isinstance(occupancy, (int, float)) or not math.isfinite(occupancy):
            raise ValueError("Occupazione del contesto non valida.")
        if not 0 < occupancy < 1:
            raise ValueError("L'occupazione deve essere inferiore al limite tecnico.")
        for j, window in enumerate(windows, start=1):
            if not isinstance(window, dict) or window.get("window") != j:
                raise ValueError("Indice quartile non coerente.")
            speed = window.get("decode_tps")
            if (not isinstance(speed, (int, float)) or isinstance(speed, bool)
                or not math.isfinite(speed) or speed <= 0):
                raise ValueError("Velocita' di quartile mancante o non valida.")
            if not isinstance(window.get("decode_steps"), int) or window["decode_steps"] <= 0:
                raise ValueError("Numero di passi decode non valido.")
        indexed[ident] = trial

    comparisons = []
    prompts = {t["prompt_tokens"] for t in trials}
    fractions = {t["final_context_token_fraction"] for t in trials}
    if len(prompts) != 1 or len(fractions) != 1:
        raise ValueError("Contesti occupati non confrontabili fra varianti.")
    for run in range(1, rounds + 1):
        rows = [indexed[(run, position)] for position in range(1, 5)]
        baseline = [r for r in rows if r["variant"] == "baseline"]
        candidate = [r for r in rows if r["variant"] == "candidate"]
        if len(baseline) != 2 or len(candidate) != 2:
            raise ValueError("Ogni ciclo richiede due worker per variante.")

        mean_base = [
            statistics.mean(r["decode_window_details"][idx]["decode_tps"] for r in baseline)
            for idx in range(4)
        ]
        mean_candidate = [
            statistics.mean(r["decode_window_details"][idx]["decode_tps"] for r in candidate)
            for idx in range(4)
        ]
        differences = [
            round(mean_candidate[i] - mean_base[i], 3) for i in range(4)
        ]
        comparisons.append({
            "round": run,
            "baseline_quartile_tps": [round(x, 3) for x in mean_base],
            "candidate_quartile_tps": [round(x, 3) for x in mean_candidate],
            "candidate_minus_baseline_quartile_tps": differences,
            "last_quartile_delta_tps": differences[3],
            "baseline_last_minus_first_tps": round(mean_base[3] - mean_base[0], 3),
            "candidate_last_minus_first_tps": round(
                mean_candidate[3] - mean_candidate[0], 3
            ),
        })

    return {
        "prompt_tokens": next(iter(prompts)),
        "ending_context_fraction": next(iter(fractions)),
        "rounds": comparisons,
        "median_candidate_minus_baseline_by_quartile_tps": [
            round(statistics.median(
                r["candidate_minus_baseline_quartile_tps"][i]
                for r in comparisons), 3)
            for i in range(4)
        ],
        "last_quartile_positive_rounds": sum(
            r["last_quartile_delta_tps"] > 0 for r in comparisons
        ),
        "last_quartile_delta_median_tps": round(statistics.median(
            r["last_quartile_delta_tps"] for r in comparisons
        ), 3),
        "last_quartile_delta_range_tps": [
            min(r["last_quartile_delta_tps"] for r in comparisons),
            max(r["last_quartile_delta_tps"] for r in comparisons),
        ],
        "descriptive_only": True,
        "note": (
            "Il controbilanciamento ABBA/BAAB riduce soltanto la deriva "
            "temporale approssimativamente lineare fra worker. Quartili "
            "e frequenze GPU non separano causalmente il costo della KV cache, "
            "il throttling, il contenuto del prompt o il percorso dei kernel."
        ),
    }
