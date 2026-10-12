"""Decode throughput by generated-token position in diagnostic CUDA runs.

Time stamps are recorded after each generator yield. The first yield
samples from already-evaluated logits, so the first evaluated decode step
is the interval between timestamps[0] and timestamps[1].
Window rates are measured from those intervals; no constant per-token
speed is assumed. Occupancy is a token-count diagnostic, not KV allocation.
"""

import math


def summarize_decode_windows(
    timestamps: list[float], prompt_tokens: int, n_ctx: int, windows: int = 4
) -> dict:
    if (not isinstance(prompt_tokens, int) or prompt_tokens < 1
        or not isinstance(n_ctx, int) or n_ctx <= prompt_tokens
        or not isinstance(windows, int) or not 2 <= windows <= 8):
        raise ValueError("Numero di token o contesto non valido.")
    if len(timestamps) < windows + 1:
        raise ValueError("Troppi segmenti per i token decodificati.")
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in timestamps):
        raise ValueError("Tempo di generazione non finito.")
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("Tempo di generazione non monotono.")
    steps = len(timestamps) - 1
    result = []
    for index in range(windows):
        left = index * steps // windows
        right = (index + 1) * steps // windows
        elapsed = timestamps[right] - timestamps[left]
        start_position = prompt_tokens + left
        end_position = prompt_tokens + right
        result.append({
            "window": index + 1,
            "decode_steps": right - left,
            "elapsed_seconds": round(elapsed, 4),
            "decode_tps": round((right - left) / elapsed, 2),
            "starting_context_tokens": start_position,
            "ending_context_tokens": end_position,
            "ending_context_fraction": round(end_position / n_ctx, 4),
        })
    return {
        "window_count": windows,
        "decode_steps_total": steps,
        "prompt_tokens": prompt_tokens,
        "probe_n_ctx": n_ctx,
        "generated_tokens": len(timestamps),
        "end_context_fraction": round(
            (prompt_tokens + len(timestamps) - 1) / n_ctx, 4
        ),
        "windows": result,
        "first_window_tps": result[0]["decode_tps"],
        "last_window_tps": result[-1]["decode_tps"],
        "last_minus_first_tps": round(
            result[-1]["decode_tps"] - result[0]["decode_tps"], 3
        ),
        "measurement_note": (
            "Ogni finestra misura gli intervalli effettivi tra yield consecutivi. "
            "La prima generazione dopo eval e' solo sampling e non conta "
            "come passo decode. L'occupazione esprime posizioni token, "
            "non la memoria KV allocata. La velocita' include possibili "
            "cambiamenti di frequenza/temperatura e lavoro variabile."
        ),
    }
