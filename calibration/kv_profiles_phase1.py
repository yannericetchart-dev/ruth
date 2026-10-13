"""Phase 1 experiments: seven Flash Attention / KV precision profiles.

Weights and KV stay GPU-resident. The historical f16/f16, FA-off reference
is not recalibrated or overwritten by this experimental sweep.
"""
import argparse
import json
from pathlib import Path

from model import ModelManager
from .gpu_context import search_gpu_context
from .profile_pipeline import atomic_json


PROFILES = (
    ("fa_off_k_q8_0_v_f16", False, "q8_0", "f16"),
    ("fa_off_k_q4_0_v_f16", False, "q4_0", "f16"),
    ("fa_on_k_f16_v_f16", True, "f16", "f16"),
    ("fa_on_k_q8_0_v_q8_0", True, "q8_0", "q8_0"),
    ("fa_on_k_q8_0_v_q4_0", True, "q8_0", "q4_0"),
    ("fa_on_k_q4_0_v_q8_0", True, "q4_0", "q8_0"),
    ("fa_on_k_q4_0_v_q4_0", True, "q4_0", "q4_0"),
)


def run_profiles(*, model_id: str = "gemma-2-9b-it",
                 source: str | None = None, baseline_ctx: int = 257,
                 max_ctx: int | None = None, tokens: int = 16,
                 batch: int = 32, timeout: int = 300,
                 confirm_rounds: int = 2,
                 output: Path = Path("phase1_kv_profiles_from_258.json"),
                 rerun: bool = False, search=search_gpu_context,
                 progress=None) -> dict:
    """Start each profile at baseline_ctx + 1; stop if that first probe fails."""
    if (not isinstance(baseline_ctx, int) or isinstance(baseline_ctx, bool)
        or baseline_ctx < 63 or not 1 <= tokens <= 32 or batch < 1):
        raise ValueError("Parametri Fase 1 KV non validi.")
    min_ctx = baseline_ctx + 1
    if max_ctx is not None and max_ctx < min_ctx:
        raise ValueError("max_ctx deve essere almeno min_ctx.")
    if model_id not in ModelManager.CATALOG:
        raise ValueError(f"Modello non supportato: {model_id}")
    output = Path(output).expanduser().resolve()
    identity = {
        "model_id": model_id, "source": source,
        "baseline_ctx": baseline_ctx, "min_ctx": min_ctx,
        "max_ctx": max_ctx, "tokens": tokens, "batch": batch,
        "timeout": timeout, "confirm_rounds": confirm_rounds,
    }
    report = {
        "schema": "ruth-phase1-gpu-kv-profiles-v2",
        "parameters": identity,
        "original_reference": {
            "flash_attn": False, "type_k": "f16", "type_v": "f16",
            "n_ctx": baseline_ctx,
            "note": "Riferimento storico non rieseguito; ogni nuovo profilo parte da baseline + 1.",
        },
        "profiles": [], "all_profiles_completed": False,
        "note": ("Full GPU weights + GPU KV, token prompt + output = n_ctx. "
                 "Prima prova sempre a baseline_ctx + 1; fallimento iniziale "
                 "arresta il profilo, senza ulteriori n_ctx. "
                 "Configurazioni non supportate sono riportate separatamente. "
                 "Nessun profilo operativo o reference originale viene sovrascritto."),
    }
    if output.exists() and not rerun:
        try:
            previous = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ValueError("Report precedente non valido; usare --rerun.")
        if (previous.get("schema") != report["schema"]
            or previous.get("parameters") != identity):
            raise ValueError("Parametri diversi dal report salvato; usare --rerun.")
        report["profiles"] = [
            p for p in previous.get("profiles", [])
            if p.get("id") in {row[0] for row in PROFILES}
        ]

    cached = {row["id"]: row for row in report["profiles"]}
    report["profiles"] = []
    for identifier, flash, k, v in PROFILES:
        if progress is not None:
            progress({"event": "profile_start", "profile": identifier})
        old = cached.get(identifier)
        if old is not None and old.get("status") == "confirmed":
            report["profiles"].append(old)
            continue
        try:
            result = search(
                model_id, source=source, min_ctx=min_ctx, max_ctx=max_ctx,
                tokens=tokens, batch=batch, timeout=timeout,
                confirm_rounds=confirm_rounds, flash_attn=flash,
                type_k=k, type_v=v, persist_reference=False,
                progress=(lambda event: progress({"profile": identifier, **event}))
                if progress is not None else None,
            )
            state = ("confirmed" if result.get("success") is True
                     else "unsupported_or_inconclusive")
        except Exception as exc:
            result = {"success": False,
                      "error": f"{type(exc).__name__}: {exc}"}
            state = "unsupported_or_inconclusive"
        report["profiles"].append({
            "id": identifier, "flash_attn": flash,
            "type_k": k, "type_v": v,
            "status": state,
            "max_gpu_n_ctx": result.get("confirmed_max_gpu_resident_n_ctx")
            if result.get("success") is True else None,
            "result": result,
        })
        atomic_json(output, report)
        if progress is not None:
            progress({"event": "profile_end", "profile": identifier,
                      "status": state,
                      "n_ctx": report["profiles"][-1]["max_gpu_n_ctx"]})
    report["all_profiles_completed"] = len(report["profiles"]) == len(PROFILES)
    atomic_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fase 1: massimo contesto full-CUDA per sette profili KV/Flash Attention"
    )
    parser.add_argument("--model", default="gemma-2-9b-it")
    parser.add_argument("--source")
    parser.add_argument("--baseline-ctx", type=int, default=257,
                        help="Riferimento full-GPU FP16/FP16 senza FA; primo probe = valore + 1")
    parser.add_argument("--max-ctx", type=int)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--confirm-rounds", type=int, default=2)
    parser.add_argument("--output", type=Path,
                        default=Path("phase1_kv_profiles_from_258.json"))
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    try:
        report = run_profiles(
            model_id=args.model, source=args.source, baseline_ctx=args.baseline_ctx,
            max_ctx=args.max_ctx, tokens=args.tokens,
            batch=args.batch, timeout=args.timeout,
            confirm_rounds=args.confirm_rounds, output=args.output,
            rerun=args.rerun,
            progress=lambda event: print(
                event.get("profile", "?"), event.get("event", "?"),
                event.get("n_ctx", ""),
                event.get("failure_class", ""), flush=True,
            ),
        )
        print("Profilo KV | esito | massimo n_ctx")
        for row in report["profiles"]:
            print(f"{row['id']} | {row['status']} | {row['max_gpu_n_ctx']}")
        print("Report:", args.output)
        return 0 if all(p["status"] == "confirmed" for p in report["profiles"]) else 1
    except Exception as exc:
        print(f"Fase 1 KV: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
