"""Phase 1 experiment: all supported f16/q8_0/q4_0 KV combinations.

Each non-reference profile searches the full-CUDA context frontier, starting
at the already measured reference n_ctx + 1. Do not test an inferior K/V
precision pair once a componentwise-superior pair with the SAME Flash
Attention setting is confirmed at the model's architectural context limit.
"""

import argparse
import json
from pathlib import Path

from model import ModelManager
from .gpu_context import search_gpu_context
from .profile_pipeline import atomic_json


# Off: V quantization requires Flash Attention; on: all 3x3 combinations.
# The reference profile is present in the matrix, but never re-executed.
# Within each FA mode, superior precision profiles precede dominated ones.
PROFILES = (
    ("fa_off_k_f16_v_f16", False, "f16", "f16"),
    ("fa_off_k_q8_0_v_f16", False, "q8_0", "f16"),
    ("fa_off_k_q4_0_v_f16", False, "q4_0", "f16"),
    ("fa_on_k_f16_v_f16", True, "f16", "f16"),
    ("fa_on_k_f16_v_q8_0", True, "f16", "q8_0"),
    ("fa_on_k_q8_0_v_f16", True, "q8_0", "f16"),
    ("fa_on_k_q8_0_v_q8_0", True, "q8_0", "q8_0"),
    ("fa_on_k_f16_v_q4_0", True, "f16", "q4_0"),
    ("fa_on_k_q4_0_v_f16", True, "q4_0", "f16"),
    ("fa_on_k_q8_0_v_q4_0", True, "q8_0", "q4_0"),
    ("fa_on_k_q4_0_v_q8_0", True, "q4_0", "q8_0"),
    ("fa_on_k_q4_0_v_q4_0", True, "q4_0", "q4_0"),
)

PRECISION_RANK = {"q4_0": 0, "q8_0": 1, "f16": 2}
REFERENCE_ID = "fa_off_k_f16_v_f16"
STATUS_SKIP = "skipped_dominated_at_model_limit"


def precision_dominates(winner: dict, candidate: tuple) -> bool:
    """Strict Pareto dominance for K/V, comparing only the same FA mode.

    Precision is ordered per component. Mixed configurations such as
    f16/q4_0 and q8_0/q8_0 are incomparable and must both be tested.
    """
    _, flash, k, v = candidate
    if flash != winner["flash_attn"]:
        return False
    wk, wv = winner["type_k"], winner["type_v"]
    return (PRECISION_RANK[wk] >= PRECISION_RANK[k]
            and PRECISION_RANK[wv] >= PRECISION_RANK[v]
            and (wk != k or wv != v))


def model_limit_reached(profile: dict) -> bool:
    """Never infer model limit from the caller's optional --max-ctx bound."""
    if profile.get("status") != "confirmed":
        return False
    result = profile.get("result") or {}
    limit = result.get("model_declared_context_length")
    maximum = profile.get("max_gpu_n_ctx")
    return (type(limit) is int and type(maximum) is int
            and limit >= 64 and maximum == limit
            and result.get("success") is True)


def run_profiles(*, model_id: str = "gemma-2-9b-it",
                 source: str | None = None, baseline_ctx: int = 257,
                 max_ctx: int | None = None, tokens: int = 16,
                 batch: int = 32, timeout: int = 300,
                 confirm_rounds: int = 2,
                 output: Path = Path("phase1_kv_profiles_complete.json"),
                 rerun: bool = False, search=search_gpu_context,
                 progress=None) -> dict:
    """Run each non-dominated profile and checkpoint after each decision."""
    if (type(baseline_ctx) is not int or baseline_ctx < 63
        or type(tokens) is not int or not 1 <= tokens <= 32
        or type(batch) is not int or batch < 1
        or type(confirm_rounds) is not int or confirm_rounds < 1
        or type(timeout) is not int or timeout < 1):
        raise ValueError("Parametri Fase 1 KV non validi.")
    min_ctx = baseline_ctx + 1
    if max_ctx is not None and (type(max_ctx) is not int or max_ctx < min_ctx):
        raise ValueError("max_ctx deve essere almeno baseline_ctx + 1.")
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
        "schema": "ruth-phase1-gpu-kv-profiles-v3",
        "parameters": identity,
        "original_reference": {
            "flash_attn": False, "type_k": "f16", "type_v": "f16",
            "n_ctx": baseline_ctx,
            "note": "Full-GPU historical reference, not re-executed.",
        },
        "profiles": [], "all_profiles_completed": False,
        "note": (
            "All 12 allowed FA/KV combinations; 1 historic reference + 11 "
            "experimental profiles. Each probe <= GGUF architecture limit; "
            "start at baseline+1. At the architecture limit, skip only "
            "K/V componentwise dominated profiles with identical FA setting. "
            "The skip optimizes context-capacity search, not throughput or quality. "
            "The original reference and operational profiles remain unchanged."
        ),
    }
    if output.exists() and not rerun:
        try:
            previous = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ValueError("Report precedente non valido; usare --rerun.")
        if (previous.get("schema") != report["schema"]
            or previous.get("parameters") != identity):
            raise ValueError(
                "Report precedente con schema/parametri diversi: "
                "usare un nuovo --output o --rerun."
            )
        cached = {
            p["id"]: p for p in previous.get("profiles", [])
            if isinstance(p, dict) and p.get("id") in {
                row[0] for row in PROFILES
            }
        }
    else:
        cached = {}

    def record(profile: dict) -> None:
        report["profiles"].append(profile)
        atomic_json(output, report)
        if progress is not None:
            progress({
                "event": "profile_end", "profile": profile["id"],
                "status": profile["status"],
                "n_ctx": profile["max_gpu_n_ctx"],
                "dominated_by": profile.get("dominated_by"),
            })

    for candidate in PROFILES:
        identifier, flash, k, v = candidate
        if progress is not None:
            progress({"event": "profile_start", "profile": identifier})

        # Existing FP16/FP16 reference remains an explicit row in the
        # complete matrix; no CUDA worker and no fake new measurements.
        if identifier == REFERENCE_ID:
            record({
                "id": identifier, "flash_attn": flash,
                "type_k": k, "type_v": v,
                "status": "reference_reused",
                "max_gpu_n_ctx": baseline_ctx,
                "result": None,
                "reference_provenance": "previous_phase1_full_gpu_measurement",
            })
            continue

        # Reuse complete earlier evidence, but recompute all skip decisions
        # based on confirmed model ceilings from this exact runtime identity.
        old = cached.get(identifier)
        if old is not None and old.get("status") == "confirmed":
            record(old)
            continue

        dominator = next((
            p for p in report["profiles"]
            if model_limit_reached(p) and precision_dominates(p, candidate)
        ), None)
        if dominator is not None:
            record({
                "id": identifier, "flash_attn": flash,
                "type_k": k, "type_v": v,
                "status": STATUS_SKIP, "max_gpu_n_ctx": None,
                "dominated_by": dominator["id"],
                "model_declared_context_length": dominator["result"][
                    "model_declared_context_length"
                ],
                "reason": (
                    "A same-Flash-Attention configuration with equal or better "
                    "K and V precision already reached the model context cap."
                ),
                "result": None,
            })
            continue

        try:
            result = search(
                model_id, source=source, min_ctx=min_ctx, max_ctx=max_ctx,
                tokens=tokens, batch=batch, timeout=timeout,
                confirm_rounds=confirm_rounds, flash_attn=flash,
                type_k=k, type_v=v, persist_reference=False,
                progress=(lambda event: progress({
                    "profile": identifier, **event
                })) if progress is not None else None,
            )
            state = ("confirmed" if result.get("success") is True
                     else "unsupported_or_inconclusive")
        except Exception as exc:
            result = {
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            state = "unsupported_or_inconclusive"
        record({
            "id": identifier, "flash_attn": flash,
            "type_k": k, "type_v": v, "status": state,
            "max_gpu_n_ctx": result.get("confirmed_max_gpu_resident_n_ctx")
            if result.get("success") is True else None,
            "result": result,
        })

    report["all_profiles_completed"] = len(report["profiles"]) == len(PROFILES)
    atomic_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fase 1: 12 combinazioni KV/Flash Attention (riferimento incluso)"
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
                        default=Path("phase1_kv_profiles_complete.json"))
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
        return 0 if all(p["status"] in ("confirmed", "reference_reused", STATUS_SKIP)
                        for p in report["profiles"]) else 1
    except Exception as exc:
        print(f"Fase 1 KV: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
