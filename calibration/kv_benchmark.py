"""Fixed-context Phase 1 KV benchmarks: GPU throughput + CPU-KV-isolated quality.

The input Phase 1 reference report determines which actual GPU-KV configurations
can be measured at 512/768/1280. Quality uses FP16 vs quantized KV with both KV
buffers in CPU RAM, same FA, weights on GPU, identical corpus prefixes. This
isolates quantization better than comparing GPU-KV to CPU-KV, but the quality
measure is NOT a direct measurement of GPU-KV numerical error.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tempfile

from model import ModelManager
from model.inspect import inspect_gguf
from .baseline_discovery import native_confirmation
from .gpu_context import context_limit, probe_context, verify_native_kv_config
from .mapping_phase import verify_mapping_kv_runtime
from .native_memory import parse_native_memory
from .mapping_grouping import (group_inventory, native_multi_group_trial)
from .mapping_native_validation import verify_override_logs
from .profile_pipeline import atomic_json
from .worker_failure import analyze_worker_failure


DEFAULT_CONTEXTS = (512, 768, 1280)
SCORE_WEIGHTS = {
    "mean_kl": 0.30,
    "sampled_ppl": 0.20,
    "decode": 0.25,
    "prefill": 0.15,
    "headroom": 0.10,
}


def eligible_profiles(phase1: dict, n_ctx: int) -> list[dict]:
    """Never extrapolate GPU residency from more compressed KV formats."""
    return [
        p for p in phase1.get("profiles", [])
        if p.get("status") == "confirmed"
        and type(p.get("max_gpu_n_ctx")) is int
        and p["max_gpu_n_ctx"] >= n_ctx
    ]


def _finite_positive(value) -> bool:
    return (isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def measure_speed(model_path: Path, model_id: str, profile: dict,
                  *, n_ctx: int, tokens: int, repetitions: int,
                  timeout: int) -> dict:
    observations = []
    for i in range(repetitions):
        point = probe_context(
            model_path, n_ctx, generated_tokens=tokens,
            batch=32, timeout=timeout, model_id=model_id,
            flash_attn=profile["flash_attn"],
            type_k=profile["type_k"], type_v=profile["type_v"],
        )
        record = {
            "repetition": i + 1, "success": point.get("success") is True,
            "failure_class": point.get("failure_class"),
            "error": point.get("error"),
            "native_verified": (
                point.get("full_gpu_native_verification", {}).get("verified") is True
                and point.get("gpu_runtime_validation", {}).get("verified") is True
                and point.get("kv_config_native_verification", {}).get("verified") is True
            ),
            "measurements": {
                k: (point.get("measurements") or {}).get(k)
                for k in (
                    "prefill_tokens_per_second", "decode_tokens_per_second",
                    "prefill_s", "decode_s", "prompt_tokens",
                    "generated_tokens", "total_tokens", "vram_peak_sampled_mib",
                )
            },
        }
        record["success"] = record["success"] and record["native_verified"]
        observations.append(record)
        if not record["success"]:
            break

    result = {"success": len(observations) == repetitions
              and all(p["success"] for p in observations),
              "repetitions": observations}
    if result["success"]:
        vals = [p["measurements"] for p in observations]
        result["median_prefill_tok_s"] = round(statistics.median(
            x["prefill_tokens_per_second"] for x in vals
        ), 3)
        result["median_decode_tok_s"] = round(statistics.median(
            x["decode_tokens_per_second"] for x in vals
        ), 3)
        result["median_vram_peak_mib"] = round(statistics.median(
            x["vram_peak_sampled_mib"] for x in vals
        ), 2)
        result["max_vram_peak_mib"] = round(max(
            x["vram_peak_sampled_mib"] for x in vals
        ), 2)
    return result


def quality_subprocess(*, model_path: Path, corpus: Path, n_ctx: int,
                       flash_attn: bool, type_k: str, type_v: str,
                       points: int, reference: Path | None,
                       output: Path, timeout: int,
                       group_ids: list[str] | None = None) -> dict:
    command = [
        sys.executable, "-m", "calibration.kv_quality_worker",
        "--model-path", str(model_path), "--corpus", str(corpus),
        "--n-ctx", str(n_ctx), "--flash-attn", "on" if flash_attn else "off",
        "--type-k", type_k, "--type-v", type_v, "--points", str(points),
        "--output", str(output),
    ]
    if reference:
        command.extend(["--reference", str(reference)])
    if group_ids:
        command.extend(["--group-ids", json.dumps(sorted(group_ids))])
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "failure_class": "timeout",
                "error": f"Timeout qualità {timeout}s"}
    try:
        result = json.loads(proc.stdout)
        if not isinstance(result, dict):
            raise ValueError("Worker non ha restituito un oggetto JSON.")
    except (ValueError, TypeError):
        result = {
            "success": False, "error": "Protocollo worker qualità non valido.",
        }

    layout = inspect_gguf(model_path, n_ctx=n_ctx)
    if group_ids:
        inventory = {g["id"]: g for g in group_inventory(layout)
                     if g["eligible_for_trial"]}
        if len(set(group_ids)) != len(group_ids) or any(
            gid not in inventory for gid in group_ids
        ):
            raise ValueError("Lista gruppi qualità incoerente con GGUF.")
        trial = native_multi_group_trial(
            layout, [inventory[gid] for gid in group_ids]
        )
        placement = verify_override_logs(trial, proc.stderr)
    else:
        placement = native_confirmation(layout, proc.stderr)
    native = parse_native_memory(proc.stderr)
    kv_residency = verify_mapping_kv_runtime(native, host_kv=True)
    kv_type = verify_native_kv_config(
        proc.stderr, flash_attn=flash_attn, type_k=type_k, type_v=type_v,
    )
    failure = analyze_worker_failure(
        proc.stderr, result.get("error"), proc.returncode,
        result.get("success") is True,
    )
    result["native_checks"] = {
        "weights_gpu": placement["verified"],
        "kv_cpu": kv_residency["verified"],
        "kv_type_flash": kv_type["verified"],
    }
    result["failure_diagnostics"] = {
        "worker_returncode": proc.returncode,
        "signal": (
            signal.Signals(-proc.returncode).name
            if proc.returncode < 0 and -proc.returncode in signal.Signals._value2member_map_
            else None
        ),
        "failure_class": failure.get("failure_hint"),
        "error": result.get("error"),
    }
    result["success"] = (
        proc.returncode == 0 and result.get("success") is True
        and all(result["native_checks"].values())
        and (reference is not None or output.is_file())
        and (reference is None or isinstance(result.get("quality"), dict))
    )
    if not result["success"]:
        result["error"] = result.get("error") or (
            "Log nativi non confermano pesi GPU, KV CPU o formato KV/FA."
        )
    return result


def score_context(rows: list[dict], *, vram_total_mib: float,
                  kl_scale: float, target_headroom_mib: float) -> None:
    """Score is strictly within ONE context, not a cross-context quality ranking."""
    valid = [
        row for row in rows
        if row.get("speed", {}).get("success") is True
        and row.get("quality", {}).get("success") is True
        and _finite_positive(row["speed"].get("median_prefill_tok_s"))
        and _finite_positive(row["speed"].get("median_decode_tok_s"))
        and _finite_positive(row["quality"]["quality"].get("reference_sampled_ppl"))
        and _finite_positive(row["quality"]["quality"].get("candidate_sampled_ppl"))
        and isinstance(row["quality"]["quality"].get("mean_kl_nats"), (int, float))
        and row["quality"]["quality"]["mean_kl_nats"] >= 0
        and math.isfinite(row["quality"]["quality"]["mean_kl_nats"])
    ]
    if not valid:
        return
    top_prefill = max(r["speed"]["median_prefill_tok_s"] for r in valid)
    top_decode = max(r["speed"]["median_decode_tok_s"] for r in valid)
    for row in valid:
        q = row["quality"]["quality"]
        speed = row["speed"]
        headroom = vram_total_mib - speed["max_vram_peak_mib"]
        metrics = {
            "mean_kl": math.exp(-q["mean_kl_nats"] / kl_scale),
            "sampled_ppl": min(
                1.0, q["reference_sampled_ppl"] / q["candidate_sampled_ppl"]
            ),
            "decode": speed["median_decode_tok_s"] / top_decode,
            "prefill": speed["median_prefill_tok_s"] / top_prefill,
            "headroom": max(0.0, min(1.0, headroom / target_headroom_mib)),
        }
        row["diagnostic_score"] = round(
            100 * sum(SCORE_WEIGHTS[key] * metrics[key] for key in SCORE_WEIGHTS), 2
        )
        row["score_components"] = {
            key: round(value, 5) for key, value in metrics.items()
        }
        row["vram_headroom_mib"] = round(headroom, 2)
        # A sampled margin target is not proof of operational stability.
        row["headroom_target_met"] = headroom >= target_headroom_mib


def run_benchmark(*, phase1_report: Path, corpus: Path,
                  model_path: Path | None = None,
                  contexts: tuple[int, ...] = DEFAULT_CONTEXTS,
                  output: Path = Path("kv_quality_speed_benchmarks.json"),
                  repetitions: int = 3, decode_tokens: int = 16,
                  quality_points: int = 64, timeout: int = 600,
                  kl_scale: float = 0.01,
                  headroom_target_mib: float = 64.0,
                  progress=None) -> dict:
    if (repetitions < 1 or decode_tokens < 1 or decode_tokens > 32
        or quality_points < 2 or timeout < 1
        or not _finite_positive(kl_scale)
        or not _finite_positive(headroom_target_mib)):
        raise ValueError("Parametri benchmark non validi.")
    phase1_report = Path(phase1_report).resolve()
    corpus = Path(corpus).resolve()
    output = Path(output).resolve()
    source = json.loads(phase1_report.read_text(encoding="utf-8"))
    if source.get("schema") != "ruth-phase1-gpu-kv-profiles-v3":
        raise ValueError("Serve report Fase 1 KV v3.")
    selection = source["parameters"]
    model_id = selection["model_id"]
    if model_path is None:
        baseline = ModelManager().selected_baseline(
            model_id, source=selection.get("source")
        )
        if not baseline:
            raise FileNotFoundError(
                "Baseline GGUF non trovata; specificare --model-path."
            )
        model_path = Path(baseline["local_path"])
    model_path = Path(model_path).resolve()
    if not contexts:
        raise ValueError("Nessun contesto richiesto.")
    layout = inspect_gguf(model_path, n_ctx=min(contexts))
    limit = context_limit(layout)
    if any(
        type(ctx) is not int or ctx < 64 or ctx > limit
        for ctx in contexts
    ):
        raise ValueError(f"Contesti richiesti fuori limite GGUF={limit}.")
    corpus_bytes = corpus.read_bytes()
    if len(corpus_bytes) > 10_000_000 or not corpus_bytes:
        raise ValueError("Corpus vuoto o maggiore di 10 MB.")
    corpus_hash = hashlib.sha256(corpus_bytes).hexdigest()
    vram_total = float(
        next(
            (p["result"]["gpu_total_mib"] for p in source["profiles"]
             if p.get("result") and p["result"].get("gpu_total_mib")),
            0.0,
        )
    )
    if vram_total <= 0:
        raise ValueError("Dimensione VRAM non presente nel report Fase 1.")

    report = {
        "schema": "ruth-kv-fixed-quality-speed-v1",
        "source_phase1_sha256": hashlib.sha256(
            phase1_report.read_bytes()
        ).hexdigest(),
        "corpus_sha256": corpus_hash, "model_id": model_id,
        "model_path": str(model_path), "model_context_limit": limit,
        "contexts": list(contexts), "repetitions": repetitions,
        "decode_tokens": decode_tokens, "quality_positions": quality_points,
        "quality_kv_location": "cpu_for_both_fp16_and_quantized",
        "speed_kv_location": "gpu",
        "score_weights": SCORE_WEIGHTS, "kl_scale_nats": kl_scale,
        "headroom_target_mib": headroom_target_mib,
        "score_note": (
            "Diagnostic score only. 30% exp(-KL/kl_scale), "
            "20% min(1, PPL_ref/PPL_quant), 25% decode rate, "
            "15% prefill rate, 10% VRAM headroom. Rates normalized by the "
            "fastest measured candidate AT THE SAME n_ctx. "
            "Quality is teacher-forced sampled positions (not full-corpus PPL). "
            "CPU KV in BOTH quality runs isolates quantization; the CUDA "
            "GPU-KV candidate is not assumed numerically identical. "
            "Not a quality guarantee or an operational profile."
        ),
        "results": [], "completed": False,
    }
    with tempfile.TemporaryDirectory(prefix="ruth-kv-quality-") as directory:
        workspace = Path(directory)
        for ctx in contexts:
            selected = eligible_profiles(source, ctx)
            ctx_row = {"n_ctx": ctx, "eligible_profiles": len(selected),
                       "profiles": []}
            report["results"].append(ctx_row)
            reference_files = {}
            reference_errors = {}
            for profile in selected:
                pid = profile["id"]
                if progress:
                    progress(f"n_ctx={ctx} | {pid}")
                speed = measure_speed(
                    model_path, model_id, profile, n_ctx=ctx,
                    tokens=decode_tokens, repetitions=repetitions,
                    timeout=timeout,
                )
                flash = profile["flash_attn"]
                if flash not in reference_files and flash not in reference_errors:
                    ref_path = workspace / f"baseline_ctx{ctx}_fa{int(flash)}.npz"
                    base = quality_subprocess(
                        model_path=model_path, corpus=corpus, n_ctx=ctx,
                        flash_attn=flash, type_k="f16", type_v="f16",
                        points=quality_points, reference=None,
                        output=ref_path, timeout=timeout,
                    )
                    if base["success"]:
                        reference_files[flash] = ref_path
                    else:
                        reference_errors[flash] = base
                if flash in reference_errors:
                    quality = {
                        "success": False,
                        "error": "Baseline FP16 in CPU KV non verificata.",
                        "reference_failure": reference_errors[flash],
                    }
                else:
                    quality = quality_subprocess(
                        model_path=model_path, corpus=corpus, n_ctx=ctx,
                        flash_attn=flash, type_k=profile["type_k"],
                        type_v=profile["type_v"], points=quality_points,
                        reference=reference_files[flash],
                        output=workspace / "unused.npz", timeout=timeout,
                    )
                ctx_row["profiles"].append({
                    "id": pid, "flash_attn": flash,
                    "type_k": profile["type_k"], "type_v": profile["type_v"],
                    "phase1_max_gpu_n_ctx": profile["max_gpu_n_ctx"],
                    "speed": speed, "quality": quality,
                    "diagnostic_score": None,
                })
                atomic_json(output, report)
            score_context(
                ctx_row["profiles"], vram_total_mib=vram_total,
                kl_scale=kl_scale, target_headroom_mib=headroom_target_mib,
            )
            atomic_json(output, report)
    report["completed"] = True
    atomic_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark KV: tok/s + PPL campionata + KL")
    parser.add_argument("--phase1-report", type=Path,
                        default=Path("phase1_kv_profiles_complete.json"))
    parser.add_argument("--corpus", type=Path, required=True,
                        help="Testo UTF-8 fisso, almeno 1280 token per il test completo")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--contexts", default="512,768,1280")
    parser.add_argument("--output", type=Path,
                        default=Path("kv_quality_speed_benchmarks.json"))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--quality-points", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--kl-scale", type=float, default=0.01)
    parser.add_argument("--headroom-target-mib", type=float, default=64)
    args = parser.parse_args()
    try:
        contexts = tuple(int(x) for x in args.contexts.split(","))
        result = run_benchmark(
            phase1_report=args.phase1_report, corpus=args.corpus,
            model_path=args.model_path, contexts=contexts,
            output=args.output, repetitions=args.repetitions,
            decode_tokens=args.decode_tokens,
            quality_points=args.quality_points, timeout=args.timeout,
            kl_scale=args.kl_scale,
            headroom_target_mib=args.headroom_target_mib,
            progress=lambda line: print(line, flush=True),
        )
        for row in result["results"]:
            print(f"n_ctx={row['n_ctx']} | {len(row['profiles'])} candidati")
            for p in sorted(row["profiles"],
                            key=lambda x: x["diagnostic_score"] or -1,
                            reverse=True):
                qual = p["quality"].get("quality") or {}
                print(
                    f" {p['id']} | prefill={p['speed'].get('median_prefill_tok_s')} "
                    f"decode={p['speed'].get('median_decode_tok_s')} "
                    f"PPL={qual.get('candidate_sampled_ppl')} "
                    f"KL={qual.get('mean_kl_nats')} "
                    f"score={p['diagnostic_score']}"
                )
        print("Report:", args.output)
        return 0 if all(
            p["diagnostic_score"] is not None
            for group in result["results"] for p in group["profiles"]
        ) else 1
    except Exception as exc:
        print(f"Benchmark KV: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
