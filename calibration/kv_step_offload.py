"""Phase 2: recover GPU-resident 512 context via CPU tensor-group combinations.

Read phase-1 measured KV buffer steps, use serialized group byte sizes only
to PRIORITIZE, not certify, candidate combinations. Native CUDA inference at
n_ctx=512 with GPU KV and exact CPU tensor overrides is the acceptance gate.
After confirmation, run the same throughput protocol and sampled PPL/KL as
the existing full-GPU benchmark at n_ctx=512.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import tempfile

from model import ModelManager
from model.inspect import inspect_gguf

from .gpu_context import context_limit, probe_context
from .mapping_grouping import (
    group_inventory, native_multi_group_trial, native_group_trial,
)
from .mapping_phase import (
    BASELINE_MODE, probe_native_mapping,
)
from .kv_benchmark import (
    quality_subprocess, score_context,
)
from .profile_pipeline import atomic_json


def blocked_profiles(report: dict, *, current: int = 257,
                     target: int = 512) -> list[dict]:
    """Do not assume a failed 258 probe proves 256 is the last valid n_ctx."""
    result = []
    for p in report.get("profiles", []):
        if p.get("status") == "reference_reused":
            maximum = p.get("max_gpu_n_ctx")
            if type(maximum) is int and maximum < target:
                result.append(p)
        elif p.get("status") == "confirmed":
            maximum = p.get("max_gpu_n_ctx")
            if type(maximum) is int and maximum < target:
                result.append(p)
        elif p.get("status") == "unsupported_or_inconclusive":
            search = p.get("result") or {}
            attempts = search.get("attempts", [])
            # Include only the specific tested first frontier (baseline+1).
            if any(a.get("requested_n_ctx") == current + 1 and
                   a.get("success") is False and
                   a.get("failure_class") in (
                       "native_cuda_oom",
                       "native_cuda_resource_allocation_failure",
                   ) for a in attempts):
                result.append(p)
    return result


def enumerate_group_candidates(groups: list[dict], *, proxy_mib: float | None,
                               safety_mib: float, max_groups: int,
                               max_candidates: int) -> list[list[dict]]:
    """Rank plausible combinations, preserve singletons and bound search cost."""
    eligible = [g for g in groups if g.get("eligible_for_trial") is True]
    if not eligible or max_groups < 1 or max_candidates < 1:
        return []
    if proxy_mib is not None and (not math.isfinite(proxy_mib) or proxy_mib < -0.05):
        raise ValueError("Costo nativo KV incoerente.")
    needed = max(0., proxy_mib or 0.) + safety_mib
    # The estimate may be ZERO for 257->512 because both fall into the
    # same allocation block: do not skip candidate testing in that case.
    combos = []
    for depth in range(1, min(max_groups, len(eligible)) + 1):
        for picked in itertools.combinations(eligible, depth):
            mib = sum(g["gguf_unique_weight_mib"] for g in picked)
            compute = sum(
                g.get("compute_proxy_gops_per_token") or 0 for g in picked
            )
            # Insufficient approximate bytes remain eligible in a fallback
            # only if no group combination reaches needed.
            combos.append({
                "groups": list(picked), "mib": mib,
                "compute": compute, "fits_proxy": mib >= needed,
            })
    preferred = [c for c in combos if c["fits_proxy"]]
    candidates = preferred if preferred else combos
    candidates.sort(key=lambda c: (
        len(c["groups"]), c["compute"], max(0, c["mib"] - needed),
        [g["id"] for g in c["groups"]],
    ))
    return [c["groups"] for c in candidates[:max_candidates]]


def speed_trial(model_path: Path, model_id: str, trial: dict,
                *, n_ctx: int, tokens: int, timeout: int,
                flash_attn: bool, type_k: str, type_v: str,
                repetitions: int) -> dict:
    samples = []
    for _ in range(repetitions):
        result = probe_native_mapping(
            model_path, model_id, trial,
            n_ctx=n_ctx, tokens=tokens, timeout=timeout,
            flash_attn=flash_attn, type_k=type_k, type_v=type_v,
        )
        samples.append({
            "success": result.get("success") is True,
            "failure_class": result.get("failure_class"),
            "worker_signal": result.get("worker_signal"),
            "measurements": result.get("measurements"),
            "native_host_tensor_count": trial["host_tensor_count"],
            "native_placement_verified": (
                result.get("native_placement", {}).get("verified") is True
            ),
            "native_gpu_kv_verified": (
                result.get("gpu_runtime_validation", {}).get("verified") is True
            ),
            "native_kv_type_verified": (
                result.get("kv_config_native_verification", {}).get("verified")
                is True
            ),
        })
        if not result.get("success"):
            break
    ok = len(samples) == repetitions and all(p["success"] for p in samples)
    result = {"success": ok, "repetitions": samples}
    if ok:
        values = [s["measurements"] for s in samples]
        result.update({
            "median_prefill_tok_s": round(statistics.median(
                m["prefill_tokens_per_s"] for m in values), 3),
            "median_decode_tok_s": round(statistics.median(
                m["decode_tokens_per_s"] for m in values), 3),
            "max_vram_peak_mib": max(
                m["vram_peak_sampled_mib"] for m in values),
            "median_vram_peak_mib": round(statistics.median(
                m["vram_peak_sampled_mib"] for m in values), 2),
            "confirmed_n_ctx": n_ctx, "confirmed_runs": repetitions,
            "native_group_ids": trial["group_ids"],
            "native_host_tensors": trial["host_tensor_names"],
        })
    return result


def run_step_offload(*, phase1_report: Path,
                     memory_steps_report: Path,
                     benchmark_report: Path,
                     corpus: Path,
                     model_path: Path | None = None,
                     output: Path = Path("phase2_kv_step_offload_512.json"),
                     target_ctx: int = 512,
                     reference_ctx: int = 257,
                     max_groups: int = 3,
                     max_candidates: int = 20,
                     max_validated: int = 3,
                     repetitions: int = 3,
                     tokens: int = 16,
                     quality_points: int = 64,
                     timeout: int = 600,
                     safety_mib: float = 64.0,
                     progress=None) -> dict:
    if (target_ctx <= reference_ctx or reference_ctx < 64
        or tokens < 1 or tokens > 32
        or repetitions < 2 or max_candidates < 1 or max_validated < 1
        or max_groups < 1 or quality_points < 2 or timeout < 1
        or not math.isfinite(safety_mib) or safety_mib < 0):
        raise ValueError("Parametri Fase 2 KV non validi.")
    source_path = Path(phase1_report).resolve()
    memory_path = Path(memory_steps_report).resolve()
    previous_path = Path(benchmark_report).resolve()
    corpus = Path(corpus).resolve()
    phase1 = json.loads(source_path.read_text(encoding="utf-8"))
    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    benchmark = json.loads(previous_path.read_text(encoding="utf-8"))
    if (phase1.get("schema") != "ruth-phase1-gpu-kv-profiles-v3"
        or memory.get("schema") != "ruth-kv-native-memory-steps-v1"
        or benchmark.get("schema") != "ruth-kv-fixed-quality-speed-v1"):
        raise ValueError("Report Fase1, KV steps o benchmark incompatibili.")
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if (source_hash != memory.get("source_phase1_sha256")
        or source_hash != benchmark.get("source_phase1_sha256")):
        raise ValueError("Report non originati dalla stessa Fase 1.")
    if hashlib.sha256(corpus.read_bytes()).hexdigest() != benchmark["corpus_sha256"]:
        raise ValueError("Il corpus non corrisponde al benchmark precedente.")
    model_id = phase1["parameters"]["model_id"]
    if model_id != memory["model_id"] or model_id != benchmark["model_id"]:
        raise ValueError("Modello incoerente nei report.")
    if model_path is None:
        model_path = Path(memory["model_path"])
    model_path = Path(model_path).resolve()
    if str(model_path) != memory["model_path"] or str(model_path) != benchmark["model_path"]:
        raise ValueError("Percorso GGUF diverso da quello dei benchmark.")
    layout = inspect_gguf(model_path, n_ctx=target_ctx)
    maximum = context_limit(layout)
    if target_ctx > maximum or target_ctx not in benchmark["contexts"]:
        raise ValueError("Target oltre GGUF o assente dal benchmark.")
    compare = next(x for x in benchmark["results"]
                   if x["n_ctx"] == target_ctx)
    existing = compare["profiles"]
    base = blocked_profiles(phase1, current=reference_ctx, target=target_ctx)
    groups = group_inventory(layout)
    memory_by_id = {p["id"]: p for p in memory["profiles"]}

    gpu_total = float(
        next((p["result"]["gpu_total_mib"] for p in phase1["profiles"]
              if p.get("result") and p["result"].get("gpu_total_mib")), 0.)
    )
    if gpu_total <= 0:
        raise ValueError("VRAM totale assente dal report.")
    report = {
        "schema": "ruth-phase2-kv-step-recovery-v1",
        "model_id": model_id, "n_ctx": target_ctx,
        "reference_n_ctx": reference_ctx, "model_limit": maximum,
        "phase1_sha256": source_hash,
        "native_steps_sha256": hashlib.sha256(memory_path.read_bytes()).hexdigest(),
        "baseline_benchmark_sha256": hashlib.sha256(previous_path.read_bytes()).hexdigest(),
        "corpus_sha256": benchmark["corpus_sha256"],
        "max_groups": max_groups, "max_candidates": max_candidates,
        "max_validated": max_validated, "repetitions": repetitions,
        "tokens": tokens, "margin_reserve_proxy_mib": safety_mib,
        "quality_kv_location": "cpu",
        "speed_kv_location": "gpu",
        "existing_full_gpu_comparison": [{
            "id": p["id"],
            "diagnostic_score": p["diagnostic_score"],
            "speed": {k: p["speed"].get(k) for k in (
                "median_prefill_tok_s", "median_decode_tok_s", "max_vram_peak_mib",
            )},
            "quality": (p["quality"].get("quality") or {}),
        } for p in existing],
        "profiles": [], "completed": False,
        "note": (
            "GGUF group bytes and CPU-KV buffer changes are heuristic "
            "planning only. Actual CUDA n_ctx=512 with exact placement and "
            "GPU KV, repeated independently, is mandatory. Sampled quality "
            "uses matched CPU KV and same corpus versus full-GPU-weight FP16 "
            "reference. Scores are recomputed with 512 full-GPU candidates "
            "and must never be interpreted as operational safety."
        ),
    }

    with tempfile.TemporaryDirectory(prefix="ruth-kv-phase2-") as work:
        temp = Path(work)
        for profile in base:
            pid = profile["id"]
            measured = memory_by_id.get(pid)
            if measured is None:
                report["profiles"].append({
                    "id": pid, "success": False,
                    "error": "Misurazione del passo KV non presente.",
                    "candidates": [],
                })
                atomic_json(Path(output), report)
                continue
            kv_proxy = measured.get("additional_kv_mib_257_to_512_host_proxy")
            if kv_proxy is None:
                report["profiles"].append({
                    "id": pid, "success": False,
                    "error": "Delta KV CPU 257->512 non misurabile.",
                    "candidates": [],
                })
                atomic_json(Path(output), report)
                continue
            # Measure a same-context GPU-resident all-weight baseline:
            # the byte estimate is never silently treated as measured VRAM.
            ref_gpu = probe_context(
                model_path, reference_ctx, generated_tokens=tokens,
                batch=32, timeout=timeout, model_id=model_id,
                flash_attn=profile["flash_attn"],
                type_k=profile["type_k"], type_v=profile["type_v"],
            )
            baseline_vram = (
                (ref_gpu.get("measurements") or {}).get("vram_peak_sampled_mib")
                if ref_gpu.get("success") is True else None
            )
            selected = enumerate_group_candidates(
                groups, proxy_mib=kv_proxy, safety_mib=safety_mib,
                max_groups=max_groups, max_candidates=max_candidates,
            )
            flash = profile["flash_attn"]
            k, v = profile["type_k"], profile["type_v"]
            phase_row = {
                "id": pid, "flash_attn": flash,
                "type_k": k, "type_v": v,
                "previous_status": profile["status"],
                "kv_allocation_257_to_512_host_proxy_mib": kv_proxy,
                "candidate_group_count": len(selected),
                "same_ctx_full_gpu_reference_verified": baseline_vram is not None,
                "same_ctx_full_gpu_reference_peak_mib": baseline_vram,
                "candidates": [], "success": False,
            }
            report["profiles"].append(phase_row)

            ref_path = temp / f"ref_fa{int(flash)}_{pid}.npz"
            reference = quality_subprocess(
                model_path=model_path, corpus=corpus, n_ctx=target_ctx,
                flash_attn=flash, type_k="f16", type_v="f16",
                points=quality_points, reference=None, output=ref_path,
                timeout=timeout,
            )
            phase_row["quality_reference_valid"] = reference.get("success") is True
            if not reference.get("success"):
                phase_row["quality_reference_failure"] = reference
            validated = 0
            for chosen in selected:
                trial = native_multi_group_trial(layout, chosen)
                if progress:
                    progress(f"{pid} | {','.join(trial['group_ids'])}")
                # Cheap preflight also measures net change in actual NVML
                # peak at the SAME context (weights+KV+scratch together).
                preflight = probe_native_mapping(
                    model_path, model_id, trial, n_ctx=reference_ctx,
                    tokens=tokens, timeout=timeout, flash_attn=flash,
                    type_k=k, type_v=v,
                )
                preflight_peak = (
                    (preflight.get("measurements") or {}).get("vram_peak_sampled_mib")
                    if preflight.get("success") is True else None
                )
                net_saving = (
                    round(baseline_vram - preflight_peak, 2)
                    if baseline_vram is not None and preflight_peak is not None
                    else None
                )
                measure = speed_trial(
                    model_path, model_id, trial, n_ctx=target_ctx,
                    tokens=tokens, timeout=timeout, flash_attn=flash,
                    type_k=k, type_v=v,
                    repetitions=(repetitions if preflight.get("success") else 0),
                ) if preflight.get("success") else {
                    "success": False,
                    "error": "Preflight n_ctx iniziale fallito.",
                    "failure_class": preflight.get("failure_class"),
                    "repetitions": [],
                }
                row = {
                    "id": trial["id"], "host_group_ids": trial["group_ids"],
                    "host_tensor_names": trial["host_tensor_names"],
                    "host_tensor_count": trial["host_tensor_count"],
                    "gguf_host_weight_mib_proxy": round(
                        trial["host_gguf_bytes"] / 1048576, 3
                    ),
                    "preflight_same_ctx": {
                        "n_ctx": reference_ctx,
                        "success": preflight.get("success") is True,
                        "net_vram_peak_saved_vs_full_gpu_mib": net_saving,
                        "baseline_vram_peak_mib": baseline_vram,
                        "candidate_vram_peak_mib": preflight_peak,
                        "note": "Delta NVML misura il sistema completo a pari n_ctx,"
                                " non i soli byte fisici di tensori spostati.",
                    },
                    "speed": measure, "quality": {"success": False},
                    "diagnostic_score": None,
                }
                if measure["success"] and reference.get("success"):
                    row["quality"] = quality_subprocess(
                        model_path=model_path, corpus=corpus, n_ctx=target_ctx,
                        flash_attn=flash, type_k=k, type_v=v,
                        points=quality_points, reference=ref_path,
                        output=temp / "unused.npz", timeout=timeout,
                        group_ids=trial["group_ids"],
                    )
                row["success"] = (
                    measure["success"] and row["quality"].get("success") is True
                )
                phase_row["candidates"].append(row)
                if row["success"]:
                    validated += 1
                    phase_row["success"] = True
                atomic_json(Path(output), report)
                if validated >= max_validated:
                    break

            # Reuse only existing validated 512 rows to create a comparable
            # normalization population; no new GGUF profiles are promoted.
            full_gpu = [{
                "id": r["id"], "speed": r["speed"], "quality": r["quality"],
            } for r in existing if r.get("speed", {}).get("success") and
                             r.get("quality", {}).get("success")]
            score_context(
                full_gpu + phase_row["candidates"],
                vram_total_mib=gpu_total, kl_scale=benchmark["kl_scale_nats"],
                target_headroom_mib=benchmark["headroom_target_mib"],
            )
            phase_row["normalized_full_gpu_scores"] = [{
                "id": r["id"], "diagnostic_score": r.get("diagnostic_score"),
            } for r in full_gpu]
            atomic_json(Path(output), report)
    report["completed"] = True
    atomic_json(Path(output), report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fase 2: sblocca 512 token con combinazioni di gruppi CPU"
    )
    parser.add_argument("--phase1-report", type=Path,
                        default=Path("phase1_kv_profiles_complete.json"))
    parser.add_argument("--memory-steps-report", type=Path,
                        default=Path("phase1_kv_memory_steps.json"))
    parser.add_argument("--benchmark-report", type=Path,
                        default=Path("kv_quality_speed_benchmarks.json"))
    parser.add_argument("--corpus", type=Path,
                        default=Path("calibration/corpora/kv_eval_it_v1.txt"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("phase2_kv_step_offload_512.json"))
    parser.add_argument("--target-ctx", type=int, default=512)
    parser.add_argument("--reference-ctx", type=int, default=257)
    parser.add_argument("--max-groups", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--max-validated", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--quality-points", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--safety-mib", type=float, default=64.0)
    args = parser.parse_args()
    try:
        result = run_step_offload(
            phase1_report=args.phase1_report,
            memory_steps_report=args.memory_steps_report,
            benchmark_report=args.benchmark_report, corpus=args.corpus,
            model_path=args.model_path, output=args.output,
            target_ctx=args.target_ctx, reference_ctx=args.reference_ctx,
            max_groups=args.max_groups, max_candidates=args.max_candidates,
            max_validated=args.max_validated, repetitions=args.repetitions,
            tokens=args.tokens, quality_points=args.quality_points,
            timeout=args.timeout, safety_mib=args.safety_mib,
            progress=lambda line: print(line, flush=True),
        )
        for row in result["profiles"]:
            print(
                row["id"], "success=", row["success"],
                "candidates=", len(row["candidates"]),
            )
        print("Report:", args.output)
        return 0 if result["completed"] and all(
            p["success"] for p in result["profiles"]
        ) else 1
    except Exception as exc:
        print(f"Fase 2 KV steps: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
