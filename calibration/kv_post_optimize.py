"""Optimize already-confirmed GPU-KV contexts WITHOUT a VRAM reserve constraint.

Neighborhood: remove one CPU tensor, replace one CPU tensor, or add one CPU
tensor. Rank solely by decode/prefill speed measured at the SAME target n_ctx.
VRAM is recorded but never part of the objective or a minimum-headroom gate.
All GPU-KV candidates must pass independent native CUDA runs. The verified
seed is retained; incomplete runs resume via saved checkpoint with a higher
--max-tests without re-probing identical configurations.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

from model.inspect import inspect_gguf

from .gpu_context import context_limit
from .kv_adaptive_tensor_offload import (
    _probe_summary, _measurement_key, summarize_probe_repeats, discovery_groups
)
from .mapping_grouping import native_tensor_trial
from .mapping_phase import probe_native_mapping
from .kv_benchmark import quality_subprocess, score_context
from .profile_pipeline import atomic_json


SCHEMA = "ruth-phase2-post-optimize-speed-v1"


def objective(metrics: dict, *, decode_weight: float = .7) -> float:
    """Dimensionless weighted log speed, NOT weighted memory headroom."""
    if not 0. <= decode_weight <= 1.:
        raise ValueError("Peso decode tra 0 e 1.")
    d, p = metrics["decode_tok_s"], metrics["prefill_tok_s"]
    if any(not isinstance(v, (int, float)) or not math.isfinite(v)
           or v <= 0 for v in (d, p)):
        raise ValueError("Velocita non valida.")
    return decode_weight * math.log(d) + (1-decode_weight) * math.log(p)


def pareto_front(rows: list[dict]) -> list[dict]:
    """Two-speed (decode/prefill) Pareto frontier, no VRAM dimension."""
    good = [r for r in rows if r.get("success") is True
            and isinstance(r.get("metrics"), dict)]
    result = []
    for a in good:
        d, p = a["metrics"]["decode_tok_s"], a["metrics"]["prefill_tok_s"]
        dominated = any(
            b is not a
            and b["metrics"]["decode_tok_s"] >= d
            and b["metrics"]["prefill_tok_s"] >= p
            and (b["metrics"]["decode_tok_s"] > d
                 or b["metrics"]["prefill_tok_s"] > p)
            for b in good
        )
        if not dominated:
            result.append(a)
    return result


def candidate_pool(layout: dict, source: dict, *,
                   max_candidates: int) -> list[str]:
    """Prefer measurements at anchor and same suffix, then larger tensors.

    No inference is made that anchor rankings transfer perfectly to 512.
    They only prioritize which alternatives to verify first.
    """
    all_groups = discovery_groups(layout)
    known = set()
    for group in all_groups:
        known.update(group["tensor_names"])
    preferred = set(source["learned_groups"][-1]["cumulative_host_tensor_names"])
    suffixes = {name.split(".", 2)[-1] for name in preferred}
    size = {t["name"]: t["bytes"] for t in layout["tensors"]}
    screening_score = {}
    for stage in source.get("stages", []):
        for row in stage.get("individual_screenings", []):
            name = row.get("tensor_name")
            marginal = row.get("marginal") or {}
            if name in known and marginal.get("eligible") is True:
                screening_score[name] = max(
                    screening_score.get(name, -1e20),
                    marginal.get("efficiency") or 0.,
                )
    candidates = sorted(known - preferred, key=lambda name: (
        name.split(".", 2)[-1] not in suffixes,
        -screening_score.get(name, -1.0),
        -size[name],
        name,
    ))
    return candidates[:max_candidates]


def neighbors(current: list[str], pool: list[str]):
    """Deterministic order for checkpoint replay and bounded exploration."""
    occupied = set(current)
    seen = set()
    for removed in sorted(occupied):
        names = tuple(sorted(occupied - {removed}))
        if names not in seen:
            seen.add(names)
            yield {"operation": "remove", "removed": removed,
                   "added": None, "host_tensor_names": list(names)}
    for added in pool:
        if added in occupied:
            continue
        for removed in sorted(occupied):
            names = tuple(sorted((occupied - {removed}) | {added}))
            if names not in seen:
                seen.add(names)
                yield {"operation": "swap", "removed": removed,
                       "added": added, "host_tensor_names": list(names)}
        names = tuple(sorted(occupied | {added}))
        if names not in seen:
            seen.add(names)
            yield {"operation": "add", "removed": None, "added": added,
                   "host_tensor_names": list(names)}


def _valid_seed(source: dict, stage: dict, *, target: int,
                confirmation_runs: int) -> dict:
    if (source.get("schema") != "ruth-phase2-adaptive-tensor-greedy-v3"
        or source.get("completed") is not True
        or stage.get("successful") is not True
        or stage.get("measured_target_n_ctx") != target):
        raise ValueError("Serve un gruppo confermato nel report greedy v3.")
    verified = stage.get("validation") or {}
    samples = verified.get("samples") or []
    if (verified.get("success") is not True
        or verified.get("n_ctx") != target
        or verified.get("repetitions", 0) < confirmation_runs
        or len(samples) < confirmation_runs
        or not all(
            s.get("success") is True
            and s.get("native_placement_verified") is True
            and s.get("gpu_kv_verified") is True
            and s.get("native_kv_type_verified") is True
            for s in samples
        )):
        raise ValueError("La seed non ha la verifica nativa CUDA richiesta.")
    if (sorted(verified.get("host_tensor_names", [])) !=
        sorted(stage.get("confirmed_host_tensor_names", []))):
        raise ValueError("Seed: placement diverso dai tensori confermati.")
    return verified


def run_post_optimize(
    *, source_report: Path, output: Path, model_path: Path | None = None,
    target_ctx: int = 512, max_tests: int = 160, max_rounds: int = 2,
    max_candidates: int = 48, confirm_repeats: int = 3,
    decode_weight: float = .7, timeout: int = 600,
    benchmark_report: Path | None = None, corpus: Path | None = None,
    quality_points: int = 64, rerun: bool = False,
    progress=None, probe=probe_native_mapping,
) -> dict:
    if (target_ctx < 64 or max_tests < 1 or max_rounds < 1
        or max_candidates < 1 or confirm_repeats < 3
        or confirm_repeats > 10 or timeout < 1
        or quality_points < 2 or not math.isfinite(decode_weight)
        or not 0. <= decode_weight <= 1.):
        raise ValueError("Parametri ottimizzazione post-conferma non validi.")

    source_path = Path(source_report).resolve()
    output = Path(output).resolve()
    if output == source_path:
        raise ValueError("Non sovrascrivere il report greedy di origine.")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    params = source.get("parameters") or {}
    matching_stages = [
        s for s in source.get("stages", [])
        if s.get("measured_target_n_ctx") == target_ctx
    ]
    if len(matching_stages) != 1:
        raise ValueError("Target assente o ambiguo nel report iniziale.")
    seed = _valid_seed(source, matching_stages[0], target=target_ctx,
                       confirmation_runs=3)
    if confirm_repeats != seed["repetitions"]:
        raise ValueError(
            "Il numero di ripetizioni deve coincidere con le conferme "
            "native della configurazione iniziale."
        )
    if params.get("confirm_repeats", 0) < 3:
        raise ValueError("Riferimento con meno di tre conferme CUDA.")
    model_id = params["model_id"]
    model_path = (Path(model_path) if model_path is not None
                  else Path(params["model_path"])).resolve()
    if str(model_path) != params["model_path"]:
        raise ValueError("GGUF differente da quello del report greedy.")
    layout = inspect_gguf(model_path, n_ctx=target_ctx)
    if target_ctx > context_limit(layout):
        raise ValueError("Target oltre limite GGUF.")
    baseline_names = sorted(seed["host_tensor_names"])
    trial = native_tensor_trial(layout, baseline_names)
    if trial["id"] != seed["mapping_id"]:
        raise ValueError("Mapping seed non corrisponde al GGUF corrente.")
    if not baseline_names:
        raise ValueError("Nessun tensore iniziale da ottimizzare.")

    benchmark = None
    corpus_path = Path(corpus).resolve() if corpus is not None else None
    benchmark_sha = None
    if benchmark_report is not None:
        if corpus_path is None:
            raise ValueError("--benchmark-report richiede --corpus.")
        benchmark_file = Path(benchmark_report).resolve()
        benchmark = json.loads(benchmark_file.read_text("utf-8"))
        benchmark_sha = hashlib.sha256(benchmark_file.read_bytes()).hexdigest()
        if (benchmark.get("schema") != "ruth-kv-fixed-quality-speed-v1"
            or benchmark_sha != params.get("benchmark_sha256")
            or benchmark.get("model_path") != str(model_path)
            or benchmark.get("corpus_sha256") != hashlib.sha256(
                corpus_path.read_bytes()).hexdigest()
            or target_ctx not in benchmark.get("contexts", [])):
            raise ValueError("Benchmark/corpus non corrispondono al report greedy.")

    base_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    pool = candidate_pool(layout, source, max_candidates=max_candidates)
    identity = {
        "source_sha256": base_hash, "model_path": str(model_path),
        "target_n_ctx": target_ctx,
        "max_rounds": max_rounds, "max_candidates": max_candidates,
        "confirm_repeats": confirm_repeats,
        "decode_weight": decode_weight,
        "timeout": timeout,
        "generated_tokens": params["generated_tokens"],
        "benchmark_sha256": benchmark_sha,
        "quality_points": quality_points,
        "max_tests": max_tests,
    }
    prior = None
    if output.exists() and not rerun:
        prior = json.loads(output.read_text("utf-8"))
        prior_params = prior.get("parameters") or {}
        comparable = (
            prior.get("schema") == SCHEMA
            and {k: v for k, v in prior_params.items() if k != "max_tests"}
                == {k: v for k, v in identity.items() if k != "max_tests"}
        )
        if not comparable or max_tests < prior_params.get("max_tests", 0):
            raise ValueError("Checkpoint incompatibile; consentito solo aumentare --max-tests.")
        if prior.get("completed") is True:
            return prior
        measured = prior.get("measured_configs") or {}
    else:
        measured = {}
    if not isinstance(measured, dict) or len(measured) > max_tests:
        raise ValueError("Checkpoint non valido.")
    # Reuse target-context observations from the verified greedy run.
    # An earlier native CUDA OOM is a documented failed configuration,
    # not a positive benchmark. Do not consume new trials replaying it.
    old_measurements = {}
    for row in source.get("measured_configs", {}).values():
        if (not isinstance(row, dict)
            or row.get("n_ctx") != target_ctx
            or not isinstance(row.get("host_tensor_names"), list)
            or not row.get("host_tensor_names")):
            continue
        names = sorted(row["host_tensor_names"])
        if native_tensor_trial(layout, names)["id"] != row.get("mapping_id"):
            continue
        success = row.get("success") is True
        if success and (
            row.get("repetitions") != confirm_repeats
            or not all(sample.get("success") is True
                       and sample.get("native_placement_verified") is True
                       and sample.get("gpu_kv_verified") is True
                       for sample in row.get("samples", []))
        ):
            continue
        if not success and row.get("failure_class") != "native_cuda_oom":
            continue
        key = _measurement_key(target_ctx, params["generated_tokens"],
                               confirm_repeats, names)
        old_measurements[key] = row

    report = {
        "schema": SCHEMA, "parameters": identity,
        "status": "in_progress", "completed": False,
        "objective": {
            "metric": "weighted_log_throughput",
            "decode_weight": decode_weight,
            "prefill_weight": 1 - decode_weight,
            "vram_minimum_headroom_mib": None,
            "vram_objective_weight": 0,
            "note": ("Nessuna soglia VRAM imposta. La verifica nativa "
                     "CUDA a n_ctx target resta obbligatoria."),
        },
        "seed": seed, "candidate_tensor_pool": pool,
        "measured_configs": measured,
        "reused_source_configs": len(old_measurements),
        "rounds": [],
        "best_host_tensor_names": baseline_names,
        "best": seed,
        "pareto_front": [],
        "quality": None,
        "quality_warning": None,
    }

    def save():
        atomic_json(output, report)

    repeats = confirm_repeats
    tokens = params["generated_tokens"]

    def evaluate(names: list[str]) -> dict:
        key = _measurement_key(target_ctx, tokens, repeats, names)
        if sorted(names) == baseline_names and repeats == seed["repetitions"]:
            return seed
        if key in measured:
            return measured[key]
        if key in old_measurements:
            return old_measurements[key]
        if len(measured) >= max_tests:
            return {"success": False, "budget_exhausted": True}
        placement = native_tensor_trial(layout, sorted(names))
        rows = []
        for _ in range(repeats):
            result = probe(model_path, model_id, placement, n_ctx=target_ctx,
                           tokens=tokens, timeout=timeout,
                           flash_attn=False, type_k="f16", type_v="f16")
            row = _probe_summary(result)
            rows.append(row)
            if not row["success"]:
                break
        summary = summarize_probe_repeats(rows, repeats)
        summary["n_ctx"] = target_ctx
        summary["host_tensor_names"] = sorted(names)
        summary["mapping_id"] = placement["id"]
        measured[key] = summary
        save()
        if progress:
            progress({
                "n_ctx": target_ctx, "cpu_tensors": len(names),
                "success": summary["success"],
                "tests_used": len(measured), "test_limit": max_tests,
            })
        return summary

    best = seed
    current = seed
    for iteration in range(max_rounds):
        record = {
            "iteration": iteration + 1,
            "start_host_tensor_names": current["host_tensor_names"],
            "tested": [], "completed_scan": True,
            "selected_host_tensor_names": None,
        }
        report["rounds"].append(record)
        contender = current
        contender_score = objective(current["metrics"], decode_weight=decode_weight)
        for neighbor in neighbors(current["host_tensor_names"], pool):
            sample = evaluate(neighbor["host_tensor_names"])
            if sample.get("budget_exhausted"):
                record["completed_scan"] = False
                break
            row = {
                "operation": neighbor["operation"],
                "removed": neighbor["removed"], "added": neighbor["added"],
                "host_tensor_names": neighbor["host_tensor_names"],
                "success": sample.get("success") is True,
                "failure_class": sample.get("failure_class"),
            }
            if sample.get("success"):
                score = objective(sample["metrics"],
                                  decode_weight=decode_weight)
                row["objective"] = round(score, 8)
                row["prefill_tok_s"] = sample["metrics"]["prefill_tok_s"]
                row["decode_tok_s"] = sample["metrics"]["decode_tok_s"]
                row["vram_headroom_mib"] = round(
                    sample["metrics"]["vram_total_mib"]
                    - sample["metrics"]["vram_peak_mib"], 2
                )
                if score > contender_score + 1e-8:
                    contender, contender_score = sample, score
            record["tested"].append(row)
        record["selected_host_tensor_names"] = contender["host_tensor_names"]
        record["best_score"] = round(contender_score, 8)
        current = contender
        if objective(current["metrics"], decode_weight=decode_weight) > (
            objective(best["metrics"], decode_weight=decode_weight) + 1e-8
        ):
            best = current
        report["best"] = best
        report["best_host_tensor_names"] = best["host_tensor_names"]
        candidate_rows = [seed] + list(measured.values())
        report["pareto_front"] = sorted(
            [{
                "host_tensor_names": row["host_tensor_names"],
                "prefill_tok_s": row["metrics"]["prefill_tok_s"],
                "decode_tok_s": row["metrics"]["decode_tok_s"],
                "vram_headroom_mib": round(
                    row["metrics"]["vram_total_mib"]
                    - row["metrics"]["vram_peak_mib"], 2
                ),
            } for row in pareto_front(candidate_rows)],
            key=lambda x: (-x["decode_tok_s"], -x["prefill_tok_s"]),
        )
        save()
        if not record["completed_scan"]:
            report["status"] = "budget_exhausted_best_valid"
            break
        if contender["host_tensor_names"] == record["start_host_tensor_names"]:
            report["status"] = "local_optimum_found"
            break
    else:
        report["status"] = "round_limit_reached"

    report["completed"] = report["status"] != "budget_exhausted_best_valid"
    if benchmark is not None and report["completed"]:
        with tempfile.TemporaryDirectory(prefix="ruth-post-quality-") as directory:
            tmp = Path(directory)
            # Same CPU-KV reference protocol as the historical fixed benchmark.
            control = quality_subprocess(
                model_path=model_path, corpus=corpus_path, n_ctx=target_ctx,
                flash_attn=False, type_k="f16", type_v="f16",
                points=quality_points, reference=None,
                output=tmp / "reference.npz", timeout=timeout,
            )
            if control.get("success"):
                quality = quality_subprocess(
                    model_path=model_path, corpus=corpus_path,
                    n_ctx=target_ctx, flash_attn=False,
                    type_k="f16", type_v="f16", points=quality_points,
                    reference=tmp / "reference.npz",
                    output=tmp / "unused.npz", timeout=timeout,
                    tensor_names=best["host_tensor_names"],
                )
            else:
                quality = {"success": False, "reference_failure": control}
            report["quality"] = quality
            if quality.get("success"):
                ctx = next(
                    row for row in benchmark["results"]
                    if row["n_ctx"] == target_ctx
                )
                population = [
                    {"id": p["id"], "speed": p["speed"], "quality": p["quality"]}
                    for p in ctx["profiles"]
                ]
                optimized = {
                    "id": "post_optimized",
                    "speed": {
                        "success": True,
                        "median_prefill_tok_s": best["metrics"]["prefill_tok_s"],
                        "median_decode_tok_s": best["metrics"]["decode_tok_s"],
                        "max_vram_peak_mib": best["metrics"]["vram_peak_mib"],
                    },
                    "quality": quality,
                }
                score_context(
                    population + [optimized],
                    vram_total_mib=best["metrics"]["vram_total_mib"],
                    kl_scale=benchmark["kl_scale_nats"],
                    target_headroom_mib=benchmark["headroom_target_mib"],
                )
                report["diagnostic_score"] = optimized.get("diagnostic_score")
                report["historical_scores_recomputed"] = [
                    {"id": x["id"], "diagnostic_score": x.get("diagnostic_score")}
                    for x in population
                ]
            else:
                report["quality_warning"] = (
                    "Qualita non verificata; il miglior candidato e solo "
                    "una soluzione di performance CUDA, non un profilo operativo."
                )
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description="Ottimizzazione speed senza margine VRAM.")
    parser.add_argument("--source-report", type=Path,
                        default=Path("phase2_adaptive_tensor_groups_v3.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("phase2_post_optimize_512.json"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--target-ctx", type=int, default=512)
    parser.add_argument("--max-tests", type=int, default=160)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=48)
    parser.add_argument("--confirm-repeats", type=int, default=3)
    parser.add_argument("--decode-weight", type=float, default=.7)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--benchmark-report", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--quality-points", type=int, default=64)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    try:
        result = run_post_optimize(
            source_report=args.source_report, output=args.output,
            model_path=args.model_path, target_ctx=args.target_ctx,
            max_tests=args.max_tests, max_rounds=args.max_rounds,
            max_candidates=args.max_candidates,
            confirm_repeats=args.confirm_repeats,
            decode_weight=args.decode_weight, timeout=args.timeout,
            benchmark_report=args.benchmark_report,
            corpus=args.corpus, quality_points=args.quality_points,
            rerun=args.rerun,
            progress=lambda data: print(json.dumps(data), flush=True),
        )
        print(json.dumps({
            "status": result["status"],
            "tests_used": len(result["measured_configs"]),
            "best_tensor_names": result["best_host_tensor_names"],
            "prefill_tok_s": result["best"]["metrics"]["prefill_tok_s"],
            "decode_tok_s": result["best"]["metrics"]["decode_tok_s"],
            "vram_peak_mib": result["best"]["metrics"]["vram_peak_mib"],
            "output": str(args.output),
        }, indent=2))
        return 0 if result["completed"] else 1
    except Exception as exc:
        print(f"Ruth post optimization: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
