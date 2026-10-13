"""Adaptive tensor-level Phase 2: empirical family/block -> individual -> freeze.

An experimental greedy search, NOT an optimal solver. Every decision depends
on actual same-n_ctx GPU-resident inference, independently checked native
placement, NVML peak, prefill and decode. No CPU/host-only estimate is accepted
as a successful CUDA context. Final per-step snapshots preserve learned tensor
bundles, not only homologous GGUF families.

Baseline: FA off, K/V f16, 257 tokens (historical measured reference).
Targets: explicitly requested ascending contexts, e.g. 512,768.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile

from model import ModelManager
from model.inspect import inspect_gguf

from .gpu_context import context_limit
from .mapping_grouping import group_inventory, native_tensor_trial, BLOCK_TENSOR
from .mapping_phase import probe_native_mapping
from .kv_benchmark import quality_subprocess, score_context
from .profile_pipeline import atomic_json


SCHEMA = "ruth-phase2-adaptive-tensor-greedy-v1"
FA = False
K_TYPE = "f16"
V_TYPE = "f16"


def discovery_groups(layout: dict) -> list[dict]:
    """Include both original homologous families and WHOLE transformer blocks."""
    families = [{
        "id": g["id"], "kind": "family",
        "tensor_names": list(g["tensor_names"]),
        "gguf_unique_weight_mib": g["gguf_unique_weight_mib"],
    } for g in group_inventory(layout) if g["eligible_for_trial"]]
    arch = layout["metadata"]["general.architecture"]
    count = layout["metadata"][f"{arch}.block_count"]
    by_block = {i: [] for i in range(count)}
    for tensor in layout["tensors"]:
        match = BLOCK_TENSOR.fullmatch(tensor["name"])
        if match and int(match.group(1)) < count:
            by_block[int(match.group(1))].append(tensor["name"])
    for index, names in by_block.items():
        if names:
            families.append({
                "id": f"block:{index}", "kind": "block",
                "tensor_names": sorted(names),
                "gguf_unique_weight_mib": round(sum(
                    t["bytes"] for t in layout["tensors"]
                    if t["name"] in set(names)
                ) / 1048576, 3),
            })
    return families


def marginal_value(current: dict, candidate: dict, *,
                   min_gain_mib: float = 0.25,
                   prefill_weight: float = 1.,
                   decode_weight: float = 2.) -> dict:
    """Ranking heuristic; all values are measured at SAME n_ctx."""
    result = {"eligible": False, "gain_mib": None, "prefill_loss_pct": None,
              "decode_loss_pct": None, "combined_loss_pct": None,
              "efficiency": None}
    if not current.get("success") or not candidate.get("success"):
        return result
    a = current["metrics"]
    b = candidate["metrics"]
    speed_fields = ("prefill_tok_s", "decode_tok_s", "vram_peak_mib")
    if any(
        not isinstance(a.get(key), (int, float))
        or not isinstance(b.get(key), (int, float))
        or not math.isfinite(a[key]) or not math.isfinite(b[key])
        or a[key] <= 0 or b[key] <= 0 for key in speed_fields
    ):
        return result
    gain = a["vram_peak_mib"] - b["vram_peak_mib"]
    prefill = 100 * (1 - b["prefill_tok_s"] / a["prefill_tok_s"])
    decode = 100 * (1 - b["decode_tok_s"] / a["decode_tok_s"])
    penalty = (
        prefill_weight * max(0., prefill) +
        decode_weight * max(0., decode)
    )
    ca, cb = a.get("combined_time_s"), b.get("combined_time_s")
    combined_pct = (
        100 * (cb / ca - 1)
        if isinstance(ca, (int, float)) and ca > 0
        and isinstance(cb, (int, float)) and cb > 0
        else None
    )
    result.update({
        "gain_mib": round(gain, 3),
        "combined_loss_pct": round(combined_pct, 3)
                             if combined_pct is not None else None,
        "prefill_loss_pct": round(prefill, 3),
        "decode_loss_pct": round(decode, 3),
        # Unit: MiB / (1 + weighted loss in percentage points).
        "efficiency": round(gain / (1 + penalty), 6)
                      if gain > 0 else 0.,
        "eligible": gain >= min_gain_mib,
        "metric": "measured_net_NVML_mib_per_weighted_speed_penalty",
    })
    return result


def _probe_summary(probe: dict) -> dict:
    m = probe.get("measurements") or {}
    return {
        "success": probe.get("success") is True,
        "failure_class": probe.get("failure_class"),
        "error": probe.get("error"),
        "native_placement_verified": (
            probe.get("native_placement", {}).get("verified") is True
        ),
        "gpu_kv_verified": (
            probe.get("gpu_runtime_validation", {}).get("verified") is True
        ),
        "native_kv_type_verified": (
            probe.get("kv_config_native_verification", {}).get("verified")
            is True
        ),
        "prefill_tok_s": m.get("prefill_tokens_per_s"),
        "decode_tok_s": m.get("decode_tokens_per_s"),
        "vram_peak_mib": m.get("vram_peak_sampled_mib"),
        "vram_total_mib": m.get("vram_total_mib"),
        "prompt_tokens": m.get("prompt_tokens"),
        "generated_tokens": m.get("generated_tokens"),
        "total_tokens": m.get("total_tokens"),
        "host_peak_rss_mib": m.get("host_peak_rss_mib"),
    }


def summarize_probe_repeats(rows: list[dict], expected: int) -> dict:
    success = len(rows) == expected and all(row["success"] for row in rows)
    result = {"success": success, "repetitions": len(rows), "samples": rows}
    if success:
        result["metrics"] = {
            "prefill_tok_s": round(statistics.median(
                row["prefill_tok_s"] for row in rows
            ), 3),
            "decode_tok_s": round(statistics.median(
                row["decode_tok_s"] for row in rows
            ), 3),
            "vram_peak_mib": max(row["vram_peak_mib"] for row in rows),
            "vram_total_mib": statistics.median(
                row["vram_total_mib"] for row in rows
            ),
        }
    else:
        result["failure_class"] = next(
            (row.get("failure_class") for row in rows if not row["success"]),
            "incomplete_repeats",
        )
    return result


def _measurement_key(ctx: int, tokens: int, repetitions: int,
                     host_names: list[str]) -> str:
    data = json.dumps(
        [ctx, tokens, repetitions, sorted(host_names)],
        separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:24]


def run_adaptive_tensor_search(
    *, model_path: Path, model_id: str = "gemma-2-9b-it",
    phase1_report: Path, memory_steps_report: Path,
    targets: tuple[int, ...] = (512,),
    baseline_ctx: int = 257, generated_tokens: int = 16,
    scouting_repeats: int = 2, confirm_repeats: int = 3,
    max_tests: int = 250, max_frozen: int = 40, max_backtracks: int = 2,
    min_gain_mib: float = 0.25, timeout: int = 600,
    output: Path = Path("phase2_adaptive_tensor_groups.json"),
    benchmark_report: Path | None = None, corpus: Path | None = None,
    quality_points: int = 64, rerun: bool = False,
    progress=None, probe=probe_native_mapping,
) -> dict:
    """Greedy group screening followed by singleton marginal probes.

    Each step starts from the actually validated previous n_ctx, not from
    an extrapolated memory frontier. Rollbacks are bounded, and reports
    checkpoint individual empirical measurements for deterministic resumption.
    """
    if (not targets or any(type(t) is not int for t in targets)
        or tuple(sorted(set(targets))) != targets
        or targets[0] <= baseline_ctx
        or not 64 <= baseline_ctx
        or not 1 <= generated_tokens <= 32
        or not 1 <= scouting_repeats <= 10
        or not 2 <= confirm_repeats <= 10
        or max_tests < 1 or max_frozen < 1 or max_backtracks < 0
        or timeout < 1 or quality_points < 2
        or not math.isfinite(min_gain_mib) or min_gain_mib < 0):
        raise ValueError("Parametri greedy non validi.")
    model_path = Path(model_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    phase_path = Path(phase1_report).resolve()
    steps_path = Path(memory_steps_report).resolve()
    phase_data = json.loads(phase_path.read_text("utf-8"))
    steps = json.loads(steps_path.read_text("utf-8"))
    sha_phase = hashlib.sha256(phase_path.read_bytes()).hexdigest()
    if (phase_data.get("schema") != "ruth-phase1-gpu-kv-profiles-v3"
        or steps.get("schema") != "ruth-kv-native-memory-steps-v1"
        or steps.get("source_phase1_sha256") != sha_phase
        or phase_data["parameters"]["model_id"] != model_id
        or steps["model_id"] != model_id):
        raise ValueError("Report Fase 1 e misure KV non corrispondono.")
    layout = inspect_gguf(model_path, n_ctx=targets[-1])
    limit = context_limit(layout)
    if targets[-1] > limit:
        raise ValueError("Target oltre limite architetturale GGUF.")
    groups = discovery_groups(layout)
    if not groups:
        raise RuntimeError("Nessun blocco idoneo per il mapping individuale.")
    names_available = {t for g in groups for t in g["tensor_names"]}
    all_group_lookup = {g["id"]: g for g in groups}
    base_phase = next((
        p for p in phase_data.get("profiles", [])
        if p.get("id") == "fa_off_k_f16_v_f16"
    ), None)
    if not base_phase or base_phase.get("max_gpu_n_ctx") != baseline_ctx:
        raise ValueError("Baseline storica FP16/FP16 FA OFF incoerente.")
    step_profile = next((
        p for p in steps.get("profiles", [])
        if p.get("id") == "fa_off_k_f16_v_f16"
    ), None)
    if not step_profile:
        raise ValueError("Misurazioni KV per FP16/FP16 FA OFF mancanti.")

    comparison = None
    comparison_sha = None
    corpus_path = Path(corpus).resolve() if corpus else None
    if benchmark_report is not None:
        if corpus_path is None:
            raise ValueError("Serve --corpus per PPL/KL.")
        comp_path = Path(benchmark_report).resolve()
        comparison = json.loads(comp_path.read_text("utf-8"))
        comparison_sha = hashlib.sha256(comp_path.read_bytes()).hexdigest()
        if (comparison.get("schema") != "ruth-kv-fixed-quality-speed-v1"
            or comparison.get("source_phase1_sha256") != sha_phase
            or comparison.get("model_id") != model_id
            or comparison.get("model_path") != str(model_path)
            or comparison.get("corpus_sha256") != hashlib.sha256(
                corpus_path.read_bytes()).hexdigest()
            or any(t not in comparison["contexts"] for t in targets)):
            raise ValueError("Benchmark precedente non comparabile.")

    identity = {
        "model_id": model_id, "model_path": str(model_path),
        "phase1_sha256": sha_phase,
        "steps_sha256": hashlib.sha256(steps_path.read_bytes()).hexdigest(),
        "benchmark_sha256": comparison_sha,
        "targets": list(targets), "baseline_ctx": baseline_ctx,
        "generated_tokens": generated_tokens,
        "scouting_repeats": scouting_repeats,
        "confirm_repeats": confirm_repeats,
        "max_tests": max_tests, "max_frozen": max_frozen,
        "max_backtracks": max_backtracks, "min_gain_mib": min_gain_mib,
        "timeout": timeout, "quality_points": quality_points,
    }
    if output.exists() and not rerun:
        previous = json.loads(output.read_text("utf-8"))
        if (previous.get("schema") != SCHEMA
            or previous.get("parameters") != identity):
            raise ValueError("Checkpoint incompatibile; usare --rerun.")
        if previous.get("completed"):
            return previous
        # Incomplete searches are deterministically replayed from saved
        # measured configs; no saved success is silently reinterpreted.
        cached = previous.get("measured_configs") or {}
    else:
        cached = {}
    report = {
        "schema": SCHEMA, "parameters": identity, "completed": False,
        "status": "in_progress", "model_context_limit": limit,
        "native_kv_step_evidence": {
            "observed_edge_gcd_tokens":
                step_profile.get("observed_edge_gcd_tokens"),
            "steps": step_profile.get("steps"),
            "host_257_to_512_kv_proxy_mib":
                step_profile.get("additional_kv_mib_257_to_512_host_proxy"),
            "note": "GPU VRAM change verified independently by NVML; "
                    "KV buffer steps are not the total incremental cost.",
        },
        "scanned_families": [{
            "id": g["id"], "kind": g["kind"],
            "tensor_count": len(g["tensor_names"]),
        } for g in groups],
        "measured_configs": cached, "stages": [],
        "learned_groups": [], "final_host_tensor_names": [],
        "note": (
            "Greedy rank, not exhaustive optimum. Each probe starts in an "
            "isolated CUDA process. Native CPU-tensor and GPU-KV placement "
            "must validate. Group level screens inform ONLY search order; "
            "singles are measured in context of frozen tensor set. "
            "Tentative moves are rolled back on bounded dead ends. A learned "
            "bundle is locked only after repeated successful target inference."
        ),
    }

    def save():
        atomic_json(output, report)

    def measurement(names: list[str], ctx: int, reps: int) -> dict:
        if not set(names).issubset(names_available):
            raise ValueError("Tensore fuori dai blocchi GGUF.")
        key = _measurement_key(ctx, generated_tokens, reps, names)
        if key in cached:
            return cached[key]
        if len(cached) >= max_tests:
            return {"success": False, "budget_exhausted": True}
        trial = native_tensor_trial(layout, sorted(names))
        rows = []
        for _ in range(reps):
            result = probe(
                model_path, model_id, trial, n_ctx=ctx,
                tokens=generated_tokens, timeout=timeout,
                flash_attn=FA, type_k=K_TYPE, type_v=V_TYPE,
            )
            row = _probe_summary(result)
            # The mapping probe already checks all native evidence; the
            # captured flags are retained for each repeat as diagnostics.
            rows.append(row)
            if not row["success"]:
                break
        summary = summarize_probe_repeats(rows, reps)
        if summary["success"]:
            speeds = summary["metrics"]
            speeds["combined_time_s"] = round(
                (ctx - generated_tokens) / speeds["prefill_tok_s"]
                + generated_tokens / speeds["decode_tok_s"], 6
            )
        summary["n_ctx"] = ctx
        summary["host_tensor_names"] = sorted(names)
        summary["mapping_id"] = trial["id"]
        cached[key] = summary
        save()
        if progress:
            progress({
                "event": "measured", "n_ctx": ctx,
                "cpu_tensors": len(names), "success": summary["success"],
                "tests_used": len(cached), "test_limit": max_tests,
            })
        return summary

    frozen = []
    anchor = baseline_ctx
    with tempfile.TemporaryDirectory(prefix="ruth-tensor-greedy-") as temp_dir:
        temp = Path(temp_dir)
        for target in targets:
            stage = {
                "from_n_ctx": anchor, "to_n_ctx": target,
                "start_host_tensor_names": list(frozen),
                "group_screenings": [], "individual_screenings": [],
                "frozen_tensors": [], "rejected_branches": [],
                "status": "in_progress", "successful": False,
            }
            report["stages"].append(stage)
            save()
            current = measurement(frozen, anchor, scouting_repeats)
            if not current["success"]:
                stage["status"] = "anchor_not_reproducible"
                save()
                break
            anchor_reference = current
            history = []
            rejected = set()
            ignored_groups = set()
            active_group = None
            backtracks = 0
            target_verified = False
            # A previously verified placement might already fit target.
            while True:
                if len(frozen) > max_frozen:
                    stage["status"] = "max_frozen_tensors_reached"
                    break
                attempt = measurement(frozen, target, confirm_repeats)
                if attempt.get("budget_exhausted"):
                    stage["status"] = "budget_exhausted"
                    break
                if attempt["success"]:
                    stage["validation"] = attempt
                    stage["status"] = "confirmed_gpu_resident"
                    target_verified = True
                    break

                if len(frozen) >= max_frozen:
                    stage["status"] = "max_frozen_tensors_reached"
                    break

                if active_group is None:
                    choices = []
                    state_id = tuple(sorted(frozen))
                    for group in groups:
                        names = sorted(
                            set(frozen) | set(group["tensor_names"])
                        )
                        if (len(names) == len(frozen)
                            or (state_id, group["id"]) in ignored_groups):
                            continue
                        trial = measurement(names, anchor, scouting_repeats)
                        if trial.get("budget_exhausted"):
                            break
                        marginal = marginal_value(
                            current, trial, min_gain_mib=min_gain_mib
                        )
                        record = {
                            "anchor_n_ctx": anchor,
                            "frozen_before": list(frozen),
                            "group_id": group["id"],
                            "kind": group["kind"],
                            "candidate_tensor_count":
                                len(names) - len(frozen),
                            "marginal": marginal,
                            "probe_success": trial.get("success") is True,
                        }
                        stage["group_screenings"].append(record)
                        if marginal["eligible"]:
                            choices.append((marginal["efficiency"], group))
                        save()
                    if choices:
                        choices.sort(key=lambda x: (-x[0], x[1]["id"]))
                        active_group = choices[0][1]
                        continue

                    # Bounded backtrack: try an alternative frozen tensor
                    # instead of declaring the greedy branch globally optimal.
                    if history and backtracks < max_backtracks:
                        last = history.pop()
                        if last.get("tensor") is not None:
                            rejected.add((
                                tuple(sorted(last["before"])), last["tensor"]
                            ))
                        stage["rejected_branches"].append({
                            "tensor": last["tensor"],
                            "source_group": last["group_id"],
                            "mode": last.get("mode", "single"),
                            "host_tensor_names_before": last["before"],
                            "reason": "no_more_positive_marginal_candidates",
                        })
                        frozen = list(last["before"])
                        stage["frozen_tensors"] = [
                            f for f in stage["frozen_tensors"]
                            if f["step_index"] <= len(history)
                        ]
                        current = measurement(
                            frozen, anchor, scouting_repeats
                        )
                        active_group = None
                        ignored_groups = (
                            {(tuple(sorted(frozen)), last["group_id"])}
                            if last.get("mode") == "atomic_group_fallback"
                            else set()
                        )
                        backtracks += 1
                        save()
                        continue
                    stage["status"] = (
                        "budget_exhausted"
                        if len(cached) >= max_tests else
                        "no_more_positive_marginal_candidates"
                    )
                    break

                pending = sorted(
                    set(active_group["tensor_names"]) - set(frozen)
                )
                individual_candidates = []
                base_names = list(frozen)
                for name in pending:
                    if (tuple(sorted(frozen)), name) in rejected:
                        continue
                    trial = measurement(
                        sorted(set(frozen) | {name}),
                        anchor, scouting_repeats
                    )
                    if trial.get("budget_exhausted"):
                        break
                    marginal = marginal_value(
                        current, trial, min_gain_mib=min_gain_mib
                    )
                    record = {
                        "anchor_n_ctx": anchor,
                        "frozen_before": base_names,
                        "selected_group_id": active_group["id"],
                        "tensor_name": name, "marginal": marginal,
                        "probe_success": trial.get("success") is True,
                    }
                    stage["individual_screenings"].append(record)
                    if marginal["eligible"]:
                        individual_candidates.append(
                            (marginal["efficiency"], name, trial, marginal)
                        )
                    save()
                if not individual_candidates:
                    # A coarse CUDA allocator may show ZERO incremental
                    # saving per individual tensor but a positive saving
                    # after moving the whole family. Preserve that measured
                    # interaction as an atomic learned bundle rather than
                    # erroneously rejecting all of its members.
                    bundled_names = sorted(set(frozen) | set(pending))
                    bundle_probe = (
                        measurement(bundled_names, anchor, scouting_repeats)
                        if (len(pending) > 1 and
                            len(bundled_names) <= max_frozen) else
                        {"success": False}
                    )
                    bundle_delta = marginal_value(
                        current, bundle_probe, min_gain_mib=min_gain_mib
                    )
                    if bundle_delta["eligible"]:
                        before = list(frozen)
                        history.append({
                            "before": before, "tensor": None,
                            "group_id": active_group["id"],
                            "mode": "atomic_group_fallback",
                        })
                        frozen = bundled_names
                        current = bundle_probe
                        stage["frozen_tensors"].append({
                            "step_index": len(history),
                            "mode": "atomic_group_fallback",
                            "tensors": pending,
                            "source_group": active_group["id"],
                            "anchor_n_ctx": anchor,
                            "measured_marginal": bundle_delta,
                            "host_tensor_names": list(frozen),
                            "measured_peak_vram_mib":
                                current["metrics"]["vram_peak_mib"],
                        })
                        active_group = None
                        ignored_groups = set()
                        save()
                        continue
                    ignored_groups.add((
                        tuple(sorted(frozen)), active_group["id"]
                    ))
                    active_group = None
                    if len(cached) >= max_tests:
                        stage["status"] = "budget_exhausted"
                        break
                    continue

                individual_candidates.sort(key=lambda x: (-x[0], x[1]))
                _, selected_name, best_trial, marginal = individual_candidates[0]
                history.append({
                    "before": list(frozen),
                    "tensor": selected_name,
                    "group_id": active_group["id"],
                })
                frozen = sorted(set(frozen) | {selected_name})
                current = best_trial
                stage["frozen_tensors"].append({
                    "step_index": len(history),
                    "tensor": selected_name,
                    "source_group": active_group["id"],
                    "anchor_n_ctx": anchor,
                    "measured_marginal": marginal,
                    "host_tensor_names": list(frozen),
                    "measured_peak_vram_mib":
                        current["metrics"]["vram_peak_mib"],
                })
                ignored_groups = set()
                save()

            if not target_verified:
                # Never freeze failed stage across 256-token frontier.
                frozen = stage["start_host_tensor_names"]
                report["final_host_tensor_names"] = list(frozen)
                stage["tentative_moves_not_promoted"] = True
                save()
                break

            saved_names = list(frozen)
            stage["confirmed_host_tensor_names"] = saved_names
            stage["cumulative_host_gguf_mib"] = round(
                native_tensor_trial(layout, saved_names)["host_gguf_bytes"]
                / 1048576, 3
            )
            stage["net_vram_saved_vs_stage_anchor_mib"] = round(
                anchor_reference["metrics"]["vram_peak_mib"]
                - current["metrics"]["vram_peak_mib"], 3
            ) if anchor_reference.get("success") and current.get("success") else None
            stage["measured_target_n_ctx"] = target
            bundle = {
                "id": f"learned_ctx_{target}",
                "target_n_ctx": target,
                "incremental_tensors": sorted(
                    set(saved_names) - set(stage["start_host_tensor_names"])
                ),
                "cumulative_host_tensor_names": saved_names,
                "native_mapping_id": native_tensor_trial(
                    layout, saved_names
                )["id"],
                "anchor_n_ctx": anchor,
                "confirmed_native_gpu_kv": True,
                "confirmation_rounds": confirm_repeats,
                "observed_step_tokens": target - anchor,
                "group_origin_path": [
                    item["source_group"] for item in stage["frozen_tensors"]
                ],
            }
            report["learned_groups"].append(bundle)
            report["final_host_tensor_names"] = saved_names
            save()

            if comparison is not None:
                ctx_existing = next(
                    c for c in comparison["results"]
                    if c["n_ctx"] == target
                )["profiles"]
                ref_path = temp / f"ref_{target}.npz"
                fp16_control = quality_subprocess(
                    model_path=model_path, corpus=corpus_path,
                    n_ctx=target, flash_attn=FA, type_k=K_TYPE, type_v=V_TYPE,
                    points=quality_points, reference=None,
                    output=ref_path, timeout=timeout,
                )
                stage["quality_fp16_reference_verified"] = (
                    fp16_control.get("success") is True
                )
                if fp16_control.get("success"):
                    candidate_quality = quality_subprocess(
                        model_path=model_path, corpus=corpus_path, n_ctx=target,
                        flash_attn=FA, type_k=K_TYPE, type_v=V_TYPE,
                        points=quality_points, reference=ref_path,
                        output=temp / "unused.npz", timeout=timeout,
                        tensor_names=saved_names,
                    )
                else:
                    candidate_quality = {
                        "success": False, "reference_failure": fp16_control,
                    }
                stage["quality"] = candidate_quality
                speed = {
                    "success": True,
                    "median_prefill_tok_s": attempt["metrics"]["prefill_tok_s"],
                    "median_decode_tok_s": attempt["metrics"]["decode_tok_s"],
                    "max_vram_peak_mib": attempt["metrics"]["vram_peak_mib"],
                }
                stage["speed"] = speed
                synthetic = {
                    "id": bundle["id"], "speed": speed,
                    "quality": candidate_quality,
                }
                existing_rows = [{
                    "id": p["id"], "speed": p["speed"], "quality": p["quality"]
                } for p in ctx_existing]
                score_context(
                    existing_rows + [synthetic],
                    vram_total_mib=attempt["metrics"]["vram_total_mib"],
                    kl_scale=comparison["kl_scale_nats"],
                    target_headroom_mib=comparison["headroom_target_mib"],
                )
                stage["diagnostic_score"] = synthetic.get("diagnostic_score")
                stage["existing_scores_recomputed"] = [{
                    "id": p["id"], "score": p.get("diagnostic_score")
                } for p in existing_rows]
            stage["successful"] = True
            anchor = target
            save()

    report["completed"] = all(s["successful"] for s in report["stages"]) and (
        len(report["stages"]) == len(targets)
    )
    report["status"] = (
        "confirmed_all_requested_contexts"
        if report["completed"] else "incomplete_or_unverified"
    )
    save()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ruth: greedy per famiglie/blocchi e singoli tensori"
    )
    parser.add_argument("--phase1-report", type=Path,
                        default=Path("phase1_kv_profiles_complete.json"))
    parser.add_argument("--memory-steps-report", type=Path,
                        default=Path("phase1_kv_memory_steps.json"))
    parser.add_argument("--model", default="gemma-2-9b-it")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--targets", default="512")
    parser.add_argument("--baseline-ctx", type=int, default=257)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--scouting-repeats", type=int, default=2)
    parser.add_argument("--confirm-repeats", type=int, default=3)
    parser.add_argument("--max-tests", type=int, default=250)
    parser.add_argument("--max-frozen", type=int, default=40)
    parser.add_argument("--max-backtracks", type=int, default=2)
    parser.add_argument("--min-gain-mib", type=float, default=0.25)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output", type=Path,
                        default=Path("phase2_adaptive_tensor_groups.json"))
    parser.add_argument("--benchmark-report", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--quality-points", type=int, default=64)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    try:
        model = args.model_path
        if model is None:
            selection = ModelManager().selected_baseline(args.model)
            if not selection:
                raise FileNotFoundError("Baseline GGUF assente.")
            model = Path(selection["local_path"])
        report = run_adaptive_tensor_search(
            model_path=model, model_id=args.model,
            phase1_report=args.phase1_report,
            memory_steps_report=args.memory_steps_report,
            targets=tuple(int(x) for x in args.targets.split(",")),
            baseline_ctx=args.baseline_ctx,
            generated_tokens=args.tokens,
            scouting_repeats=args.scouting_repeats,
            confirm_repeats=args.confirm_repeats,
            max_tests=args.max_tests, max_frozen=args.max_frozen,
            max_backtracks=args.max_backtracks,
            min_gain_mib=args.min_gain_mib, timeout=args.timeout,
            output=args.output,
            benchmark_report=args.benchmark_report, corpus=args.corpus,
            quality_points=args.quality_points, rerun=args.rerun,
            progress=lambda x: print(json.dumps(x), flush=True),
        )
        for stage in report["stages"]:
            print(
                f"{stage['from_n_ctx']} -> {stage['to_n_ctx']}: "
                f"{stage['status']}; tensori temporanei "
                f"{len(stage['frozen_tensors'])}"
            )
        print(f"Report: {args.output}")
        return 0 if report["completed"] else 1
    except Exception as exc:
        print(f"Greedy tensor search: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
