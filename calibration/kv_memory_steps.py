"""Measure native llama.cpp KV allocation steps without assuming 256-token rounding.

All-weights-GPU / CPU-KV probes let us inspect allocations even when GPU KV
cannot fit. CPU-reported KV buffer changes are a *proxy* for GPU KV bytes, not
a direct measurement of total freed VRAM. GPU-KV native records from historical
phase-1 successes are retained separately if available.
"""
import argparse
import hashlib
import json
from pathlib import Path

from model import ModelManager
from model.inspect import inspect_gguf

from .gpu_context import context_limit
from .mapping_grouping import native_group_trial
from .mapping_phase import BASELINE_MODE, probe_native_mapping
from .profile_pipeline import atomic_json


def kv_bytes_mib(native: dict, *, domain: str) -> float | None:
    rows = [
        row for row in (native or {}).get("native_buffer_records", [])
        if row.get("category") == "kv" and row.get("domain") == domain
    ]
    if not rows:
        return None
    return round(sum(row["reported_mib"] for row in rows), 4)


def changes_at_boundary(readings: dict, boundary: int) -> dict:
    """Distinguish the step crossing b->b+1 from b-1->b."""
    first = readings.get(str(boundary - 1))
    middle = readings.get(str(boundary))
    last = readings.get(str(boundary + 1))
    before = None if first is None or middle is None else round(middle - first, 4)
    after = None if middle is None or last is None else round(last - middle, 4)
    return {
        "boundary_tokens": boundary,
        "before_step_mib": before,
        "crossing_step_mib": after,
        "native_step_observed": after is not None and after > 0.01
                                and (before is None or after > before + 0.01),
        "measurement_domain": "cpu_host_kv",
        "gpu_kv_increment_mib": None,
        "note": "Memoria allocata KV CPU dai log nativi, proxy per GPU, non delta VRAM NVML.",
    }


def run_memory_steps(*, phase1_report: Path,
                     model_path: Path | None = None,
                     output: Path = Path("phase1_kv_memory_steps.json"),
                     boundaries: tuple[int, ...] = (256, 512, 1280),
                     timeout: int = 600, tokens: int = 16,
                     progress=None, probe=probe_native_mapping) -> dict:
    path = Path(phase1_report).resolve()
    source = json.loads(path.read_text(encoding="utf-8"))
    if source.get("schema") != "ruth-phase1-gpu-kv-profiles-v3":
        raise ValueError("Richiesto report KV Fase 1 v3.")
    params = source["parameters"]
    model_id = params["model_id"]
    if model_path is None:
        selected = ModelManager().selected_baseline(
            model_id, source=params.get("source")
        )
        if not selected:
            raise FileNotFoundError("Baseline GGUF non disponibile.")
        model_path = Path(selected["local_path"])
    model_path = Path(model_path).resolve()
    layout = inspect_gguf(model_path, n_ctx=256)
    limit = context_limit(layout)
    if not boundaries or any(
        type(b) is not int or b < 64 or b + 1 > limit for b in boundaries
    ):
        raise ValueError("Boundary oltre il limite architetturale GGUF.")
    trial = native_group_trial(layout, None)
    trial["id"] = BASELINE_MODE

    profiles = [{
        "id": p["id"], "flash_attn": p["flash_attn"],
        "type_k": p["type_k"], "type_v": p["type_v"],
    } for p in source["profiles"] if p["status"] in (
        "confirmed", "unsupported_or_inconclusive", "reference_reused",
    )]
    report = {
        "schema": "ruth-kv-native-memory-steps-v1",
        "source_phase1_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_id": model_id, "model_path": str(model_path),
        "model_context_limit": limit,
        "boundaries": list(boundaries), "profiles": [],
        "note": ("Native CPU-KV measured allocation is a layout proxy, not GPU "
                 "VRAM cost. Verified successful GPU-KV observations from phase 1 "
                 "are shown separately. No extrapolation beyond GGUF context cap."),
    }
    for profile in profiles:
        if progress:
            progress(profile["id"])
        contexts = sorted({
            c for b in boundaries for c in (b-1, b, b+1)
            if 64 <= c <= limit
        } | {min(limit, 257), min(limit, 512)})
        readings = {}
        probes = []
        for ctx in contexts:
            result = probe(
                model_path, model_id, trial, n_ctx=ctx,
                tokens=min(tokens, ctx - 1), timeout=timeout,
                flash_attn=profile["flash_attn"],
                type_k=profile["type_k"], type_v=profile["type_v"],
            )
            measured = (kv_bytes_mib(
                result.get("native_memory"), domain="host_cpu"
            ) if result.get("success") else None)
            if measured is None:
                measured = (kv_bytes_mib(
                    result.get("native_memory"), domain="host_cuda_pinned"
                ) if result.get("success") else None)
            probes.append({
                "n_ctx": ctx, "success": result.get("success") is True,
                "kv_native_host_mib": measured,
                "failure_class": result.get("failure_class"),
                "vram_peak_sampled_mib":
                    (result.get("measurements") or {}).get("vram_peak_sampled_mib"),
            })
            if measured is not None:
                readings[str(ctx)] = measured
        steps = [changes_at_boundary(readings, b) for b in boundaries]
        phase1 = next((p for p in source["profiles"]
                       if p["id"] == profile["id"]), None)
        gpu_measurements = []
        for attempt in (phase1 or {}).get("result", {}).get("attempts", []):
            if attempt.get("success") is not True:
                continue
            native = attempt.get("native_memory")
            value = kv_bytes_mib(native, domain="gpu_vram")
            if value is not None:
                gpu_measurements.append({
                    "n_ctx": attempt.get("requested_n_ctx"),
                    "gpu_kv_native_mib": value,
                })
        by_ctx = {v["n_ctx"]: v["gpu_kv_native_mib"] for v in gpu_measurements}
        for step in steps:
            b = step["boundary_tokens"]
            if b in by_ctx and b+1 in by_ctx:
                step["gpu_kv_increment_mib"] = round(
                    by_ctx[b+1] - by_ctx[b], 4
                )
        target = None
        if "257" in readings and "512" in readings:
            target = round(readings["512"] - readings["257"], 4)
        report["profiles"].append({
            **profile, "probes": probes, "native_host_kv_mib_by_n_ctx": readings,
            "steps": steps, "additional_kv_mib_257_to_512_host_proxy": target,
            "verified_gpu_kv_native_observations": gpu_measurements,
        })
        atomic_json(Path(output), report)
    report["completed"] = True
    atomic_json(Path(output), report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Misura step nativi KV per Fase 2")
    parser.add_argument("--phase1-report", type=Path,
                        default=Path("phase1_kv_profiles_complete.json"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("phase1_kv_memory_steps.json"))
    parser.add_argument("--boundaries", default="256,512,1280")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    try:
        result = run_memory_steps(
            phase1_report=args.phase1_report, model_path=args.model_path,
            output=args.output,
            boundaries=tuple(int(x) for x in args.boundaries.split(",")),
            timeout=args.timeout, progress=lambda x: print("KV profile", x, flush=True)
        )
        for row in result["profiles"]:
            print(row["id"], [
                (s["boundary_tokens"], s["crossing_step_mib"])
                for s in row["steps"]
            ])
        return 0 if all(
            p["additional_kv_mib_257_to_512_host_proxy"] is not None
            for p in result["profiles"]
        ) else 1
    except Exception as exc:
        print(f"KV native steps: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
