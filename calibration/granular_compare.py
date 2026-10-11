"""Compare baseline CUDA offload to an automatically selected FFN tensor override.

This experimental comparison does NOT select an operational n_ctx or save
a model profile. It uses the same 512-token context for both mapping tests.
"""

import argparse
import json
import statistics
import sys

from hardware import GPUManager
from model import ModelManager

from .benchmark import benchmark


def compare(gpu_layers: int = 32, runs: int = 2, tokens: int = 128) -> dict:
    if gpu_layers < 1 or not 1 <= runs <= 5 or not 64 <= tokens <= 256:
        raise ValueError("Richiesti gpu_layers positivo, runs 1-5, tokens 64-256.")

    layout = ModelManager().inspect(n_ctx=512)
    meta = layout["metadata"]
    arch = meta["general.architecture"]
    blocks = meta[f"{arch}.block_count"]
    if not 1 <= gpu_layers <= blocks + 1:
        raise ValueError(f"Offload {gpu_layers} incompatibile con {blocks} blocchi.")

    # Generate a candidate from the actual GGUF tensor inventory, not a
    # hard-coded tensor name. Keep output placement as the backend default.
    first_gpu_block = max(0, blocks + 1 - gpu_layers)
    gpu_blocks = set(range(first_gpu_block, blocks))
    eligible = [
        item for item in layout["tensors"]
        if item["block"] in gpu_blocks and item["group"] == "feed_forward"
        and item["name"].endswith(".ffn_down.weight")
    ]
    if not eligible:
        raise RuntimeError("Nessun tensore FFN down eleggibile nei blocchi GPU.")

    # Test the last transformer block first, matching the verified
    # native override. Further optimizer steps will explore other mappings.
    target = max(eligible, key=lambda t: t["block"])
    tensor = target["name"]

    report = {
        "phase": "granular_mapping_performance_comparison",
        "engine": "llama.cpp",
        "method": "llama-cpp-python",
        "model": "gemma-2-9b-it",
        "gpu_layers": gpu_layers,
        "mapping_candidates": [
            {"id": "baseline", "cpu_tensor": None},
            {"id": "ffn_tensor_on_host", "cpu_tensor": tensor,
             "gguf_tensor_bytes": target["bytes"]},
        ],
        "probe_n_ctx": 512,
        "selected_n_ctx": None,
        "generation_test_tokens": tokens,
        "repetitions_per_candidate": runs,
        "trials": [],
        "comparison": None,
        "success": False,
        "profile_generated": False,
        "note": (
            "Un solo tensore FFN scelto dal GGUF: verifica l'effetto di un "
            "override granulare su decode, prefill e VRAM. "
            "Non e' una ricerca completa e non dimostra equivalenza di qualita'."
        ),
    }

    for iteration in range(runs):
        for entry in report["mapping_candidates"]:
            status = GPUManager.calibration_status()
            if not status["ready"]:
                report["error"] = (
                    f"GPU non pronta: {status['errors']}; {status['foreign_processes']}"
                )
                return report

            measurement = benchmark(
                gpu_layers=gpu_layers,
                max_tokens=tokens,
                cpu_tensor=entry["cpu_tensor"],
            )
            metrics = measurement.get("measurements") or {}
            report["trials"].append({
                "run": iteration + 1,
                "candidate": entry["id"],
                "success": measurement.get("success") is True,
                "decode_tokens_per_second": metrics.get("decode_tokens_per_second"),
                "prefill_tokens_per_second": metrics.get("prefill_tokens_per_second"),
                "vram_peak_sampled_mib": metrics.get("vram_peak_sampled_mib"),
                "mapping_verification": measurement.get("mapping_verification"),
                "error": measurement.get("error"),
                "failure_type": measurement.get("failure_type"),
            })
            if not measurement.get("success"):
                report["error"] = (
                    f"Prova {entry['id']} #{iteration + 1} fallita: "
                    f"{measurement.get('error') or measurement.get('failure_type')}"
                )
                return report

    def median(field, candidate):
        values = [
            trial[field] for trial in report["trials"]
            if trial["candidate"] == candidate and trial[field] is not None
        ]
        if len(values) != runs:
            raise RuntimeError(f"Misurazioni incomplete: {candidate}, {field}")
        return round(statistics.median(values), 2)

    base_decode = median("decode_tokens_per_second", "baseline")
    host_decode = median("decode_tokens_per_second", "ffn_tensor_on_host")
    base_vram = median("vram_peak_sampled_mib", "baseline")
    host_vram = median("vram_peak_sampled_mib", "ffn_tensor_on_host")
    report["comparison"] = {
        "baseline_decode_tps": base_decode,
        "override_decode_tps": host_decode,
        "decode_percent_change": (
            round(100 * (host_decode - base_decode) / base_decode, 2)
            if base_decode > 0 else None
        ),
        "baseline_prefill_tps": median("prefill_tokens_per_second", "baseline"),
        "override_prefill_tps": median("prefill_tokens_per_second", "ffn_tensor_on_host"),
        "baseline_peak_vram_mib": base_vram,
        "override_peak_vram_mib": host_vram,
        "sampled_vram_savings_mib": round(base_vram - host_vram, 1),
    }
    report["success"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confronto diagnostico tra mapping standard e override FFN"
    )
    parser.add_argument("--gpu-layers", type=int, default=32)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()

    try:
        result = compare(args.gpu_layers, args.runs, args.tokens)
    except Exception as exc:
        result = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
