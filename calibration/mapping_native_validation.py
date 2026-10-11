"""Native validation of bidirectional GGUF tensor placement candidates.

Run exactly one candidate per isolated worker. This verifies CUDA/host
buffer choices and functional inference; it is not a speed optimizer,
a persistent profile, or a context search.
"""

import argparse
import json
import os
import re
import resource
import signal
import subprocess
import sys

from hardware import GPUManager
from model import ModelManager

from .granular_probe import _buffer_domain
from .mapping_candidates import propose_mapping_candidates


_OVERRIDE_LINE = re.compile(
    r"tensor\s+(?P<name>\S+)\s+\([^\n]*?\)\s+"
    r"buffer type overridden to\s+(?P<buffer>\S+)"
)


def choose_native_trials(report: dict) -> list[dict]:
    """Use small attention tensors to cover both directions and a swap.

    Selection is diagnostic, NOT a performance rank. The GGUF byte counts
    are used only to bound allocations in this initial native test.
    """
    candidates = report["candidates"]
    selected = []
    for direction in ("gpu_to_host", "host_to_gpu", "swap"):
        matches = [
            candidate for candidate in candidates
            if candidate["direction"] == direction
            and candidate["family"] == "attention"
            and candidate["kind"] in ("single_tensor", "tensor_swap")
        ]
        if not matches:
            raise RuntimeError(f"Nessun candidato attention per {direction}.")
        if direction == "gpu_to_host":
            chosen = min(matches, key=lambda c: (
                c["gguf_bytes_to_host"], c["id"]
            ))
        elif direction == "host_to_gpu":
            chosen = min(matches, key=lambda c: (
                c["gguf_bytes_to_gpu"], c["id"]
            ))
        else:
            chosen = min(matches, key=lambda c: (
                c["gguf_bytes_to_host"] + c["gguf_bytes_to_gpu"],
                abs(c["estimated_vram_weight_delta_bytes"]),
                c["id"],
            ))
        selected.append(chosen)
    return selected


def verify_override_logs(candidate: dict, logs: str) -> dict:
    """Verify native buffer assignment for every requested *logical* tensor.

    Gemma 2's tied token embedding has two logical uses under the same name,
    so it requires TWO native log occurrences, both on the target device.
    """
    from collections import defaultdict

    expected = {
        **{name: "host" for name in candidate["host_tensor_names"]},
        **{name: "gpu" for name in candidate["gpu_tensor_names"]},
    }
    occurrences = candidate.get("expected_native_log_occurrences") or {}
    found = defaultdict(list)
    for match in _OVERRIDE_LINE.finditer(logs):
        name, buffer_name = match.group("name", "buffer")
        found[name].append(buffer_name)

    per_tensor = []
    for name, wanted in sorted(expected.items()):
        buffer_names = found.get(name, [])
        domains = [_buffer_domain(buffer) for buffer in buffer_names]
        expected_count = occurrences.get(name, 1)
        matched = len(domains) == expected_count and all(
            domain in ("host_cpu", "host_cuda_pinned")
            if wanted == "host" else domain == "gpu_vram"
            for domain in domains
        )
        per_tensor.append({
            "name": name,
            "requested": wanted,
            "native_buffer": buffer_names[0] if buffer_names else None,
            "native_buffers": buffer_names,
            "native_domain": domains[0] if domains else "not_logged",
            "native_domains": domains,
            "native_occurrences": len(buffer_names),
            "expected_native_occurrences": expected_count,
            "verified_by_log": matched,
        })

    output_layer_gpu_default_log = "offloading output layer to GPU" in logs
    count_exact = set(found) == set(expected) and all(
        len(found[name]) == occurrences.get(name, 1) for name in expected
    )
    duplicate_logged_names = [
        name for name, buffers in found.items()
        if len(buffers) > occurrences.get(name, 1)
    ]
    target = candidate.get("output_weight_target")
    # For a tied output, the native *default* offload log is not proof of
    # physical output placement. Trust only the matching override lines.
    if target in ("host", "gpu"):
        output_placement_verified = (
            count_exact and bool(per_tensor) and all(
                item["verified_by_log"] for item in per_tensor
            )
        )
        output_placement_source = "tied_weight_native_override_logs"
    else:
        output_placement_verified = output_layer_gpu_default_log
        output_placement_source = "default_offload_log"
    verified = (
        bool(per_tensor) and count_exact and output_placement_verified
        and all(item["verified_by_log"] for item in per_tensor)
    )
    return {
        "output_gpu_verified_by_log": (
            output_layer_gpu_default_log if target not in ("host", "gpu") else
            (verified if target == "gpu" else False)
        ),
        "default_output_layer_gpu_log": output_layer_gpu_default_log,
        "output_weight_target": target or "default",
        "output_placement_verified": output_placement_verified,
        "output_placement_source": output_placement_source,
        "requested_tensor_count": len(expected),
        "native_logged_tensor_count": len(found),
        "native_logged_occurrence_count": sum(map(len, found.values())),
        "native_logged_names_exact": count_exact,
        "duplicate_logged_names": duplicate_logged_names,
        "tensors": per_tensor,
        "verified": verified,
        "evidence_note": (
            "Buffer verificati nei log DEBUG, non byte effettivamente allocati. "
            "CUDA_Host indica RAM host. Per token_embd.weight su Gemma 2 "
            "servono due occorrenze native: input e output sono accoppiati. "
            "La stringa 'offloading output layer to GPU' non prova il "
            "placement dell'output quando il peso condiviso e' sovrascritto."
        ),
    }


def _worker(layers: int, candidate_id: str) -> dict:
    from inference.adapters.llama_cpp_python import LlamaCppPythonMappingAdapter

    status = GPUManager.calibration_status()
    if not status["ready"]:
        raise RuntimeError(
            f"GPU non disponibile: {status['errors']}; {status['foreign_processes']}"
        )
    model = ModelManager()
    path = model.get("gemma-2-9b-it")
    if path is None:
        raise FileNotFoundError("GGUF Gemma 2 non installato.")

    gpu = status["gpu"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible not in (gpu["uuid"], str(gpu["index"])):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES non compatibile: {visible}")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu["uuid"])

    layout = model.inspect("gemma-2-9b-it", n_ctx=512)
    proposals = propose_mapping_candidates(layout, layers)
    candidate = next(
        (item for item in proposals["candidates"] if item["id"] == candidate_id),
        None,
    )
    if candidate is None or not candidate["rules"]:
        raise ValueError(f"Candidato sconosciuto o senza override: {candidate_id}")

    adapter = LlamaCppPythonMappingAdapter()
    llm = None
    try:
        llm, matched = adapter.load(
            path, n_gpu_layers=layers, rules=candidate["rules"], n_ctx=512
        )
        outcome = llm.create_chat_completion(
            messages=[{
                "role": "user",
                "content": "Ciao Ruth. Rispondi con una breve frase in italiano.",
            }],
            max_tokens=32,
            temperature=0.0,
        )
        choices = outcome.get("choices") or []
        reply = (choices[0]["message"].get("content") or "").strip() if choices else ""
        usage = outcome.get("usage") or {}
        if not reply or usage.get("completion_tokens", 0) < 1:
            raise RuntimeError("Generazione funzionale non riuscita.")
        return {
            "worker_ok": True,
            "candidate_id": candidate_id,
            "native_rules": matched,
            "completion_tokens": usage.get("completion_tokens"),
            "response_excerpt": reply[:120],
        }
    finally:
        if llm is not None:
            llm.close()


def validate_native(layers: int = 32, timeout: int = 180,
                    tied_output: str | None = None,
                    output_norm_host: bool = False) -> dict:
    if not isinstance(layers, int) or layers < 1 or timeout < 1:
        raise ValueError("Layer e timeout devono essere positivi.")
    layout = ModelManager().inspect("gemma-2-9b-it", n_ctx=512)
    proposals = propose_mapping_candidates(layout, layers)
    if tied_output is not None and output_norm_host:
        raise ValueError("Seleziona una sola modalita' di prova output.")
    if output_norm_host:
        selected = [
            c for c in proposals["candidates"]
            if c["kind"] == "single_tensor"
            and c["host_tensor_names"] == ["output_norm.weight"]
        ]
        if len(selected) != 1:
            raise RuntimeError("Candidato output_norm.weight host assente/ambiguo.")
    elif tied_output is None:
        selected = choose_native_trials(proposals)
    else:
        if tied_output not in ("host", "gpu"):
            raise ValueError("tied_output deve essere host oppure gpu.")
        selected = [
            c for c in proposals["candidates"]
            if c.get("output_weight_target") == tied_output
        ]
        if len(selected) != 1:
            raise RuntimeError(
                f"Candidato output condiviso {tied_output} assente/ambiguo."
            )

    report = {
        "phase": "bidirectional_native_mapping_validation",
        "model": "gemma-2-9b-it",
        "engine": "llama.cpp",
        "method": "llama-cpp-python",
        "probe_n_ctx": 512,
        "selected_n_ctx": None,
        "gpu_layers": layers,
        "candidate_count_available": proposals["candidate_count"],
        "output_placement_trial": (
            tied_output if tied_output is not None
            else "norm_host" if output_norm_host else None
        ),
        "trials": [],
        "success": False,
        "profile_generated": False,
        "note": (
            "Prove diagnostiche isolate, opzionalmente su output condiviso. "
            "Verifica placement da log nativi + generazione; "
            "nessuna misura di velocita' o selezione di profilo."
        ),
    }

    for candidate in selected:
        status = GPUManager.calibration_status()
        if not status["ready"]:
            report["error"] = (
                f"GPU cambiata: {status['errors']}; {status['foreign_processes']}"
            )
            return report

        command = [
            sys.executable, "-m", "calibration.mapping_native_validation",
            "--worker", "--gpu-layers", str(layers),
            "--candidate-id", candidate["id"],
        ]
        try:
            process = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            report["trials"].append({
                "candidate_id": candidate["id"],
                "direction": candidate["direction"],
                "success": False,
                "failure_type": "timeout",
            })
            return report

        try:
            result = json.loads(process.stdout)
        except (json.JSONDecodeError, TypeError):
            result = {"worker_ok": False, "error": "Worker senza JSON valido."}

        evidence = verify_override_logs(candidate, process.stderr)
        valid = (
            process.returncode == 0 and result.get("worker_ok") is True
            and evidence["verified"]
        )
        trial = {
            "candidate_id": candidate["id"],
            "direction": candidate["direction"],
            "family": candidate["family"],
            "gguf_bytes_to_host": candidate["gguf_bytes_to_host"],
            "gguf_bytes_to_gpu": candidate["gguf_bytes_to_gpu"],
            "worker_returncode": process.returncode,
            "worker_ok": result.get("worker_ok", False),
            "completion_tokens": result.get("completion_tokens"),
            "mapping_verification": evidence,
            "success": valid,
            "error": result.get("error"),
        }
        if process.returncode < 0:
            trial["failure_type"] = "native_signal"
            try:
                trial["native_signal"] = signal.Signals(-process.returncode).name
            except ValueError:
                trial["native_signal"] = str(-process.returncode)
        elif process.returncode != 0:
            trial["failure_type"] = "worker_error"

        if not valid:
            important = [
                line.strip() for line in process.stderr.splitlines()
                if any(word in line.lower() for word in (
                    "override", "buffer type", "cuda error",
                    "out of memory", "offloading output layer",
                    "failed", "assert",
                ))
            ]
            trial["diagnostics"] = important[-12:]
            if not important:
                trial["stderr_tail"] = process.stderr[-1200:]
        report["trials"].append(trial)
        if not valid:
            report["error"] = (
                f"Mapping {candidate['id']} non validato; prove rimanenti non eseguite."
            )
            return report

    report["success"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prove native CPU/GPU e swap su mapping GGUF"
    )
    parser.add_argument("--gpu-layers", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--tied-output", choices=("host", "gpu"), default=None,
                        help="Verifica separatamente il peso condiviso input/output Gemma 2")
    parser.add_argument("--output-norm-host", action="store_true",
                        help="Sposta solo output_norm.weight sulla RAM per verifica")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--candidate-id", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    try:
        if args.worker:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            if not args.candidate_id:
                raise ValueError("ID del candidato mancante.")
            result = _worker(args.gpu_layers, args.candidate_id)
        else:
            result = validate_native(args.gpu_layers, args.timeout,
                                     tied_output=args.tied_output,
                                     output_norm_host=args.output_norm_host)
    except Exception as exc:
        result = {
            "worker_ok": False, "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.worker:
        if not result.get("worker_ok"):
            raise SystemExit(1)
    elif not result.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
