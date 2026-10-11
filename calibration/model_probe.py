"""First phase: measure a model-mapping candidate independently of n_ctx selection.

The small context is a technical prerequisite for running llama.cpp, NOT
an optimization target. Generating a validated profile and selecting its
operational context are separate later phases.
"""

import argparse
import json
import sys

from model import ModelManager

from .probe import probe
from .spec import ContextResources, GenerationResources, ModelMapping


def model_mapping_probe(gpu_layers: int = 20) -> dict:
    """Evaluate one mapping candidate on the currently available CUDA adapter."""
    if gpu_layers < 1:
        raise ValueError("I layer GPU richiesti devono essere almeno 1.")

    model_id = "gemma-2-9b-it"
    layout = ModelManager().inspect(model_id, n_ctx=512)
    metadata = layout["metadata"]
    arch = metadata.get("general.architecture")
    block_count = metadata.get(f"{arch}.block_count")
    context_limit = metadata.get(f"{arch}.context_length")

    if not isinstance(block_count, int) or not 1 <= gpu_layers <= block_count:
        raise ValueError(f"Layer richiesti {gpu_layers}; il GGUF dichiara {block_count} blocchi.")
    if not isinstance(context_limit, int) or context_limit < 128:
        raise RuntimeError("Il GGUF non dichiara un contesto tecnico sufficiente per la prova.")

    placement = ModelMapping(
        model_id=model_id,
        engine="llama.cpp",
        method="llama-cpp-python",
        backend="cuda",
        placement={"kind": "gpu_layer_offload", "n_gpu_layers": gpu_layers},
    )
    generation = GenerationResources()
    context = ContextResources(
        probe_n_ctx=min(512, context_limit),
        probe_n_batch=min(128, context_limit),
        probe_n_ubatch=min(128, context_limit),
    )

    # llama-cpp-python: this adapter currently applies contiguous block
    # offload only. Future adapters may accept per-tensor placement rules.
    if placement.method != "llama-cpp-python" or placement.backend != "cuda":
        raise NotImplementedError("Adapter di calibrazione non disponibile.")
    if generation != GenerationResources():
        raise NotImplementedError("Questa prova iniziale usa risorse di generazione fisse.")

    measurement = probe(
        gpu_layers=placement.placement["n_gpu_layers"],
        n_ctx=context.probe_n_ctx,
        n_batch=context.probe_n_batch,
        n_ubatch=context.probe_n_ubatch,
        max_tokens=64,
    )

    return {
        "phase": "model_mapping",
        "model_candidate": placement.as_dict(),
        "generation_resources": generation.as_dict(),
        "context_resources": context.as_dict(),
        "model_structure": {
            "architecture": arch,
            "block_count": block_count,
            "weights_bytes": layout["weights_bytes"],
            "model_native_ctx_limit": context_limit,
        },
        "measurement": measurement,
        "candidate_valid": measurement["success"],
        "profile_generated": False,
        "notes": [
            "n_ctx della prova e' temporaneo e NON vincola il profilo modello.",
            "La generazione e' una verifica funzionale; qualita' e prestazioni devono ancora essere validate.",
            "La selezione automatica del mapping e la ricerca di n_ctx verranno dopo questa separazione.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase modello: una prova con contesto tecnico minimo")
    parser.add_argument(
        "--gpu-layers", type=int, default=20,
        help="Candidato diagnostico; la selezione automatica verra' implementata dopo",
    )
    args = parser.parse_args()

    try:
        result = model_mapping_probe(args.gpu_layers)
    except Exception as exc:
        print(f"Prova mapping non avviata: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["candidate_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
