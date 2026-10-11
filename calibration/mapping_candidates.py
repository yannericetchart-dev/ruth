"""Generate *candidate* granular placements from real GGUF tensor metadata.

No GPU is used here. Neither a production context nor a validated profile is
selected. Runtime placement and quality must be tested in separate workers.
"""

import argparse
import json
import re

from model import ModelManager


def propose_mapping_candidates(layout: dict, n_gpu_layers: int) -> dict:
    """Enumerate FFN tensors currently mapped to GPU by default llama.cpp.

    The current llama.cpp layer policy uses one offload slot for the output,
    and places the remaining slots on the last transformer blocks.
    This policy is adapter-specific and must be re-verified for other engines.
    """
    metadata = layout.get("metadata") or {}
    architecture = metadata.get("general.architecture")
    block_count = metadata.get(f"{architecture}.block_count")
    if not isinstance(block_count, int) or block_count <= 0:
        raise ValueError("Il GGUF non dichiara un numero valido di blocchi.")
    if not isinstance(n_gpu_layers, int) or not 1 <= n_gpu_layers <= block_count + 1:
        raise ValueError(
            f"n_gpu_layers deve essere tra 1 e {block_count + 1}, incluso l'output."
        )

    gpu_block_count = min(block_count, n_gpu_layers - 1)
    first_gpu_block = block_count - gpu_block_count
    gpu_block_indices = list(range(first_gpu_block, block_count))

    tensors = layout.get("tensors") or []
    names = [tensor.get("name") for tensor in tensors]
    if len(set(names)) != len(names):
        raise ValueError("Il catalogo GGUF contiene nomi tensoriali duplicati.")

    by_block = {}
    for tensor in tensors:
        block = tensor.get("block")
        if block not in gpu_block_indices:
            continue
        if tensor.get("group") != "feed_forward":
            continue
        if not isinstance(tensor.get("bytes"), int) or tensor["bytes"] <= 0:
            continue
        name = tensor["name"]
        if not isinstance(name, str) or not name.startswith(f"blk.{block}."):
            continue
        if not name.endswith(".weight"):
            continue
        by_block.setdefault(block, []).append(tensor)

    choices = [
        {
            "id": "baseline",
            "kind": "unchanged_mapping",
            "tensor_names": [],
            "gguf_weight_bytes_to_host": 0,
            "rules": [],
        }
    ]

    for block, items in sorted(by_block.items()):
        for tensor in sorted(items, key=lambda item: item["name"]):
            name = tensor["name"]
            choices.append({
                "id": f"single:{name}",
                "kind": "single_ffn_weight_to_host",
                "tensor_names": [name],
                "gguf_weight_bytes_to_host": tensor["bytes"],
                "rules": [{"pattern": "^" + re.escape(name) + "$", "device": "cpu"}],
            })

        if len(items) > 1:
            names_in_block = sorted(tensor["name"] for tensor in items)
            exact = "|".join(re.escape(name) for name in names_in_block)
            choices.append({
                "id": f"group:blk.{block}.ffn",
                "kind": "ffn_weight_group_to_host",
                "tensor_names": names_in_block,
                "gguf_weight_bytes_to_host": sum(t["bytes"] for t in items),
                "rules": [{"pattern": "^(?:" + exact + ")$", "device": "cpu"}],
            })

    # This is a listing order, not a performance-based profile ranking.
    candidates = [choices[0]] + sorted(
        choices[1:],
        key=lambda c: (-c["gguf_weight_bytes_to_host"], c["id"]),
    )
    # Guard against accidentally targeting Gemma's tied embedding/output.
    for candidate in candidates:
        for rule in candidate["rules"]:
            matched = [name for name in names if re.search(rule["pattern"], name)]
            if matched != candidate["tensor_names"]:
                # Group names are sorted; the inventory order may differ.
                if sorted(matched) != sorted(candidate["tensor_names"]):
                    raise RuntimeError(
                        f"Override ambiguo {candidate['id']}: {matched}"
                    )
            if "token_embd.weight" in matched:
                raise RuntimeError("L'output condiviso non puo' essere spostato su host.")

    return {
        "phase": "granular_mapping_candidate_generation",
        "architecture": architecture,
        "block_count": block_count,
        "n_gpu_layers": n_gpu_layers,
        "baseline_mapping": {
            "output": "gpu_required",
            "repeating_blocks_gpu": gpu_block_indices,
            "repeating_blocks_cpu": list(range(first_gpu_block)),
        },
        "output_rule": {
            "require_gpu": True,
            "validation": "native_load_log_required",
            "gemma2_tied_tensor": "token_embd.weight" if architecture == "gemma2" else None,
        },
        "candidates": candidates,
        "candidate_count": len(candidates),
        "context_selected": False,
        "profile_generated": False,
        "notes": (
            "I byte indicano dimensioni GGUF, non risparmi di VRAM dimostrati. "
            "Si propongono spostamenti FFN GPU->host senza alterare il GGUF. "
            "Nessuna velocita', qualita' o allocazione reale e' stata misurata. "
            "Il mapping di partenza segue la politica standard del backend testato."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera mapping granulari teorici da un GGUF, senza caricare la GPU"
    )
    parser.add_argument(
        "--gpu-layers", type=int, default=32,
        help="Mapping iniziale diagnostico (32 = precedente baseline verificata)",
    )
    parser.add_argument(
        "--show", type=int, default=12,
        help="Numero di candidati visualizzati, incluso baseline; --all per tutti",
    )
    parser.add_argument("--all", action="store_true", help="Visualizza tutti i candidati")
    args = parser.parse_args()
    if args.show < 1:
        parser.error("--show deve essere positivo")

    result = propose_mapping_candidates(
        ModelManager().inspect(n_ctx=512), args.gpu_layers
    )
    if not args.all:
        result["candidates"] = result["candidates"][:args.show]
        result["displayed_candidates"] = len(result["candidates"])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
