"""Offline GGUF-driven tensor-placement candidates for llama-cpp-python.

No CUDA allocations, no quality claims, no context selection and no profile.
All candidates require native buffer verification before benchmarking.
"""

import argparse
from collections import Counter, defaultdict
import json
import re

from model import ModelManager


# Output / tied embedding are intentionally excluded: they cannot safely
# be treated as independent weights by the current Gemma 2 native adapter.
BLOCK_GROUPS = frozenset({
    "attention", "feed_forward", "normalization", "other_block",
})


def _rule(names: list[str], device: str) -> dict:
    """Build an exact-match regex (including for C++ std::regex)."""
    if not names or device not in ("cpu", "gpu"):
        raise ValueError("Regola di placement non valida.")
    escaped = "|".join(re.escape(n) for n in sorted(names))
    pattern = "^(" + escaped + ")$"
    return {"pattern": pattern, "device": device}


def _candidate(kind: str, direction: str, family: str,
               host: list[dict], gpu: list[dict], label: str) -> dict:
    """host: GPU->host; gpu: host->GPU. Byte counts are GGUF estimates."""
    host_names = sorted(t["name"] for t in host)
    gpu_names = sorted(t["name"] for t in gpu)
    host_bytes = sum(t["bytes"] for t in host)
    gpu_bytes = sum(t["bytes"] for t in gpu)
    rules = []
    if host_names:
        rules.append(_rule(host_names, "cpu"))
    if gpu_names:
        rules.append(_rule(gpu_names, "gpu"))
    return {
        "id": label,
        "kind": kind,
        "direction": direction,
        "family": family,
        "tensor_names": sorted(host_names + gpu_names),
        "host_tensor_names": host_names,
        "gpu_tensor_names": gpu_names,
        "gguf_bytes_to_host": host_bytes,
        "gguf_bytes_to_gpu": gpu_bytes,
        "estimated_vram_weight_delta_bytes": gpu_bytes - host_bytes,
        "rules": rules,
        "requires_native_verification": True,
    }


def propose_mapping_candidates(layout: dict, n_gpu_layers: int) -> dict:
    """Build individual, block-family, and balanced weight-exchange proposals.

    The default placement estimate for the verified llama.cpp adapter uses
    the final n_gpu_layers - 1 transformer blocks plus the output layer.
    This assumption must NOT be generalized to other runtimes.
    """
    metadata = layout.get("metadata") or {}
    arch = metadata.get("general.architecture")
    block_count = metadata.get(f"{arch}.block_count")
    if not isinstance(block_count, int) or block_count <= 0:
        raise ValueError("Numero dei blocchi mancante o non valido nel GGUF.")
    if not isinstance(n_gpu_layers, int) or not 1 <= n_gpu_layers <= block_count + 1:
        raise ValueError(
            f"n_gpu_layers deve essere compreso fra 1 e {block_count + 1} (output incluso)."
        )

    first_gpu_block = block_count - (n_gpu_layers - 1)
    gpu_blocks = list(range(first_gpu_block, block_count))
    cpu_blocks = list(range(first_gpu_block))
    inventory = layout.get("tensors") or []
    names = [t.get("name") for t in inventory]
    if len(names) != len(set(names)) or not all(isinstance(n, str) for n in names):
        raise ValueError("Inventario GGUF: nomi duplicati o non validi.")

    eligible = {"host": [], "gpu": []}
    for t in inventory:
        name, block = t.get("name"), t.get("block")
        if (not isinstance(block, int) or block < 0 or block >= block_count
            or t.get("group") not in BLOCK_GROUPS
            or not isinstance(t.get("bytes"), int) or t["bytes"] <= 0
            or not re.fullmatch(r"blk\." + str(block) + r"\..+\.weight", name)):
            continue

        # 'host' means already CPU in the *estimated baseline*, not an
        # explicit native verification of that tensor's current buffer.
        baseline_domain = "gpu" if block >= first_gpu_block else "host"
        eligible[baseline_domain].append(t)

    for items in eligible.values():
        items.sort(key=lambda t: t["name"])

    candidates = [_candidate(
        "baseline", "unchanged", "all", [], [], "baseline"
    )]

    by_domain_group = {"gpu": defaultdict(list), "host": defaultdict(list)}
    for domain, tensors in eligible.items():
        for tensor in tensors:
            target = "host" if domain == "gpu" else "gpu"
            direction = "gpu_to_host" if domain == "gpu" else "host_to_gpu"
            candidates.append(_candidate(
                "single_tensor", direction, tensor["group"],
                [tensor] if target == "host" else [],
                [tensor] if target == "gpu" else [],
                f"single:{direction}:{tensor['name']}",
            ))
            by_domain_group[domain][(tensor["block"], tensor["group"])].append(tensor)

    for domain in ("gpu", "host"):
        direction = "gpu_to_host" if domain == "gpu" else "host_to_gpu"
        for (block, family), tensors in sorted(by_domain_group[domain].items()):
            if len(tensors) < 2:
                continue
            candidates.append(_candidate(
                "block_family", direction, family,
                tensors if domain == "gpu" else [],
                tensors if domain == "host" else [],
                f"family:{direction}:blk.{block}:{family}",
            ))

    # Bounded swaps: for each host-resident tensor, find one GPU-resident
    # tensor in the same family that is at least as large in GGUF bytes.
    # This avoids proposing every O(N^2) pair and ensures non-positive
    # *estimated weight* VRAM change, but says nothing about runtime scratch.
    gpu_by_family = defaultdict(list)
    for t in eligible["gpu"]:
        gpu_by_family[t["group"]].append(t)

    for promote in eligible["host"]:
        evictions = [
            t for t in gpu_by_family[promote["group"]]
            if t["bytes"] >= promote["bytes"]
        ]
        if not evictions:
            continue
        evict = min(evictions, key=lambda t: (t["bytes"] - promote["bytes"], t["name"]))
        candidates.append(_candidate(
            "tensor_swap", "swap", promote["group"],
            [evict], [promote],
            f"swap:{evict['name']}->{promote['name']}",
        ))

    # Check that *every* emitted native regex matches exactly the expected
    # GGUF tensor names, and protects ALL output / embedding tensors.
    protected = {name for name in names if (
        name.startswith("token_embd.") or name.startswith("output.")
    )}
    for c in candidates:
        seen = set()
        for rule in c["rules"]:
            actual = {name for name in names if re.search(rule["pattern"], name)}
            expected = set(c["host_tensor_names"] if rule["device"] == "cpu"
                           else c["gpu_tensor_names"])
            if not actual or actual != expected or actual & protected or actual & seen:
                raise RuntimeError(f"Regole native non sicure per {c['id']}")
            seen.update(actual)
        if seen != set(c["tensor_names"]):
            raise RuntimeError(f"Inventario incompleto per {c['id']}")
        if c["direction"] == "swap" and c["estimated_vram_weight_delta_bytes"] > 0:
            raise RuntimeError(f"Scambio stimato troppo costoso: {c['id']}")

    counts = Counter(c["direction"] for c in candidates)
    families = Counter(c["family"] for c in candidates if c["direction"] != "unchanged")

    # Show diverse candidates first without pretending that this is a
    # performance ranking or making a static GGUF size a quality metric.
    by_direction = defaultdict(list)
    for c in candidates[1:]:
        by_direction[c["direction"]].append(c)
    preview = [candidates[0]]
    for family in ("attention", "feed_forward", "normalization", "other_block"):
        for direction in ("gpu_to_host", "host_to_gpu", "swap"):
            found = next((c for c in by_direction[direction] if c["family"] == family), None)
            if found is not None:
                preview.append(found)

    return {
        "phase": "granular_mapping_candidate_generation",
        "architecture": arch,
        "block_count": block_count,
        "n_gpu_layers": n_gpu_layers,
        "baseline_mapping": {
            "output": "gpu_required",
            "repeating_blocks_gpu": gpu_blocks,
            "repeating_blocks_cpu": cpu_blocks,
            "assumption": "verified_llama_cpp_default_output_first_then_last_blocks",
        },
        "output_rule": {
            "require_gpu": True,
            "protected_names": sorted(protected),
            "validation": "native_load_log_required",
        },
        "supported_families": sorted(BLOCK_GROUPS),
        "directions": ["gpu_to_host", "host_to_gpu", "swap"],
        "candidate_count": len(candidates),
        "counts_by_direction": dict(sorted(counts.items())),
        "counts_by_family": dict(sorted(families.items())),
        "preview": preview,
        "candidates": candidates,
        "context_selected": False,
        "profile_generated": False,
        "notes": (
            "Contesto operativo non selezionato. Posizioni di partenza stimate dalla "
            "regola di offload standard del runtime: occorre verificarle su hardware. "
            "Byte GGUF non equivalgono a VRAM usata o liberata. Scambi limitati "
            "al singolo tensore per famiglia e a delta pesi non positivo, "
            "senza garanzie su allocazioni temporanee o stabilita'. "
            "Nessuna ottimizzazione, qualita' o velocita' e' stata validata."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Proposte statiche bidirezionali di mapping, senza utilizzare CUDA"
    )
    parser.add_argument("--gpu-layers", type=int, default=32)
    parser.add_argument("--show", type=int, default=8,
                        help="Quanti esempi diversi mostrare (baseline inclusa)")
    parser.add_argument("--all", action="store_true",
                        help="Stampa l'intero elenco, non soltanto l'anteprima")
    args = parser.parse_args()
    if args.show < 1:
        parser.error("--show deve essere positivo.")

    result = propose_mapping_candidates(
        ModelManager().inspect(n_ctx=512), args.gpu_layers
    )
    if args.all:
        result.pop("preview")
    else:
        result["preview"] = result["preview"][:args.show]
        result["displayed_candidates"] = len(result["preview"])
        result.pop("candidates")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
