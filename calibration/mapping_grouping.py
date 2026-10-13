"""GGUF-derived homologous tensor-group ranking for Ruth mapping phase 2.

Classify repeating blk.N.<suffix> tensors without model-specific allowlists.
Everything below is a screening heuristic, never a measured CUDA saving or
performance estimate. Shared, repeated native and non-block weights are
accounted for separately and not silently counted once per occurrence.
"""

from collections import defaultdict
import math
import re

from .baseline_discovery import all_gpu_rules


BLOCK_TENSOR = re.compile(r"^blk\.(\d+)\.(.+)$")
MIB = 1048576
MIN_GROUP_MIB = 16.0
MAX_TRIAL_GROUPS = 3


def group_inventory(layout: dict) -> list[dict]:
    metadata = layout.get("metadata") or {}
    arch = metadata.get("general.architecture")
    blocks = metadata.get(f"{arch}.block_count")
    if not isinstance(blocks, int) or isinstance(blocks, bool) or blocks < 1:
        raise ValueError("Mapping: numero blocchi GGUF non verificabile.")
    tensors = layout.get("tensors") or []
    if not tensors:
        raise ValueError("Mapping: GGUF senza tensori.")
    names = [tensor.get("name") for tensor in tensors]
    if not all(isinstance(x, str) and x for x in names) or len(names) != len(set(names)):
        raise ValueError("Mapping: tensori GGUF invalidi o duplicati tra shard.")
    _, all_gpu = all_gpu_rules(layout)
    occurrences = all_gpu["expected_native_log_occurrences"]
    by_group = defaultdict(list)
    for tensor in tensors:
        name = tensor["name"]
        size = tensor.get("bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"Mapping: dimensione tensore non valida: {name}")
        match = BLOCK_TENSOR.fullmatch(name)
        if match and int(match.group(1)) < blocks:
            by_group[match.group(2)].append(tensor)
        else:
            by_group["NON_BLOCK:" + name].append(tensor)

    results = []
    for suffix, items in sorted(by_group.items()):
        in_blocks = not suffix.startswith("NON_BLOCK:")
        block_indices = sorted({
            int(BLOCK_TENSOR.fullmatch(t["name"]).group(1))
            for t in items
        }) if in_blocks else []
        names = sorted(t["name"] for t in items)
        serialized_bytes = sum(t["bytes"] for t in items)
        native_occurrences = sum(occurrences.get(name, 1) for name in names)
        shapes = [t.get("shape") for t in items]
        # Proxy for ops per decoded token. Dense matrix: ~2 per parameter.
        # Vector reductions/norms: ~5 per scalar. Unknown shapes: unknown.
        compute_proxy = 0
        shape_known = True
        matrix_count = 0
        for t, shape in zip(items, shapes):
            if (not isinstance(shape, (list, tuple)) or not shape
                or any(not isinstance(d, int) or d <= 0 for d in shape)):
                shape_known = False
                continue
            elems = math.prod(shape)
            compute_proxy += (2 if len(shape) >= 2 else 5) * elems
            if len(shape) >= 2:
                matrix_count += 1
        compute_gops = compute_proxy / 1e9 if shape_known else None
        mib = serialized_bytes / MIB
        # Rank intentionally balances meaningful GGUF bytes vs compute
        # exposure and number of calls. It CANNOT predict PCIe transfers,
        # cache behavior or runtime freed VRAM. Such effects need benchmarks.
        score = (
            round(mib / (1 + 8 * compute_gops + 0.03 * native_occurrences), 5)
            if in_blocks and compute_gops is not None else None
        )
        coupling = (
            "native_repeated_or_shared"
            if any(occurrences.get(name, 1) > 1 for name in names)
            else "none_detected"
        )
        eligible = (
            in_blocks and shape_known and score is not None
            and coupling == "none_detected" and mib >= MIN_GROUP_MIB
        )
        results.append({
            "id": ("group:" + suffix if in_blocks else suffix),
            "tensor_suffix": suffix if in_blocks else None,
            "tensor_names": names,
            "tensor_count": len(names),
            "block_indices": block_indices,
            "block_coverage": len(block_indices),
            "total_model_blocks": blocks,
            "gguf_unique_weight_bytes": serialized_bytes,
            "gguf_unique_weight_mib": round(mib, 3),
            "native_occurrences_expected": native_occurrences,
            "native_duplicate_uses": native_occurrences - len(names),
            "native_coupling": coupling,
            "gguf_types": sorted({str(t.get("type", "unknown")) for t in items}),
            "matrix_tensor_count": matrix_count,
            "compute_proxy_ops_per_token": compute_proxy if shape_known else None,
            "compute_proxy_gops_per_token": (
                round(compute_gops, 5) if compute_gops is not None else None
            ),
            "estimated_transfer_cost": "unknown_until_native_benchmark",
            "ranking_score_heuristic": score,
            "ranking_provenance": (
                "serialized_GGUF_bytes_and_shape_ops_proxy_not_measured_speed"
            ),
            "eligible_for_trial": eligible,
            "skip_reason": (
                None if eligible else
                "shared_or_non_block" if not in_blocks or coupling != "none_detected"
                else "incomplete_shape_metadata" if not shape_known
                else "below_minimum_memory_benefit"
            ),
        })
    eligible = sorted(
        (g for g in results if g["eligible_for_trial"]),
        key=lambda g: (-g["ranking_score_heuristic"],
                       -g["gguf_unique_weight_bytes"], g["id"]),
    )
    for position, group in enumerate(eligible, 1):
        group["priority_rank"] = position
    return sorted(
        results, key=lambda g: (
            not g["eligible_for_trial"], g.get("priority_rank", 10**6), g["id"]
        )
    )


def choose_group_trials(groups: list[dict],
                        max_groups: int = MAX_TRIAL_GROUPS) -> list[dict]:
    if not isinstance(max_groups, int) or max_groups < 1:
        raise ValueError("Mapping: massimo numero prove non valido.")
    return [g for g in groups if g["eligible_for_trial"]][:max_groups]


def native_group_trial(layout: dict, group: dict | None) -> dict:
    """Build verified coverage for baseline or a homologous group on CPU."""
    _, expected = all_gpu_rules(layout)
    all_names = sorted(t["name"] for t in layout["tensors"])
    host = sorted(group["tensor_names"]) if group is not None else []
    if not set(host).issubset(set(all_names)) or len(host) != len(set(host)):
        raise ValueError("Mapping: gruppo non corrisponde al GGUF.")
    name_to_device = {name: "cpu" if name in host else "gpu" for name in all_names}
    # Exact names prevent overlap between CPU and GPU and permit native
    # occurrence-count validation, including tied Gemma and repeated rope.
    rules = [{
        "pattern": "^" + re.escape(name) + "$",
        "device": name_to_device[name],
    } for name in all_names]
    # Gemma tied output + input use one physical tensor with two native
    # occurrences; both must remain GPU in this initial group experiment.
    return {
        "id": group["id"] if group else "baseline",
        "rules": rules,
        "host_tensor_names": host,
        "gpu_tensor_names": [name for name in all_names if name not in host],
        "expected_native_log_occurrences":
            expected["expected_native_log_occurrences"],
        "output_weight_target": "gpu",
        "host_tensor_count": len(host),
        "gpu_tensor_count": len(all_names) - len(host),
        "host_gguf_bytes": (group["gguf_unique_weight_bytes"] if group else 0),
        "group": group["id"] if group else None,
    }
