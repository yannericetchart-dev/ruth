"""Conservative CPU-placement planning annotations for GGUF tensor inventories.

Only tensor names, block order, and known native override constraints are
observed. This does not reconstruct the compute graph or verify CPU mapping.
"""

import re
from collections import defaultdict


def annotate_matrix(report: dict) -> dict:
    """Add explicit per-tensor selectors, related weights and coupling notes."""
    arch = report["architecture"]
    layers = report["layers"]
    all_rows = [t for layer in layers for t in layer["tensors"]]
    all_rows += report["non_block_tensors"]
    names = [t["name"] for t in all_rows]
    if len(names) != len(set(names)):
        raise ValueError("Matrice tensor: nomi GGUF ripetuti.")
    name_set = set(names)
    tied = arch == "gemma2" and "token_embd.weight" in name_set and "output.weight" not in name_set
    block_names = {
        layer["index"]: [t["name"] for t in layer["tensors"]]
        for layer in layers
    }
    total_blocks = report["block_count"]
    relationships = []
    constraints = []

    for layer in layers:
        index = layer["index"]
        layer["module_type"] = "transformer_block"
        layer["previous_block"] = index - 1 if index else None
        layer["next_block"] = index + 1 if index + 1 < total_blocks else None
        layer["cpu_prefix_planning_candidate"] = True
        layer["placement_granularity"] = (
            "contiguous_prefix_via_n_gpu_layers_or_regex_tensor_overrides"
        )
        layer["runtime_cpu_placement_verified"] = False
        layer["same_block_tensor_names"] = block_names[index]
        if index + 1 < total_blocks:
            relationships.append({
                "type": "sequential_block_order",
                "from": f"blk.{index}", "to": f"blk.{index+1}",
                "evidence": "gguf_block_numbering",
                "runtime_execution_graph_verified": False,
            })

    subsystem_groups = defaultdict(list)
    for row in all_rows:
        n = row["name"]
        match = re.match(r"^blk\.(\d+)\.(.+)$", n)
        idx = int(match.group(1)) if match else None
        if match:
            subsystem = row["group"]
            subsystem_groups[(idx, subsystem)].append(n)

    for row in all_rows:
        n = row["name"]
        match = re.match(r"^blk\.(\d+)\.(.+)$", n)
        idx = int(match.group(1)) if match else None
        group = row["group"]
        tied_use = tied and n == "token_embd.weight"
        replicated = (
            n == "rope_freqs.weight"
            and arch in ("llama", "qwen2")
        )
        related = sorted(
            other for other in subsystem_groups.get((idx, group), [])
            if other != n
        ) if idx is not None else []
        row["tensor_role"] = (
            n.rsplit(".", 1)[0].split(".")[-1]
            if n.endswith(".weight") else n.split(".")[-1]
        )
        row["block_index"] = idx
        row["kind"] = group
        row["exact_cpu_override_regex"] = "^" + re.escape(n) + "$"
        row["cpu_placement_candidate"] = True
        row["cpu_placement_tested"] = False
        row["cpu_placement_status"] = "candidate_not_native_verified"
        row["placement_constraint"] = (
            "gemma2_tied_embedding_and_output_must_move_together"
            if tied_use else (
                "rope_frequency_tensor_reused_by_multiple_native_blocks"
                if replicated else "independent_name_selector_not_execution_independence"
            )
        )
        row["related_tensors_same_subsystem"] = related
        row["neighbor_block_indices"] = (
            ([idx - 1] if idx else [])
            + ([idx + 1] if idx is not None and idx + 1 < total_blocks else [])
        ) if idx is not None else []
        row["relation_evidence"] = (
            "gguf_name_and_architecture_heuristic_not_computation_graph"
        )
        if tied_use:
            constraints.append({
                "kind": "tied_weight",
                "tensor": n,
                "logical_uses": ["input_embedding", "output_projection"],
                "native_placement": "single_regex_both_uses",
                "verified_at_runtime": False,
            })
        if replicated:
            constraints.append({
                "kind": "multiple_native_uses",
                "tensor": n,
                "expected_native_occurrences": total_blocks,
                "native_placement": "verify_every_use_before_accepting_mapping",
                "verified_at_runtime": False,
            })

    report["schema_version"] = 2
    report["tensor_index"] = sorted(
        all_rows, key=lambda t: (t["block_index"] is None,
                                 t["block_index"] if t["block_index"] is not None else -1,
                                 t["name"])
    )
    report["relationships"] = relationships
    report["placement_constraints"] = constraints
    report["tensor_cpu_candidates_count"] = len(all_rows)
    report["tensor_cpu_native_verified_count"] = 0
    report["dependency_semantics"] = (
        "Only same-block association, named functional group and sequential "
        "block indices; this is NOT a verified computation/dependency graph."
    )
    report["mapping_implementation_status"] = "da fare"
    return report
