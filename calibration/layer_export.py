"""Human-readable exports for tensor-level planning (no runtime actions)."""

import csv
import json
from pathlib import Path


def export_matrix_tables(report: dict, destination: Path) -> dict:
    """Create deterministic CSV summaries alongside the detailed JSON report."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if report.get("schema_version", 0) < 2:
        raise ValueError("CSV layer matrix richiede schema 2.")

    def write(name, fields, rows):
        target = destination / name
        temp = target.with_suffix(".csv.tmp")
        try:
            with temp.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=fields, extrasaction="ignore"
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow({
                        key: (json.dumps(value, ensure_ascii=False)
                              if isinstance(value, (list, dict)) else value)
                        for key, value in row.items()
                    })
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
        return str(target)

    tensors = [
        {
            "tensor_name": t["name"],
            "block_index": t["block_index"],
            "module_group": t["group"],
            "tensor_role": t["tensor_role"],
            "gguf_type": t["type"],
            "shape": t["shape"],
            "elements": t.get("elements"),
            "gguf_offset_bytes": t.get("gguf_offset_bytes"),
            "offset_source": t.get("offset_source"),
            "shard_index": t.get("shard_index"),
            "shard_file": t.get("shard_file"),
            "size_bytes": t["bytes"],
            "size_mib": t["mib"],
            "cpu_candidate_unverified": t["cpu_placement_candidate"],
            "cpu_placement_tested": t["cpu_placement_tested"],
            "placement_constraint": t["placement_constraint"],
            "exact_cpu_override_regex": t["exact_cpu_override_regex"],
            "related_tensor_names": t["related_tensors_same_subsystem"],
            "neighbor_blocks": t["neighbor_block_indices"],
        }
        for t in report["tensor_index"]
    ]
    blocks = [
        {
            "block_index": layer["index"],
            "module_type": layer["module_type"],
            "tensor_count": layer["tensor_count"],
            "size_bytes": layer["serialized_weight_bytes"],
            "size_mib": layer["serialized_weight_mib"],
            "previous_block": layer["previous_block"],
            "next_block": layer["next_block"],
            "tensor_groups": layer["groups"],
            "tensor_names": layer["same_block_tensor_names"],
            "cpu_prefix_candidate_unverified":
                layer["cpu_prefix_planning_candidate"],
        }
        for layer in report["layers"]
    ]
    paths = {
        "tensors_csv": write(
            "layer_matrix_tensors.csv", list(tensors[0]) if tensors else [],
            tensors
        ),
        "blocks_csv": write(
            "layer_matrix_blocks.csv", list(blocks[0]) if blocks else [],
            blocks
        ),
        "links_csv": write(
            "layer_matrix_links.csv",
            ["type", "from", "to", "evidence",
             "runtime_execution_graph_verified"],
            report["relationships"]
        ),
        "constraints_csv": write(
            "layer_matrix_constraints.csv",
            ["kind", "tensor", "logical_uses", "native_placement",
             "expected_native_occurrences", "verified_at_runtime"],
            report["placement_constraints"]
        ),
    }
    return paths
