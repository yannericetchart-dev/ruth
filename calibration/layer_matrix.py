"""Read-only per-layer weight matrix for an already verified GGUF baseline.

This is a metadata inventory, NOT layer placement or CPU inference.
Serialized GGUF tensor bytes are not a measurement of VRAM recovered by
offloading; KV cache, CUDA workspaces, transfer buffers and runtime
placement are all outside this report.
"""

from collections import defaultdict
from pathlib import Path

from inference.engine_catalog import engine_info
from model.inspect import inspect_gguf, gguf_file_set


def analyze_layer_matrix(baseline: dict, *, model_id: str,
                         engine: str = "llama.cpp") -> dict:
    """Enumerate every tensor and contiguous transformer block CPU scenarios.

    Only open a model that the baseline phase already fully verified on CUDA.
    Do not load llama.cpp, migrate weights, change KV, or allocate GPU memory.
    """
    if (not isinstance(baseline, dict)
        or baseline.get("native_verified_all_gpu") is not True
        or baseline.get("model_id") != model_id):
        raise RuntimeError("Matrice layer: baseline full-CUDA non verificata.")
    info = engine_info(engine)
    if not info["enabled"]:
        raise NotImplementedError(
            f"Matrice layer per {engine}: da fare, nessun fallback."
        )
    path = Path(baseline.get("local_path") or "/NONEXISTENT").expanduser()
    if (path.name != baseline.get("filename")
        or not path.is_file()):
        raise RuntimeError("Matrice layer: GGUF baseline assente o non coerente.")
    shards = gguf_file_set(path)
    expected = baseline.get("shard_filenames")
    if expected and [p.name for p in shards] != expected:
        raise RuntimeError(
            "Matrice layer: shard effettivi diversi dalla baseline verificata."
        )

    layout = inspect_gguf(path, n_ctx=baseline.get("baseline_probe_n_ctx", 64))
    arch = layout["metadata"].get("general.architecture")
    if arch != baseline.get("architecture"):
        raise RuntimeError("Matrice layer: architettura GGUF diversa dalla baseline.")
    block_count = layout["metadata"].get(f"{arch}.block_count")
    if (not isinstance(block_count, int) or isinstance(block_count, bool)
        or block_count < 1):
        raise RuntimeError("Matrice layer: numero blocchi GGUF non verificabile.")

    tensors = layout["tensors"]
    total = layout["weights_bytes"]
    if total <= 0 or sum(t["bytes"] for t in tensors) != total:
        raise RuntimeError("Matrice layer: somma tensori non coerente.")

    layers_by_index = defaultdict(list)
    outside = []
    for tensor in tensors:
        index = tensor["block"]
        if index is None:
            outside.append(tensor)
        elif isinstance(index, int) and 0 <= index < block_count:
            layers_by_index[index].append(tensor)
        else:
            raise RuntimeError("Matrice layer: indice blocco fuori intervallo.")
    if set(layers_by_index) != set(range(block_count)):
        missing = sorted(set(range(block_count)) - set(layers_by_index))
        raise RuntimeError(
            f"Matrice layer: catalogo blocchi incompleto: {missing}."
        )

    def record(tensor: dict) -> dict:
        return {
            "name": tensor["name"], "type": tensor["type"],
            "shape": tensor["shape"], "bytes": tensor["bytes"],
            "mib": round(tensor["bytes"] / 1048576, 3),
            "group": tensor["group"],
        }

    layers = []
    for index in range(block_count):
        entries = sorted(layers_by_index[index], key=lambda t: t["name"])
        byte_count = sum(t["bytes"] for t in entries)
        groups = {}
        for tensor in entries:
            g = groups.setdefault(tensor["group"], {"tensors": 0, "bytes": 0})
            g["tensors"] += 1
            g["bytes"] += tensor["bytes"]
        layers.append({
            "index": index,
            "tensor_count": len(entries),
            "serialized_weight_bytes": byte_count,
            "serialized_weight_mib": round(byte_count / 1048576, 3),
            "percent_of_model_weights": round(100 * byte_count / total, 4),
            "groups": groups,
            "tensors": [record(t) for t in entries],
            "current_baseline_location": "gpu_verified",
            "future_cpu_placement_evaluated": False,
        })

    outside_bytes = sum(t["bytes"] for t in outside)
    layer_bytes = sum(row["serialized_weight_bytes"] for row in layers)
    if outside_bytes + layer_bytes != total:
        raise RuntimeError("Matrice layer: somma blocchi/altro incoerente.")

    # llama.cpp n_gpu_layers moves a contiguous suffix to the GPU;
    # these hypothetical CPU prefixes describe serialized bytes only.
    scenarios = []
    if info["model_weight_layer_cpu_offload"]:
        counts = sorted({count for count in (1, 2, 4, 8, 16, block_count)
                         if count <= block_count})
        cumulative = 0
        for k in range(1, block_count + 1):
            cumulative += layers[k - 1]["serialized_weight_bytes"]
            if k in counts:
                scenarios.append({
                    "hypothetical_cpu_prefix_blocks": k,
                    "block_indices": list(range(k)),
                    "serialized_block_weight_bytes": cumulative,
                    "serialized_block_weight_mib": round(cumulative / 1048576, 3),
                    "percent_of_model_weights": round(100 * cumulative / total, 3),
                    "not_predicted_vram_savings": True,
                    "measured": False,
                    "executable_mapping": False,
                })

    return {
        "phase": "layer_matrix",
        "status": "enumeration_only",
        "success": True,
        "model_id": model_id,
        "source": baseline.get("source"),
        "repository": baseline.get("repository"),
        "quantization": baseline.get("quantization"),
        "filename": baseline["filename"],
        "model_path": str(path.resolve()),
        "shard_filenames": [p.name for p in shards],
        "architecture": arch,
        "engine": engine,
        "baseline_native_verified_all_gpu": True,
        "block_count": block_count,
        "tensor_count": len(tensors),
        "serialized_weight_bytes": total,
        "serialized_weight_mib": round(total / 1048576, 3),
        "block_serialized_weight_bytes": layer_bytes,
        "non_block_serialized_weight_bytes": outside_bytes,
        "non_block_tensors": [record(t) for t in sorted(
            outside, key=lambda t: t["name"
        ])],
        "layers": layers,
        "cpu_prefix_scenarios": scenarios,
        "cpu_offload_executed": False,
        "gpu_allocated_by_this_phase": False,
        "operational_profile_changed": False,
        "notes": [
            "Solo enumerazione del GGUF esistente; nessun kernel CUDA o placement CPU.",
            "Byte = payload tensor GGUF serializzato, non VRAM liberabile.",
            "Embedding/output, KV, workspace e buffer runtime non inclusi "
            "negli scenari di prefisso CPU.",
            "Spostamenti CPU futuri richiedono misurazione separata di "
            "VRAM, RAM, trasferimenti e prestazioni.",
        ],
    }
