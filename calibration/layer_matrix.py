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

    return _matrix_from_layout(
        layout, baseline, model_id=model_id, engine=engine,
        model_path=str(path.resolve()),
        shards=[p.name for p in shards],
        baseline_verified=True,
    )


def analyze_remote_header_layout(
    layout: dict, *, model_id: str, repository: str,
    filename: str, quantization: str, engine: str = "llama.cpp",
) -> dict:
    """Analyze remote GGUF header only. Never claim full-GPU verification."""
    if engine != "llama.cpp":
        raise NotImplementedError("Matrice per altri engine: da fare.")
    baseline = {
        "model_id": model_id, "source": "bartowski",
        "repository": repository, "filename": filename,
        "architecture": layout["metadata"].get("general.architecture"),
        "quantization": quantization,
    }
    return _matrix_from_layout(
        layout, baseline, model_id=model_id, engine=engine,
        model_path=f"hf://{repository}/{filename}",
        shards=[filename], baseline_verified=False,
    )


def _matrix_from_layout(
    layout: dict, baseline: dict, *, model_id: str, engine: str,
    model_path: str, shards: list[str], baseline_verified: bool
) -> dict:
    info = engine_info(engine)
    arch = layout["metadata"].get("general.architecture")
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
            "elements": tensor.get("elements"),
            "mib": round(tensor["bytes"] / 1048576, 3),
            "gguf_offset_bytes": tensor.get("offset"),
            "offset_source": tensor.get("offset_source"),
            "shard_index": tensor.get("shard_index"),
            "shard_file": tensor.get("shard_file"),
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

    report = {
        "phase": "layer_matrix",
        "status": "enumeration_only",
        "success": True,
        "model_id": model_id,
        "source": baseline.get("source"),
        "repository": baseline.get("repository"),
        "quantization": baseline.get("quantization"),
        "filename": baseline["filename"],
        "model_path": model_path,
        "shard_filenames": shards,
        "architecture": arch,
        "engine": engine,
        "baseline_native_verified_all_gpu": baseline_verified,
        "provenance": (
            "local_cuda_baseline_verified" if baseline_verified
            else "remote_gguf_header_only_no_runtime_verification"
        ),
        "block_count": block_count,
        "tensor_count": len(tensors),
        "serialized_weight_bytes": total,
        "serialized_weight_mib": round(total / 1048576, 3),
        "block_serialized_weight_bytes": layer_bytes,
        "non_block_serialized_weight_bytes": outside_bytes,
        "non_block_tensors": [
            record(t) for t in sorted(outside, key=lambda t: t["name"])
        ],
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

    from .layer_dependencies import annotate_matrix
    return annotate_matrix(report)


def remote_huggingface_matrices(
    models: list[str], *, output_root: Path | None = None,
    catalog_inventory=None, ranged_header=None,
) -> list[dict]:
    """Catalog four HF models at their published default Q4_K_M GGUF format.

    Network operations fetch model listings and GGUF headers only.
    Does NOT run baseline, perform CUDA probes, or download model weights.
    """
    from model import ModelManager
    from .gguf_catalog import inventory, remote_header
    from .profile_pipeline import atomic_json
    from .layer_export import export_matrix_tables

    catalog_inventory = catalog_inventory or inventory
    ranged_header = ranged_header or remote_header
    root = (Path(output_root).expanduser().resolve()
            if output_root else
            ModelManager().root / "calibration_runs" / "remote_hf_matrices")
    reports = []
    for model_id in models:
        spec = ModelManager.source_spec(model_id, "bartowski")
        name = spec["filename"]
        candidates = catalog_inventory(spec["repository"])
        matches = [item for item in candidates if item["filename"] == name]
        if len(matches) != 1:
            raise RuntimeError(
                f"{model_id}: GGUF di riferimento {name} non presente nel catalogo."
            )
        candidate = matches[0]
        if candidate.get("quantization") != spec["quantization"]:
            raise RuntimeError(
                f"{model_id}: quantizzazione della variante non coerente."
            )
        layout = ranged_header(
            spec["repository"], name, candidate["file_bytes"],
            include_tensors=True,
        )
        if layout["metadata"].get("general.architecture") != spec["architecture"]:
            raise RuntimeError(
                f"{model_id}: architettura remota diversa dalla dichiarata."
            )
        if layout["tensor_count"] != len(layout["tensors"]):
            raise RuntimeError(f"{model_id}: conteggio tensori remoto incoerente.")
        report = analyze_remote_header_layout(
            layout, model_id=model_id, repository=spec["repository"],
            filename=name, quantization=spec["quantization"],
        )
        path = root / model_id / (name + ".json")
        atomic_json(path, report)
        report["table_paths"] = export_matrix_tables(report, path.parent)
        report["report_path"] = str(path)
        reports.append(report)
    return reports


def local_baseline_matrices(
    models: list[str], *, source: str = "all",
    output_root: Path | None = None,
) -> list[dict]:
    """Read existing verified local baselines; never discover/download one."""
    import json
    from .phase1_audit import audit_all

    reports = []
    inspected = audit_all(model="all", source=source, output_root=output_root)
    for state in inspected:
        if state["model_id"] not in models:
            continue
        if not state["full_gpu_baseline_verified"]:
            raise RuntimeError(
                f"{state['model_id']} / {state['source']}: baseline e GGUF "
                "full-CUDA non ancora verificati: "
                + "; ".join(state["errors"])
            )
        folder = Path(state["run_directory"])
        original = json.loads(
            (folder / "baseline.json").read_text(encoding="utf-8")
        )
        selected = original["baseline"]
        report = analyze_layer_matrix(
            selected, model_id=state["model_id"]
        )
        from .profile_pipeline import atomic_json
        from .layer_export import export_matrix_tables
        # Write a separate analysis report, never overwrite the resumable
        # pipeline's own layer_matrix.json or alter its completed state.
        dest = folder / "local_matrix_analysis"
        target = dest / "layer_matrix.json"
        atomic_json(target, report)
        report["table_paths"] = export_matrix_tables(report, dest)
        report["report_path"] = str(target)
        reports.append(report)
    return reports


def main(argv: list[str] | None = None) -> int:
    import argparse
    from model import ModelManager

    parser = argparse.ArgumentParser(
        description="Matrici GGUF locali con baseline verificata "
                    "(oppure --remote-header per ricerca solo header)."
    )
    parser.add_argument("--model", default="all",
                        choices=["all", *sorted(ModelManager.CATALOG)])
    parser.add_argument("--source", default="all",
                        choices=["all", "bartowski", "kaggle"])
    parser.add_argument("--remote-header", action="store_true",
                        help="Solo header HF: nessuna prova full-CUDA locale.")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args(argv)
    models = sorted(ModelManager.CATALOG) if args.model == "all" else [args.model]
    errors = 0
    if args.remote_header:
        if args.source not in ("bartowski", "all"):
            parser.error("--remote-header e' disponibile solo per Bartowski HF.")
        for model_id in models:
            try:
                reports = remote_huggingface_matrices(
                    [model_id], output_root=args.output_root
                )
            except Exception as exc:
                errors += 1
                print(f"[{model_id}] BLOCCATO: {type(exc).__name__}: {exc}")
                continue
            for report in reports:
                print(
                    f"[{report['model_id']}] HEADER HF: "
                    f"{report['block_count']} layer, {report['tensor_count']} tensori "
                    f"| {report['serialized_weight_mib']} MiB serializzati "
                    f"| {report['report_path']}"
                )
        print("Solo header remoti, nessun GGUF scaricato o validazione CUDA.")
        return 1 if errors else 0

    # Audit every registered pair, but only selected model/source reports
    # are exported. No CLI path here can download model weights.
    from .phase1_audit import model_source_pairs
    for model_id, source in model_source_pairs(args.source, args.model):
        try:
            reports = local_baseline_matrices(
                [model_id], source=source, output_root=args.output_root
            )
            if len(reports) != 1:
                raise RuntimeError("Baseline GGUF locale non trovata.")
        except Exception as exc:
            errors += 1
            print(f"[{model_id} / {source}] NON VERIFICATO: "
                  f"{type(exc).__name__}: {exc}")
            continue
        report = reports[0]
        print(
            f"[{model_id} / {source}] LOCALE: "
            f"{report['quantization']}, {report['block_count']} layer, "
            f"{report['tensor_count']} tensori, "
            f"{report['serialized_weight_mib']} MiB | {report['report_path']}"
        )
    print("Nessun download GGUF, test CUDA o modifica al profilo operativo.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
