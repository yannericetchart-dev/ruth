"""Safely compose independently generated GGUF native-placement candidates.

This only merges exact tensor rules, and does not predict runtime placement
benefit or infer VRAM from GGUF size. Every composite must be native-verified.
"""

def compose_candidates(proposals: dict, candidate_ids: list[str]) -> dict:
    if not candidate_ids or any(not isinstance(x, str) or not x for x in candidate_ids):
        raise ValueError("Una composizione richiede ID di candidati validi.")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Nessun candidato puo' essere ripetuto.")
    index = {candidate["id"]: candidate for candidate in proposals["candidates"]}
    if len(index) != len(proposals["candidates"]):
        raise ValueError("ID candidati duplicati nel generatore.")

    originals = []
    used = set()
    host, gpu = [], []
    rules = []
    expected_occurrences = {}
    output_target = None
    delta = 0

    for id_ in candidate_ids:
        item = index.get(id_)
        if item is None or not item["rules"]:
            raise ValueError(f"Candidato non valido o baseline: {id_}")
        overlap = used.intersection(item["tensor_names"])
        if overlap:
            raise ValueError(f"Override sovrapposti: {sorted(overlap)}")
        used.update(item["tensor_names"])
        originals.append(item)
        host.extend(item["host_tensor_names"])
        gpu.extend(item["gpu_tensor_names"])
        rules.extend(item["rules"])

        if item.get("output_weight_target") is not None:
            if output_target is not None:
                raise ValueError("Piu' di un placement accoppiato di output.")
            output_target = item["output_weight_target"]
        occurrences = item.get("expected_native_log_occurrences") or {}
        expected_occurrences.update(occurrences)
        estimated = item.get("estimated_vram_weight_delta_bytes")
        if estimated is None:
            delta = None
        elif delta is not None:
            delta += estimated

    if set(host).intersection(gpu) or len(rules) == 0:
        raise ValueError("Composizione non valida.")
    # Exact native regexes are generated upstream. No reshaping, no
    # implicit promotion, no backend-specific tensor alias inference.
    result = {
        "id": "composite:" + "+".join(candidate_ids),
        "kind": "composite",
        "direction": "mixed",
        "family": "mixed",
        "component_ids": list(candidate_ids),
        "tensor_names": sorted(used),
        "host_tensor_names": sorted(host),
        "gpu_tensor_names": sorted(gpu),
        "rules": rules,
        "gguf_bytes_to_host": sum(c["gguf_bytes_to_host"] for c in originals),
        "gguf_bytes_to_gpu": sum(c["gguf_bytes_to_gpu"] for c in originals),
        "estimated_vram_weight_delta_bytes": delta,
        "requires_native_verification": True,
    }
    if output_target is not None:
        result["output_weight_target"] = output_target
        result["expected_native_log_occurrences"] = expected_occurrences
    return result
