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
        # Every homologous block tensor family is safe to TRY when
        # native placement can be checked: neither a small GGUF payload,
        # unknown FLOP proxy nor repeated native uses is a reason to skip.
        # Repeated occurrences are verified individually in native logs.
        # Non-block/tied tensors remain separately inventoried rather than
        # silently moving a coupled input/output together.
        eligible = in_blocks
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
                "non_block_or_coupled_tensor" if not in_blocks else None
            ),
        })
    eligible = sorted(
        (g for g in results if g["eligible_for_trial"]),
        key=lambda g: (
            g["ranking_score_heuristic"] is None,
            -(g["ranking_score_heuristic"] or 0),
            -g["gguf_unique_weight_bytes"], g["id"],
        ),
    )
    for position, group in enumerate(eligible, 1):
        group["priority_rank"] = position
    return sorted(
        results, key=lambda g: (
            not g["eligible_for_trial"], g.get("priority_rank", 10**6), g["id"]
        )
    )


def choose_group_trials(groups: list[dict]) -> list[dict]:
    """Try EVERY eligible homologous block family in ranked order."""
    return [g for g in groups if g["eligible_for_trial"]]


def native_group_trial(layout: dict, group: dict | None) -> dict:
    """Build verified coverage for baseline or a homologous group on CPU."""
    _, expected = all_gpu_rules(layout)
    all_names = sorted(t["name"] for t in layout["tensors"])
    host = sorted(group["tensor_names"]) if group is not None else []
    if not set(host).issubset(set(all_names)) or len(host) != len(set(host)):
        raise ValueError("Mapping: gruppo non corrisponde al GGUF.")
    name_to_device = {name: "cpu" if name in host else "gpu" for name in all_names}
    # Efficient native regex: ONE expression per homologous tensor family
    # across all transformer blocks, not hundreds of regexes per model.
    # Non-block tensors remain independently addressable, including the
    # physical Gemma tied embedding which has TWO native uses.
    arch = layout["metadata"]["general.architecture"]
    blocks = layout["metadata"][f"{arch}.block_count"]
    by_suffix = defaultdict(list)
    non_block = []
    for name in all_names:
        match = BLOCK_TENSOR.fullmatch(name)
        if match and int(match.group(1)) < blocks:
            by_suffix[match.group(2)].append(name)
        else:
            non_block.append(name)
    rules = []
    for suffix, names in sorted(by_suffix.items()):
        targets = {name_to_device[name] for name in names}
        if len(targets) != 1:
            raise RuntimeError(
                "Mapping: gruppo omologo diviso tra CPU e GPU: " + suffix
            )
        rules.append({
            "pattern": r"^blk\.[0-9]+\." + re.escape(suffix) + "$",
            "device": targets.pop(),
        })
    for name in sorted(non_block):
        rules.append({
            "pattern": "^" + re.escape(name) + "$",
            "device": "gpu",
        })
    covered = {}
    for rule in rules:
        matching = [name for name in all_names
                    if re.search(rule["pattern"], name)]
        if not matching:
            raise RuntimeError("Mapping: override nativo senza tensori.")
        for name in matching:
            if name in covered:
                raise RuntimeError("Mapping: override nativi sovrapposti.")
            covered[name] = rule["device"]
    if covered != name_to_device:
        raise RuntimeError("Mapping: copertura fisica tensori incompleta.")
    # All instances of a repeated GGUF name must appear in the native
    # logs and be placed on the same requested device.
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


def native_multi_group_trial(layout: dict, groups: list[dict]) -> dict:
    """One native mapping with the union of disjoint homologous CPU groups.

    Serialized GGUF bytes are only a prioritization proxy, never measured
    VRAM freed. Uses native_group_trial's exhaustive GPU/CPU rule coverage.
    """
    if not groups:
        raise ValueError("Combinazione gruppi vuota.")
    ids = [g.get("id") for g in groups]
    if len(ids) != len(set(ids)) or any(
        not isinstance(gid, str) or not gid.startswith("group:") for gid in ids
    ):
        raise ValueError("Gruppi duplicati o ID non validi.")
    if any(g.get("eligible_for_trial") is not True for g in groups):
        raise ValueError("Gruppo non idoneo a spostamento CPU.")
    names = [name for group in groups for name in group["tensor_names"]]
    if len(names) != len(set(names)):
        raise ValueError("Gruppi con tensori sovrapposti.")
    merged = {
        "id": "groups:" + "+".join(sorted(ids)),
        "tensor_names": names,
        "gguf_unique_weight_bytes": sum(
            g["gguf_unique_weight_bytes"] for g in groups
        ),
    }
    trial = native_group_trial(layout, merged)
    trial["group_ids"] = sorted(ids)
    trial["groups"] = [{
        "id": g["id"],
        "gguf_unique_weight_mib": g["gguf_unique_weight_mib"],
        "tensor_count": g["tensor_count"],
        "tensor_names": g["tensor_names"],
    } for g in groups]
    return trial


def native_tensor_trial(layout: dict, host_tensor_names: list[str]) -> dict:
    """Build exhaustive native overrides for ANY subset of block tensors.

    Homogeneous block suffix families remain one regex for efficiency; split
    families use an exact regex per physical tensor. Non-block and tied
    embedding/output weights intentionally remain GPU-resident.
    """
    import hashlib
    import json

    _, expected = all_gpu_rules(layout)
    tensors = layout.get("tensors") or []
    all_names = sorted(t["name"] for t in tensors)
    if len(all_names) != len(set(all_names)):
        raise ValueError("GGUF con nomi tensori duplicati.")
    if (not isinstance(host_tensor_names, list)
        or any(not isinstance(name, str) for name in host_tensor_names)
        or len(host_tensor_names) != len(set(host_tensor_names))):
        raise ValueError("Lista tensori CPU malformata o duplicata.")
    host = sorted(host_tensor_names)
    # Critical: the empty-subset control MUST be the exact Phase-1
    # all-GPU tensor mapping. The regular subset algorithm expands GPU
    # overrides by suffix; doing that for the baseline could introduce
    # additional native allocation overhead at the VRAM frontier.
    if not host:
        baseline_rules, baseline_native = all_gpu_rules(layout)
        identifier = "tensors:" + hashlib.sha256(
            json.dumps([], ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:20]
        return {
            "id": identifier, "rules": baseline_rules,
            "host_tensor_names": [],
            "gpu_tensor_names": all_names,
            "expected_native_log_occurrences":
                baseline_native["expected_native_log_occurrences"],
            "output_weight_target": "gpu",
            "host_tensor_count": 0,
            "gpu_tensor_count": len(all_names),
            "host_gguf_bytes": 0,
            "group": None,
            "mapping_kind": "exact_phase1_all_gpu_control",
        }
    names_set = set(all_names)
    metadata = layout["metadata"]
    arch = metadata["general.architecture"]
    blocks = metadata[f"{arch}.block_count"]
    for name in host:
        match = BLOCK_TENSOR.fullmatch(name)
        if (name not in names_set or match is None
            or int(match.group(1)) >= blocks):
            raise ValueError(
                "Mapping individuale: consentiti soltanto tensori di blocco: "
                + name
            )

    placement = {name: "cpu" if name in host else "gpu" for name in all_names}
    by_suffix = defaultdict(list)
    non_block = []
    for name in all_names:
        match = BLOCK_TENSOR.fullmatch(name)
        if match and int(match.group(1)) < blocks:
            by_suffix[match.group(2)].append(name)
        else:
            non_block.append(name)

    rules = []
    for suffix, names in sorted(by_suffix.items()):
        domains = {placement[name] for name in names}
        if len(domains) == 1:
            rules.append({
                "pattern": r"^blk\.[0-9]+\." + re.escape(suffix) + "$",
                "device": domains.pop(),
            })
        else:
            for name in sorted(names):
                rules.append({
                    "pattern": "^" + re.escape(name) + "$",
                    "device": placement[name],
                })
    for name in sorted(non_block):
        rules.append({
            "pattern": "^" + re.escape(name) + "$",
            "device": "gpu",
        })

    covered = {}
    for rule in rules:
        matching = [
            name for name in all_names
            if re.search(rule["pattern"], name)
        ]
        if not matching:
            raise RuntimeError("Regola mapping senza corrispondenze.")
        for name in matching:
            if name in covered:
                raise RuntimeError("Regole mapping sovrapposte.")
            covered[name] = rule["device"]
    if covered != placement:
        raise RuntimeError("Mapping individuale non copre il GGUF.")

    identifier = "tensors:" + hashlib.sha256(
        json.dumps(host, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:20]
    sizes = {t["name"]: t["bytes"] for t in tensors}
    return {
        "id": identifier, "rules": rules,
        "host_tensor_names": host,
        "gpu_tensor_names": [name for name in all_names if name not in host],
        "expected_native_log_occurrences":
            expected["expected_native_log_occurrences"],
        "output_weight_target": "gpu",
        "host_tensor_count": len(host),
        "gpu_tensor_count": len(all_names) - len(host),
        "host_gguf_bytes": sum(sizes[name] for name in host),
        "group": None,
        "mapping_kind": "individual_tensor_subset",
    }
