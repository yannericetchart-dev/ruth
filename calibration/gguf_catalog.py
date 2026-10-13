"""Remote GGUF inventory and necessary-condition weight-only GPU preflight.

The quality order is the publisher's coarse quantization hierarchy, *not* a
measured model-quality score. Read the GGUF header over ranged file access,
calculate exact serialized tensor bytes from each tensor's quantization type
and shape. Never reject on file size alone: it contains metadata and padding.
"""

import re
import struct

REPOSITORY = "bartowski/gemma-2-9b-it-GGUF"


def repo_prefix(repo_id: str) -> str:
    """Find the GGUF filename prefix from the registered model catalog."""
    from model import ModelManager

    matches = [
        spec["gguf_prefix"] for spec in ModelManager.CATALOG.values()
        if spec["repository"] == repo_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Repository GGUF non registrato o ambiguo: {repo_id}")
    return matches[0]
# Coarse high-to-low fidelity convention derived from the publisher's
# descriptions. It is a PRIORITY, not a measured quality score. Variants
# without direct published comparisons stay explicitly heuristic.
QUALITY_ORDER = (
    "f32", "f16", "Q8_0_L", "Q8_0", "Q6_K_L", "Q6_K",
    "Q5_K_L", "Q5_K_M", "Q5_K_S", "Q5_1", "Q5_0", "Q4_K_L", "Q4_K_M",
    "Q4_K_S", "IQ4_NL", "IQ4_XS", "Q4_0", "Q3_K_XL", "Q3_K_L", "Q3_K_M",
    "IQ3_M", "IQ3_S", "Q3_K_S", "IQ3_XS", "Q2_K_L", "Q2_K",
    "IQ3_XXS", "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS",
    "IQ1_M", "IQ1_S",
)
_QUALITY_RANK = {q.casefold(): i for i, q in enumerate(QUALITY_ORDER)}
# Standard llama-quantize *command* aliases, NOT equivalent GGUF bytes.
# Bare Q5_K means Q5_K_M, never Bartowski's mixed Q5_K_L recipe.
_STANDARD_QUANTIZATION_ALIASES = {
    "q3_k": "Q3_K_M",
    "q4_k": "Q4_K_M",
    "q5_k": "Q5_K_M",
}
# Mixed-precision variants inherit their principal weight format's quality
# tier and are tried before the plain version of the SAME tier. The suffix
# alone cannot quantify any actual perplexity/accuracy gain.
_PRECISION_SUFFIX_PRIORITY = (
    ("-f32", -40), ("-fp32", -40),
    ("-f16", -30), ("-fp16", -30),
    ("-q8", -20), ("-q8_0", -20),
)


def quantization_priority(label: str) -> dict:
    """Resolve family/precision suffix without hiding unknown quantizations.

    The interval of 100 between main quantization families ensures that a
    mixed-precision variant cannot leap into a stronger *base* family just
    because its filename contains f16/f32. Such cross-family comparisons
    require model-quality benchmarks rather than naming heuristics.
    """
    if not isinstance(label, str) or not label.strip():
        return {
            "quality_priority": None, "quality_rank_confidence": "unranked",
            "quality_base": None, "quality_modifier": None,
        }
    folded = label.casefold()
    folded = {"fp16": "f16", "fp32": "f32"}.get(folded, folded)
    canonical = _STANDARD_QUANTIZATION_ALIASES.get(folded)
    if canonical is not None:
        return {
            "quality_priority": _QUALITY_RANK[canonical.casefold()] * 100,
            "quality_rank_confidence": "standard_llama_quantize_alias",
            "quality_base": canonical,
            "quality_modifier": "alias",
        }
    rank = _QUALITY_RANK.get(folded)
    if rank is not None:
        return {
            "quality_priority": rank * 100,
            "quality_rank_confidence": "coarse_family_prior",
            "quality_base": QUALITY_ORDER[rank],
            "quality_modifier": None,
        }
    for suffix, boost in _PRECISION_SUFFIX_PRIORITY:
        if folded.endswith(suffix):
            parent = folded[:-len(suffix)]
            parent = _STANDARD_QUANTIZATION_ALIASES.get(
                parent, parent
            ).casefold()
            parent_rank = _QUALITY_RANK.get(parent)
            if parent_rank is not None:
                return {
                    "quality_priority": parent_rank * 100 + boost,
                    "quality_rank_confidence": "mixed_precision_heuristic",
                    "quality_base": QUALITY_ORDER[parent_rank],
                    "quality_modifier": suffix[1:],
                }
    return {
        "quality_priority": None,
        "quality_rank_confidence": "unranked",
        "quality_base": None,
        "quality_modifier": None,
    }
_SCALAR_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4,
                 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def gguf_payload_bytes(stream, file_bytes: int, *, include_tensors: bool = False) -> dict:
    """Read v2/v3 GGUF tensor byte sizes without downloading tensor payload.

    Sizes come from GGML quantization block characteristics and tensor
    shapes, not file length or a safety factor. They describe serialized
    weights: the native runtime's possible repacking or temporary buffers
    must still be measured during actual loading.
    """
    from gguf import GGMLQuantizationType, GGML_QUANT_SIZES
    def read(n):
        data = stream.read(n)
        if len(data) != n:
            raise ValueError("Header GGUF incompleto.")
        return data

    def u32():
        return struct.unpack("<I", read(4))[0]

    def u64():
        return struct.unpack("<Q", read(8))[0]

    def skip(n):
        if n < 0 or n > file_bytes:
            raise ValueError("Dimensione GGUF non plausibile.")
        stream.seek(n, 1)

    def string(want=False):
        n = u64()
        if n > file_bytes:
            raise ValueError("Stringa GGUF troppo lunga.")
        if want:
            if n > 65536:
                raise ValueError("Nome/valore GGUF troppo lungo per inventario remoto.")
            return read(n).decode("utf-8")
        skip(n)

    def value(kind, depth=0, want=False):
        if depth > 4:
            raise ValueError("Array GGUF troppo annidato.")
        if kind in _SCALAR_BYTES:
            size = _SCALAR_BYTES[kind]
            return read(size) if size <= 8 else None
        if kind == 8:
            return string(want=want)
        if kind == 9:
            item_kind, length = u32(), u64()
            if length > file_bytes:
                raise ValueError("Array GGUF non plausibile.")
            if item_kind in _SCALAR_BYTES:
                skip(length * _SCALAR_BYTES[item_kind])
            else:
                for _ in range(length):
                    value(item_kind, depth + 1)
            return None
        raise ValueError(f"Tipo metadato GGUF non supportato: {kind}")

    if read(4) != b"GGUF":
        raise ValueError("Firma GGUF non valida.")
    version = u32()
    if version not in (2, 3):
        raise ValueError(f"Versione GGUF non supportata: {version}")
    tensor_count, field_count = u64(), u64()
    if tensor_count == 0 or tensor_count > 1_000_000 or field_count > 1_000_000:
        raise ValueError("Header GGUF con conteggi anomali.")
    alignment = 32
    metadata = {}
    for _ in range(field_count):
        key = string(want=True)
        kind = u32()
        wanted = include_tensors and (
            key == "general.architecture" or key.endswith(".block_count")
        )
        raw = value(kind, want=wanted)
        if wanted and raw is not None:
            if key == "general.architecture" and kind == 8:
                metadata[key] = raw
            elif key.endswith(".block_count") and kind in (0, 1, 2, 3, 4, 5, 10, 11):
                metadata[key] = int.from_bytes(raw, "little")
        if key == "general.alignment":
            if kind not in (4, 10):
                raise ValueError("Tipo alignment GGUF non valido.")
            alignment = int.from_bytes(raw, "little")
    exact_weight_bytes = 0
    catalog = []
    for _ in range(tensor_count):
        name = string(want=include_tensors)
        dims = u32()
        if dims < 1 or dims > 8:
            raise ValueError("Dimensioni di tensore GGUF non plausibili.")
        elements = 1
        shape = []
        for _ in range(dims):
            length = u64()
            if length < 1 or length > 1_000_000_000:
                raise ValueError("Forma del tensore non valida.")
            elements *= length
            if include_tensors:
                shape.append(length)
        quant_type = GGMLQuantizationType(u32())
        block_size, type_size = GGML_QUANT_SIZES[quant_type]
        relative_offset = u64()  # Offset relativo all'inizio dei dati.
        if elements % block_size:
            raise ValueError("Tensor count incompatibile con blocchi quantizzati.")
        nbytes = elements // block_size * type_size
        exact_weight_bytes += nbytes
        if include_tensors:
            from model.inspect import _kind
            group, block = _kind(name)
            catalog.append({
                "name": name, "shape": shape,
                "type": quant_type.name, "elements": elements,
                "bytes": nbytes, "offset": relative_offset,
                "offset_source": "gguf_tensor_data_relative",
                "shard_index": 0, "shard_file": None,
                "group": group, "block": block,
            })
    if alignment < 1 or alignment > 4096 or alignment & (alignment - 1):
        raise ValueError("Alignment GGUF non valido.")
    position = stream.tell()
    data_offset = (position + alignment - 1) // alignment * alignment
    if data_offset >= file_bytes:
        raise ValueError("Offset pesi GGUF non valido.")
    result = {
        "version": version,
        "tensor_count": tensor_count,
        "data_offset_bytes": data_offset,
        "serialized_payload_bytes_upper_bound": file_bytes - data_offset,
        "exact_serialized_tensor_bytes": exact_weight_bytes,
        "precheck_kind": "quantization_and_shape_exact_serialized_weights",
    }
    if include_tensors:
        if any(
            item["offset"] + item["bytes"] > file_bytes - data_offset
            for item in catalog
        ):
            raise ValueError("Header GGUF: payload tensore oltre fine file.")
        if len(catalog) != len({item["name"] for item in catalog}):
            raise ValueError("Catalogo remoto GGUF: tensori duplicati.")
        if not metadata.get("general.architecture"):
            raise ValueError("Catalogo remoto GGUF: architettura assente.")
        arch = metadata["general.architecture"]
        if not metadata.get(f"{arch}.block_count"):
            raise ValueError("Catalogo remoto GGUF: conteggio blocchi assente.")
        result.update({
            "metadata": metadata, "tensors": catalog,
            "weights_bytes": exact_weight_bytes,
            "file_bytes": file_bytes, "shard_count": 1,
        })
    return result


def inventory(repo_id: str = REPOSITORY) -> list[dict]:
    """Fetch all single-file GGUF siblings, ordered by expected fidelity."""
    from huggingface_hub import HfApi
    prefix = repo_prefix(repo_id)
    info = HfApi().model_info(repo_id, files_metadata=True)
    found = []
    for entry in info.siblings:
        name = entry.rfilename
        if "/" in name or not name.lower().endswith(".gguf"):
            continue
        match = re.fullmatch(
            re.escape(prefix) + r"(.+)\.gguf", name, re.IGNORECASE
        )
        if not match:
            continue
        q = match.group(1)
        # ARM-specific prepacked Q4_0 formats are not suitable for an
        # all-CUDA baseline on the RTX 4050. Preserve them in the catalog.
        arm_only = q.casefold() in {
            "q4_0_4_4", "q4_0_4_8", "q4_0_8_8"
        }
        classification = quantization_priority(
            "Q4_0" if arm_only else q
        )
        size = entry.size
        if not isinstance(size, int) or size <= 0:
            raise RuntimeError(f"Dimensione remota mancante per {name}.")
        found.append({
            "repo_id": repo_id,
            "filename": name,
            "quantization": q,
            **classification,
            "cuda_baseline_eligible": not arm_only,
            "compatibility_note": (
                "prepacked_arm_only_not_cuda_baseline" if arm_only else None
            ),
            "file_bytes": size,
            "ranking_basis": "coarse_quantization_family_plus_mixed_precision_heuristic",
        })
    if not found:
        raise RuntimeError("Nessun GGUF singolo trovato nel repository.")
    # Inventory must ALWAYS be observable, even if a new type is unknown.
    # Unranked entries stay in the catalog; the full-GPU *selector* refuses
    # to crown a winner until the unknown types have been reviewed.
    return sorted(
        found,
        key=lambda x: (
            x["quality_priority"] is None,
            x["quality_priority"] if x["quality_priority"] is not None else 0,
            x["filename"],
        ),
    )


def remote_header(repo_id: str, filename: str, file_bytes: int, *,
                  include_tensors: bool = False) -> dict:
    """Inspect remote header only via ranged access; never fetch full weights."""
    from huggingface_hub import HfFileSystem
    with HfFileSystem().open(f"{repo_id}/{filename}", "rb") as remote:
        return gguf_payload_bytes(remote, file_bytes, include_tensors=include_tensors)


def preflight_locally_inspected_weights(layout: dict, total_vram_bytes: int) -> dict:
    """Only exact GGUF tensor n_bytes > total VRAM permits a hard exclusion."""
    exact = sum(t["bytes"] for t in layout["tensors"])
    return {
        "exact_weight_bytes": exact,
        "certainly_cannot_fit_all_gpu_weights": exact > total_vram_bytes,
        "reason": (
            "exact_gguf_tensor_bytes_exceed_physical_vram"
            if exact > total_vram_bytes else None
        ),
    }
