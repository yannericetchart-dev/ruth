"""Remote GGUF inventory and necessary-condition weight-only GPU preflight.

The quality order is the publisher's coarse quantization hierarchy, *not* a
measured model-quality score. Read the GGUF header over ranged file access,
calculate exact serialized tensor bytes from each tensor's quantization type
and shape. Never reject on file size alone: it contains metadata and padding.
"""

import re
import struct

REPOSITORY = "bartowski/gemma-2-9b-it-GGUF"
PREFIX = "gemma-2-9b-it-"
# Published high-to-low fidelity convention. Adjacent IQ/K variants may
# exchange rank on specific tasks; no exact quality ordering is claimed.
QUALITY_ORDER = (
    "f32", "Q8_0", "Q6_K_L", "Q6_K", "Q5_K_L", "Q5_K_M",
    "Q5_K_S", "Q4_K_L", "Q4_K_M", "Q4_K_S", "IQ4_XS",
    "Q3_K_L", "Q3_K_M", "IQ3_M", "Q3_K_S", "IQ3_XS",
    "Q2_K_L", "Q2_K", "IQ3_XXS", "IQ2_M",
)
_QUALITY_RANK = {q.lower(): i for i, q in enumerate(QUALITY_ORDER)}
_SCALAR_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4,
                 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def gguf_payload_bytes(stream, file_bytes: int) -> dict:
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
            return read(n).decode("utf-8")
        skip(n)

    def value(kind, depth=0):
        if depth > 4:
            raise ValueError("Array GGUF troppo annidato.")
        if kind in _SCALAR_BYTES:
            size = _SCALAR_BYTES[kind]
            return read(size) if size <= 8 else None
        if kind == 8:
            string()
            return None
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
    for _ in range(field_count):
        key = string(want=True)
        kind = u32()
        raw = value(kind)
        if key == "general.alignment":
            if kind not in (4, 10):
                raise ValueError("Tipo alignment GGUF non valido.")
            alignment = int.from_bytes(raw, "little")
    exact_weight_bytes = 0
    for _ in range(tensor_count):
        string()
        dims = u32()
        if dims < 1 or dims > 8:
            raise ValueError("Dimensioni di tensore GGUF non plausibili.")
        elements = 1
        for _ in range(dims):
            length = u64()
            if length < 1 or length > 1_000_000_000:
                raise ValueError("Forma del tensore non valida.")
            elements *= length
        quant_type = GGMLQuantizationType(u32())
        block_size, type_size = GGML_QUANT_SIZES[quant_type]
        u64()  # Offset relativo all'inizio dei dati dei tensori.
        if elements % block_size:
            raise ValueError("Tensor count incompatibile con blocchi quantizzati.")
        exact_weight_bytes += elements // block_size * type_size
    if alignment < 1 or alignment > 4096 or alignment & (alignment - 1):
        raise ValueError("Alignment GGUF non valido.")
    position = stream.tell()
    data_offset = (position + alignment - 1) // alignment * alignment
    if data_offset >= file_bytes:
        raise ValueError("Offset pesi GGUF non valido.")
    return {
        "version": version,
        "tensor_count": tensor_count,
        "data_offset_bytes": data_offset,
        "serialized_payload_bytes_upper_bound": file_bytes - data_offset,
        "exact_serialized_tensor_bytes": exact_weight_bytes,
        "precheck_kind": "quantization_and_shape_exact_serialized_weights",
    }


def inventory(repo_id: str = REPOSITORY) -> list[dict]:
    """Fetch all single-file GGUF siblings, ordered by expected fidelity."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo_id, files_metadata=True)
    found = []
    for entry in info.siblings:
        name = entry.rfilename
        if "/" in name or not name.lower().endswith(".gguf"):
            continue
        match = re.fullmatch(
            re.escape(PREFIX) + r"(.+)\.gguf", name, re.IGNORECASE
        )
        if not match:
            continue
        q = match.group(1)
        rank = _QUALITY_RANK.get(q.lower())
        size = entry.size
        if not isinstance(size, int) or size <= 0:
            raise RuntimeError(f"Dimensione remota mancante per {name}.")
        found.append({
            "repo_id": repo_id,
            "filename": name,
            "quantization": q,
            "quality_priority": rank,
            "file_bytes": size,
            "ranking_basis": "coarse_publisher_quantization_hierarchy",
        })
    if not found:
        raise RuntimeError("Nessun GGUF singolo trovato nel repository.")
    # Unranked entries are never silently treated as lower quality:
    # halt and require a ranking update before selecting a baseline.
    unknown = [x["filename"] for x in found if x["quality_priority"] is None]
    if unknown:
        raise RuntimeError(
            f"Quantizzazioni sconosciute, aggiornare ordine prima della selezione: {unknown}"
        )
    return sorted(found, key=lambda x: (x["quality_priority"], x["filename"]))


def remote_header(repo_id: str, filename: str, file_bytes: int) -> dict:
    """Inspect remote header only via ranged access; never fetch full weights."""
    from huggingface_hub import HfFileSystem
    with HfFileSystem().open(f"{repo_id}/{filename}", "rb") as remote:
        return gguf_payload_bytes(remote, file_bytes)


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
