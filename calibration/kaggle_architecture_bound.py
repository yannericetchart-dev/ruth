"""Source-scoped, conservative tensor-size floors for standard Qwen2.5 7B.

Architecture constants: official Qwen/Qwen2.5-7B-Instruct config.json
https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json

This is NOT a GGUF header or a model-file-size heuristic. For canonical
publisher Qwen 2.5 7B Instruct, count only necessarily present dense matrices:
  - 28 blocks of gate/up/down projections,
  - 28 blocks of Q/K/V/output attention projections,
  - ONE token embedding matrix (even if the output head is tied).
Ignore norms, biases, output head and all runtime memory, ensuring a
conservative *lower* bound for standard FP16 and Q8_0 tensor storage.
Never extrapolate this bound to arbitrary Kaggle publishers or quantizations.

Even under the more conservative ONE embedding assumption, Q8_0 needs
more than 6141 MiB of weight payload alone; FP16 needs far more.
"""

from model import ModelManager

_SOURCE_REPO = "qwen-lm/qwen2.5"
_SOURCE_HANDLE_PREFIX = _SOURCE_REPO + "/gguf/7b-instruct"
_MODEL_ID = "qwen-2.5-7b-instruct"
_MIB = 1048576

# Official Qwen2.5-7B-Instruct architecture, excluding all optional matrices.
_NUM_LAYERS = 28
_HIDDEN = 3584
_MLP_INTERMEDIATE = 18944
_ATTENTION_HEADS = 28
_KV_HEADS = 4
_VOCAB = 152064

_MLP_MATRIX_PARAMS = _NUM_LAYERS * 3 * _HIDDEN * _MLP_INTERMEDIATE
_HEAD_DIM = _HIDDEN // _ATTENTION_HEADS
_ATTENTION_MATRIX_PARAMS = _NUM_LAYERS * (
    2 * _HIDDEN * _HIDDEN + 2 * _HIDDEN * _KV_HEADS * _HEAD_DIM
)
_SINGLE_EMBEDDING_PARAMS = _VOCAB * _HIDDEN
_MINIMUM_DENSE_PARAMS = (
    _MLP_MATRIX_PARAMS + _ATTENTION_MATRIX_PARAMS + _SINGLE_EMBEDDING_PARAMS
)

# Bits/weight below actual Q8_0 block storage (which also needs scales).
_MIN_BITS_BY_QUANTIZATION = {"fp16": 16, "q8_0": 8}


def conservative_kaggle_weight_floor(model_id: str, candidate: dict,
                                     total_vram_bytes: int) -> dict | None:
    """Skip only recognized, canonical candidates whose tensor floor > VRAM.

    This argument assumes the Qwen publisher's variant follows the standard
    Qwen2.5-7B-Instruct model architecture and the named FP16/Q8_0 encoding.
    On publisher/model/handle/format ambiguity, do not skip: return None.
    """
    if not isinstance(total_vram_bytes, int) or total_vram_bytes <= 0:
        raise ValueError("Dimensione VRAM non valida.")
    if model_id != _MODEL_ID:
        return None
    spec = ModelManager.source_spec(model_id, "kaggle")
    if spec["repository"] != _SOURCE_REPO:
        return None
    if candidate.get("repo_id") != _SOURCE_REPO:
        return None
    handle = candidate.get("kaggle_handle")
    if (not isinstance(handle, str)
        or not (handle == _SOURCE_HANDLE_PREFIX
                or (handle.startswith(_SOURCE_HANDLE_PREFIX + "/")
                    and handle[len(_SOURCE_HANDLE_PREFIX) + 1:].isdigit()))):
        return None
    quant = str(candidate.get("quantization") or "").casefold()
    bits = _MIN_BITS_BY_QUANTIZATION.get(quant)
    if bits is None:
        return None
    bound_bytes = _MINIMUM_DENSE_PARAMS * bits // 8
    return {
        "evidence": "standard_qwen2_5_7b_architecture_dense_matrix_floor",
        "source_architecture": "Qwen/Qwen2.5-7B-Instruct/config.json",
        "assumptions": (
            "Standard Qwen2.5-7B-Instruct shapes and named GGUF "
            "FP16 or Q8_0 encoding; one embedding matrix only; "
            "no metadata or runtime CUDA allocation included."
        ),
        "quantization": quant,
        "minimum_dense_parameters": _MINIMUM_DENSE_PARAMS,
        "minimum_bits_per_weight": bits,
        "minimum_weight_bytes": bound_bytes,
        "minimum_weight_mib": round(bound_bytes / _MIB, 1),
        "total_vram_mib": round(total_vram_bytes / _MIB, 1),
        "exceeds_total_vram": bound_bytes > total_vram_bytes,
    }
