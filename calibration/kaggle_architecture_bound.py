"""Data-driven conservative CUDA weight floors for standard dense decoder GGUFs.

No model-specific parameter counts: load an explicitly trusted upstream
config.json declared by the Kaggle source and compute mandatory matrix
parameters. Only known dense, gated-MLP/GQA decoder structures can use the
bound; unsupported architectures and unrelated variants return None.

A source config is an *assumption* about the model in a GGUF: it is not a
remote GGUF tensor header. The bound is deliberately limited to mandatory
dense matrices and never uses Kaggle download size to infer CUDA viability.
"""

import json
from pathlib import Path

from model import ModelManager

MIB = 1048576

# These are minimum bits per weight, not exact GGUF storage estimates.
MIN_BITS_PER_WEIGHT = {"fp16": 16, "q8_0": 8}
# Dense decoder families with Q/K/V/O and gated up/down MLP projections.
DENSE_GATED_GQA = frozenset(("qwen2", "llama", "mistral"))


def read_architecture_reference(spec: dict) -> dict | None:
    """Fetch only a small official config.json, never model weights.

    Configuration is opt-in per registered Kaggle source. A broken declared
    reference is an error, not permission to guess or download giant weights.
    """
    repo = spec.get("architecture_reference_repo")
    if not repo:
        return None
    if not isinstance(repo, str) or repo.count("/") != 1:
        raise ValueError("Repository di riferimento architetturale non valido.")
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo_id=repo, filename="config.json"))
    if path.stat().st_size > 1_048_576:
        raise ValueError("Config architetturale non ragionevolmente piccolo.")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("model_type") != spec["architecture"]:
        raise ValueError("Config architetturale incompatibile con sorgente GGUF.")
    return {"config": config, "repo": repo}


def mandatory_dense_parameter_floor(config: dict) -> int | None:
    """Count only mandatory tensor entries, ignoring optional parameters.

    Valid for known dense decoder architectures, not MoE/multimodal/tied
    special variants. Never extend the inference to an unknown model type.
    """
    if not isinstance(config, dict) or config.get("model_type") not in DENSE_GATED_GQA:
        return None
    if any(config.get(k) is not None for k in (
        "num_local_experts", "num_experts", "num_experts_per_tok",
        "vision_config", "audio_config", "text_config",
    )):
        return None
    fields = (
        "num_hidden_layers", "hidden_size", "intermediate_size",
        "num_attention_heads", "num_key_value_heads", "vocab_size",
    )
    values = [config.get(name) for name in fields]
    if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in values):
        return None
    layers, hidden, intermediate, heads, kv_heads, vocab = values
    if hidden % heads or kv_heads > heads or heads > 1024 or layers > 10000:
        return None
    head_dim = config.get("head_dim", hidden // heads)
    if not isinstance(head_dim, int) or head_dim < 1 or head_dim > hidden:
        return None
    # Includes ONE embedding matrix and the gated up/down projections.
    # Q and O are hidden x hidden; K and V are hidden x kv_heads*head_dim.
    return (
        vocab * hidden
        + layers * (
            3 * hidden * intermediate
            + 2 * hidden * hidden
            + 2 * hidden * kv_heads * head_dim
        )
    )


def conservative_kaggle_weight_floor(
    model_id: str, candidate: dict, total_vram_bytes: int, *,
    reference: dict | None = None,
) -> dict | None:
    """For any configured Kaggle model, estimate a *minimum* tensor payload.

    A reference must explicitly match the Kaggle source/variant. Missing
    or ambiguous evidence returns None so the candidate is not excluded.
    """
    if not isinstance(total_vram_bytes, int) or total_vram_bytes <= 0:
        raise ValueError("Dimensione VRAM non valida.")
    if reference is None:
        return None
    try:
        spec = ModelManager.source_spec(model_id, "kaggle")
    except ValueError:
        return None
    source_handle = spec.get("kaggle_handle")
    requested_handle = candidate.get("kaggle_handle")
    filename = candidate.get("filename")
    prefix = spec.get("gguf_prefix")
    if (
        candidate.get("repo_id") != spec.get("repository")
        or reference.get("repo") != spec.get("architecture_reference_repo")
        or not isinstance(source_handle, str)
        or not isinstance(requested_handle, str)
        or not (
            requested_handle == source_handle
            or (
                requested_handle.startswith(source_handle + "/")
                and requested_handle[len(source_handle) + 1:].isdigit()
            )
        )
        or not isinstance(filename, str)
        or not isinstance(prefix, str)
        or not filename.casefold().startswith(prefix.casefold())
        or not filename.casefold().endswith(".gguf")
        or candidate.get("filename_matches_model_prefix") is False
        or candidate.get("split_complete") is False
    ):
        return None
    bits = MIN_BITS_PER_WEIGHT.get(
        str(candidate.get("quantization") or "").casefold()
    )
    if bits is None:
        return None
    config = reference.get("config")
    if not isinstance(config, dict) or config.get("model_type") != spec.get("architecture"):
        return None
    parameters = mandatory_dense_parameter_floor(config)
    if parameters is None:
        return None
    bytes_floor = parameters * bits // 8
    return {
        "evidence": "configured_reference_dense_decoder_mandatory_matrix_floor",
        "source_architecture": reference["repo"] + "/config.json",
        "assumptions": (
            "Kaggle model/variant matches configured upstream dense-decoder "
            "config; standard FP16 or Q8_0 encoding; only necessary matrices "
            "counted; GGUF tensor header not yet downloaded or verified."
        ),
        "quantization": str(candidate["quantization"]).casefold(),
        "minimum_dense_parameters": parameters,
        "minimum_bits_per_weight": bits,
        "minimum_weight_bytes": bytes_floor,
        "minimum_weight_mib": round(bytes_floor / MIB, 1),
        "total_vram_mib": round(total_vram_bytes / MIB, 1),
        "exceeds_total_vram": bytes_floor > total_vram_bytes,
    }
