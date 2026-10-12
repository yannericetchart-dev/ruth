"""Declarative inference-engine capability matrix (planning, not runnable code).

For pending engines these entries capture the user's supplied offload table.
They are NOT installation checks, runtime probes, feature detection, or a
promise that a given model can execute. No adapter is imported here.
"""

ENGINE_CATALOG = {
    "llama.cpp": {
        "label": "llama.cpp",
        "method": "llama_cpp_python",
        "status": "implementato",
        "enabled": True,
        "model_weight_layer_cpu_offload": True,
        "weight_offload_detail": "Totale o parziale per layer",
        "kv_cache_host_offload": False,
        "kv_offload_detail": "Nessuno swapping KV su host nella matrice fornita",
        "model_formats": ("GGUF",),
        "format_notes": "GGUF",
        "profile_phases": ("inventory", "baseline", "layer_matrix", "context_gpu"),
    },
    "exllamav2": {
        "label": "ExLlamaV2",
        "method": "exllamav2",
        "status": "da fare",
        "enabled": False,
        "model_weight_layer_cpu_offload": False,
        "weight_offload_detail": "Pesi integralmente in VRAM",
        "kv_cache_host_offload": False,
        "kv_offload_detail": "Nessuno swapping KV nativo su RAM",
        "model_formats": ("EXL2", "GPTQ"),
        "format_notes": "EXL2, GPTQ",
        "profile_phases": (),
    },
    "vllm": {
        "label": "vLLM",
        "method": "vllm",
        "status": "da fare",
        "enabled": False,
        "model_weight_layer_cpu_offload": False,
        "weight_offload_detail": "No nella tabella; CPU-only OpenVINO/x86 fuori scope",
        "kv_cache_host_offload": True,
        "kv_offload_detail": "Swapping asincrono RAM o storage condiviso",
        "model_formats": ("AWQ", "GPTQ", "FP16", "BF16"),
        "format_notes": "No EXL2",
        "profile_phases": (),
    },
    "aphrodite": {
        "label": "Aphrodite Engine",
        "method": "aphrodite",
        "status": "da fare",
        "enabled": False,
        "model_weight_layer_cpu_offload": False,
        "weight_offload_detail": "Nativo GPU; niente offload layer nella tabella",
        "kv_cache_host_offload": True,
        "kv_offload_detail": "Configurabile con variabili d'ambiente dedicate",
        "model_formats": ("EXL2", "GPTQ", "AWQ"),
        "format_notes": "EXL2, GPTQ, AWQ",
        "profile_phases": (),
    },
    "tensorrt-llm": {
        "label": "TensorRT-LLM",
        "method": "tensorrt_llm",
        "status": "da fare",
        "enabled": False,
        "model_weight_layer_cpu_offload": False,
        "weight_offload_detail": "Richiede un TensorRT engine compilato per GPU",
        "kv_cache_host_offload": True,
        "kv_offload_detail": "Host offloading su memoria CPU pinned",
        "model_formats": ("TensorRT engine compilato",),
        "format_notes": "Conversione/compilazione da Hugging Face",
        "profile_phases": (),
    },
}


def engine_info(engine: str) -> dict:
    """Return metadata for a registered engine; never initialize it."""
    try:
        return dict(ENGINE_CATALOG[engine])
    except KeyError:
        raise ValueError(
            f"Motore sconosciuto: {engine}. "
            f"Disponibili: {', '.join(ENGINE_CATALOG)}."
        ) from None


def validate_engine_method(engine: str, method: str) -> dict:
    """Fail before hardware discovery for future/unsupported adapters."""
    info = engine_info(engine)
    if not info["enabled"]:
        raise NotImplementedError(
            f"{info['label']}: adapter/profilazione da fare. "
            "Le capacita' sono censite ma nessuna fase e' eseguibile. "
            "Nessun fallback su llama.cpp."
        )
    if method != info["method"]:
        raise ValueError(
            f"Metodo non supportato per {engine}: {method}. "
            f"Disponibile: {info['method']}. Nessun fallback automatico."
        )
    return info
