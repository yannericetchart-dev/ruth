"""Configuration and stable identity for machine-specific Ruth calibrations.

Source, model, engine and inference method are explicit user choices.
GPU backend/VRAM and RAM are derived from the actual host; no user-supplied
VRAM number or backend is trusted. Future adapters are declared but disabled until implemented.
"""

from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from pathlib import Path

from hardware import GPUManager
from model import ModelManager
from inference.engine_catalog import validate_engine_method


SUPPORTED_PAIRS = {
    ("llama.cpp", "llama_cpp_python"): {"backend": "cuda"},
}


def _installed(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _ram_total_mib() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    raise RuntimeError("RAM totale non rilevabile da /proc/meminfo.")


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def validate_choices(model_id: str, source: str = "bartowski",
                     engine: str = "llama.cpp",
                     method: str = "llama_cpp_python") -> dict:
    """Validate user options independently of hardware probing."""
    item = ModelManager.source_spec(model_id, source)
    # Check declared future engines before testing the method combination:
    # --engine vllm must say 'da fare', not accidentally use llama_cpp_python.
    validate_engine_method(engine, method)
    if (engine, method) not in SUPPORTED_PAIRS:
        raise ValueError(
            f"Motore/metodo non implementati: {engine}/{method}. "
            "Nessun fallback automatico."
        )
    return item


def resolve_identity(model_id: str, source: str | None = None,
                     engine: str = "llama.cpp",
                     method: str = "llama_cpp_python",
                     status: dict | None = None,
                     kaggle_handle: str | None = None) -> dict:
    """Validate available adapters and resolve one reproducible configuration."""
    if model_id not in ModelManager.CATALOG:
        raise ValueError(f"Modello non registrato: {model_id}")
    source = source or ModelManager.CATALOG[model_id]["source"]
    item = validate_choices(model_id, source, engine, method)
    selected_repository = item["repository"]
    if source == "kaggle":
        kaggle_handle = kaggle_handle or item.get("kaggle_handle")
        if not kaggle_handle:
            raise ValueError("Fonte Kaggle senza handle GGUF verificabile.")
        from .kaggle_source import validate_handle
        selected_repository = "/".join(kaggle_handle.split("/")[:2])
        validate_handle(kaggle_handle, selected_repository)
    elif kaggle_handle is not None:
        raise ValueError("--kaggle-handle richiede la fonte kaggle.")
    support = SUPPORTED_PAIRS[(engine, method)]
    if status is None:
        status = GPUManager.calibration_status()
    if not status.get("ready"):
        raise RuntimeError(
            f"GPU non pronta: {status.get('errors')}; "
            f"{status.get('foreign_processes')}"
        )
    gpu = status.get("gpu") or {}
    memory = status.get("memory") or {}
    backend = "cuda" if gpu.get("vendor") == "nvidia" else None
    if backend != support["backend"]:
        raise RuntimeError(
            f"Backend rilevato '{backend}' non supportato da {engine}/{method}. "
            "Nessuna scelta CUDA forzata su hardware non NVIDIA."
        )
    vram_mib = memory.get("total_mib")
    if (not isinstance(vram_mib, (int, float))
        or isinstance(vram_mib, bool) or vram_mib <= 0
        or not gpu.get("uuid")):
        raise RuntimeError("Identita' GPU o capacita' VRAM non attendibile.")

    choices = {
        "source": source,
        "model_id": model_id,
        "repository": selected_repository,
        "architecture": item["architecture"],
        "engine": engine,
        "method": method,
        "kaggle_handle": kaggle_handle,
    }
    hardware = {
        "backend": backend,
        "gpu_vendor": gpu["vendor"],
        "gpu_name": gpu["name"],
        "gpu_uuid": gpu["uuid"],
        "gpu_index": gpu["index"],
        "vram_total_mib": vram_mib,
        "ram_total_mib": _ram_total_mib(),
        "cpu_model": _cpu_model(),
        "cpu_architecture": platform.machine(),
    }
    software = {
        "llama_cpp_python": _installed("llama-cpp-python"),
        "python": platform.python_version(),
        "linux": platform.system(),
    }
    identity = {
        "schema_version": 1,
        "choices": choices,
        "hardware": hardware,
        "software": software,
        # Preserve the existing CUDA reference settings and identity key;
        # layer_matrix reads the existing baseline and does not alter them.
        "reference_runtime": {
            "baseline_n_ctx": 64, "n_batch": 32,
            "context_generated_tokens": 8,
            "context_n_batch": 32,
            "context_confirm_rounds": 2,
            "offload_kqv": True, "flash_attn": False,
            "kv_type_k": "default_f16", "kv_type_v": "default_f16",
        },
    }
    stable = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    identity["profile_key"] = sha256(stable.encode()).hexdigest()[:24]
    return identity
