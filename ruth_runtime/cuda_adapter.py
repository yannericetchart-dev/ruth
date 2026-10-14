"""Restricted Ruth CUDA llama.cpp adapter; no disk or user-provided cache.

This module imports NO llama_cpp or diskcache on import. The adapter is a
narrow future runtime boundary, not proof of GPU inference or a replacement
for a native execution test. Do not expose raw llama_cpp.Llama to consumers.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Callable, Mapping, Any

from .contracts import StackSelection
from .install_plan import WHEEL_VERSION

INITIAL_STACK = StackSelection("gpu", "nvidia", "cuda", "llama.cpp", "llama_cpp_python")
_ALLOWED_OPTIONS = frozenset({"n_ctx", "n_batch", "n_ubatch"})


class UnsafeRuntimeOptions(ValueError):
    """Ruth refuses cache access or attempts to disable mandatory GPU offload."""


class UnverifiedCudaEngine(RuntimeError):
    """Fail closed: binding absent, wrong version, or CUDA offload unavailable."""


def _options(raw: Mapping[str, object]) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        raise UnsafeRuntimeOptions("Options must be an explicit mapping")
    keys = set(raw)
    extras = keys - _ALLOWED_OPTIONS
    if extras:
        raise UnsafeRuntimeOptions(
            "Unsupported or potentially unsafe option(s): " +
            ", ".join(sorted(str(x) for x in extras)))
    valid: dict[str, int] = {}
    for name in _ALLOWED_OPTIONS:
        if name in raw:
            value = raw[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise UnsafeRuntimeOptions(name + " must be a positive integer")
            valid[name] = value
    n_ctx = valid.get("n_ctx", 512)
    n_batch = valid.get("n_batch", 128)
    n_ubatch = valid.get("n_ubatch", 128)
    if n_batch > n_ctx or n_ubatch > n_batch:
        raise UnsafeRuntimeOptions("Require n_ubatch <= n_batch <= n_ctx")
    return {"n_ctx": n_ctx, "n_batch": n_batch, "n_ubatch": n_ubatch}


class _NoDiskCacheSession:
    """Only exposes generate/close; consumers cannot call set_cache through API."""

    __slots__ = ("_llm", "_closed")

    def __init__(self, llm: Any):
        self._llm = llm
        self._closed = False

    @property
    def cache_disabled(self) -> bool:
        return not self._closed and getattr(self._llm, "cache", object()) is None

    def _completion(self, prompt: str, *, max_tokens: int) -> dict:
        if self._closed:
            raise UnverifiedCudaEngine("Inference session already closed")
        if not self.cache_disabled:
            raise UnsafeRuntimeOptions("Llama cache must remain disabled")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("A nonempty text prompt is required")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) \
                or not 1 <= max_tokens <= 1024:
            raise ValueError("max_tokens must be between 1 and 1024")
        result = self._llm.create_completion(
            prompt=prompt, max_tokens=max_tokens, stream=False)
        return result

    def generate(self, prompt: str, *, max_tokens: int = 32) -> str:
        return str(self._completion(prompt, max_tokens=max_tokens)["choices"][0]["text"])

    def generate_report(self, prompt: str, *, max_tokens: int = 32) -> dict:
        """Return actual API token count, not a heuristic based on characters."""
        result = self._completion(prompt, max_tokens=max_tokens)
        usage = result.get("usage") or {}
        tokens = usage.get("completion_tokens")
        if type(tokens) is not int or tokens < 0:
            raise UnverifiedCudaEngine("Native completion token count not available")
        return {"completion_tokens": tokens,
                "text": str(result["choices"][0]["text"])}

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._llm.close()


@dataclass(frozen=True)
class CudaLlamaAdapter:
    """Strict, versioned CUDA-only initialization adapter.

    Does not create LlamaDiskCache, diskcache.Cache or load cached pickle.
    Requires a real native backend with the supported GPU-offload symbol.
    """
    load_backend: Callable[[str], Any] = importlib.import_module
    native_diagnostics: bool = False

    def load(
        self, model_path: Path, selection: StackSelection,
        options: Mapping[str, object],
    ) -> _NoDiskCacheSession:
        if selection != INITIAL_STACK:
            raise UnverifiedCudaEngine("Only NVIDIA/CUDA/llama.cpp is implemented")
        selected = _options(options)
        model = Path(model_path)
        if model.is_symlink() or not model.is_file() or model.suffix.lower() != ".gguf":
            raise ValueError("Model must be an existing regular GGUF file, not a symlink")
        try:
            backend = self.load_backend("llama_cpp")
        except (ImportError, OSError) as exc:
            raise UnverifiedCudaEngine("CUDA llama_cpp Python binding unavailable") from exc
        if getattr(backend, "__version__", None) != WHEEL_VERSION:
            raise UnverifiedCudaEngine(
                "Unexpected llama_cpp binding version; refuse runtime")
        offload = getattr(backend, "llama_supports_gpu_offload", None)
        if not callable(offload) or not offload():
            raise UnverifiedCudaEngine("CUDA GPU offload not supported by binding")
        constructor = getattr(backend, "Llama", None)
        if not callable(constructor):
            raise UnverifiedCudaEngine("llama_cpp.Llama constructor unavailable")
        llm = constructor(
            model_path=str(model.resolve()),
            n_gpu_layers=-1,  # request all possible layers on GPU
            main_gpu=0,
            n_ctx=selected["n_ctx"],
            n_batch=selected["n_batch"],
            n_ubatch=selected["n_ubatch"],
            vocab_only=False,
            verbose=self.native_diagnostics,
        )
        if getattr(llm, "cache", object()) is not None:
            try:
                llm.close()
            finally:
                raise UnsafeRuntimeOptions("Native Llama created a cache unexpectedly")
        return _NoDiskCacheSession(llm)
