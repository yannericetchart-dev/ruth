"""Granular GGUF tensor placement for llama-cpp-python 0.3.36 (Linux).

llama.cpp supports tensor_buft_overrides, but Llama(...) does not expose it.
The injection is scoped to model construction. Use this adapter in an isolated
calibration worker: temporarily patching a Python class is not thread-safe.
"""

import ctypes
import re
import threading
from pathlib import Path

from model import ModelManager

_PATCH_LOCK = threading.Lock()


class _TensorBufferOverride(ctypes.Structure):
    _fields_ = [("pattern", ctypes.c_char_p), ("buft", ctypes.c_void_p)]


def _backend_functions():
    """Resolve backend symbols only from libraries shipped with this venv."""
    import llama_cpp
    import llama_cpp.llama_cpp as api

    directory = Path(llama_cpp.__file__).parent / "lib"
    handles = [api._lib]
    for filename in sorted(directory.glob("libggml*.so*")):
        try:
            handles.append(ctypes.CDLL(str(filename), mode=ctypes.RTLD_GLOBAL))
        except OSError:
            continue

    def symbol(name, argument_types, return_type):
        for handle in handles:
            try:
                fn = getattr(handle, name)
            except AttributeError:
                continue
            fn.argtypes = argument_types
            fn.restype = return_type
            return fn
        raise RuntimeError(
            f"Simbolo nativo {name} non trovato nelle librerie GGML del venv. "
            "Questa build non puo' applicare mapping granulari."
        )

    count = symbol("ggml_backend_dev_count", [], ctypes.c_size_t)
    get = symbol("ggml_backend_dev_get", [ctypes.c_size_t], ctypes.c_void_p)
    name = symbol("ggml_backend_dev_name", [ctypes.c_void_p], ctypes.c_char_p)
    buffer_type = symbol("ggml_backend_dev_buffer_type", [ctypes.c_void_p], ctypes.c_void_p)
    cpu_type = symbol("ggml_backend_cpu_buffer_type", [], ctypes.c_void_p)

    devices = []
    for index in range(count()):
        dev = get(index)
        label = name(dev).decode("utf-8", errors="replace")
        if label.upper().startswith("CUDA"):
            devices.append((label, dev))

    if len(devices) != 1:
        raise RuntimeError(
            f"Adapter sperimentale: prevista esattamente una GPU CUDA, "
            f"rilevate {[x[0] for x in devices]}."
        )

    gpu = buffer_type(devices[0][1])
    cpu = cpu_type()
    if not gpu or not cpu:
        raise RuntimeError("Impossibile ottenere i buffer CPU/CUDA dal runtime GGML.")
    return {"cpu": cpu, "gpu": gpu}, handles, devices[0][0]


class LlamaCppPythonMappingAdapter:
    method = "llama-cpp-python"
    backend = "cuda"

    @staticmethod
    def capabilities() -> dict:
        from importlib.metadata import version

        current = version("llama-cpp-python")
        return {
            "method": LlamaCppPythonMappingAdapter.method,
            "backend": LlamaCppPythonMappingAdapter.backend,
            "version": current,
            "supported": current == "0.3.36",
            "placements": ["cpu", "gpu"],
            "granularity": ["contiguous_blocks", "regex_tensors"],
            "requires_isolated_process": True,
            "tied_embedding_output": "same_regex_both_uses",
            "output_gpu_required": False,
        }

    def load(self, path: Path, *, n_gpu_layers: int, rules: list[dict], n_ctx: int = 512):
        """Load a model with native per-tensor placement rules.

        Rules are {"pattern": regex, "device": "cpu"|"gpu"}. A pattern
        must match a real GGUF tensor. Gemma 2's tied token_embd.weight can be placed only as a coupled
        input/output mapping, never independently with a name regex.
        """
        from llama_cpp import Llama
        from llama_cpp import _internals

        caps = self.capabilities()
        if not caps["supported"]:
            raise RuntimeError(
                f"Adapter validato solo per llama-cpp-python 0.3.36; installato {caps['version']}."
            )
        if n_gpu_layers < 1:
            raise ValueError("n_gpu_layers deve essere positivo in questo esperimento; "
                                 "l'output puo' essere collocato su host tramite override.")
        if not rules:
            raise ValueError("Nessun override richiesto.")
        if not _PATCH_LOCK.acquire(blocking=False):
            raise RuntimeError("Un altro caricamento granulare e' gia' in corso.")

        try:
            layout = ModelManager().inspect(n_ctx=n_ctx) if path.resolve() == ModelManager().get() else None
            if layout is None:
                from model.inspect import inspect_gguf

                layout = inspect_gguf(path, n_ctx=n_ctx)
            all_names = [t["name"] for t in layout["tensors"]]
            targets = []
            for rule in rules:
                if set(rule) != {"pattern", "device"} or rule["device"] not in {"cpu", "gpu"}:
                    raise ValueError(f"Regola non valida: {rule}")
                pattern = rule["pattern"]
                if not isinstance(pattern, str) or not pattern:
                    raise ValueError("Pattern regex vuoto/non valido.")
                try:
                    regex = re.compile(pattern)
                except re.error as exc:
                    raise ValueError(f"Regex non valida {pattern!r}: {exc}") from None
                matching = [n for n in all_names if regex.search(n)]
                if not matching:
                    raise ValueError(f"Override senza tensori corrispondenti: {pattern}")
                tied = "token_embd.weight" in matching
                if tied:
                    if (layout["metadata"].get("general.architecture") != "gemma2"
                        or matching != ["token_embd.weight"]
                        or pattern not in (r"^(token_embd\.weight)$",
                                           r"^token_embd\.weight$")):
                        raise ValueError(
                            "Embedding/output condivisi: e' consentito solo "
                            "l'override esatto e accoppiato di token_embd.weight."
                        )
                targets.append({
                    "pattern": pattern, "device": rule["device"],
                    "tensors": matching, "tied_input_output": tied,
                })

            # Llama() calls llama_backend_init() before constructing
            # internals.LlamaModel. Resolve GGML devices at that point.
            original = _internals.LlamaModel
            kept = {}
            def injected_model(*, path_model, params, verbose):
                buffers, handles, gpu_name = _backend_functions()
                entries = (_TensorBufferOverride * (len(targets) + 1))()
                for idx, item in enumerate(targets):
                    entries[idx].pattern = item["pattern"].encode("utf-8")
                    entries[idx].buft = buffers[item["device"]]
                # The last element is NULL-terminated.
                params.tensor_buft_overrides = ctypes.addressof(entries)
                kept["native_overrides"] = entries
                kept["native_handles"] = handles
                kept["gpu_name"] = gpu_name
                return original(path_model=path_model, params=params, verbose=verbose)

            _internals.LlamaModel = injected_model
            try:
                llm = Llama(
                    model_path=str(path), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                    n_batch=128, n_ubatch=128,
                    offload_kqv=True, flash_attn=False, verbose=True,
                )
            finally:
                _internals.LlamaModel = original

            llm._ruth_override_refs = kept
            return llm, targets
        finally:
            _PATCH_LOCK.release()
