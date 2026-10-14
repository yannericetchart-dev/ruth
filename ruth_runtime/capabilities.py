"""Matrice dichiarativa prevista dall'utente; nessuna disponibilità inventata.

TARGET significa "da sviluppare e testare per primo", *non* utilizzabile.
TO DO significa "presente nel catalogo", senza implementazione.
Una successiva fase registrerà risultati dei probe e verifiche hardware
separatamente da questo catalogo statico.
"""

from .contracts import (
    Capability, ComponentKind as K, DevelopmentState as S,
    StackAssessment, StackSelection, UnsupportedStackError,
)


# GNU/Linux + x86_64 sono vincoli del progetto, non adapter runtime.
PLATFORM_OS = "GNU/Linux"
PLATFORM_ARCH = "x86_64"


def _group(kind: K, target: tuple[str, ...], todo: tuple[str, ...]) -> tuple[Capability, ...]:
    return tuple(Capability(kind, name, S.TARGET) for name in target) + tuple(
        Capability(kind, name, S.TO_DO) for name in todo
    )


CATALOG: tuple[Capability, ...] = (
    _group(K.CPU_VENDOR, ("intel",), ("amd",))
    + _group(K.GPU_VENDOR, ("nvidia",), ("amd", "intel"))
    + _group(K.GPU_BACKEND, ("cuda",), (
        "vulkan", "rocm", "hip", "sycl", "oneapi", "xpu", "openvino-gpu",
    ))
    + _group(K.CPU_BACKEND, ("ggml-cpu",), (
        "onednn", "onemkl", "openvino-cpu", "onnxruntime-cpu",
        "vllm-cpu", "blis", "openblas",
    ))
    + _group(K.INFERENCE_ENGINE, ("llama.cpp",), (
        "vllm", "onnxruntime", "pytorch", "openvino", "mlc-llm",
        "iree", "sglang",
    ))
    + _group(K.INITIALIZATION, ("llama_cpp_python",), (
        "local-server", "system-library", "cli",
    ))
)

# Proposta iniziale: *solo* queste due combinazioni saranno sviluppate per prime.
# Non sono utilizzabili finché i test hardware nativi non le certificano.
INITIAL_STACKS: tuple[StackSelection, ...] = (
    StackSelection("gpu", "nvidia", "cuda", "llama.cpp", "llama_cpp_python"),
    StackSelection("cpu", "intel", "ggml-cpu", "llama.cpp", "llama_cpp_python"),
)


def components(kind: K | None = None) -> tuple[Capability, ...]:
    """Elenca capacità dichiarative (non la disponibilità sulla macchina)."""
    return tuple(c for c in CATALOG if kind is None or c.kind == kind)


def inspect_stack(selection: StackSelection) -> StackAssessment:
    """Distingue target pianificati e stack non specificati; non prova nulla."""
    lookup = {(c.kind, c.identifier): c for c in CATALOG}
    if selection.device_kind == "gpu":
        vendor_kind, backend_kind = K.GPU_VENDOR, K.GPU_BACKEND
    elif selection.device_kind == "cpu":
        vendor_kind, backend_kind = K.CPU_VENDOR, K.CPU_BACKEND
    else:
        raise UnsupportedStackError("Tipo dispositivo non riconosciuto: " + selection.device_kind)

    for kind, name in (
        (vendor_kind, selection.device_vendor),
        (backend_kind, selection.backend),
        (K.INFERENCE_ENGINE, selection.engine),
        (K.INITIALIZATION, selection.initialization),
    ):
        if (kind, name) not in lookup:
            raise UnsupportedStackError(f"Componente sconosciuto: {kind.value}={name}")

    if selection in INITIAL_STACKS:
        return StackAssessment(selection, S.TARGET,
                               "Primo target di implementazione; test nativi non ancora eseguiti")
    return StackAssessment(selection, S.TO_DO,
                           "Combinazione non implementata o non qualificata; TO DO")
