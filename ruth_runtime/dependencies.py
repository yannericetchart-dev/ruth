"""Read-only prerequisite diagnostics for NVIDIA/CUDA/llama.cpp/llama_cpp_python.

No installation and no CUDA/model initialization; a binding capability check is
isolated in a child process. GPU execution is NOT verified at this stage.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable

from .hardware import HardwareReport, inspect_linux_hardware


CUDA_WHEEL_GUIDE = (
    "Crea un ambiente isolato: python3 -m venv .venv; source .venv/bin/activate; "
    "python -m pip install --upgrade pip. Per un wheel CUDA scegli una versione "
    "compatibile con GPU, Python e driver dalla guida ufficiale: "
    "https://llama-cpp-python.readthedocs.io/en/stable/ "
    "(CUDA 13.0: indice cu130, se supportato). "
    "In alternativa installa CUDA Toolkit, CMake e compilatore C++ e compila "
    'con CMAKE_ARGS="-DGGML_CUDA=on" python -m pip install llama-cpp-python. '
    "Controlla sempre il backend GPU dopo l'installazione."
)

BINDING_PROBE = (
    "from llama_cpp.llama_cpp import llama_supports_gpu_offload; "
    "print('RUTH_OFFLOAD=' + ('yes' if llama_supports_gpu_offload() else 'no'))"
)


@dataclass(frozen=True)
class DependencyCheck:
    name: str
    required_now: bool
    status: str  # ok / missing / incompatible / probe_failed / optional_missing
    detail: str
    remediation: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class DependencyReport:
    checks: tuple[DependencyCheck, ...]

    @property
    def ready_for_next_probe(self) -> bool:
        return all(c.ok for c in self.checks if c.required_now)

    def to_dict(self) -> dict:
        return {
            "ready_for_next_probe": self.ready_for_next_probe,
            "inference_verified": False,
            "checks": [asdict(c) for c in self.checks],
        }


def _run_probe(run: Callable[..., object], executable: str) -> tuple[str, str]:
    try:
        result = run([executable, "-c", BINDING_PROBE], text=True,
                     capture_output=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return "probe_failed", f"Probe isolato non avviabile: {type(exc).__name__}"
    if result.returncode != 0:
        return "probe_failed", "Import nativo non riuscito o subprocess terminato con errore"
    lines = result.stdout.splitlines()
    if "RUTH_OFFLOAD=yes" in lines:
        return "ok", "Binding importabile; llama.cpp dichiara GPU offload disponibile (non prova inferenza)"
    if "RUTH_OFFLOAD=no" in lines:
        return "incompatible", "Binding presente ma compilato senza supporto GPU offload"
    return "probe_failed", "Il probe non ha restituito una risposta valida"


def inspect_dependencies(
    hardware: HardwareReport | None = None,
    *,
    python_version: tuple[int, ...] | None = None,
    executable: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
    package_version: Callable[[str], str] = metadata.version,
    run: Callable[..., object] = subprocess.run,
) -> DependencyReport:
    """No network, writes, installs, GPU benchmark or model imports in this process.

    Compiler/nvcc/cmake/pip are preparatory tools, not runtime requirements
    for an already working CUDA wheel. Later phases extend the registry.
    """
    hardware = hardware if hardware is not None else inspect_linux_hardware()
    python_version = python_version if python_version is not None else sys.version_info[:3]
    executable = executable if executable is not None else sys.executable
    checks: list[DependencyCheck] = []
    add = lambda n, req, status, detail, fix=None: checks.append(DependencyCheck(
        n, req, status, detail, fix))

    add("GNU/Linux x86_64", True, "ok" if hardware.supported_platform else "incompatible",
        f"OS={hardware.os_name}, arch={hardware.architecture}",
        None if hardware.supported_platform else "Serve un host GNU/Linux x86_64.")
    py_ok = tuple(python_version[:2]) >= (3, 10)
    add("Python >=3.10", True, "ok" if py_ok else "incompatible",
        "Python " + ".".join(str(v) for v in python_version),
        None if py_ok else "Installa Python >= 3.10 e crea un ambiente virtuale dedicato.")
    smi = which("nvidia-smi")
    add("nvidia-smi", True, "ok" if smi else "missing",
        smi or "Comando non trovato nel PATH",
        None if smi else "Verifica installazione e PATH del driver NVIDIA; consulta la documentazione della distribuzione GNU/Linux.")
    identified = [g for g in hardware.gpus if g.vendor == "nvidia"
                  and g.driver and g.uuid]
    add("GPU/driver NVIDIA", True, "ok" if identified else "missing",
        (str(len(identified)) + " GPU con identificazione nativa via nvidia-smi"
         if identified else "Nessuna GPU NVIDIA con driver e UUID verificati"),
        None if identified else "Esegui nvidia-smi; controlla driver, permessi e accessibilità GPU. Non reinstallare il driver alla cieca.")

    try:
        installed = package_version("llama-cpp-python")
    except metadata.PackageNotFoundError:
        installed = None
    except Exception as exc:
        add("llama-cpp-python", True, "probe_failed",
            f"Impossibile leggere la versione del pacchetto: {type(exc).__name__}", CUDA_WHEEL_GUIDE)
        installed = "ERROR"
    if installed is None:
        add("llama-cpp-python", True, "missing",
            "Binding non installato nell'interprete Python corrente", CUDA_WHEEL_GUIDE)
    elif installed != "ERROR":
        add("llama-cpp-python", True, "ok", "Versione distribuzione " + installed)
        status, detail = _run_probe(run, executable)
        add("llama.cpp GPU offload", True, status, detail,
            None if status == "ok" else CUDA_WHEEL_GUIDE)

    for name, command, fix in (
        ("pip", "pip", "Per Ubuntu/Debian: sudo apt install python3-pip python3-venv; verificare nell'ambiente virtuale."),
        ("CMake", "cmake", "Per Ubuntu/Debian: sudo apt install cmake; richiesto soltanto per compilare."),
        ("C++ compiler", "c++", "Per Ubuntu/Debian: sudo apt install build-essential; richiesto soltanto per compilare."),
        ("CUDA Toolkit nvcc", "nvcc", "Installa CUDA Toolkit seguendo https://developer.nvidia.com/cuda-downloads; necessario per compilare da sorgente, non per ogni wheel precompilato."),
    ):
        path = which(command)
        add(name, False, "ok" if path else "optional_missing",
            path or "Non trovato (opzionale finché non è richiesta una build da sorgente)",
            None if path else fix)
    return DependencyReport(tuple(checks))


def main() -> int:
    data = inspect_dependencies().to_dict()
    print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if data["ready_for_next_probe"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
