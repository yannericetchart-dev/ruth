"""Read-only CUDA Toolkit 13.2 preflight for native Ruth CUDA builds.

Not the same as "CUDA Version" from nvidia-smi (driver capability).
No installation, environment changes, compiler invocation or GPU allocation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Callable, Mapping

CUDA_TOOLKIT_VERSION = (13, 2)
CUDA_ARCHITECTURE = "89"
TOOLKIT_PACKAGE = "cuda-toolkit-13-2"
NVIDIA_INSTALL_GUIDE = (
    "https://docs.nvidia.com/cuda/archive/13.2.0/cuda-installation-guide-linux/"
)
TOOLKIT_SETUP_GUIDE = (
    "Preferisci il CUDA Toolkit 13.2 senza reinstallare il driver NVIDIA. "
    "Su Ubuntu/Debian, dopo avere configurato e verificato il repository NVIDIA "
    "compatibile con la distribuzione: 'apt-cache policy cuda-toolkit-13-2' "
    "e 'apt-get -s install cuda-toolkit-13-2' (simulazione, senza installare). "
    "Solo con autorizzazione esplicita: 'sudo apt install cuda-toolkit-13-2'. "
    "Non usare il metapacchetto 'cuda' che può installare o aggiornare il driver. "
    "Istruzioni ufficiali: " + NVIDIA_INSTALL_GUIDE
)
_NVCC_RELEASE = re.compile(r"\brelease\s+(\d+)\.(\d+)\b")


@dataclass(frozen=True)
class ToolkitCheck:
    status: str  # ok, missing, incompatible, probe_failed, missing_build_tools
    expected_version: str
    detected_version: str | None
    nvcc_path: str | None
    cmake_path: str | None
    cpp_compiler_path: str | None
    cuda_architecture: str
    native_build_prerequisites: bool
    detail: str
    remediation: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def parse_nvcc_version(output: str) -> tuple[int, int] | None:
    """Parse 'nvcc --version' rather than guessing from the driver."""
    match = _NVCC_RELEASE.search(output)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _locate_nvcc(
    which: Callable[[str], str | None],
    exists: Callable[[Path], bool],
    environ: Mapping[str, str],
) -> str | None:
    # Prefer an explicit versioned CUDA installation; don't accidentally use
    # another toolkit merely because /usr/local/cuda points to it.
    for path in (
        Path("/usr/local/cuda-13.2/bin/nvcc"),
        *(Path(environ[key]) / "bin" / "nvcc"
          for key in ("CUDA_HOME", "CUDA_PATH") if environ.get(key)),
    ):
        if exists(path):
            return str(path)
    return which("nvcc")


def inspect_cuda_toolkit(
    *,
    which: Callable[[str], str | None] = shutil.which,
    exists: Callable[[Path], bool] = Path.is_file,
    environ: Mapping[str, str] | None = None,
    run: Callable[..., object] = subprocess.run,
) -> ToolkitCheck:
    """A missing Toolkit is expected on an unprepared host; no auto-fallback."""
    env = os.environ if environ is None else environ
    nvcc = _locate_nvcc(which, exists, env)
    cmake = which("cmake")
    cxx = which("c++")
    version: tuple[int, int] | None = None

    if not nvcc:
        status, detail = "missing", (
            "nvcc non rilevato. CUDA 13.2 nel driver non significa Toolkit installato."
        )
    else:
        try:
            result = run([nvcc, "--version"], capture_output=True, text=True,
                         check=False, timeout=8)
            if result.returncode != 0:
                status, detail = "probe_failed", "nvcc non ha completato il controllo."
            else:
                version = parse_nvcc_version(result.stdout + "\n" + result.stderr)
                if version is None:
                    status, detail = "probe_failed", "Versione Toolkit non riconoscibile dall'output nvcc."
                elif version != CUDA_TOOLKIT_VERSION:
                    status, detail = "incompatible", (
                        "Toolkit trovato: " + ".".join(map(str, version))
                        + "; Ruth richiede 13.2 per la build primaria fissata."
                    )
                else:
                    status, detail = "ok", "CUDA Toolkit 13.2 verificato tramite nvcc."
        except (OSError, subprocess.SubprocessError) as exc:
            status, detail = "probe_failed", "Impossibile eseguire nvcc: " + type(exc).__name__

    if status == "ok" and (not cmake or not cxx):
        status = "missing_build_tools"
        detail = "Toolkit 13.2 presente, ma CMake e/o compilatore C++ assenti."
    prereqs = status == "ok"
    fix = None if prereqs else TOOLKIT_SETUP_GUIDE
    if status == "missing_build_tools":
        fix = (
            "Per Ubuntu/Debian, verifica i repository e installa CMake/compilatore "
            "solo con autorizzazione: sudo apt install cmake build-essential. "
            + TOOLKIT_SETUP_GUIDE
        )
    return ToolkitCheck(
        status=status, expected_version="13.2",
        detected_version=".".join(map(str, version)) if version else None,
        nvcc_path=nvcc, cmake_path=cmake, cpp_compiler_path=cxx,
        cuda_architecture=CUDA_ARCHITECTURE,
        native_build_prerequisites=prereqs,
        detail=detail, remediation=fix,
    )
