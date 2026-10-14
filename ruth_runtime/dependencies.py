"""Read-only diagnostics restricted to Ruth's project-local .venv.

Python itself must first be made available on GNU/Linux (see bootstrap.sh).
No host pip probe; no model loading, installations, downloads or CUDA init.
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
from .requirements import (
    MIN_NVIDIA_DRIVER, MIN_CUDA_VERSION, driver_meets_policy,
    parse_driver_version, parse_reported_cuda_version, cuda_meets_policy,
    render_version,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_GUIDE = (
    "Un venv non può installare l'interprete Python sul sistema. Se necessario "
    "installa Python >=3.10 e il modulo venv tramite il gestore pacchetti "
    "(solo dopo autorizzazione). Esegui 'sh bootstrap.sh --plan' e poi "
    "'sh bootstrap.sh --create-venv'. Usa sempre '.venv/bin/python' "
    "e non installare dipendenze Python globalmente."
)
CUDA_BINDING_GUIDE = (
    "Installa llama-cpp-python SOLO nella .venv di Ruth dopo aver selezionato "
    "una versione/build CUDA compatibile e averla bloccata. Documentazione: "
    "https://llama-cpp-python.readthedocs.io/en/stable/ . "
    "Se scegli la build da sorgente prepara CUDA Toolkit/nvcc, CMake e "
    "compilatore C++; usa CMAKE_ARGS='-DGGML_CUDA=on' "
    "'.venv/bin/python -m pip install llama-cpp-python==<VERSIONE_APPROVATA>'. "
    "La sola installazione non prova l'inferenza; evita wheel CPU-only."
)
BINDING_PROBE = (
    "from llama_cpp.llama_cpp import llama_supports_gpu_offload; "
    "print('RUTH_OFFLOAD=' + ('yes' if llama_supports_gpu_offload() else 'no'))"
)


@dataclass(frozen=True)
class DependencyCheck:
    name: str
    required_now: bool
    status: str  # ok/missing/incompatible/probe_failed/optional_missing/blocked
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
        return "ok", "Binding importabile; llama.cpp dichiara GPU offload (non è prova di inferenza)"
    if "RUTH_OFFLOAD=no" in lines:
        return "incompatible", "Binding presente ma privo del supporto GPU offload"
    return "probe_failed", "Il probe non ha restituito una risposta valida"


def inspect_dependencies(
    hardware: HardwareReport | None = None,
    *,
    python_version: tuple[int, ...] | None = None,
    executable: str | None = None,
    project_root: Path | None = None,
    venv_prefix: Path | None = None,
    which: Callable[[str], str | None] = shutil.which,
    package_version: Callable[[str], str] = metadata.version,
    run: Callable[..., object] = subprocess.run,
    smi_run: Callable[..., object] = subprocess.run,
) -> DependencyReport:
    """Inspect only the active project .venv, never host site-packages.

    Inject venv_prefix in tests to simulate an isolated project venv.
    Production mode always validates the real prefix and pyvenv.cfg marker.
    """
    hardware = hardware if hardware is not None else inspect_linux_hardware()
    python_version = python_version if python_version is not None else sys.version_info[:3]
    executable = executable if executable is not None else sys.executable
    project_root = Path(project_root) if project_root is not None else PROJECT_ROOT
    expected_venv = (project_root / ".venv").resolve()
    actual_prefix = Path(venv_prefix) if venv_prefix is not None else Path(sys.prefix)
    correct_prefix = actual_prefix.resolve() == expected_venv
    cfg = expected_venv / "pyvenv.cfg"
    try:
        config_text = cfg.read_text(encoding="utf-8") if cfg.is_file() else ""
    except OSError:
        config_text = ""
    allows_global = any(
        line.strip().lower().replace(" ", "") == "include-system-site-packages=true"
        for line in config_text.splitlines()
    )
    has_venv_marker = (
        venv_prefix is not None  # test injection, never used by main()
        or (cfg.is_file() and sys.prefix != sys.base_prefix)
    )
    active_venv = (
        correct_prefix and has_venv_marker and not allows_global
        and not (project_root / ".venv").is_symlink()
    )

    checks: list[DependencyCheck] = []

    def add(name: str, required: bool, status: str, detail: str,
            fix: str | None = None) -> None:
        checks.append(DependencyCheck(name, required, status, detail, fix))

    add("GNU/Linux x86_64", True,
        "ok" if hardware.supported_platform else "incompatible",
        f"OS={hardware.os_name}, arch={hardware.architecture}",
        None if hardware.supported_platform else "Serve GNU/Linux x86_64.")
    python_ok = tuple(python_version[:2]) >= (3, 10)
    add("Python >=3.10 (Ruth .venv)", True,
        "ok" if python_ok and active_venv else ("missing" if not active_venv else "incompatible"),
        "Versione " + ".".join(map(str, python_version)) + (
            "; interprete appartenente al progetto" if active_venv
            else "; non stai usando l'interprete della .venv di Ruth"),
        None if python_ok and active_venv else VENV_GUIDE)
    add("Ruth .venv", True, "ok" if active_venv else "missing",
        "Ambiente progetto verificato" if active_venv else
        "Richiesta .venv del repository; pacchetti host ignorati",
        None if active_venv else VENV_GUIDE)

    smi = which("nvidia-smi")
    add("nvidia-smi", True, "ok" if smi else "missing",
        smi or "Comando NVIDIA non presente nel PATH",
        None if smi else "Installa/verifica il driver NVIDIA con la procedura della tua distribuzione; poi esegui nvidia-smi.")
    gpus = [d for d in hardware.gpus if d.vendor == "nvidia" and d.driver and d.uuid]
    versions = [parse_driver_version(d.driver) for d in gpus]
    driver_ok = bool(versions) and all(
        value is not None and value >= MIN_NVIDIA_DRIVER for value in versions
    )
    if not gpus:
        driver_status = "missing"
        driver_detail = "Nessuna GPU NVIDIA con driver e UUID verificati"
    elif not driver_ok:
        driver_status = "incompatible"
        driver_detail = ("Driver rilevati: " + ", ".join(str(d.driver) for d in gpus)
                         + "; minimo richiesto: " + render_version(MIN_NVIDIA_DRIVER))
    else:
        driver_status = "ok"
        driver_detail = (str(len(gpus)) + " GPU NVIDIA, driver: "
                         + ", ".join(sorted({str(d.driver) for d in gpus})))
    add("GPU/driver NVIDIA", True, driver_status, driver_detail,
        None if driver_status == "ok" else
        ("Richiesta policy Ruth: driver NVIDIA >= " + render_version(MIN_NVIDIA_DRIVER)
         + " (ramo R595, CUDA 13.2). Se hai appena aggiornato il driver, "
         "riavvia e riprova prima di reinstallare. Per l'installazione usa "
         "il gestore driver della distribuzione e le istruzioni NVIDIA, "
         "con autorizzazione esplicita; non modificare i driver automaticamente."))

    if not smi or not driver_ok:
        add("CUDA supportata dal driver", True, "blocked",
            "Serve prima un driver NVIDIA verificato conforme a R595/CUDA 13.2",
            "Risolvi il controllo GPU/driver NVIDIA prima di verificare CUDA.")
    else:
        try:
            smi_result = smi_run([smi], capture_output=True, text=True,
                                 timeout=8, check=False)
            if smi_result.returncode != 0:
                cuda_status, cuda_detail = "probe_failed", (
                    "nvidia-smi non ha completato correttamente la verifica CUDA")
            else:
                cuda_version = parse_reported_cuda_version(smi_result.stdout)
                cuda_status = ("ok" if cuda_meets_policy(cuda_version)
                               else "incompatible" if cuda_version is not None
                               else "probe_failed")
                cuda_detail = (
                    "Massima versione CUDA dichiarata dal driver: "
                    + (render_version(cuda_version) if cuda_version else "sconosciuta")
                    + "; minimo Ruth: " + render_version(MIN_CUDA_VERSION)
                    + ". Non certifica CUDA Toolkit né inferenza.")
        except (OSError, subprocess.SubprocessError) as exc:
            cuda_status = "probe_failed"
            cuda_detail = "Impossibile verificare CUDA tramite nvidia-smi: " + type(exc).__name__
        add("CUDA supportata dal driver", True, cuda_status, cuda_detail,
            None if cuda_status == "ok" else
            "Richiesto supporto CUDA >=13.2 nel driver NVIDIA. Riavvia se hai appena aggiornato "
            "il driver; controlla nvidia-smi, versioni e compatibilità prima di installare.")

    if not active_venv or not python_ok:
        add("llama-cpp-python", True, "blocked",
            "Controllo sospeso: serve prima Python compatibile dentro Ruth .venv",
            VENV_GUIDE)
    else:
        try:
            installed = package_version("llama-cpp-python")
        except metadata.PackageNotFoundError:
            installed = None
        except Exception as exc:
            add("llama-cpp-python", True, "probe_failed",
                f"Non è stato possibile ottenere la versione: {type(exc).__name__}",
                CUDA_BINDING_GUIDE)
            installed = "ERROR"
        if installed is None:
            add("llama-cpp-python", True, "missing",
                "Pacchetto assente nella .venv di Ruth", CUDA_BINDING_GUIDE)
        elif installed != "ERROR":
            add("llama-cpp-python", True, "ok", "Versione " + installed)
            status, detail = _run_probe(run, executable)
            add("llama.cpp GPU offload", True, status, detail,
                None if status == "ok" else CUDA_BINDING_GUIDE)

    # Build tools are host prerequisites only when compiling from source.
    # An existing compatible CUDA wheel does not require them. No pip probe.
    optional_tools = (
        ("CMake", "cmake", "Per Debian/Ubuntu: sudo apt install cmake (solo build da sorgente)."),
        ("Compilatore C++", "c++", "Per Debian/Ubuntu: sudo apt install build-essential (solo build da sorgente)."),
        ("CUDA Toolkit nvcc", "nvcc", "Toolchain NVIDIA: https://developer.nvidia.com/cuda-downloads (solo build da sorgente)."),
    )
    for name, cmd, fix in optional_tools:
        path = which(cmd)
        add(name, False, "ok" if path else "optional_missing",
            path or "Assente: richiesto soltanto se si compilerà da sorgente",
            None if path else fix)
    return DependencyReport(tuple(checks))


def main() -> int:
    report = inspect_dependencies()
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.ready_for_next_probe else 1


if __name__ == "__main__":
    raise SystemExit(main())
