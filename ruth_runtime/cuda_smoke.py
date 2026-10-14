"""Phase 02: compile and run a CUDA-only kernel on the expected NVIDIA GPU.

Default CLI is read-only. --run explicitly compiles a temporary binary with
nvcc for sm_89, runs it and verifies the kernel result and PCI identity.
No downloads, pip, system installation, model loading or CPU fallback.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Callable

from .dependencies import DependencyReport, inspect_dependencies
from .hardware import HardwareReport, inspect_linux_hardware, normalize_pci_address
from .toolkit import ToolkitCheck, inspect_cuda_toolkit

SOURCE = Path(__file__).resolve().parent / "native" / "cuda_smoke.cu"
_SUCCESS = re.compile(
    r"^RUTH_CUDA_SMOKE_OK device=(\d+) arch=(\d+)\.(\d+) "
    r"pci=(\S+) elements=(\d+)$"
)
_REQUIRED = frozenset({
    "GNU/Linux x86_64", "Python >=3.10 (Ruth .venv)",
    "Ruth .venv", "nvidia-smi", "GPU/driver NVIDIA",
    "CUDA supportata dal driver",
})


@dataclass(frozen=True)
class CudaSmokeReport:
    status: str
    ready: bool
    compiled: bool
    executed: bool
    gpu_verified: bool
    cuda_architecture: str
    pci_address: str | None
    source: str
    compiler: str | None
    description: str
    error: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _result(
    status: str, ready: bool, compiled: bool, executed: bool, verified: bool,
    pci: str | None, nvcc: str | None, description: str, error: str | None = None,
    source: Path = SOURCE,
) -> CudaSmokeReport:
    return CudaSmokeReport(
        status=status, ready=ready, compiled=compiled, executed=executed,
        gpu_verified=verified, cuda_architecture="sm_89", pci_address=pci,
        source=str(source), compiler=nvcc, description=description,
        error=error,
    )


def interpret_kernel_output(output: str, expected_pci: str) -> tuple[bool, str]:
    lines = [line.strip() for line in output.splitlines()
             if line.strip().startswith("RUTH_CUDA_SMOKE_OK")]
    if len(lines) != 1:
        return False, "Token di successo CUDA assente, duplicato o non valido"
    match = _SUCCESS.fullmatch(lines[0])
    if not match:
        return False, "Risposta di successo CUDA non conforme al protocollo"
    index, major, minor, pci, elements = match.groups()
    if (index, major, minor, elements) != ("0", "8", "9", "256"):
        return False, "Il risultato non corrisponde a device 0 / sm_89 / 256 elementi"
    observed = normalize_pci_address(pci)
    expected = normalize_pci_address(expected_pci)
    if observed is None or expected is None or observed != expected:
        return False, (
            f"Identità PCI GPU differente: runtime={observed}, inventario={expected}"
        )
    return True, "Kernel CUDA eseguito e risultati validati sulla GPU PCI attesa"


def cuda_smoke(
    *,
    execute: bool = False,
    hardware: HardwareReport | None = None,
    dependencies: DependencyReport | None = None,
    toolkit: ToolkitCheck | None = None,
    source: Path = SOURCE,
    runner: Callable[..., object] = subprocess.run,
) -> CudaSmokeReport:
    """Run only when requested; ephemeral native binary under OS tempdir."""
    hardware = hardware if hardware is not None else inspect_linux_hardware()
    dependencies = (
        dependencies if dependencies is not None
        else inspect_dependencies(hardware=hardware)
    )
    toolkit = toolkit if toolkit is not None else inspect_cuda_toolkit()
    checks = {c.name: c.status for c in dependencies.checks}
    missing = sorted(name for name in _REQUIRED if checks.get(name) != "ok")
    supported = [
        g for g in hardware.gpus
        if g.vendor == "nvidia" and g.driver and g.uuid and g.pci_address
    ]
    pci = supported[0].pci_address if len(supported) == 1 else None
    if missing:
        return _result("blocked", False, False, False, False, pci,
                       toolkit.nvcc_path, "Preflight GNU/Linux/NVIDIA non superato",
                       "Controlli non OK: " + ", ".join(missing), source)
    if not toolkit.native_build_prerequisites:
        return _result("blocked", False, False, False, False, pci,
                       toolkit.nvcc_path, "CUDA Toolkit 13.2 non pronto",
                       (toolkit.detail + "; " + (toolkit.remediation or "")), source)
    if len(supported) != 1:
        return _result("blocked", False, False, False, False, None,
                       toolkit.nvcc_path, "Serve una sola GPU NVIDIA identificata per PCI",
                       f"GPU NVIDIA identificabili: {len(supported)}", source)
    if not source.is_file():
        return _result("blocked", False, False, False, False, pci,
                       toolkit.nvcc_path, "Sorgente CUDA smoke test assente",
                       str(source), source)
    if not execute:
        return _result("ready", True, False, False, False, pci,
                       toolkit.nvcc_path, "Pronto per --run; nessun kernel eseguito", source=source)

    # Neither source code nor generated binaries are written to the repository.
    # subprocess shell=False by design.
    try:
        with tempfile.TemporaryDirectory(prefix="ruth-cuda-smoke-") as temp:
            executable = Path(temp) / "cuda_smoke"
            command = [
                str(toolkit.nvcc_path), "-O2", "-std=c++17", "-arch=sm_89",
                "-o", str(executable), str(source.resolve()),
            ]
            build = runner(command, cwd=temp, capture_output=True,
                           text=True, check=False, timeout=120)
            if build.returncode != 0:
                return _result("compile_failed", True, False, False, False, pci,
                               toolkit.nvcc_path, "Compilazione CUDA fallita",
                               (build.stderr or build.stdout or "nvcc returned nonzero")[-3000:], source)
            if not executable.is_file():
                return _result("compile_failed", True, False, False, False, pci,
                               toolkit.nvcc_path, "nvcc non ha prodotto l'eseguibile",
                               "Binary missing despite exit code zero", source)
            observed = runner([str(executable)], cwd=temp, capture_output=True,
                              text=True, check=False, timeout=30)
            if observed.returncode != 0:
                return _result("execution_failed", True, True, False, False, pci,
                               toolkit.nvcc_path, "Il processo CUDA non è riuscito",
                               (observed.stderr or observed.stdout or "process exited nonzero")[-3000:], source)
            valid, message = interpret_kernel_output(observed.stdout, pci)
            if not valid:
                return _result("verification_failed", True, True, True, False,
                               pci, toolkit.nvcc_path,
                               "Risultato CUDA non verificato", message, source)
            return _result("verified", True, True, True, True, pci,
                           toolkit.nvcc_path, message, source=source)
    except (OSError, subprocess.SubprocessError) as exc:
        return _result("probe_failed", True, False, False, False, pci,
                       toolkit.nvcc_path, "Impossibile completare la prova CUDA",
                       type(exc).__name__ + ": " + str(exc), source)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ruth native CUDA sm_89 smoke test")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--plan", action="store_true",
                         help="Read-only preflight (default)")
    actions.add_argument("--run", action="store_true",
                         help="Explicitly compile and execute a temporary CUDA kernel")
    args = parser.parse_args()
    result = cuda_smoke(execute=args.run)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if result.status in (("verified",) if args.run else ("ready",)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
