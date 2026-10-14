"""Strict NVIDIA Ruth hardware eligibility; separate Toolkit, driver, backend.

Non-destructive read-only telemetry. Architecture is identified by CUDA compute
capability, never by a GPU name substring or an assumed VRAM class.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Callable

from .dependencies import PROJECT_ROOT
from .toolkit import parse_nvcc_version

POLICY = Path("catalog/nvidia_architecture_policy.json")
_CC = re.compile(r"^\d{1,2}\.\d$")
_VER = re.compile(r"^(\d+)\.(\d+)(?:\.\d+)?$")
_DRV = re.compile(r"^\d{3,4}(?:\.\d+){1,3}$")
_PCI = re.compile(r"^[0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]$")
_UUID = re.compile(r"^GPU-[a-zA-Z0-9-]{12,64}$")
_CUDA_SMI = re.compile(r"CUDA Version:\s*(\d+\.\d+)")


def _version(s: str | None) -> tuple[int, int] | None:
    m = _VER.fullmatch(s or "")
    return (int(m[1]), int(m[2])) if m else None


def load_architecture_policy(root: Path = PROJECT_ROOT) -> dict:
    data = json.loads((Path(root) / POLICY).read_text(encoding="utf-8"))
    if data.get("schema") != "ruth-nvidia-architecture-policy-v1":
        raise ValueError("Unsupported hardware architecture policy")
    ids = set()
    capabilities = set()
    for arch in data["architectures"]:
        if (arch["id"] in ids or arch["tensor_core_generation"] < 4
                or not arch["compute_capabilities"] or not arch["vram_classes_gib"]
                or _version(arch["min_cuda"]) is None):
            raise ValueError("Invalid or duplicate Ruth architecture")
        ids.add(arch["id"])
        for cc in arch["compute_capabilities"]:
            if not _CC.fullmatch(cc) or cc in capabilities:
                raise ValueError("Invalid or ambiguous compute capability")
            capabilities.add(cc)
        if any(type(x) is not int or x < 1 for x in arch["vram_classes_gib"]):
            raise ValueError("Invalid VRAM class")
    return data


def classify_hardware(
    *, compute_capability: str | None, total_mib: int | None,
    driver_version: str | None, driver_cuda_max: str | None,
    toolkit_cuda: str | None, backend_cuda: str | None,
    policy: dict,
) -> dict:
    """Eligibility does not attest kernel cubin/PTX or usable inference."""
    problems, warnings = [], []
    matching = [a for a in policy["architectures"]
                if compute_capability in a["compute_capabilities"]]
    arch = matching[0] if len(matching) == 1 else None
    if arch is None:
        problems.append("unsupported_or_unverified_compute_capability")
    min_cuda = _version(arch["min_cuda"]) if arch else None
    latest = _version(policy["latest_reviewed_cuda"])
    drvmax = _version(driver_cuda_max)
    backend = _version(backend_cuda)
    toolkit = _version(toolkit_cuda)
    if drvmax is None:
        problems.append("driver_cuda_capability_unverified")
    if not driver_version or not _DRV.fullmatch(driver_version):
        problems.append("driver_version_unverified")
    cls = None
    if arch and type(total_mib) is int and total_mib > 0:
        diff = [(abs(total_mib - x*1024)/(x*1024),x)
                for x in arch["vram_classes_gib"]]
        deviation, candidate = min(diff)
        if deviation <= policy["nominal_vram_tolerance_percent"]/100:
            cls = candidate
    if cls is None:
        problems.append("vram_class_not_allowed_or_unverified")

    if min_cuda:
        if drvmax and drvmax < min_cuda:
            problems.append("driver_cuda_below_architecture_minimum")
        if backend is None:
            problems.append("backend_cuda_version_unverified")
        elif backend < min_cuda:
            problems.append("backend_cuda_below_architecture_minimum")
        if toolkit is not None and toolkit < min_cuda:
            problems.append("toolkit_cuda_below_architecture_minimum")
    if drvmax and backend and drvmax < backend:
        problems.append("driver_cuda_below_backend_cuda")
    if drvmax and toolkit and drvmax < toolkit:
        problems.append("driver_cuda_below_installed_toolkit")
    if toolkit is None:
        warnings.append("nvcc_toolkit_not_detected_backend_version_checked_separately")
    if latest:
        if drvmax and drvmax < latest:
            warnings.append("driver_max_cuda_older_than_latest_reviewed")
        if toolkit and toolkit < latest:
            warnings.append("installed_toolkit_older_than_latest_reviewed")
        if backend and backend < latest:
            warnings.append("pinned_backend_older_than_latest_reviewed_not_error")
    if drvmax:
        branch = policy["driver_compatibility"]["corresponding_driver_branch_by_cuda_minor"].get(
            driver_cuda_max, None)
        if branch is not None and driver_version and _DRV.fullmatch(driver_version):
            if int(driver_version.split(".")[0]) < branch:
                problems.append("driver_branch_too_old_for_reported_cuda")
        if driver_version and _DRV.fullmatch(driver_version):
            minbranch = policy["driver_compatibility"]["min_major_driver_branch"].get(str(drvmax[0]))
            if minbranch is not None and int(driver_version.split(".")[0]) < minbranch:
                problems.append("driver_branch_incompatible_with_cuda_major")
    return {
        "status": "hardware_eligible_native_backend_unverified" if not problems else "hardware_rejected_or_unverified",
        "architecture": arch["id"] if arch else None,
        "compute_capability": compute_capability,
        "tensor_core_generation": arch["tensor_core_generation"] if arch else None,
        "vram_class_gib": cls,
        "allowed_vram_classes_gib": arch["vram_classes_gib"] if arch else [],
        "min_cuda": arch["min_cuda"] if arch else None,
        "driver_cuda_max": driver_cuda_max,
        "driver_version": driver_version,
        "toolkit_cuda": toolkit_cuda,
        "backend_cuda": backend_cuda,
        "latest_reviewed_cuda": policy["latest_reviewed_cuda"],
        "transformer_engine_capability_policy": arch["tensor_engine"] if arch else None,
        "backend_kernel_architecture_verified": False,
        "errors": problems, "warnings": warnings,
    }


def inspect_host_hardware(
    *, project_root: Path = PROJECT_ROOT,
    run: Callable[..., Any] = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
    backend_cuda: str = "13.2",
) -> dict:
    """Read-only nvidia-smi and optional nvcc probe, never initializes CUDA."""
    policy = load_architecture_policy(project_root)
    result = run([
        "nvidia-smi", "--query-gpu=uuid,pci.bus_id,name,compute_cap,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, check=False, timeout=12)
    if result.returncode:
        raise ValueError("Unable to inspect NVIDIA architecture")
    lines = [x.strip() for x in result.stdout.splitlines() if x.strip()]
    if len(lines) != 1:
        raise ValueError("Exactly one NVIDIA GPU is supported")
    pieces = [x.strip() for x in lines[0].split(",")]
    if (len(pieces) != 6 or not _UUID.fullmatch(pieces[0])
            or not _PCI.fullmatch(pieces[1])
            or not pieces[4].isdecimal()):
        raise ValueError("NVIDIA architecture telemetry incomplete")
    summary = run(["nvidia-smi"], capture_output=True, text=True,
                  check=False, timeout=12)
    found = _CUDA_SMI.search(summary.stdout) if summary.returncode == 0 else None
    toolkit = None
    nvcc = which("nvcc")
    if nvcc:
        compiled = run([nvcc, "--version"], capture_output=True, text=True,
                       check=False, timeout=10)
        if compiled.returncode == 0:
            ver = parse_nvcc_version(compiled.stdout + "\n" + compiled.stderr)
            if ver:
                toolkit = f"{ver[0]}.{ver[1]}"
    checked = classify_hardware(
        compute_capability=pieces[3], total_mib=int(pieces[4]),
        driver_version=pieces[5], driver_cuda_max=found[1] if found else None,
        toolkit_cuda=toolkit, backend_cuda=backend_cuda, policy=policy,
    )
    return {**checked, "gpu_uuid": pieces[0], "pci_bus": pieces[1],
            "gpu_name": pieces[2], "total_mib": int(pieces[4]),
            "nvcc_path": nvcc, "telemetry_read_only": True}
