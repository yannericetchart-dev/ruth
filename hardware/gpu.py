import csv
import subprocess
from pathlib import Path


class GPUManager:
    """Rileva i dispositivi GPU. Le metriche VRAM sono implementate per NVIDIA."""

    VENDORS = {
        "0x10de": "nvidia",
        "0x1002": "amd",
        "0x8086": "intel",
    }

    @staticmethod
    def detect() -> list[dict]:
        gpus = []
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            for row in csv.reader(result.stdout.splitlines()):
                if len(row) != 5:
                    continue
                index, uuid, name, total, free = (item.strip() for item in row)
                gpus.append({
                    "vendor": "nvidia",
                    "name": name,
                    "index": int(index),
                    "uuid": uuid,
                    "vram_total_mb": int(total),
                    "vram_free_mb": int(free),
                })
        except (FileNotFoundError, subprocess.SubprocessError, ValueError):
            pass

        # Identificazione degli altri produttori (implementazione backend futura).
        # Anche NVIDIA viene segnalata se il driver/nvidia-smi non è disponibile.
        nvidia_found = any(gpu["vendor"] == "nvidia" for gpu in gpus)
        for vendor_file in sorted(Path("/sys/class/drm").glob("card[0-9]*/device/vendor")):
            try:
                vendor_id = vendor_file.read_text().strip().lower()
                vendor = GPUManager.VENDORS.get(vendor_id, "other")
                if vendor == "nvidia" and nvidia_found:
                    continue
                device_id = str(vendor_file.parent.resolve())
                if any(gpu.get("device_id") == device_id for gpu in gpus):
                    continue
                gpus.append({
                    "vendor": vendor,
                    "name": vendor.upper() + " GPU",
                    "device_id": device_id,
                    "index": None,
                    "uuid": None,
                    "vram_total_mb": None,
                    "vram_free_mb": None,
                })
            except OSError:
                continue

        return gpus

    @staticmethod
    def select(vendor: str = "nvidia") -> dict | None:
        candidates = [
            gpu for gpu in GPUManager.detect()
            if gpu["vendor"] == vendor and gpu["vram_free_mb"] is not None
        ]
        return max(candidates, key=lambda gpu: gpu["vram_free_mb"], default=None)

    @staticmethod
    def calibration_status(allowed_pids=None) -> dict:
        """Stato GPU NVIDIA per la calibrazione, senza cambiare il runtime.

        Per impostazione predefinita solo il processo corrente appartiene a Ruth.
        Il chiamante può autorizzare altri PID espliciti (es. worker di Ruth).
        I processi grafici non vengono esclusi automaticamente.
        """
        import os

        report = {
            "ready": False,
            "gpu": GPUManager.select("nvidia"),
            "memory": None,
            "processes": [],
            "foreign_processes": [],
            "errors": [],
        }
        gpu = report["gpu"]
        if gpu is None:
            report["errors"].append("Nessuna GPU NVIDIA accessibile tramite nvidia-smi.")
            return report

        try:
            import pynvml as nvml  # Fornito da nvidia-ml-py nel venv.
        except ImportError:
            report["errors"].append("NVML Python non disponibile: riesegui ./setup_env.sh.")
            return report

        owned = {os.getpid()} | {int(pid) for pid in (allowed_pids or ())}
        mib = 1024 * 1024

        try:
            nvml.nvmlInit()
        except nvml.NVMLError as exc:
            report["errors"].append(f"Inizializzazione NVML fallita: {exc}")
            return report

        try:
            handle = nvml.nvmlDeviceGetHandleByUUID(gpu["uuid"])
            info = nvml.nvmlDeviceGetMemoryInfo(handle)

            # Compute e grafica devono essere interrogati entrambi:
            # non consideriamo affidabile una lista parziale dei processi.
            found = {}
            for kind, query in (
                ("compute", nvml.nvmlDeviceGetComputeRunningProcesses),
                ("graphics", nvml.nvmlDeviceGetGraphicsRunningProcesses),
            ):
                try:
                    processes = query(handle)
                except nvml.NVMLError as exc:
                    report["errors"].append(f"Elenco processi {kind} non disponibile: {exc}")
                    continue

                for proc in processes:
                    pid = int(proc.pid)
                    try:
                        used_bytes = int(proc.usedGpuMemory)
                    except (TypeError, ValueError):
                        used_bytes = -1
                    # NVML_VALUE_NOT_AVAILABLE può essere un UINT64_MAX.
                    used_mib = (
                        None if used_bytes < 0 or used_bytes >= (1 << 63)
                        else round(used_bytes / mib, 1)
                    )
                    item = found.setdefault(pid, {
                        "pid": pid,
                        "name": None,
                        "types": [],
                        "vram_mib": None,
                        "owner": "ruth" if pid in owned else "external",
                    })
                    item["types"].append(kind)
                    # Un processo può comparire in entrambe le categorie.
                    # Non sommare due volte la sua memoria.
                    if used_mib is not None:
                        item["vram_mib"] = max(item["vram_mib"] or 0, used_mib)

            for pid, item in sorted(found.items()):
                try:
                    item["name"] = Path(f"/proc/{pid}/comm").read_text().strip()
                except OSError:
                    item["name"] = f"pid:{pid}"
                report["processes"].append(item)
                if item["owner"] == "external":
                    report["foreign_processes"].append(item)

            used_by_processes = sum(
                p["vram_mib"] or 0 for p in report["processes"]
            )
            used_mib = round(info.used / mib, 1)
            report["memory"] = {
                "total_mib": round(info.total / mib, 1),
                "free_mib": round(info.free / mib, 1),
                "used_mib": used_mib,
                "reported_process_mib": round(used_by_processes, 1),
                # Stima: memoria di driver/display, non attribuita o non
                # visibile per PID; le metriche NVML non sono sempre additive.
                "unattributed_mib_estimate": round(max(0, used_mib - used_by_processes), 1),
            }
            report["ready"] = not report["errors"] and not report["foreign_processes"]

        except (nvml.NVMLError, OSError, ValueError) as exc:
            report["errors"].append(f"Ispezione GPU incompleta: {exc}")
        finally:
            nvml.nvmlShutdown()

        return report
