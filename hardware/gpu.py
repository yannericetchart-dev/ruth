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
