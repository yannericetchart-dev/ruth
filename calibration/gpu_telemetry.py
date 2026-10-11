"""Optional NVML GPU telemetry during diagnostic benchmark phases.

Temperature, SM clock and board power are observational: missing device
support is reported as null, never treated as a zero or a benchmark failure.
"""

import math
import statistics


def gpu_snapshot(nvml, handle) -> dict:
    readings = {}
    probes = {
        "gpu_temperature_c": lambda: nvml.nvmlDeviceGetTemperature(
            handle, nvml.NVML_TEMPERATURE_GPU
        ),
        "sm_clock_mhz": lambda: nvml.nvmlDeviceGetClockInfo(
            handle, nvml.NVML_CLOCK_SM
        ),
        "power_watts": lambda: nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
    }
    for name, probe in probes.items():
        try:
            value = float(probe())
            readings[name] = round(value, 3) if math.isfinite(value) and value >= 0 else None
        except (AttributeError, nvml.NVMLError, OSError, NotImplementedError):
            readings[name] = None
    return readings


def summarize_gpu_telemetry(phase_samples: dict[str, list[dict]]) -> dict:
    result = {}
    for phase, samples in phase_samples.items():
        phase_result = {"sample_count": len(samples)}
        for key in ("gpu_temperature_c", "sm_clock_mhz", "power_watts"):
            values = [
                s[key] for s in samples
                if isinstance(s.get(key), (int, float))
                and math.isfinite(s[key])
            ]
            phase_result[key] = ({
                "sample_count": len(values),
                "minimum": min(values),
                "median": round(statistics.median(values), 3),
                "maximum": max(values),
            } if values else None)
        result[phase] = phase_result
    return result
