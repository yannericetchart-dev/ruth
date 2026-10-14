"""Recognize *worker infrastructure* failures, never confuse them with CUDA OOM.

An interpreter being unable to import a calibration module is not evidence
about a GPU tensor mapping. The marker is intentionally specific: other
worker exceptions, generic SIGABRT, and genuine native CUDA failures are
not silently retried by the checkpoint repair logic.
"""
import re


_MODULE_IMPORT_ERROR = re.compile(
    r"No module named .?calibration(?:[.][A-Za-z_][A-Za-z_0-9]*)+"
    r"|Error while finding module specification for .?calibration[.]",
    re.IGNORECASE,
)


def _stderr_from_failure(row: dict) -> str:
    if not isinstance(row, dict):
        return ""
    details = row.get("failure_diagnostics") or {}
    return str(details.get("stderr_tail") or "")


def worker_import_failure(row: dict) -> bool:
    """Identify a broken Python worker import, including aggregate samples."""
    if not isinstance(row, dict) or row.get("success") is not False:
        return False
    if _MODULE_IMPORT_ERROR.search(_stderr_from_failure(row)):
        return True
    samples = row.get("samples") or []
    return any(
        isinstance(sample, dict)
        and _MODULE_IMPORT_ERROR.search(_stderr_from_failure(sample))
        for sample in samples
    )


def recover_worker_import_failures(entries: dict) -> dict:
    """Archive, then remove only import-corrupted outcomes for safe retry.

    Inference and load-only histories are retained for auditability.
    Genuine CUDA OOM / verification failures and successful probes remain
    untouched. Repeated runs are idempotent.
    """
    recovered_loads = []
    recovered_inferences = []
    for tensor, entry in entries.items():
        load = entry.get("load_only")
        infer = entry.get("inference")
        if worker_import_failure(infer):
            entry.setdefault("inference_history", []).append(
                entry.pop("inference")
            )
            recovered_inferences.append(tensor)
        if worker_import_failure(load):
            if "inference" in entry:
                raise ValueError(
                    "Checkpoint contraddittorio: inference on failed load: "
                    + tensor
                )
            entry.setdefault("load_only_history", []).append(
                entry.pop("load_only")
            )
            recovered_loads.append(tensor)
    return {
        "recovered_load_only_count": len(recovered_loads),
        "recovered_inference_count": len(recovered_inferences),
        "recovered_load_only_tensors": recovered_loads,
        "recovered_inference_tensors": recovered_inferences,
    }
