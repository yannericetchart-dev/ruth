"""Conservative *empirical* CUDA-futility observation for singleton sweeps.

No CUDA OOM or SIGABRT by itself can prove that a different tensor will
not succeed. This optional, resumable STOP is a time-saving heuristic,
NEVER a certified exclusion nor a completed census.

Evidence comes from same-target native failures. The f16/256 savings
are only used to locate a large priority-band boundary and never as
an upper bound on inferencing at the new context.
"""


def _verified_cuda_memory_failure(entry: dict) -> bool:
    """True only for explicit native CUDA resource/OOM failure evidence."""
    if not isinstance(entry, dict):
        return False
    load = entry.get("load_only") or {}
    if load.get("success") is False:
        # A failed load can establish that *this* mapping does not work,
        # but not that any other candidate cannot.
        if load.get("failure_class") not in (
            "native_cuda_oom", "native_cuda_resource_allocation_failure"
        ):
            return False
        return bool(load.get("native_evidence_available")
                    or load.get("suspected_cuda_oom")
                    or load.get("native_cuda_resource_allocation_failed"))
    if load.get("success") is not True:
        return False
    infer = entry.get("inference") or {}
    if infer.get("success") is not False or infer.get("failure_class") not in (
        "native_cuda_oom", "native_cuda_resource_allocation_failure"
    ):
        return False
    for sample in infer.get("samples") or []:
        if sample.get("success") is True:
            return False
        if sample.get("failure_class") not in (
            "native_cuda_oom", "native_cuda_resource_allocation_failure"
        ):
            return False
        details = sample.get("failure_diagnostics") or {}
        # The old worker already emits an exact CUDA allocation failure
        # class. A generic SIGABRT / timeout / arbitrary exception is NOT
        # accepted. The new worker provides stronger native evidence.
        if not (
            details.get("native_cuda_resource_allocation_failed")
            or details.get("suspected_cuda_oom")
            or sample.get("worker_signal") == "SIGABRT"
               and sample.get("failure_class") ==
               "native_cuda_resource_allocation_failure"
            or sample.get("failure_class") == "native_cuda_oom"
        ):
            return False
    return bool(infer.get("samples"))


def observed_cuda_futility(
    ranked: list[dict], entries: dict,
    *, witness_count: int = 12,
    min_gain_drop_mib: float = 12.0,
) -> dict | None:
    """Return an explicitly NONCERTIFIED early-stop observation.

    Require all:
      - 256 post-load savings are ordered descending;
      - one large measured saving-band gap in that ranking;
      - every higher-priority candidate has target-context results;
      - >= witness_count consecutive CUDA memory failures *after* gap;
      - no success in the lower-saving tier sampled so far.

    This does NOT prove impossibility for other tensor families, and may
    stop before all workable tensors are discovered.
    """
    if not isinstance(ranked, list) or not ranked or witness_count < 2:
        return None
    gains = [row.get("measured_load_gain_256_mib") for row in ranked]
    if (any(type(value) not in (float, int) for value in gains)
        or any(left < right for left, right in zip(gains, gains[1:]))):
        return None
    gaps = [
        (gains[i-1]-gains[i], i)
        for i in range(1, len(gains))
        if gains[i-1] > gains[i]
    ]
    if not gaps:
        return None
    gap, boundary = max(gaps)
    if gap < min_gain_drop_mib:
        return None
    # Only consecutive completed candidates can be used as evidence.
    prefix = 0
    for row in ranked:
        entry = entries.get(row["tensor"]) or {}
        load = entry.get("load_only")
        if not isinstance(load, dict):
            break
        if load.get("success") is True:
            if not isinstance(entry.get("inference"), dict):
                break
        prefix += 1
    if prefix < boundary + witness_count or prefix >= len(ranked):
        return None
    tested_lower = ranked[boundary:prefix]
    if any(
        (entries.get(row["tensor"]) or {}).get("inference", {})
        .get("success") is True
        for row in tested_lower
    ):
        return None
    last = tested_lower[-witness_count:]
    if not all(
        _verified_cuda_memory_failure(entries.get(row["tensor"]))
        for row in last
    ):
        return None
    return {
        "stopped": True,
        "certified_exclusions": 0,
        "classification": "non_certified_observational_cuda_futility",
        "target_evidence_count": witness_count,
        "completed_rank_prefix": prefix,
        "next_rank": prefix + 1,
        "remaining_untested": len(ranked) - prefix,
        "priority_gain_boundary_rank": boundary,
        "priority_gain_drop_256_mib": gap,
        "next_candidate_gain_256_mib": gains[prefix],
        "next_candidate_tensor": ranked[prefix]["tensor"],
        "reason": (
            "Interruzione EURISTICA dopo errori CUDA nativi ripetuti "
            "oltre la principale discontinuita del ranking 256. "
            "NON e una prova che i rimanenti tensori non funzionino; "
            "non promuove un report parziale a census completo. "
            "Rilanciare senza --stop-on-cuda-futility per proseguire."
        ),
    }
