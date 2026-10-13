"""Parse llama.cpp's reported native buffer allocations from stderr logs.

Buffer allocations are not a complete VRAM accounting. In particular,
CUDA graph workspaces, allocator fragmentation and driver memory may not
appear in these lines. Always compare with independently sampled NVML usage.
"""

from collections import defaultdict
import re


_NATIVE_BUFFER_LINE = re.compile(
    r"(?P<buffer>\S+)\s+"
    r"(?P<kind>model|KV|compute|output)\s+buffer size\s*=\s*"
    r"(?P<mib>\d+(?:\.\d+)?)\s+MiB\b"
)


# llama.cpp may also print a K/V composition line alongside its
# allocation buffer lines. Keep it as independent evidence: the K/V payload
# totals can differ from allocator-rounded native buffer sizes.
_KV_COMPONENT_LINE = re.compile(
    r"\\bK\\s*\\((?P<type_k>f16|q8_0|q4_0)\\)\\s*:\\s*"
    r"(?P<k_mib>\\d+(?:\\.\\d+)?)\\s*MiB"
    r".*?\\bV\\s*\\((?P<type_v>f16|q8_0|q4_0)\\)\\s*:\\s*"
    r"(?P<v_mib>\\d+(?:\\.\\d+)?)\\s*MiB",
    re.I,
)


def buffer_domain(name: str) -> str:
    label = name.upper()
    if label == "CUDA_HOST":
        return "host_cuda_pinned"
    if label.startswith("CUDA") and "HOST" not in label:
        return "gpu_vram"
    if label.startswith("CPU"):
        return "host_cpu"
    return "other"


def parse_native_memory(native_stderr: str) -> dict:
    entries = []
    components = []
    totals = defaultdict(lambda: defaultdict(float))
    for line in native_stderr.splitlines():
        kv_match = _KV_COMPONENT_LINE.search(line)
        if kv_match:
            components.append({
                "type_k": kv_match.group("type_k").lower(),
                "type_v": kv_match.group("type_v").lower(),
                "k_mib": float(kv_match.group("k_mib")),
                "v_mib": float(kv_match.group("v_mib")),
            })
        match = _NATIVE_BUFFER_LINE.search(line)
        if match is None:
            continue
        name, category = match.group("buffer", "kind")
        mib = float(match.group("mib"))
        domain = buffer_domain(name)
        entries.append({
            "native_buffer": name,
            "category": category.lower(),
            "domain": domain,
            "reported_mib": mib,
        })
        totals[category.lower()][domain] += mib

    groups = {
        kind: {domain: round(mib, 2) for domain, mib in sorted(values.items())}
        for kind, values in sorted(totals.items())
    }
    return {
        "native_buffer_records": entries,
        "native_kv_component_records": components,
        "record_count": len(entries),
        "totals_mib_by_category_and_domain": groups,
        "gpu_model_buffer_mib": groups.get("model", {}).get("gpu_vram", 0.0),
        "host_model_buffer_mib": round(
            groups.get("model", {}).get("host_cpu", 0.0)
            + groups.get("model", {}).get("host_cuda_pinned", 0.0), 2
        ),
        "note": (
            "Numeri letti da output nativo llama.cpp. Non sono una misura "
            "di tutta la VRAM del processo: confrontare con i campioni NVML. "
            "Le categorie riportate dipendono dai log effettivamente disponibili."
        ),
    }
