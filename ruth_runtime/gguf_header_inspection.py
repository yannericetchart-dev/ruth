"""Strict read-only GGUF metadata and tensor-descriptor inspector.

Reads only GGUF header, metadata and tensor descriptors, NEVER weight tensors.
Requires fully pinned and SHA256-verified local model+license manifest.
No CUDA, llama.cpp library load, downloads or tokenization.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import struct

from .dependencies import PROJECT_ROOT
from .model_catalog import select_model
from .inference_probe import _valid_local_gguf

MAX_HEADER = 256 * 1024 * 1024
MAX_KEYS = 10000
MAX_TENSORS = 100000
MAX_ARRAY_ELEMENTS = 2_000_000
MAX_NAME_BYTES = 65535
VALUES = {0:("B",1),1:("b",1),2:("H",2),3:("h",2),
          4:("I",4),5:("i",4),6:("f",4),7:("B",1),
          10:("Q",8),11:("q",8),12:("d",8)}
GGML_TYPES = {0:"F32",1:"F16",30:"BF16"}
ALLOWED = {
    "general.architecture", "general.file_type", "general.alignment",
    "gemma3.block_count", "gemma3.embedding_length", "gemma3.context_length",
    "gemma3.attention.head_count", "gemma3.attention.head_count_kv",
    "gemma3.attention.key_length", "gemma3.attention.value_length",
    "gemma3.attention.key_length_swa", "gemma3.attention.value_length_swa",
    "gemma3.attention.sliding_window", "gemma3.attention.sliding_window_pattern",
}

class GGUFInspectError(ValueError):
    pass


class _Reader:
    def __init__(self, file, size: int):
        self.f = file
        self.size = size
        self.endian = "<"

    def advance(self, n: int):
        if n < 0 or self.f.tell() + n > min(self.size, MAX_HEADER):
            raise GGUFInspectError("Unsafe or truncated GGUF metadata header")
        self.f.seek(n, 1)

    def take(self, n: int) -> bytes:
        if n < 0 or self.f.tell() + n > min(self.size, MAX_HEADER):
            raise GGUFInspectError("Unsafe or truncated GGUF header")
        data = self.f.read(n)
        if len(data) != n:
            raise GGUFInspectError("Truncated GGUF metadata")
        return data

    def number(self, fmt: str):
        return struct.unpack(self.endian + fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self, capture: bool, max_chars: int = MAX_NAME_BYTES):
        n = self.number("Q")
        if n > max_chars:
            raise GGUFInspectError("GGUF key/name exceeds permitted length")
        if capture:
            return self.take(n).decode("utf-8", errors="strict")
        self.advance(n)
        return None

    def value(self, kind: int, capture: bool, depth: int = 0):
        if depth > 2:
            raise GGUFInspectError("Nested GGUF metadata exceeds supported depth")
        if kind in VALUES:
            fmt,size = VALUES[kind]
            if capture:
                return self.number(fmt)
            self.advance(size)
            return None
        if kind == 8:
            # Names of non-interesting tokens need not be retained.
            return self.string(capture, max_chars=4*1024*1024)
        if kind == 9:
            element = self.number("I")
            count = self.number("Q")
            if count > MAX_ARRAY_ELEMENTS:
                raise GGUFInspectError("Oversized GGUF metadata array")
            selected = capture and count <= 256
            if element in VALUES:
                fmt,n = VALUES[element]
                if selected:
                    return [self.number(fmt) for _ in range(count)]
                self.advance(count*n)
                return None
            if element == 8:
                # Tokenizer list is large; seek tokens rather than load them.
                if selected:
                    return [self.string(True, max_chars=4*1024*1024)
                            for _ in range(count)]
                for _ in range(count):
                    self.string(False, max_chars=4*1024*1024)
                return None
            if element == 9:
                # Nested arrays supported by bounded recursion.
                if selected:
                    return [self.value(element, True, depth+1)
                            for _ in range(count)]
                for _ in range(count):
                    self.value(element, False, depth+1)
                return None
        raise GGUFInspectError("Unsupported GGUF metadata value type")


def inspect_gguf_header(file: Path) -> dict:
    path = Path(file)
    if path.is_symlink() or not path.is_file():
        raise GGUFInspectError("Missing or unsafe GGUF artifact")
    with path.open("rb") as fh:
        r = _Reader(fh, path.stat().st_size)
        if r.take(4) != b"GGUF":
            raise GGUFInspectError("Invalid GGUF magic")
        raw_version = r.take(4)
        # The GGUF version field also encodes byte order in version 3.
        little = struct.unpack("<I", raw_version)[0]
        big = struct.unpack(">I", raw_version)[0]
        if little in (2,3):
            r.endian = "<"
            version = little
        elif big == 3:
            r.endian = ">"
            version = big
        else:
            raise GGUFInspectError("Unsupported GGUF version")
        n_tensors = r.number("Q")
        n_keys = r.number("Q")
        if not 1 <= n_tensors <= MAX_TENSORS or not 1 <= n_keys <= MAX_KEYS:
            raise GGUFInspectError("Invalid GGUF descriptor counts")
        meta = {}
        for _ in range(n_keys):
            key = r.string(True)
            if key in meta:
                raise GGUFInspectError("Duplicate GGUF metadata key")
            kind = r.number("I")
            is_known = key in ALLOWED
            value = r.value(kind, is_known)
            if is_known:
                meta[key] = value
        storage = Counter()
        for _ in range(n_tensors):
            r.string(False, max_chars=128)
            ndim = r.number("I")
            if not 1 <= ndim <= 4:
                raise GGUFInspectError("Unsupported GGUF tensor dimensions")
            for _ in range(ndim):
                if r.number("Q") < 1:
                    raise GGUFInspectError("Zero GGUF tensor dimension")
            dtype = r.number("I")
            r.number("Q")  # Weight offset: never dereferenced
            storage[GGML_TYPES.get(dtype, "GGML_TYPE_"+str(dtype))] += 1
        if meta.get("general.architecture") != "gemma3":
            raise GGUFInspectError("This inspection requires Gemma 3 architecture")
        if meta.get("general.file_type") != 24:
            raise GGUFInspectError("Expected GGML_FTYPE_MOSTLY_BF16 (24)")
        n_layers = meta.get("gemma3.block_count")
        if type(n_layers) is not int or not 1 <= n_layers <= 256:
            raise GGUFInspectError("Gemma 3 layer count not established")
        if storage["BF16"] == 0:
            raise GGUFInspectError("No BF16 tensor descriptor found")
        # No false claim about model-wide geometry: global/SWA split is
        # interpreted by the native backend after this first inspection.
        return {
            "status": "gguf_header_geometry_observed_native_crosscheck_pending",
            "gguf_version": version, "tensor_count": n_tensors,
            "metadata_fields": meta,
            "tensor_storage_counts": dict(sorted(storage.items())),
            "architecture": "gemma3", "blocks": n_layers,
            "weight_format": "BF16",
            "tensor_values_read": False,
            "cuda_initialized": False,
            "model_weights_loaded": False,
            "context_initialized": False,
            "inference_executed": False,
        }


def inspect_pinned_gguf(*, purpose: str, family: str, model: str, version: str,
                        quantization: str, project_root: Path = PROJECT_ROOT) -> dict:
    if (family, model, version, quantization) != (
        "gemma", "gemma-3", "gemma-3-1b-it", "BF16"):
        raise GGUFInspectError("Unsupported new-model metadata inspection target")
    choice = select_model(purpose, family=family, model=model, version=version,
                          format="gguf", quantization=quantization,
                          project_root=project_root)
    verified, path, reason = _valid_local_gguf(Path(project_root), choice)
    if not verified:
        raise GGUFInspectError("Pinned GGUF/manifest integrity not verified: "+str(reason))
    result = inspect_gguf_header(path)
    return {
        **result,
        "model_id": {"family":family,"model":model,"version":version},
        "gguf_sha256":choice.expected_sha256,
        "source_revision":choice.revision,
        "source_repository":choice.repo_id,
        "gguf_file_bytes":choice.expected_size_bytes,
        "source_integrity_verified": True,
        "local_only":True,
    }
