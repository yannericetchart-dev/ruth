"""Offline pinned Gemma3 GGUF tensor dtype inventory; never loads a model.

Parses bounded GGUF v3 metadata and tensor descriptors. Reads model data only
as a streamed SHA256 integrity check, never maps tensor payloads or runs CUDA.
Version/types are pinned to llama.cpp 0c1e5709.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import struct

from .dependencies import PROJECT_ROOT
from .model_catalog import select_model
from .model_download import _local_path

PINNED_REVISION = "0c1e57098bba43ac29e6e3b677cdceebdd22334f"
MAX_HEADER_BYTES = 512 * 1024 * 1024
MAX_KV_ITEMS = 100_000
MAX_TENSORS = 50_000
MAX_ARRAY_ELEMENTS = 10_000_000
MAX_STRING_BYTES = 4 * 1024 * 1024
GGML_TYPES = {
    0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",
    8:"Q8_0",9:"Q8_1",10:"Q2_K",11:"Q3_K",12:"Q4_K",
    13:"Q5_K",14:"Q6_K",15:"Q8_K",16:"IQ2_XXS",17:"IQ2_XS",
    18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",21:"IQ3_S",
    22:"IQ2_S",23:"IQ4_XS",24:"I8",25:"I16",26:"I32",
    27:"I64",28:"F64",29:"IQ1_M",30:"BF16",34:"TQ1_0",
    35:"TQ2_0",39:"MXFP4",40:"NVFP4",41:"Q1_0",42:"Q2_0",
}
ELEMENT_SIZES = {
    "F32":4,"F16":2,"BF16":2,"F64":8,
    "I8":1,"I16":2,"I32":4,"I64":8,
}
META_SCALAR_SIZE = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}


class GGUFInspectionError(ValueError):
    pass


class _Reader:
    def __init__(self, handle, size):
        self.handle = handle
        self.limit = min(size, MAX_HEADER_BYTES)
        self.size = size

    def raw(self, n: int) -> bytes:
        pos = self.handle.tell()
        if n < 0 or pos+n > self.limit:
            raise GGUFInspectionError("GGUF header exceeds bounded parser limits")
        data = self.handle.read(n)
        if len(data) != n:
            raise GGUFInspectionError("Truncated GGUF header")
        return data

    def skip(self, n: int) -> None:
        pos = self.handle.tell()
        if n < 0 or pos+n > self.limit:
            raise GGUFInspectionError("GGUF metadata skip exceeds file/header")
        self.handle.seek(n,1)

    def number(self, code: str):
        return struct.unpack("<"+code, self.raw(struct.calcsize(code)))[0]

    def string(self, *, decode: bool=True):
        n = self.number("Q")
        if n > MAX_STRING_BYTES:
            raise GGUFInspectionError("Oversized GGUF metadata string")
        if not decode:
            self.skip(n)
            return None
        try:
            return self.raw(n).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFInspectionError("Invalid GGUF UTF-8 metadata") from exc

    def metadata_value(self, typ: int, *, retain: bool=False):
        # GGUF arrays encode one element type (uint32) then length (uint64).
        # Nested arrays are not part of the pinned supported encoding.
        if typ in META_SCALAR_SIZE:
            if not retain:
                self.skip(META_SCALAR_SIZE[typ])
                return None
            codes={0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",
                   6:"f",7:"B",10:"Q",11:"q",12:"d"}
            return self.number(codes[typ])
        if typ == 8:
            return self.string(decode=retain)
        if typ == 9:
            item_typ = self.number("I")
            count = self.number("Q")
            if count > MAX_ARRAY_ELEMENTS:
                raise GGUFInspectionError("Excessive GGUF metadata array count")
            if item_typ in META_SCALAR_SIZE:
                self.skip(count*META_SCALAR_SIZE[item_typ])
            elif item_typ == 8:
                for _ in range(count):
                    self.string(decode=False)
            else:
                raise GGUFInspectionError("Unsupported nested/unknown GGUF array type")
            return None
        raise GGUFInspectionError("Unknown GGUF metadata type "+str(typ))


def classify_tensor(name: str) -> str:
    n=name.lower()
    if "attn_" in n and ("norm" in n or "q_norm" in n or "k_norm" in n):
        return "attention_norm"
    if any(x in n for x in ("attn_q", "attn_k", "attn_v",
                             "attn_output", "attn_qkv")):
        return "attention_projection"
    if "attn_" in n:
        return "attention_other"
    if "ffn_" in n or "mlp" in n:
        return "feed_forward"
    if "norm" in n:
        return "normalization"
    if "token_embd" in n or "token_embedding" in n:
        return "embedding"
    if "output" in n:
        return "output"
    return "other"


def parse_gguf_header(path: Path, *, expected_size: int | None=None) -> dict:
    """Bounded read of header, metadata and tensor descriptors only."""
    path=Path(path)
    if path.is_symlink() or not path.is_file():
        raise GGUFInspectionError("Missing or symlinked GGUF")
    size=path.stat().st_size
    if expected_size is not None and size != expected_size:
        raise GGUFInspectionError("Pinned GGUF file size mismatch")
    if size < 24:
        raise GGUFInspectionError("Truncated GGUF file")
    with path.open("rb") as f:
        r=_Reader(f,size)
        if r.raw(4) != b"GGUF":
            raise GGUFInspectionError("Not a GGUF file")
        version=r.number("I")
        if version != 3:
            raise GGUFInspectionError("Expected pinned GGUF v3")
        tensor_count=r.number("Q")
        kv_count=r.number("Q")
        if tensor_count > MAX_TENSORS or kv_count > MAX_KV_ITEMS:
            raise GGUFInspectionError("GGUF descriptor count outside safe limits")
        alignment=32
        file_type=None
        seen_keys=set()
        for _ in range(kv_count):
            key=r.string()
            if not key or key in seen_keys:
                raise GGUFInspectionError("Duplicate or empty GGUF metadata key")
            seen_keys.add(key)
            typ=r.number("I")
            value=r.metadata_value(
                typ, retain=key in ("general.alignment","general.file_type"))
            if key == "general.alignment":
                if typ not in (4,10) or type(value) is not int:
                    raise GGUFInspectionError("Invalid GGUF alignment type")
                alignment=value
            elif key == "general.file_type":
                file_type=value
        if (alignment < 8 or alignment > 4096
            or alignment & (alignment-1)):
            raise GGUFInspectionError("Invalid GGUF alignment")
        tensors=[]
        seen_names=set()
        for _ in range(tensor_count):
            name=r.string()
            if not name or name in seen_names:
                raise GGUFInspectionError("Duplicate/empty tensor name")
            seen_names.add(name)
            ndim=r.number("I")
            if ndim < 1 or ndim > 4:
                raise GGUFInspectionError("Invalid GGUF tensor rank")
            dims=[r.number("Q") for _ in range(ndim)]
            if not all(0 < x <= 10_000_000 for x in dims):
                raise GGUFInspectionError("Invalid GGUF tensor dimensions")
            dtype_id=r.number("I")
            offset=r.number("Q")
            if offset % alignment:
                raise GGUFInspectionError("Unaligned GGUF tensor offset")
            elements=1
            for dim in dims:
                elements *= dim
                if elements > 1_000_000_000_000:
                    raise GGUFInspectionError("GGUF tensor shape too large")
            dtype=GGML_TYPES.get(dtype_id,f"UNKNOWN_{dtype_id}")
            tensors.append({
                "name":name,"dtype":dtype,"dtype_id":dtype_id,
                "dims_gguf_order":dims,"elements":elements,
                "relative_data_offset":offset,
                "uncompressed_data_bytes": (
                    elements*ELEMENT_SIZES[dtype]
                    if dtype in ELEMENT_SIZES else None),
                "category":classify_tensor(name),
            })
        data_offset=(f.tell()+alignment-1)//alignment*alignment
        if data_offset > size:
            raise GGUFInspectionError("Tensor data starts past end of file")
        for t in tensors:
            start=data_offset+t["relative_data_offset"]
            n=t["uncompressed_data_bytes"]
            if start >= size or n is not None and start+n > size:
                raise GGUFInspectionError("Tensor data extends past end of file")
        return {
            "gguf_version":version,"metadata_kv_count":kv_count,
            "tensor_count":tensor_count,"general_file_type":file_type,
            "alignment":alignment,"tensor_data_offset":data_offset,
            "file_size_bytes":size,"tensors":tensors,
        }


def inspect_pinned_gguf(
    *, root: Path=PROJECT_ROOT, include_tensors: bool=False,
) -> dict:
    root=Path(root).resolve(strict=True)
    choice=select_model(
        "production",family="gemma",model="gemma-3",
        version="gemma-3-1b-it",format="gguf",quantization="BF16",
        project_root=root)
    path=_local_path(root,choice)
    result={
        "status":"precision_inventory_unverified",
        "source_revision":PINNED_REVISION,
        "model_sha256_expected":choice.expected_sha256,
        "model_sha256_observed":None,
        "production_qualified":False,
        "gpu_inference_executed":False,
    }
    parsed=parse_gguf_header(
        path,expected_size=choice.expected_size_bytes)
    digest=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):
            digest.update(chunk)
    sha=digest.hexdigest()
    result["model_sha256_observed"]=sha
    if sha != choice.expected_sha256:
        return result
    tensors=parsed.pop("tensors")
    by_type=Counter(t["dtype"] for t in tensors)
    by_type_elements=Counter()
    by_type_bytes=Counter()
    by_category=defaultdict(Counter)
    by_category_elements=defaultdict(Counter)
    all_tensor_spans=[]
    all_known_bytes=True
    for t in tensors:
        by_category[t["category"]][t["dtype"]]+=1
        by_category_elements[t["category"]][t["dtype"]]+=t["elements"]
        by_type_elements[t["dtype"]]+=t["elements"]
        byte_count=t["uncompressed_data_bytes"]
        if byte_count is None:
            all_known_bytes=False
        else:
            by_type_bytes[t["dtype"]]+=byte_count
            all_tensor_spans.append(
                (t["relative_data_offset"],
                 t["relative_data_offset"]+byte_count))
    spans=sorted(all_tensor_spans)
    cursor=0
    contiguous=bool(all_known_bytes and spans)
    for start,end in spans:
        if start != cursor or end <= start:
            contiguous=False
        cursor=end
    data_length=parsed["file_size_bytes"]-parsed["tensor_data_offset"]
    storage_verified=bool(contiguous and cursor==data_length)
    result.update(parsed)
    result.update({
        "status":"pinned_gguf_precision_inventory_verified",
        "model_file":str(path.relative_to(root)),
        "tensor_dtype_counts":dict(sorted(by_type.items())),
        "tensor_dtype_element_counts":dict(sorted(by_type_elements.items())),
        "tensor_dtype_storage_bytes":dict(sorted(by_type_bytes.items())),
        "gguf_tensor_data_region_bytes":data_length,
        "tensor_storage_bytes_accounted":sum(by_type_bytes.values()),
        "tensor_storage_layout_contiguous_verified":storage_verified,
        "tensor_storage_accounting_complete":all_known_bytes,
        "tensor_categories":{
            k:{"dtype_tensor_counts":dict(sorted(v.items())),
               "dtype_element_counts":dict(sorted(by_category_elements[k].items()))}
            for k,v in sorted(by_category.items())
        },
        "non_bf16_tensor_sample":[
            {"name":t["name"],"dtype":t["dtype"],"category":t["category"]}
            for t in tensors if t["dtype"]!="BF16"][:60],
    })
    if include_tensors:
        result["tensors"]=tensors
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description="Offline pinned BF16 GGUF tensor inventory")
    p.add_argument("--all-tensors",action="store_true",
                   help="Include all tensors with dims/type/offset")
    a=p.parse_args(argv)
    try:
        result=inspect_pinned_gguf(include_tensors=a.all_tensors)
    except (OSError,ValueError,OverflowError) as exc:
        result={"status":"precision_inventory_unverified",
                "error":type(exc).__name__+": "+str(exc)[:200],
                "production_qualified":False,"gpu_inference_executed":False}
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0 if result["status"]=="pinned_gguf_precision_inventory_verified" else 2


if __name__=="__main__":
    raise SystemExit(main())
