"""Synthetic offline GGUF v3 tensor dtype parser and pinned hash guards."""
from __future__ import annotations

import hashlib
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruth_runtime.gemma3_phase28_precision_inventory import (
    GGUFInspectionError, inspect_pinned_gguf, parse_gguf_header)


def s(value: str) -> bytes:
    data=value.encode("utf-8")
    return struct.pack("<Q",len(data))+data


def make_gguf(file: Path, *, unknown_dtype: bool=False) -> None:
    header=bytearray(b"GGUF"+struct.pack("<IQQ",3,2,3))
    header+=s("general.alignment")+struct.pack("<II",4,32)
    header+=s("general.file_type")+struct.pack("<II",4,32)
    header+=s("tokenizer.ggml.tokens")
    header+=struct.pack("<IIQ",9,8,2)+s("token")+s("test")

    header+=s("blk.0.attn_q.weight")
    header+=struct.pack("<IQQIQ",2,2,3,99 if unknown_dtype else 30,0)
    header+=s("blk.0.attn_norm.weight")
    header+=struct.pack("<IQIQ",1,3,0,32)
    # tensor 2 dtype_id overwritten below: rank + 1 dimension + dtype+ offset
    # correct layout is <I Q I Q>. dtype F32=0, offset=32.
    pad=(-len(header))%32
    file.write_bytes(header+b"\0"*pad+b"\0"*64)


class TestPhase28PrecisionInventory(unittest.TestCase):
    def test_header_dtype_categories_and_array_metadata(self):
        with TemporaryDirectory() as d:
            fp=Path(d)/"tiny.gguf"
            make_gguf(fp)
            parsed=parse_gguf_header(fp,expected_size=fp.stat().st_size)
            self.assertEqual(parsed["tensor_count"],2)
            self.assertEqual(parsed["metadata_kv_count"],3)
            self.assertEqual(parsed["alignment"],32)
            self.assertEqual([t["dtype"] for t in parsed["tensors"]],
                             ["BF16","F32"])
            self.assertEqual(parsed["tensors"][0]["elements"],6)
            self.assertEqual(parsed["tensors"][0]["uncompressed_data_bytes"],12)
            self.assertEqual(parsed["tensors"][0]["category"],
                             "attention_projection")
            self.assertEqual(parsed["tensors"][1]["category"],
                             "attention_norm")
            self.assertEqual(parsed["tensors"][1]["relative_data_offset"],32)

    def test_catalog_hash_must_match_and_unknown_dtype_not_renamed_bf16(self):
        with TemporaryDirectory() as d:
            root=Path(d)
            fp=root/"tiny.gguf"
            make_gguf(fp,unknown_dtype=True)
            sha=hashlib.sha256(fp.read_bytes()).hexdigest()
            choice=SimpleNamespace(
                expected_sha256=sha, expected_size_bytes=fp.stat().st_size)
            with patch(
                "ruth_runtime.gemma3_phase28_precision_inventory.select_model",
                return_value=choice
            ), patch(
                "ruth_runtime.gemma3_phase28_precision_inventory._local_path",
                return_value=fp
            ):
                report=inspect_pinned_gguf(root=root,include_tensors=True)
                self.assertEqual(report["status"],
                                 "pinned_gguf_precision_inventory_verified")
                self.assertEqual(report["tensor_dtype_counts"],
                                 {"F32":1,"UNKNOWN_99":1})
                self.assertFalse(report["production_qualified"])
                self.assertFalse(report["gpu_inference_executed"])
                choice.expected_sha256="0"*64
                failed=inspect_pinned_gguf(root=root)
                self.assertEqual(failed["status"],
                                 "precision_inventory_unverified")

    def test_truncated_bad_version_and_out_of_bound_metadata_rejected(self):
        with TemporaryDirectory() as d:
            fp=Path(d)/"tiny.gguf"
            make_gguf(fp)
            data=fp.read_bytes()
            fp.write_bytes(data[:90])
            with self.assertRaises(GGUFInspectionError):
                parse_gguf_header(fp)
            fp.write_bytes(data[:4]+struct.pack("<I",9)+data[8:])
            with self.assertRaisesRegex(GGUFInspectionError,"v3"):
                parse_gguf_header(fp)


if __name__=="__main__":
    unittest.main()
