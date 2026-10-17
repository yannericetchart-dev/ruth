"""Synthetic bounded GGUF block size checks without decoding tensor bytes."""
from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import struct
import unittest

from models.formats.gguf import inspect_block_layout, inspect_all_block_layouts
from models.formats.gguf.runtime.tensor_index import inspect_tensor_index
from models.formats.gguf.tests.test_index import info, image, kv

FORMATS = ((2, 18), (3, 20), (6, 22), (7, 24), (8, 34))


class BlockLayoutTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.folder = self.root / ".local" / "models"
        self.folder.mkdir(parents=True)

    def declare(self, records=None):
        if records is None:
            records = [{"id": "sample", "format": "gguf", "path": "demo.gguf"}]
        (self.folder / "catalog.json").write_text(
            json.dumps({"models": records}), encoding="utf-8"
        )

    def write(self, contents):
        (self.folder / "demo.gguf").write_bytes(contents)

    def one(self):
        return inspect_block_layout("sample", project_root=self.root)

    def test_not_configured_empty_or_no_gguf(self):
        self.assertEqual(inspect_all_block_layouts(project_root=self.root).status, "not_configured")
        self.declare([])
        self.assertEqual(inspect_all_block_layouts(project_root=self.root).status, "empty")
        self.declare([{"id": "other", "format": "safetensors", "path": "x.bin"}])
        self.assertEqual(inspect_all_block_layouts(project_root=self.root).status, "no_gguf")
        self.assertEqual(inspect_block_layout("other", project_root=self.root).status, "not_gguf")

    def test_five_supported_types_and_legacy_index_unchanged(self):
        self.declare()
        for kind, block_bytes in FORMATS:
            for version, endian in ((2, "<"), (3, "<"), (3, ">")):
                with self.subTest(kind=kind, version=version, endian=endian):
                    self.write(image([info("a", dims=(32,), kind=kind, order=endian)],
                                     order=endian, version=version, data=b"q"*block_bytes))
                    record = self.one()
                    self.assertEqual((record.status, record.tensor_count, record.size_unverified),
                                     ("layout_checked", 1, 0))
                    self.assertEqual(inspect_all_block_layouts(project_root=self.root).status, "inspected")
                    self.assertEqual(inspect_tensor_index("sample", project_root=self.root).status,
                                     "size_unverified")

    def test_multirow_tensor_uses_first_dimension_as_block_axis(self):
        self.declare()
        self.write(image([info("q4", dims=(32, 4), kind=2)], data=b"a"*(4*18)))
        self.assertEqual(self.one().status, "layout_checked")
        self.write(image([info("q4", dims=(32, 4), kind=2)], data=b"a"*(4*18-1)))
        self.assertEqual(self.one().status, "tensor_extent_out_of_file")

    def test_invalid_first_dimension_nonmultiple(self):
        self.declare()
        for kind, _ in FORMATS:
            for dims in ((31,), (31, 32), (33, 1)):
                with self.subTest(kind=kind, dims=dims):
                    self.write(image([info("q", dims=dims, kind=kind)], data=b"x"*256))
                    self.assertEqual(self.one().status, "invalid_block_dimension")

    def test_block_overlap_and_file_bounds(self):
        self.declare()
        self.write(image([info("a", dims=(64,), kind=2, offset=0),
                          info("b", dims=(32,), kind=3, offset=32)], data=b"x"*128))
        self.assertEqual(self.one().status, "tensor_overlap")
        self.write(image([info("a", dims=(64,), kind=2)], data=b"x"*35))
        self.assertEqual(self.one().status, "tensor_extent_out_of_file")
        self.write(image([info("a", dims=(32,), kind=8, offset=64)], data=b"x"*34))
        self.assertEqual(self.one().status, "tensor_offset_out_of_file")
        self.write(image([info("a", dims=(32,), kind=2, offset=1)], data=b"x"*64))
        self.assertEqual(self.one().status, "invalid_tensor_alignment")

    def test_mixed_scalar_and_quantized_and_unsupported(self):
        self.declare()
        self.write(image([info("q", dims=(32,), kind=2, offset=0),
                          info("float", dims=(4,), kind=0, offset=32)], data=b"x"*48))
        self.assertEqual(self.one().status, "layout_checked")
        self.write(image([info("unknown", dims=(256,), kind=12)], data=b"x"*80))
        report = inspect_all_block_layouts(project_root=self.root)
        self.assertEqual((report.status, report.models[0].status,
                          report.models[0].size_unverified), ("partial", "size_unverified", 1))

    def test_custom_alignment_and_big_endian(self):
        self.declare()
        for endian in ("<", ">"):
            with self.subTest(endian=endian):
                field = kv("general.alignment", 4, struct.pack(endian+"I", 64), endian)
                self.write(image([info("q", dims=(32,), kind=2, order=endian)],
                                 [field], order=endian, alignment=64, data=b"x"*18))
                self.assertEqual(self.one().status, "layout_checked")

    def test_limits_malformed_metadata_missing_and_symlinks(self):
        self.declare()
        self.assertEqual(self.one().status, "missing")
        self.assertEqual(inspect_block_layout("../demo.gguf", project_root=self.root).status, "not_declared")
        self.write(b"NOPE" + struct.pack("<IQQ", 3, 1, 0))
        self.assertEqual(self.one().status, "invalid_magic")
        self.write(b"GGUF" + struct.pack("<IQQ", 3, 16385, 0))
        self.assertEqual(self.one().status, "limit_exceeded")
        self.write(image([info("a", dims=(0,), kind=2)], data=b"x"*20))
        self.assertEqual(self.one().status, "invalid_dimensions")
        self.write(image([info("a", dims=(1<<63, 2), kind=2)], data=b"x"*32))
        self.assertEqual(self.one().status, "limit_exceeded")
        self.write(image([info("a", dims=(32,), kind=2)])[:28])
        self.assertEqual(self.one().status, "truncated_index")
        self.declare([{"id": "sample", "format": "gguf", "path": "../escape.gguf"}])
        self.assertEqual(inspect_all_block_layouts(project_root=self.root).status, "invalid")
        self.declare()
        (self.folder / "demo.gguf").unlink()
        target = self.root / "outside.gguf"
        target.write_bytes(image([info("q", dims=(32,), kind=2)], data=b"x"*18))
        try:
            (self.folder / "demo.gguf").symlink_to(target)
        except (NotImplementedError, OSError):
            self.skipTest("symlinks not available")
        self.assertEqual(inspect_all_block_layouts(project_root=self.root).status, "invalid")

    def test_report_immutable_and_no_payload_or_tensor_names(self):
        self.declare()
        self.write(image([info("sensitive-name", dims=(32,), kind=2)], data=b"z"*18))
        result = self.one()
        self.assertEqual(result.status, "layout_checked")
        self.assertNotIn("sensitive-name", repr(result))
        self.assertFalse(hasattr(result, "path"))
        with self.assertRaises(FrozenInstanceError):
            result.status = "modified"


if __name__ == "__main__":
    unittest.main()
