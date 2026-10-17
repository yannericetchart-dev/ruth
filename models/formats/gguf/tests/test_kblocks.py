"""Synthetic read-only GGML K-block descriptors, no model downloads."""
from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import struct
import unittest
from unittest.mock import patch

from models.formats.gguf import (
    inspect_kblock_layout, inspect_all_kblock_layouts,
    inspect_block_layout, inspect_tensor_index,
)
from models.formats.gguf.tests.test_index import image, info, kv

_K = ((10, 84), (11, 110), (12, 144), (13, 176), (14, 210), (15, 292))


class KBlockTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.local = self.root / ".local" / "models"
        self.local.mkdir(parents=True)

    def declare(self, records=None):
        rows = [{"id": "synthetic", "format": "gguf", "path": "sample.gguf"}] if records is None else records
        (self.local / "catalog.json").write_text(json.dumps({"models": rows}), encoding="utf-8")

    def write(self, data):
        (self.local / "sample.gguf").write_bytes(data)

    def inspect(self):
        return inspect_kblock_layout("synthetic", project_root=self.root)

    def test_catalog_states_and_no_free_paths(self):
        self.assertEqual(inspect_all_kblock_layouts(project_root=self.root).status, "not_configured")
        self.declare([])
        self.assertEqual(inspect_all_kblock_layouts(project_root=self.root).status, "empty")
        self.declare([{"id": "else", "format": "safetensors", "path": "some.bin"}])
        self.assertEqual(inspect_all_kblock_layouts(project_root=self.root).status, "no_gguf")
        self.assertEqual(inspect_kblock_layout("else", project_root=self.root).status, "not_gguf")
        self.assertEqual(inspect_kblock_layout("/tmp/host.gguf", project_root=self.root).status, "not_declared")

    def test_all_six_types_endianness_and_v2_v3(self):
        self.declare()
        for kind, byte_count in _K:
            for order, version in (("<", 2), ("<", 3), (">", 3)):
                with self.subTest(kind=kind, order=order, version=version):
                    self.write(image([info("k", dims=(256,), kind=kind, order=order)],
                                     order=order, version=version, data=b"k" * byte_count))
                    row = self.inspect()
                    self.assertEqual((row.status, row.tensor_count, row.size_unverified),
                                     ("kblocks_checked", 1, 0))
                    self.assertEqual(inspect_all_kblock_layouts(project_root=self.root).status, "inspected")
                    # Previously integrated checks must NOT gain implicit K-quant support.
                    self.assertEqual(inspect_tensor_index("synthetic", project_root=self.root).status, "size_unverified")
                    self.assertEqual(inspect_block_layout("synthetic", project_root=self.root).status, "size_unverified")

    def test_multiple_rows_and_first_dimension_divisible(self):
        self.declare()
        for kind, size in _K:
            with self.subTest(kind=kind):
                self.write(image([info("k", dims=(256, 3), kind=kind)], data=b"k" * (3 * size)))
                self.assertEqual(self.inspect().status, "kblocks_checked")
                self.write(image([info("k", dims=(256, 3), kind=kind)], data=b"k" * (3 * size - 1)))
                self.assertEqual(self.inspect().status, "tensor_extent_out_of_file")

    def test_rejects_wrong_first_dimension(self):
        self.declare()
        for kind, _ in _K:
            for dims in ((255,), (257, 2), (128, 4)):
                with self.subTest(kind=kind, dims=dims):
                    self.write(image([info("k", dims=dims, kind=kind)], data=b"x" * 512))
                    self.assertEqual(self.inspect().status, "invalid_block_dimension")

    def test_rejects_overlap_and_out_of_file(self):
        self.declare()
        self.write(image([info("a", dims=(512,), kind=10, offset=0),
                          info("b", dims=(256,), kind=12, offset=160)], data=b"x" * 480))
        self.assertEqual(self.inspect().status, "tensor_overlap")
        self.write(image([info("a", dims=(512,), kind=10, offset=0)], data=b"x" * 167))
        self.assertEqual(self.inspect().status, "tensor_extent_out_of_file")
        self.write(image([info("a", dims=(256,), kind=10, offset=128)], data=b"x" * 84))
        self.assertEqual(self.inspect().status, "tensor_offset_out_of_file")

    def test_mixed_supported_and_unknown_layouts(self):
        self.declare()
        self.write(image([info("q4", dims=(32,), kind=2, offset=0),
                          info("k2", dims=(256,), kind=10, offset=32)], data=b"x" * (32 + 84)))
        self.assertEqual(self.inspect().status, "kblocks_checked")
        self.write(image([info("unknown", dims=(256,), kind=16)], data=b"x" * 128))
        row = self.inspect()
        self.assertEqual((row.status, row.size_unverified), ("size_unverified", 1))
        self.assertEqual(inspect_all_kblock_layouts(project_root=self.root).status, "partial")

    def test_alignment_malformed_files_and_limits(self):
        self.declare()
        for order in ("<", ">"):
            with self.subTest(order=order):
                align = kv("general.alignment", 4, struct.pack(order+"I", 64), order)
                self.write(image([info("k", dims=(256,), kind=10, order=order)],
                                 fields=[align], order=order, alignment=64, data=b"x" * 84))
                self.assertEqual(self.inspect().status, "kblocks_checked")
        for data, status in (
            (b"NOPE" + struct.pack("<IQQ", 3, 1, 0), "invalid_magic"),
            (b"GGUF" + struct.pack("<IQQ", 3, 16385, 0), "limit_exceeded"),
            (image([info("k", dims=(0,), kind=10)], data=b"x" * 84), "invalid_dimensions"),
            (image([info("k", dims=(1 << 63, 2), kind=10)], data=b"x" * 84), "limit_exceeded"),
            (image([info("k", dims=(256,), kind=10)])[:28], "truncated_index"),
        ):
            with self.subTest(status=status):
                self.write(data)
                self.assertEqual(self.inspect().status, status)

    def test_symlinks_rejected_and_immutable_report(self):
        self.declare()
        self.assertEqual(self.inspect().status, "missing")
        target = self.root / "not-catalogued.gguf"
        target.write_bytes(image([info("k", dims=(256,), kind=10)], data=b"k" * 84))
        try:
            (self.local / "sample.gguf").symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        self.assertEqual(inspect_all_kblock_layouts(project_root=self.root).status, "invalid")
        (self.local / "sample.gguf").unlink()
        self.write(image([info("secret-tensor", dims=(256,), kind=10)], data=b"k" * 84))
        row = self.inspect()
        self.assertNotIn("secret-tensor", repr(row))
        self.assertFalse(hasattr(row, "reference"))
        with self.assertRaises(FrozenInstanceError):
            row.status = "changed"

    def test_synthetic_fixture_never_reaches_payload_reader(self):
        self.declare()
        data = image([info("k", dims=(256,), kind=10)], data=b"m" * 84)
        self.write(data)
        with patch("models.formats.gguf.runtime.tensor_index._inspect", wraps=None) as _unused:
            # Confirm API result without exposing tensor bytes in its report.
            row = self.inspect()
        self.assertEqual(row.status, "kblocks_checked")
        self.assertNotIn("m" * 20, repr(row))


if __name__ == "__main__":
    unittest.main()
