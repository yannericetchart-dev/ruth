"""Synthetic GGUF metadata tests: bounded, catalog-only, read-only."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import struct
import unittest
from unittest.mock import patch

from models.formats.gguf import inspect_metadata, inspect_all_metadata


def string(data, order="<"):
    return struct.pack(order + "Q", len(data)) + data


def entry(key, kind, value, order="<"):
    return string(key.encode("ascii"), order) + struct.pack(order + "I", kind) + value


def text(key, value, order="<"):
    return entry(key, 8, string(value.encode("utf-8"), order), order)


def integer(key, value, order="<", kind=4):
    return entry(key, kind, struct.pack(order + ("I" if kind == 4 else "Q"), value), order)


def payload(items, version=3, order="<", tensor_count=0):
    return b"GGUF" + struct.pack(order + "IQQ", version, tensor_count, len(items)) + b"".join(items)


class MetadataTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.base = self.root / ".local" / "models"
        self.base.mkdir(parents=True)

    def declare(self, rows=None):
        if rows is None:
            rows = [{"id": "demo", "format": "gguf", "path": "sample.gguf"}]
        (self.base / "catalog.json").write_text(json.dumps({"models": rows}), encoding="utf-8")

    def write(self, data, name="sample.gguf"):
        (self.base / name).write_bytes(data)

    def test_missing_and_empty_catalog(self):
        self.assertEqual(inspect_all_metadata(project_root=self.root).status, "not_configured")
        self.declare([])
        self.assertEqual(inspect_all_metadata(project_root=self.root).status, "empty")
        self.assertEqual(inspect_metadata("demo", project_root=self.root).status, "not_declared")

    def test_v2_v3_and_big_endian(self):
        self.declare()
        for version, order in ((2, "<"), (3, "<"), (3, ">")):
            with self.subTest(version=version, order=order):
                self.write(payload([
                    integer("llama.context_length", 8192, order),
                    text("general.architecture", "llama", order),
                    integer("general.file_type", 15, order),
                ], version, order))
                row = inspect_metadata("demo", project_root=self.root)
                self.assertEqual((row.status, row.architecture, row.file_type, row.context_length),
                                 ("metadata_read", "llama", 15, 8192))
                self.assertEqual(inspect_all_metadata(project_root=self.root).status, "inspected")

    def test_missing_optional_fields_remain_unknown(self):
        self.declare()
        self.write(payload([text("general.architecture", "qwen2")]))
        row = inspect_metadata("demo", project_root=self.root)
        self.assertEqual(row.status, "metadata_read")
        self.assertIsNone(row.file_type)
        self.assertIsNone(row.context_length)

    def test_does_not_invent_context_from_unrelated_architecture(self):
        self.declare()
        self.write(payload([text("general.architecture", "llama"), integer("qwen2.context_length", 8192)]))
        self.assertIsNone(inspect_metadata("demo", project_root=self.root).context_length)

    def test_skips_scalar_string_and_nested_array(self):
        self.declare()
        # ARRAY of two ARRAYs, each containing one STRING (without capturing payload).
        one = struct.pack("<IQ", 8, 1) + string(b"token")
        nested = struct.pack("<IQ", 9, 2) + one + one
        self.write(payload([
            entry("tokenizer.ggml.tokens", 9, nested),
            entry("general.unknown", 7, b"\x01"),
            text("general.architecture", "llama"),
            integer("llama.context_length", 2048),
        ]))
        row = inspect_metadata("demo", project_root=self.root)
        self.assertEqual((row.status, row.context_length), ("metadata_read", 2048))

    def test_rejects_truncation_and_magic(self):
        self.declare()
        for data, status in [
            (b"GGUF", "truncated_metadata"),
            (b"NOPE" + struct.pack("<IQQ", 3, 0, 0), "invalid_magic"),
            (payload([text("general.architecture", "llama")])[:-1], "truncated_metadata"),
            (payload([], version=1), "unsupported_version"),
        ]:
            with self.subTest(status=status):
                self.write(data)
                self.assertEqual(inspect_metadata("demo", project_root=self.root).status, status)

    def test_rejects_bad_or_duplicate_fields(self):
        self.declare()
        tests = [
            (payload([entry("general.architecture", 4, struct.pack("<I", 9))]), "invalid_metadata"),
            (payload([text("general.architecture", "llama"), text("general.architecture", "llama")]), "invalid_metadata"),
            (payload([text("General.Architecture", "llama")]), "invalid_metadata"),
            (payload([entry("unknown.metadata", 99, b"")]), "unsupported_type"),
            (payload([text("general.architecture", "\x1b[2J")]), "invalid_metadata"),
        ]
        for data, status in tests:
            with self.subTest(status=status):
                self.write(data)
                self.assertEqual(inspect_metadata("demo", project_root=self.root).status, status)

    def test_bounded_fields_strings_arrays_and_depth(self):
        self.declare()
        oversized_count = b"GGUF" + struct.pack("<IQQ", 3, 0, 16385)
        big_string = entry("tokenizer.ggml.tokens", 8, struct.pack("<Q", 8 * 1024 * 1024 + 1))
        big_array = entry("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 4, 500001))
        deep = struct.pack("<IQ", 4, 1) + struct.pack("<I", 1)
        for _ in range(7):
            deep = struct.pack("<IQ", 9, 1) + deep
        deep_array = entry("tokenizer.ggml.tokens", 9, deep)
        for data in (oversized_count, payload([big_string]), payload([big_array]), payload([deep_array])):
            with self.subTest(length=len(data)):
                self.write(data)
                self.assertEqual(inspect_metadata("demo", project_root=self.root).status, "limit_exceeded")

    def test_invalid_manifest_missing_file_and_unlisted_paths(self):
        self.declare()
        self.assertEqual(inspect_metadata("demo", project_root=self.root).status, "missing")
        self.assertEqual(inspect_metadata("/tmp/passwd", project_root=self.root).status, "not_declared")
        self.declare([{"id": "demo", "format": "gguf", "path": "../escape.gguf"}])
        self.assertEqual(inspect_all_metadata(project_root=self.root).status, "invalid")
        self.assertEqual(inspect_all_metadata(project_root=self.root).errors, ("invalid_catalog",))

    def test_non_gguf_and_mixed_results(self):
        self.declare([{"id": "other", "format": "safetensors", "path": "other.bin"},
                      {"id": "demo", "format": "gguf", "path": "sample.gguf"}])
        self.write(payload([]))
        self.assertEqual(inspect_metadata("other", project_root=self.root).status, "not_gguf")
        self.assertEqual(inspect_all_metadata(project_root=self.root).status, "inspected")
        self.declare([{"id": "other", "format": "safetensors", "path": "other.bin"}])
        self.assertEqual(inspect_all_metadata(project_root=self.root).status, "no_gguf")

    def test_symlink_rejected(self):
        self.declare()
        target = self.root / "synthetic.gguf"
        target.write_bytes(payload([]))
        try:
            (self.base / "sample.gguf").symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks not available")
        self.assertEqual(inspect_all_metadata(project_root=self.root).status, "invalid")

    def test_metadata_does_not_expose_raw_contents(self):
        self.declare()
        self.write(payload([text("tokenizer.ggml.pre", "private-test-value"),
                            text("general.architecture", "llama")]))
        row = inspect_metadata("demo", project_root=self.root)
        self.assertEqual(row.status, "metadata_read")
        self.assertFalse(hasattr(row, "reference"))
        self.assertNotIn("private-test-value", repr(row))


if __name__ == "__main__":
    unittest.main()
