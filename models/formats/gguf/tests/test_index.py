"""Synthetic, local-only tests for bounded GGUF tensor descriptor indexes."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import struct
import unittest

from models.formats.gguf.runtime.tensor_index import inspect_tensor_index, inspect_all_tensor_indexes


def kv(key, kind, data, order="<"):
    k = key.encode("ascii")
    return struct.pack(order + "Q", len(k)) + k + struct.pack(order + "I", kind) + data


def info(name, dims=(2, 2), kind=0, offset=0, order="<"):
    raw = name.encode("utf-8")
    return (struct.pack(order + "Q", len(raw)) + raw +
            struct.pack(order + "I", len(dims)) +
            b"".join(struct.pack(order + "Q", d) for d in dims) +
            struct.pack(order + "IQ", kind, offset))


def image(entries=(), fields=(), order="<", version=3, data=b"", alignment=32):
    payload = (b"GGUF" + struct.pack(order + "IQQ", version, len(entries), len(fields)) +
               b"".join(fields) + b"".join(entries))
    if entries:
        payload += b"\x00" * (-len(payload) % alignment) + data
    return payload


class IndexTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.base = self.root / ".local" / "models"
        self.base.mkdir(parents=True)

    def declare(self, records=None):
        records = [{"id": "demo", "format": "gguf", "path": "test.gguf"}] if records is None else records
        (self.base / "catalog.json").write_text(json.dumps({"models": records}), encoding="utf-8")

    def write(self, payload):
        (self.base / "test.gguf").write_bytes(payload)

    def single(self):
        return inspect_tensor_index("demo", project_root=self.root)

    def test_unconfigured_empty_and_no_gguf(self):
        self.assertEqual(inspect_all_tensor_indexes(project_root=self.root).status, "not_configured")
        self.declare([])
        self.assertEqual(inspect_all_tensor_indexes(project_root=self.root).status, "empty")
        self.declare([{"id": "other", "format": "safetensors", "path": "other.bin"}])
        self.assertEqual(inspect_all_tensor_indexes(project_root=self.root).status, "no_gguf")
        self.assertEqual(inspect_tensor_index("other", project_root=self.root).status, "not_gguf")

    def test_versions_and_byte_order_and_custom_alignment(self):
        self.declare()
        for order, version, alignment in (("<", 2, 32), ("<", 3, 64), (">", 3, 32)):
            with self.subTest(order=order, version=version, alignment=alignment):
                alignment_kv = kv("general.alignment", 4, struct.pack(order + "I", alignment), order)
                self.write(image([info("tensor.0", kind=0, order=order)], [alignment_kv],
                                 order=order, version=version, data=b"a" * 16, alignment=alignment))
                row = self.single()
                self.assertEqual((row.status, row.tensor_count, row.size_unverified),
                                 ("index_checked", 1, 0))
                self.assertEqual(inspect_all_tensor_indexes(project_root=self.root).status, "inspected")

    def test_no_tensor_is_valid_metadata_only(self):
        self.declare()
        self.write(image([], []))
        self.assertEqual(self.single().status, "index_checked")
        self.assertEqual(self.single().tensor_count, 0)

    def test_multiple_non_overlapping_descriptors_in_any_order(self):
        self.declare()
        self.write(image([info("z", dims=(4,), offset=32),
                          info("a", dims=(4,), offset=0)], data=b"x" * 48))
        self.assertEqual(self.single().status, "index_checked")

    def test_unknown_quantized_layout_is_uncertified(self):
        self.declare()
        self.write(image([info("q", dims=(32,), kind=2)], data=b"q" * 32))
        item = self.single()
        self.assertEqual((item.status, item.size_unverified), ("size_unverified", 1))
        self.assertEqual(inspect_all_tensor_indexes(project_root=self.root).status, "partial")

    def test_duplicate_tensor_names_and_offsets(self):
        self.declare()
        for rows, code in (
            ([info("a"), info("a", offset=32)], "duplicate_tensor_name"),
            ([info("a"), info("b")], "duplicate_tensor_offset")
        ):
            with self.subTest(code=code):
                self.write(image(rows, data=b"z" * 64))
                self.assertEqual(self.single().status, code)

    def test_dimensions_empty_and_overflow(self):
        self.declare()
        for dimensions, code in (
            ((), "invalid_dimensions"),
            ((0, 1), "invalid_dimensions"),
            ((1, 1, 1, 1, 1), "invalid_dimensions"),
            (((1 << 63), 2), "limit_exceeded")
        ):
            with self.subTest(code=code):
                self.write(image([info("a", dims=dimensions)], data=b"z" * 64))
                self.assertEqual(self.single().status, code)

    def test_offsets_not_aligned_or_outside_file_or_overlapping(self):
        self.declare()
        for rows, data, code in (
            ([info("a", offset=1)], b"z" * 96, "invalid_tensor_alignment"),
            ([info("a", offset=64)], b"z" * 16, "tensor_offset_out_of_file"),
            ([info("a", dims=(16,))], b"z" * 16, "tensor_extent_out_of_file"),
            ([info("a", dims=(16,)), info("b", offset=32)], b"z" * 96, "tensor_overlap")
        ):
            with self.subTest(code=code):
                self.write(image(rows, data=data))
                self.assertEqual(self.single().status, code)

    def test_invalid_alignment_and_metadata(self):
        self.declare()
        for entries, fields, code in (
            ([info("a")], [kv("general.alignment", 4, struct.pack("<I", 7))], "invalid_alignment"),
            ([info("a")], [kv("general.alignment", 8, b"")], "invalid_alignment"),
            ([info("a")], [kv("general.alignment", 4, struct.pack("<I", 32))] * 2, "invalid_alignment"),
            ([info("a")], [kv("unknown.key", 99, b"")], "unsupported_type")
        ):
            with self.subTest(code=code):
                self.write(image(entries, fields, data=b"z" * 16))
                self.assertEqual(self.single().status, code)

    def test_truncation_invalid_magic_and_version(self):
        self.declare()
        good = image([info("a")], data=b"z" * 16)
        for contents, code in (
            (b"GGUF", "truncated_metadata"),
            (b"NOPE" + good[4:], "invalid_magic"),
            (good[:24 + 8 + 2], "truncated_index"),
            (image([info("a")], version=1, data=b"z" * 16), "unsupported_version"),
        ):
            with self.subTest(code=code):
                self.write(contents)
                self.assertEqual(self.single().status, code)

    def test_filename_and_count_limits(self):
        self.declare()
        for index, code in (
            (info("a" * 65), "invalid_tensor_name"),
            (info("a")[:-1], "truncated_index"),
        ):
            with self.subTest(code=code):
                self.write(image([index], data=b"z" * 16) if code != "truncated_index"
                           else b"GGUF" + struct.pack("<IQQ", 3, 1, 0) + index)
                self.assertEqual(self.single().status, code)
        self.write(b"GGUF" + struct.pack("<IQQ", 3, 16385, 0))
        self.assertEqual(self.single().status, "limit_exceeded")

    def test_catalog_restrictions_missing_and_symlink(self):
        self.declare()
        self.assertEqual(self.single().status, "missing")
        self.assertEqual(inspect_tensor_index("/tmp/host", project_root=self.root).status, "not_declared")
        target = self.root / "outside.gguf"
        target.write_bytes(image([info("a")], data=b"z" * 16))
        try:
            (self.base / "test.gguf").symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unsupported")
        self.assertEqual(inspect_all_tensor_indexes(project_root=self.root).status, "invalid")
        self.declare([{"id": "demo", "format": "gguf", "path": "../invalid.gguf"}])
        self.assertEqual(inspect_all_tensor_indexes(project_root=self.root).errors, ("invalid_catalog",))

    def test_no_payload_data_leak_in_report(self):
        self.declare()
        self.write(image([info("sensitive-token")], data=b"z" * 16))
        result = self.single()
        self.assertEqual(result.status, "index_checked")
        self.assertFalse(hasattr(result, "path"))
        self.assertNotIn("sensitive-token", repr(result))


if __name__ == "__main__":
    unittest.main()
