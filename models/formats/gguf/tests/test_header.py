"""Only declared local GGUF headers may be inspected; no model loading."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import struct
import unittest
from unittest.mock import patch

from models.formats.gguf import inspect_header, inspect_headers


class HeaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / ".local" / "models"
        self.base.mkdir(parents=True)

    def declare(self, *entries):
        (self.base / "catalog.json").write_text(json.dumps({"models": list(entries)}), encoding="utf-8")

    def model(self, identifier="demo", fmt="gguf", name="demo.gguf"):
        return {"id": identifier, "format": fmt, "path": name}

    def write(self, data, name="demo.gguf"):
        (self.base / name).write_bytes(data)

    def test_manifest_absent_and_empty_are_not_errors(self):
        self.assertEqual(inspect_headers(project_root=self.root).status, "not_configured")
        self.declare()
        self.assertEqual(inspect_headers(project_root=self.root).status, "empty")

    def test_header_v2_and_v3_and_big_endian(self):
        self.declare(self.model())
        for version, fmt, order in ((2, "<", "little"), (3, "<", "little"), (3, ">", "big")):
            with self.subTest(version=version, order=order):
                self.write(b"GGUF" + struct.pack(fmt + "IQQ", version, 7, 9) + b"ignored")
                row = inspect_header("demo", project_root=self.root)
                self.assertEqual((row.status, row.version, row.tensor_count, row.metadata_kv_count, row.byte_order),
                                 ("header_read", version, 7, 9, order))
                self.assertEqual(inspect_headers(project_root=self.root).status, "inspected")

    def test_invalid_magic_truncation_unsupported_version(self):
        self.declare(self.model())
        for data, status in ((b"GGUF", "truncated_header"),
                             (b"NOPE" + struct.pack("<IQQ", 3, 1, 0), "invalid_magic"),
                             (b"GGUF" + struct.pack("<IQQ", 99, 1, 0), "unsupported_version"),
                             (b"GGUF" + struct.pack("<IQQ", 1, 1, 0), "unsupported_version")):
            with self.subTest(status=status):
                self.write(data)
                self.assertEqual(inspect_header("demo", project_root=self.root).status, status)
                self.assertEqual(inspect_headers(project_root=self.root).status, "partial")

    def test_missing_files_and_no_gguf(self):
        self.declare(self.model())
        self.assertEqual(inspect_header("demo", project_root=self.root).status, "missing")
        self.assertEqual(inspect_headers(project_root=self.root).status, "partial")
        self.declare(self.model(fmt="safetensors"))
        self.assertEqual(inspect_headers(project_root=self.root).status, "no_gguf")
        self.assertEqual(inspect_header("demo", project_root=self.root).status, "not_gguf")

    def test_reject_undeclared_and_unsafe_model_and_manifest(self):
        self.declare(self.model())
        self.assertEqual(inspect_header("/etc/passwd", project_root=self.root).status, "not_declared")
        self.assertEqual(inspect_header("unknown", project_root=self.root).status, "not_declared")
        self.declare(self.model(name="../escape.gguf"))
        self.assertEqual(inspect_headers(project_root=self.root).status, "invalid")
        self.assertEqual(inspect_headers(project_root=self.root).errors, ("invalid_catalog",))

    def test_reject_symlink_without_following(self):
        self.declare(self.model())
        target = self.root / "real.gguf"
        target.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 1))
        try:
            (self.base / "demo.gguf").symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks not available")
        self.assertEqual(inspect_headers(project_root=self.root).status, "invalid")

    def test_no_full_file_reads_or_disk_discovery(self):
        self.declare(self.model())
        self.write(b"GGUF" + struct.pack("<IQQ", 3, 1, 0) + b"x" * 1000)
        # The result deliberately contains no model contents, paths or tensor data.
        row = inspect_header("demo", project_root=self.root)
        self.assertEqual(row.status, "header_read")
        self.assertFalse(hasattr(row, "reference"))
        self.assertFalse(hasattr(row, "metadata"))

    def test_independent_multiple_declared_headers(self):
        self.declare(self.model(), self.model("two", "gguf", "two.gguf"))
        self.write(b"GGUF" + struct.pack("<IQQ", 3, 1, 0))
        self.write(b"BAD!" + struct.pack("<IQQ", 3, 1, 0), "two.gguf")
        report = inspect_headers(project_root=self.root)
        self.assertEqual(report.status, "partial")
        self.assertEqual([row.status for row in report.headers], ["header_read", "invalid_magic"])

if __name__ == "__main__":
    unittest.main()
