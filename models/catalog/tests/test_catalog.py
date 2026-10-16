"""Synthetic CAT-001 tests: confined files, no host discovery or downloads."""
import json
from pathlib import Path
import tempfile
import unittest

from models.catalog import ModelCatalog, ModelRecord, catalog, resolve


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.directory = self.root / ".local" / "models"

    def manifest(self, entries):
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "catalog.json").write_text(
            json.dumps({"models": entries}), encoding="utf-8")

    @staticmethod
    def entry(identifier="synthetic-model", path="sample.gguf", fmt="gguf"):
        return {"id": identifier, "path": path, "format": fmt}

    def test_unconfigured_does_not_scan(self):
        report = catalog(project_root=self.root)
        self.assertEqual((report.status, report.count), ("not_configured", 0))
        self.assertEqual(report.errors, ())

    def test_empty_manifest_is_valid(self):
        self.manifest([])
        self.assertEqual(catalog(project_root=self.root).status, "empty")

    def test_existing_file_present_unverified(self):
        self.manifest([self.entry()])
        (self.directory / "sample.gguf").write_bytes(b"synthetic")
        report = catalog(project_root=self.root)
        self.assertIsInstance(report, ModelCatalog)
        self.assertEqual((report.status, report.count), ("ready", 1))
        self.assertEqual(report.models[0].status, "present_unverified")
        self.assertEqual(report.models[0].reference, "sample.gguf")
        self.assertEqual(resolve("synthetic-model", project_root=self.root), report.models[0])

    def test_nested_file_allowed_inside_local_models(self):
        self.manifest([self.entry(path="nested/model.gguf")])
        (self.directory / "nested").mkdir()
        (self.directory / "nested/model.gguf").write_bytes(b"data")
        self.assertEqual(catalog(project_root=self.root).status, "ready")

    def test_missing_file_yields_partial(self):
        self.manifest([self.entry()])
        result = catalog(project_root=self.root)
        self.assertEqual((result.status, result.models[0].status), ("partial", "missing"))

    def test_multiple_entries_and_resolution(self):
        self.manifest([self.entry("first", "a.gguf"), self.entry("second", "b.gguf")])
        report = catalog(project_root=self.root)
        self.assertEqual(report.count, 2)
        self.assertIsNone(resolve("other", project_root=self.root))

    def test_duplicate_ids_fail_atomically(self):
        self.manifest([self.entry(), self.entry(path="other.gguf")])
        result = catalog(project_root=self.root)
        self.assertEqual((result.status, result.errors, result.count),
                         ("invalid", ("duplicate_id",), 0))

    def test_malformed_json_and_unexpected_root(self):
        self.directory.mkdir(parents=True)
        (self.directory / "catalog.json").write_text("{broken", encoding="utf-8")
        self.assertEqual(catalog(project_root=self.root).status, "invalid")
        (self.directory / "catalog.json").write_text('{"unexpected":[]}', encoding="utf-8")
        self.assertEqual(catalog(project_root=self.root).errors, ("invalid_schema",))

    def test_entry_schema_validation(self):
        self.manifest([{"id": "a", "path": "a.gguf"}])
        self.assertEqual(catalog(project_root=self.root).errors, ("invalid_entry",))

    def test_identifier_and_format_sanitized(self):
        self.manifest([self.entry(identifier="bad\nprivate")])
        self.assertEqual(catalog(project_root=self.root).errors, ("invalid_id",))
        self.manifest([self.entry(fmt="GGUF\x1b[31m")])
        self.assertEqual(catalog(project_root=self.root).errors, ("invalid_format",))

    def test_absolute_and_parent_traversal_rejected(self):
        for path in ("/tmp/private", "../private", "folder/../../outside",
                     "./model.gguf", "folder//model.gguf", "C:\\private.gguf"):
            with self.subTest(path=path):
                self.manifest([self.entry(path=path)])
                result = catalog(project_root=self.root)
                self.assertEqual(result.errors, ("unsafe_reference",))
                self.assertNotIn(path, str(result.errors))

    def test_unprintable_path_rejected(self):
        self.manifest([self.entry(path="secret\nfile.gguf")])
        self.assertEqual(catalog(project_root=self.root).errors, ("unsafe_reference",))

    def test_outside_symlink_rejected(self):
        outside = self.root / "elsewhere.gguf"
        outside.write_bytes(b"private")
        self.manifest([self.entry()])
        (self.directory / "sample.gguf").symlink_to(outside)
        self.assertEqual(catalog(project_root=self.root).errors, ("unsafe_reference",))

    def test_symlinked_parent_rejected(self):
        outside = self.root / "elsewhere"
        outside.mkdir()
        self.manifest([self.entry(path="shortcut/model.gguf")])
        (self.directory / "shortcut").symlink_to(outside, target_is_directory=True)
        self.assertEqual(catalog(project_root=self.root).errors, ("unsafe_reference",))

    def test_symlink_manifest_rejected(self):
        self.directory.mkdir(parents=True)
        payload = self.root / "elsewhere.json"
        payload.write_text('{"models":[]}', encoding="utf-8")
        (self.directory / "catalog.json").symlink_to(payload)
        self.assertEqual(catalog(project_root=self.root).errors, ("unsafe_manifest_path",))

    def test_symlinked_local_directory_rejected(self):
        target = self.root / "secret"
        target.mkdir()
        (self.root / ".local").symlink_to(target, target_is_directory=True)
        self.assertEqual(catalog(project_root=self.root).errors, ("unsafe_manifest_path",))

    def test_missing_model_does_not_infer_download(self):
        self.manifest([self.entry()])
        self.assertFalse((self.directory / "sample.gguf").exists())
        self.assertEqual(catalog(project_root=self.root).status, "partial")
        self.assertFalse((self.directory / "sample.gguf").exists())

    def test_manifest_too_large(self):
        self.directory.mkdir(parents=True)
        (self.directory / "catalog.json").write_bytes(b"x" * (1024 * 1024 + 1))
        self.assertEqual(catalog(project_root=self.root).status, "invalid")

    def test_too_many_models(self):
        self.manifest([self.entry(str(i), str(i)+".gguf") for i in range(201)])
        self.assertEqual(catalog(project_root=self.root).errors, ("too_many_models",))

    def test_reject_broken_reference_and_invalid_catalog_on_resolve(self):
        with self.assertRaises(ValueError):
            resolve("../host", project_root=self.root)
        self.manifest([self.entry(), self.entry()])
        with self.assertRaises(ValueError):
            resolve("synthetic-model", project_root=self.root)

    def test_immutable_report_and_to_dict(self):
        report = ModelCatalog("ready", (ModelRecord("x", "gguf", "x.gguf", "missing"),), ())
        with self.assertRaises((AttributeError, TypeError)):
            report.status = "changed"
        self.assertEqual(report.to_dict()["models"][0]["id"], "x")


if __name__ == "__main__":
    unittest.main()
