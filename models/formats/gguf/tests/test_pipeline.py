"""End-to-end disposable GGUF fixtures; no real weights or models."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from models.catalog import catalog
from models.formats.gguf import (
    inspect_headers, inspect_all_metadata, inspect_all_tensor_indexes,
    inspect_all_block_layouts, inspect_all_kblock_layouts)
from models.formats.gguf.runtime.fixture import _create_fixture_catalog, run_fixture_pipeline


class PipelineTests(unittest.TestCase):
    def test_synthetic_chain(self):
        result = run_fixture_pipeline()
        self.assertEqual(result.status, "passed", result)
        self.assertEqual(result.fixtures, 4)
        self.assertEqual(result.stages, (
            "CAT=ready", "GGUF=inspected", "META=inspected",
            "INDEX=partial", "BLOCKS=partial", "KBLOCKS=inspected",
            "SUMMARY=structural_checked"))

    def test_catalog_legacy_statuses_and_endianness(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(catalog(project_root=root).status, "not_configured")
            _create_fixture_catalog(root)
            self.assertEqual(tuple(x.byte_order for x in inspect_headers(project_root=root).headers),
                             ("little", "big", "little", "little"))
            self.assertEqual(tuple(x.architecture for x in inspect_all_metadata(project_root=root).models),
                             ("llama", "llama", "llama", "llama"))
            self.assertEqual(tuple(x.size_unverified for x in inspect_all_tensor_indexes(project_root=root).models),
                             (0, 1, 1, 1))
            self.assertEqual(tuple(x.size_unverified for x in inspect_all_block_layouts(project_root=root).models),
                             (0, 0, 1, 1))
            self.assertEqual(tuple(x.size_unverified for x in inspect_all_kblock_layouts(project_root=root).models),
                             (0, 0, 0, 0))

    def test_corrupt_magic_is_rejected(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _create_fixture_catalog(root)
            file = root / ".local/models/scalar.gguf"
            blob = file.read_bytes()
            file.write_bytes(b"NOPE" + blob[4:])
            self.assertEqual(inspect_headers(project_root=root).status, "partial")
            self.assertEqual(inspect_all_metadata(project_root=root).status, "partial")
            self.assertEqual(inspect_all_kblock_layouts(project_root=root).status, "partial")

    def test_truncated_kblock_extent_is_rejected(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _create_fixture_catalog(root)
            file = root / ".local/models/kquant.gguf"
            with file.open("r+b") as stream:
                stream.truncate(file.stat().st_size - 1)
            result = inspect_all_kblock_layouts(project_root=root)
            self.assertEqual(result.status, "partial")
            self.assertEqual(result.models[-1].status, "tensor_extent_out_of_file")


if __name__ == "__main__":
    unittest.main()
