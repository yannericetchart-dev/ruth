"""MIG-016: multi-tensor synthetic GGUF fixture validation, no real weights."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from models.formats.gguf.runtime.fixture import (
    _create_fixture_catalog, _synthetic_multitensor, run_fixture_pipeline,
)
from models.formats.gguf.runtime.structural_summary import inspect_structural_summaries
from models.formats.gguf.runtime.tensor_index import inspect_all_tensor_indexes
from models.formats.gguf.runtime.blocks import inspect_all_block_layouts
from models.formats.gguf.runtime.kblocks import inspect_all_kblock_layouts


class MultiTensorFixtureTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        _create_fixture_catalog(self.root)
        self.file = self.root / ".local" / "models" / "multi.gguf"

    def verdict(self):
        report = inspect_structural_summaries(project_root=self.root)
        return report.status, report.models[2]

    def test_integrated_catalog_has_four_fixtures_and_multitensor_summary(self):
        result = run_fixture_pipeline()
        self.assertEqual((result.status, result.fixtures), ("passed", 4))
        self.assertEqual(result.stages[-1], "SUMMARY=structural_checked")
        status, item = self.verdict()
        self.assertEqual(status, "structural_checked")
        self.assertEqual((item.model_id, item.status, item.tensor_count,
                          item.size_unverified, item.reason),
                         ("multi", "structural_checked", 3, 0, ""))

    def test_legacy_readers_keep_partial_sizes_until_kblocks(self):
        index = inspect_all_tensor_indexes(project_root=self.root).models[2]
        blocks = inspect_all_block_layouts(project_root=self.root).models[2]
        kblocks = inspect_all_kblock_layouts(project_root=self.root).models[2]
        self.assertEqual((index.status, index.tensor_count, index.size_unverified),
                         ("size_unverified", 3, 2))
        self.assertEqual((blocks.status, blocks.tensor_count, blocks.size_unverified),
                         ("size_unverified", 3, 1))
        self.assertEqual((kblocks.status, kblocks.tensor_count, kblocks.size_unverified),
                         ("kblocks_checked", 3, 0))

    def test_out_of_order_tensor_descriptors_validated_by_offsets(self):
        blob = _synthetic_multitensor(specs=(
            ("multi.q4", 32, 2, 32),
            ("multi.kquant", 256, 12, 64),
            ("multi.scalar", 4, 0, 0),
        ))
        self.file.write_bytes(blob)
        self.assertEqual(self.verdict()[1].status, "structural_checked")

    def test_custom_alignment_and_big_endian(self):
        self.file.write_bytes(_synthetic_multitensor(
            specs=(("multi.scalar", 4, 0, 0),
                   ("multi.q4", 32, 2, 64),
                   ("multi.kquant", 256, 12, 128)),
            alignment=64, data_size=272, order=">", version=3))
        self.assertEqual(self.verdict()[1].status, "structural_checked")

    def test_tensor_alignment_violation_rejected(self):
        self.file.write_bytes(_synthetic_multitensor(specs=(
            ("multi.scalar", 4, 0, 0),
            ("multi.q4", 32, 2, 33),
            ("multi.kquant", 256, 12, 64),
        )))
        self.assertEqual(self.verdict()[1].reason, "INDEX:invalid_tensor_alignment")

    def test_tensor_overlap_rejected(self):
        self.file.write_bytes(_synthetic_multitensor(specs=(
            ("multi.scalar", 16, 0, 0),  # 64 bytes, next starts at 32
            ("multi.q4", 32, 2, 32),
            ("multi.kquant", 256, 12, 64),
        )))
        self.assertEqual(self.verdict()[1].reason, "INDEX:tensor_overlap")

    def test_tensor_offset_outside_file_rejected(self):
        self.file.write_bytes(_synthetic_multitensor(specs=(
            ("multi.scalar", 4, 0, 0),
            ("multi.q4", 32, 2, 224),
            ("multi.kquant", 256, 12, 64),
        )))
        self.assertEqual(self.verdict()[1].reason, "INDEX:tensor_offset_out_of_file")

    def test_quantized_extent_truncated_rejected(self):
        self.file.write_bytes(_synthetic_multitensor(data_size=207))
        status, item = self.verdict()
        self.assertEqual(status, "invalid")
        self.assertEqual((item.status, item.reason),
                         ("invalid", "KBLOCKS:tensor_extent_out_of_file"))

    def test_duplicate_tensor_offsets_not_claimed_valid(self):
        self.file.write_bytes(_synthetic_multitensor(specs=(
            ("multi.scalar", 4, 0, 0),
            ("multi.q4", 32, 2, 32),
            ("multi.kquant", 256, 12, 32),
        )))
        self.assertEqual(self.verdict()[1].reason, "INDEX:duplicate_tensor_offset")

    def test_unknown_quantization_remains_unverified(self):
        self.file.write_bytes(_synthetic_multitensor(specs=(
            ("multi.scalar", 4, 0, 0),
            ("multi.unknown", 32, 16, 32),
            ("multi.kquant", 256, 12, 64),
        )))
        status, item = self.verdict()
        self.assertEqual((status, item.status, item.size_unverified),
                         ("size_unverified", "size_unverified", 1))
        self.assertEqual(item.reason, "KBLOCKS:size_unverified")

    def test_invalid_block_dimension_rejected(self):
        self.file.write_bytes(_synthetic_multitensor(specs=(
            ("multi.scalar", 4, 0, 0),
            ("multi.q4", 31, 2, 32),
            ("multi.kquant", 256, 12, 64),
        )))
        self.assertEqual(self.verdict()[1].reason, "BLOCKS:invalid_block_dimension")


if __name__ == "__main__":
    unittest.main()
