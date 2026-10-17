"""MIG-015: integrated synthetic CAT->GGUF->META->INDEX->BLOCKS->KBLOCKS->SUMMARY."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from models.formats.gguf.runtime import fixture, structural_summary as summary
from models.formats.gguf.runtime.structural_summary import GGUFStructuralReport


class FixtureSummaryTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        fixture._create_fixture_catalog(self.root)
        self.target = self.root / ".local/models/kquant.gguf"

    def test_full_synthetic_pipeline_checks_all_four_summary_verdicts(self):
        result = fixture.run_fixture_pipeline()
        self.assertEqual(result.status, "passed", result)
        self.assertEqual(result.fixtures, 4)
        self.assertEqual(result.stages[-1], "SUMMARY=structural_checked")

    def test_summary_has_four_stable_structurally_checked_models(self):
        report = summary.inspect_structural_summaries(project_root=self.root)
        self.assertEqual(report.status, "structural_checked")
        self.assertEqual(tuple((r.model_id, r.status, r.tensor_count)
                               for r in report.models),
                         (("scalar", "structural_checked", 1),
                          ("q4", "structural_checked", 1),
                          ("multi", "structural_checked", 3),
                          ("kquant", "structural_checked", 1)))

    def test_unsupported_quantization_is_not_falsely_verified(self):
        self.target.write_bytes(fixture._synthetic_gguf(
            version=3, order="<", kind=16, width=256, size=144))
        report = summary.inspect_structural_summaries(project_root=self.root)
        self.assertEqual(report.status, "size_unverified")
        self.assertEqual(report.models[-1].reason, "KBLOCKS:size_unverified")

    def test_damaged_magic_is_invalid(self):
        original = self.target.read_bytes()
        self.target.write_bytes(b"NOPE" + original[4:])
        report = summary.inspect_structural_summaries(project_root=self.root)
        self.assertEqual(report.status, "invalid")
        self.assertEqual(report.models[-1].reason, "GGUF:invalid_magic")

    def test_truncated_tensor_extent_is_invalid(self):
        with self.target.open("r+b") as stream:
            stream.truncate(self.target.stat().st_size - 1)
        report = summary.inspect_structural_summaries(project_root=self.root)
        self.assertEqual(report.status, "invalid")
        self.assertEqual(report.models[-1].reason,
                         "KBLOCKS:tensor_extent_out_of_file")

    def test_change_during_summary_is_invalid_even_after_valid_readers(self):
        original_reader = summary._inspect_kblocks

        def mutate_after_reading(record, root):
            result = original_reader(record, root)
            if record.id == "kquant":
                with self.target.open("ab") as file:
                    file.write(b"\x00")
            return result

        with patch.object(summary, "_inspect_kblocks",
                          side_effect=mutate_after_reading):
            report = summary.inspect_structural_summaries(project_root=self.root)
        self.assertEqual(report.status, "invalid")
        self.assertEqual(report.models[-1].reason,
                         "STABILITY:changed_during_inspection")

    def test_integrated_fixture_fails_if_summary_is_not_verified(self):
        # Test the actual fixture pipeline guard; the patch never changes local models.
        for bad in ("invalid", "size_unverified", "not_configured"):
            with self.subTest(bad=bad), patch.object(
                fixture, "inspect_structural_summaries",
                return_value=GGUFStructuralReport(bad),
            ):
                result = fixture.run_fixture_pipeline()
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.reason, "unexpected_report")
            self.assertEqual(result.stages[-1], "SUMMARY=" + bad)


if __name__ == "__main__":
    unittest.main()
