"""MIG-014: synthetic observable file changes during GGUF SUMMARY.

Only isolated temporary GGUF fixtures are written. Never touch a real model.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import os
import unittest

from models.formats.gguf.runtime.fixture import _create_fixture_catalog, _synthetic_gguf
from models.formats.gguf.runtime import structural_summary as summary


class StructuralStabilityTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        _create_fixture_catalog(self.root)
        self.target = self.root / ".local/models/kquant.gguf"
        self.original = summary._inspect_kblocks

    def run_summary(self):
        return summary.inspect_structural_summaries(project_root=self.root)

    def inject_after_kblocks(self, mutation):
        def change(record, root):
            result = self.original(record, root)
            if record.id == "kquant":
                mutation()
            return result
        with patch.object(summary, "_inspect_kblocks", side_effect=change):
            return self.run_summary()

    def test_unchanged_file_stays_structurally_checked(self):
        report = self.run_summary()
        self.assertEqual(report.status, "structural_checked")
        self.assertEqual(report.models[-1].status, "structural_checked")
        self.assertEqual(report.models[-1].reason, "")

    def test_length_change_after_reader_rejected(self):
        def mutate():
            with self.target.open("ab") as stream:
                stream.write(b"\x00")
        report = self.inject_after_kblocks(mutate)
        self.assertEqual(report.status, "invalid")
        self.assertEqual((report.models[-1].status, report.models[-1].reason),
                         ("invalid", "STABILITY:changed_during_inspection"))
        self.assertEqual(tuple(item.status for item in report.models[:2]),
                         ("structural_checked", "structural_checked"))

    def test_same_size_replacement_rejected_by_inode(self):
        def replace():
            candidate = self.target.with_name("replacement.tmp")
            candidate.write_bytes(self.target.read_bytes())
            os.replace(candidate, self.target)
        report = self.inject_after_kblocks(replace)
        self.assertEqual((report.status, report.models[-1].reason),
                         ("invalid", "STABILITY:changed_during_inspection"))

    def test_same_size_metadata_change_rejected(self):
        def mutate():
            info = self.target.stat()
            os.utime(self.target, ns=(info.st_atime_ns, info.st_mtime_ns - 1_000_000_000))
        report = self.inject_after_kblocks(mutate)
        self.assertEqual(report.models[-1].reason,
                         "STABILITY:changed_during_inspection")

    def test_removed_during_inspection_reports_unavailable(self):
        report = self.inject_after_kblocks(self.target.unlink)
        self.assertEqual((report.status, report.models[-1].reason),
                         ("invalid", "STABILITY:unavailable"))

    def test_replaced_by_symlink_during_inspection_rejected(self):
        def replace():
            other = self.root / "outside.gguf"
            other.write_bytes(self.target.read_bytes())
            self.target.unlink()
            self.target.symlink_to(other)
        try:
            with TemporaryDirectory() as probe:
                link = Path(probe) / "link"
                link.symlink_to(self.target)
                link.unlink()
        except (OSError, NotImplementedError):
            self.skipTest("symlinks not supported")
        report = self.inject_after_kblocks(replace)
        self.assertEqual(report.status, "invalid")
        self.assertEqual(report.models[-1].reason, "STABILITY:unavailable")

    def test_unknown_layout_still_size_unverified_when_stable(self):
        self.target.write_bytes(_synthetic_gguf(version=3, order="<",
                                               kind=16, width=256, size=144))
        report = self.run_summary()
        self.assertEqual(report.status, "size_unverified")
        self.assertEqual(report.models[-1].reason, "KBLOCKS:size_unverified")

    def test_stable_malformed_file_retains_original_reason(self):
        with self.target.open("r+b") as stream:
            stream.write(b"NOPE")
        report = self.run_summary()
        self.assertEqual(report.status, "invalid")
        self.assertEqual(report.models[-1].reason, "GGUF:invalid_magic")

    def test_unavailable_initial_snapshot_is_not_false_success(self):
        with patch.object(summary, "_fingerprint", return_value=None) as inspect:
            report = self.run_summary()
        self.assertEqual(report.status, "invalid")
        self.assertEqual(report.models[-1].reason, "STABILITY:unavailable")
        self.assertEqual(inspect.call_count, 3)


if __name__ == "__main__":
    unittest.main()
