"""MIG-013: read-only aggregate structural verdicts for catalogued GGUF."""
from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest

import main
from models.formats.gguf import inspect_structural_summaries
from models.formats.gguf.runtime.structural_summary import (
    GGUFStructuralReport, GGUFStructuralSummary,
)
from models.formats.gguf.runtime.fixture import (
    _create_fixture_catalog, _synthetic_gguf,
)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / ".local/models"

    def make(self):
        _create_fixture_catalog(self.root)

    def test_absent_empty_and_unrelated_catalog(self):
        row = inspect_structural_summaries(project_root=self.root)
        self.assertEqual((row.status, row.reason, row.count),
                         ("not_configured", "CAT:not_configured", 0))
        self.folder.mkdir(parents=True)
        manifest = self.folder / "catalog.json"
        manifest.write_text('{"models":[]}', encoding="utf-8")
        self.assertEqual(inspect_structural_summaries(project_root=self.root).status,
                         "not_configured")
        manifest.write_text(json.dumps({"models": [
            {"id":"other", "format":"safetensors", "path":"absent.bin"}
        ]}), encoding="utf-8")
        self.assertEqual(inspect_structural_summaries(project_root=self.root).reason,
                         "CAT:no_gguf")

    def test_scalar_standard_quant_and_kquant_all_verified(self):
        self.make()
        report = inspect_structural_summaries(project_root=self.root)
        self.assertEqual((report.status, report.count), ("structural_checked", 3))
        self.assertEqual(tuple(x.status for x in report.models),
                         ("structural_checked",) * 3)
        self.assertEqual(tuple(x.tensor_count for x in report.models), (1, 1, 1))
        self.assertEqual(tuple(x.size_unverified for x in report.models), (0, 0, 0))
        self.assertEqual(tuple(x.model_id for x in report.models),
                         ("scalar", "q4", "kquant"))

    def test_unknown_layout_not_misreported_as_verified(self):
        self.make()
        (self.folder / "kquant.gguf").write_bytes(
            _synthetic_gguf(version=3, order="<", kind=16, width=256, size=144))
        report = inspect_structural_summaries(project_root=self.root)
        self.assertEqual(report.status, "size_unverified")
        self.assertEqual(report.models[-1].status, "size_unverified")
        self.assertEqual(report.models[-1].reason, "KBLOCKS:size_unverified")
        self.assertEqual(report.models[-1].size_unverified, 1)
        self.assertEqual(tuple(row.status for row in report.models[:2]),
                         ("structural_checked", "structural_checked"))

    def test_invalid_header_precedes_uncertified_size(self):
        self.make()
        target = self.folder / "kquant.gguf"
        data = target.read_bytes()
        target.write_bytes(b"NOPE" + data[4:])
        report = inspect_structural_summaries(project_root=self.root)
        self.assertEqual(report.status, "invalid")
        self.assertEqual(report.models[-1].reason, "GGUF:invalid_magic")

    def test_truncated_extent_not_size_unverified(self):
        self.make()
        target = self.folder / "kquant.gguf"
        with target.open("r+b") as handle:
            handle.truncate(target.stat().st_size - 1)
        report = inspect_structural_summaries(project_root=self.root)
        self.assertEqual(report.status, "invalid")
        self.assertEqual(report.models[-1].reason,
                         "KBLOCKS:tensor_extent_out_of_file")

    def test_invalid_catalog_and_missing_declared_file(self):
        self.make()
        target = self.folder / "scalar.gguf"
        target.unlink()
        report = inspect_structural_summaries(project_root=self.root)
        self.assertEqual(report.status, "invalid")
        self.assertEqual(report.models[0].reason, "CAT:missing")
        (self.folder / "catalog.json").write_text(
            json.dumps({"models": [
                {"id":"unsafe", "format":"gguf", "path":"../escape.gguf"}
            ]}), encoding="utf-8")
        bad = inspect_structural_summaries(project_root=self.root)
        self.assertEqual((bad.status, bad.reason, bad.count),
                         ("invalid", "CAT:invalid_catalog", 0))

    def test_immutable_portable_no_tensor_data(self):
        self.make()
        report = inspect_structural_summaries(project_root=self.root)
        self.assertNotIn("fixture.tensor", repr(report))
        self.assertNotIn("general.architecture", repr(report))
        self.assertFalse(hasattr(report.models[0], "reference"))
        with self.assertRaises(FrozenInstanceError):
            report.models[0].status = "invalid"

    def test_targeted_cli_success_and_no_catalog(self):
        with patch("main.inspect_structural_summaries",
                   return_value=GGUFStructuralReport("structural_checked", (
                       GGUFStructuralSummary("sample", "structural_checked", tensor_count=2),
                   ))) as inspect:
            out = StringIO()
            with redirect_stdout(out):
                exit_code = main.main(("gguf-summary",))
        self.assertEqual(exit_code, 0)
        self.assertIn("riepilogo GGUF: structural_checked", out.getvalue())
        self.assertIn("2 descrittori", out.getvalue())
        inspect.assert_called_once_with(project_root=Path(main.__file__).resolve().parent)

    def test_targeted_cli_unknown_or_invalid_returns_nonzero(self):
        for state, result in (
            ("size_unverified", GGUFStructuralReport("size_unverified", (
                GGUFStructuralSummary("synthetic", "size_unverified",
                                      "KBLOCKS:size_unverified", 1, 1),))),
            ("invalid", GGUFStructuralReport("invalid", reason="CAT:invalid_catalog")),
        ):
            with self.subTest(state=state), patch(
                "main.inspect_structural_summaries", return_value=result
            ):
                out = StringIO()
                with redirect_stdout(out):
                    code = main.main(("gguf-summary",))
                self.assertEqual(code, 1)
                self.assertIn(state, out.getvalue())

    def test_targeted_cli_not_configured_is_valid_skip(self):
        with patch("main.inspect_structural_summaries",
                   return_value=GGUFStructuralReport("not_configured",
                                                     reason="CAT:not_configured")):
            out = StringIO()
            with redirect_stdout(out):
                code = main.main(("gguf-summary",))
        self.assertEqual(code, 0)
        self.assertIn("not_configured", out.getvalue())

    def test_default_summary_follows_fixture(self):
        # Verify the step ordering in the actual main routine, without probing
        # host hardware, Python environment or personal catalog.
        steps = []
        python = type("SyntheticPython", (), {"interpreter_in_project_venv": True})()
        with patch("main._check_python", return_value=python), \
             patch("main._check_dependencies", return_value=(0, None)), \
             patch("main._diagnose"), patch("main._check_hardware"), \
             patch("main._check_gpu"), patch("main._check_catalog", return_value=0), \
             patch("main._check_gguf", return_value=0), \
             patch("main._check_gguf_metadata", return_value=0), \
             patch("main._check_gguf_index", return_value=0), \
             patch("main._check_gguf_blocks", return_value=0), \
             patch("main._check_gguf_kblocks", return_value=0), \
             patch("main._check_fixture", side_effect=lambda: steps.append("fixture") or 0), \
             patch("main._check_gguf_summary", side_effect=lambda _: steps.append("summary") or 0):
            with redirect_stdout(StringIO()):
                code = main.main()
        self.assertEqual((code, steps), (0, ["fixture", "summary"]))


if __name__ == "__main__":
    unittest.main()
