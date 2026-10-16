"""Synthetic metadata-only dependency tests; no host inventory."""
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

from environment.dependencies import plan
from environment.runtime.contracts import PythonEnvironment


def python_report(ok: bool = True) -> PythonEnvironment:
    return PythonEnvironment(
        project_root="synthetic", interpreter="synthetic",
        python_version=(3, 12, 0), interpreter_in_project_venv=ok,
        venv_status="ok" if ok else "invalid_or_inactive",
        reasons=() if ok else ("prefix_mismatch",),
    )


class DependencyTests(TestCase):
    def test_present_without_exact_version(self):
        look = Mock(return_value="24.0")
        result = plan(["pip"], python_report(), version_lookup=look)
        self.assertEqual(result.packages[0].status, "present_unverified")
        self.assertTrue(result.all_declared_present)
        look.assert_called_once_with("pip")

    def test_exact_version_matches(self):
        result = plan(["pip==24.0"], python_report(), version_lookup=lambda _: "24.0")
        self.assertEqual(result.packages[0].status, "present_unverified")
        self.assertEqual(result.packages[0].installed_version, "24.0")

    def test_exact_version_mismatch(self):
        result = plan(["pip==23.0"], python_report(), version_lookup=lambda _: "24.0")
        self.assertEqual(result.packages[0].status, "incompatible")
        self.assertFalse(result.all_declared_present)

    def test_package_missing(self):
        def missing(_):
            raise metadata.PackageNotFoundError("synthetic")
        result = plan(["synthetic"], python_report(), version_lookup=missing)
        self.assertEqual(result.packages[0].status, "missing")
        self.assertIsNone(result.packages[0].installed_version)

    def test_inspection_failure_is_explicit(self):
        result = plan(["synthetic"], python_report(), version_lookup=lambda _: 1 / 0)
        self.assertEqual(result.packages[0].status, "inspection_failed")

    def test_invalid_python_blocks_all_metadata_lookups(self):
        look = Mock()
        result = plan(["pip", "other==1"], python_report(False), version_lookup=look)
        self.assertEqual([p.status for p in result.packages], ["blocked", "blocked"])
        self.assertFalse(result.all_declared_present)
        look.assert_not_called()

    def test_duplicate_normalized_name_rejected(self):
        with self.assertRaisesRegex(ValueError, "ripetuto"):
            plan(["my_pkg", "my.pkg"], python_report(), version_lookup=Mock())

    def test_version_range_not_supported(self):
        for spec in ["pkg>=1", "pkg~=1", "pkg<2", "pkg==", "pkg==1;extra"]:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                plan([spec], python_report(), version_lookup=Mock())

    def test_invalid_empty_list_rejected(self):
        with self.assertRaises(ValueError):
            plan([], python_report())

    def test_scalar_string_is_not_a_list(self):
        with self.assertRaises(ValueError):
            plan("pip", python_report())

    def test_to_dict_is_in_memory(self):
        report = plan(["pip"], python_report(), version_lookup=lambda _: "24.0")
        data = report.to_dict()
        self.assertEqual(data["packages"][0]["status"], "present_unverified")
        self.assertIn("python", data)

    def test_environment_facade_preserves_python_gate(self):
        import environment
        with patch("environment.runtime.inspect", return_value=python_report(False)) as inspect:
            result = environment.plan(["pip"], project_root=Path("synthetic"))
        self.assertFalse(result.all_declared_present)
        self.assertEqual(result.packages[0].status, "blocked")
        inspect.assert_called_once_with(Path("synthetic"))
