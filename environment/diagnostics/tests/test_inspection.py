"""Synthetic DIAG-001 tests: existing reports only, no host I/O."""
import unittest

from environment.diagnostics import inspect
from environment.runtime.contracts import (
    DependencyPlan, DiagnosticReport, PackageState, PythonEnvironment,
)


def python_report(ok=True, reasons=()):
    return PythonEnvironment(
        "/synthetic", "synthetic", (3, 12, 0), ok,
        "ok" if ok else "invalid_or_inactive", tuple(reasons),
    )


def packages(python, *entries):
    states = tuple(PackageState(name, name, None, status) for name, status in entries)
    return DependencyPlan(python, states, python.interpreter_in_project_venv and
                          all(x.status == "present_unverified" for x in states))


class DiagnosticsTests(unittest.TestCase):
    def test_compliant_python_and_present_unverified_not_certified(self):
        py = python_report()
        report = inspect(py, packages(py, ("pip", "present_unverified")))
        self.assertIsInstance(report, DiagnosticReport)
        self.assertEqual(report.status, "ok_unverified")
        self.assertFalse(report.issues)

    def test_python_invalid_without_dependency_report(self):
        report = inspect(python_report(False, ("prefix_mismatch", "not_virtualenv")))
        self.assertEqual(report.status, "attention_required")
        self.assertEqual([p.code for p in report.issues],
                         ["venv_prefix_mismatch", "python_not_virtualenv"])

    def test_all_known_python_reasons(self):
        reasons = ("venv_symlink", "configuration_missing", "configuration_invalid",
                   "shared_site_packages", "prefix_mismatch", "not_virtualenv",
                   "inspection_failed", "python_too_old")
        report = inspect(python_report(False, reasons))
        self.assertEqual(len(report.issues), len(reasons))
        self.assertEqual(report.status, "inspection_incomplete")
        self.assertTrue(all(p.suggested_action for p in report.issues))

    def test_unknown_reason_is_not_quoted_back(self):
        report = inspect(python_report(False, ("/private/path",)))
        self.assertEqual(report.issues[0].code, "python_problem_unclassified")
        self.assertNotIn("/private/path", str(report.to_dict()))

    def test_duplicate_reason_does_not_duplicate_issue(self):
        report = inspect(python_report(False, ("prefix_mismatch", "prefix_mismatch")))
        self.assertEqual(len(report.issues), 1)

    def test_no_reason_on_nonconforming_report(self):
        report = inspect(python_report(False))
        self.assertEqual(report.issues[0].code, "python_problem_unclassified")

    def test_missing_package(self):
        py = python_report()
        report = inspect(py, packages(py, ("pkg", "missing")))
        self.assertEqual(report.status, "attention_required")
        self.assertEqual(report.issues[0].code, "package_missing")
        self.assertEqual(report.issues[0].package, "pkg")

    def test_exact_version_mismatch(self):
        py = python_report()
        report = inspect(py, packages(py, ("pkg", "incompatible")))
        self.assertEqual(report.issues[0].code, "package_version_mismatch")

    def test_metadata_inspection_failure(self):
        py = python_report()
        report = inspect(py, packages(py, ("pkg", "inspection_failed")))
        self.assertEqual(report.status, "inspection_incomplete")
        self.assertEqual(report.issues[0].code, "package_inspection_failed")

    def test_package_blocked(self):
        py = python_report(False, ("configuration_missing",))
        report = inspect(py, packages(py, ("pip", "blocked")))
        self.assertEqual([x.code for x in report.issues],
                         ["venv_configuration_missing", "package_check_blocked"])

    def test_unknown_package_status_does_not_guess(self):
        py = python_report()
        report = inspect(py, packages(py, ("pkg", "unknown_native")))
        self.assertEqual(report.issues[0].code, "package_status_unknown")

    def test_reject_mismatched_python_dependency_snapshots(self):
        first = python_report()
        second = python_report(False, ("prefix_mismatch",))
        with self.assertRaises(ValueError):
            inspect(first, packages(second, ("pip", "blocked")))

    def test_report_is_immutable_and_transient(self):
        py = python_report()
        report = inspect(py, packages(py, ("pip", "present_unverified")))
        with self.assertRaises((AttributeError, TypeError)):
            report.status = "changed"
        self.assertEqual(report.to_dict()["issues"], ())


if __name__ == "__main__":
    unittest.main()
