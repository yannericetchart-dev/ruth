"""Official Ruth entrypoint: synthetic end-to-end orchestration tests."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import unittest
from unittest.mock import patch

import main
from environment.runtime.contracts import DependencyPlan, PackageState, PythonEnvironment
from hardware import DeviceInfo, ResourceReport, HardwareReport


def python_report(root: str, good: bool, reasons: tuple[str, ...]) -> PythonEnvironment:
    return PythonEnvironment(
        project_root=root, interpreter="synthetic",
        python_version=(3, 12, 0), interpreter_in_project_venv=good,
        venv_status="ok" if good else "invalid_or_inactive",
        reasons=reasons,
    )


def dep_report(python: PythonEnvironment, status: str = "present_unverified") -> DependencyPlan:
    return DependencyPlan(python, (
        PackageState("pip", "pip", "24.0" if status == "present_unverified" else None, status),
    ), python.interpreter_in_project_venv and status == "present_unverified")


def hw_report():
    return HardwareReport((DeviceInfo("cpu", "synthetic", 4, "available"),),
                          ResourceReport(4096, "available"), "available")


class EntrypointTests(unittest.TestCase):
    def test_ok_runs_python_then_dependencies_then_diagnostics(self):
        py = python_report("synthetic", True, ())
        out = StringIO()
        with (
            patch("main.inspect_environment", return_value=py) as inspect,
            patch("main.plan_dependencies", return_value=dep_report(py)) as plan,
            patch("main.detect_hardware", return_value=hw_report()) as hw,
            redirect_stdout(out),
        ):
            result = main.main()
        self.assertEqual(result, 0)
        inspect.assert_called_once_with(Path(main.__file__).resolve().parent)
        plan.assert_called_once_with(main.DEFAULT_CHECK_PACKAGES, project_root=Path(main.__file__).resolve().parent)
        printed = out.getvalue()
        self.assertLess(printed.index("avvio controllo ambiente Python"),
                        printed.index("avvio controllo dipendenze Python"))
        self.assertLess(printed.index("avvio controllo dipendenze Python"),
                        printed.index("avvio diagnostica ambiente"))
        self.assertIn("diagnostica: ok_unverified", printed)
        self.assertLess(printed.index("avvio diagnostica ambiente"), printed.index("avvio inventario hardware"))
        hw.assert_called_once_with()
        self.assertIn("RAM fisica totale (byte): 4096", printed)
        self.assertIn("sequenza controlli completata", printed)

    def test_invalid_python_still_diagnosed_but_packages_skipped(self):
        py = python_report("synthetic", False, ("configuration_missing", "prefix_mismatch"))
        out = StringIO()
        with (
            patch("main.inspect_environment", return_value=py),
            patch("main.plan_dependencies") as plan,
            patch("main.detect_hardware", return_value=hw_report()) as hw,
            redirect_stdout(out),
        ):
            result = main.main()
        self.assertEqual(result, 1)
        plan.assert_not_called()
        text = out.getvalue()
        self.assertIn("controllo dipendenze saltato", text)
        self.assertLess(text.index("controllo dipendenze saltato"),
                        text.index("avvio diagnostica ambiente"))
        self.assertIn("venv_configuration_missing", text)
        self.assertIn("venv_prefix_mismatch", text)
        self.assertLess(text.index("avvio diagnostica ambiente"), text.index("avvio inventario hardware"))
        hw.assert_called_once_with()

    def test_missing_dependency_explained(self):
        py = python_report("synthetic", True, ())
        out = StringIO()
        with (
            patch("main.inspect_environment", return_value=py),
            patch("main.plan_dependencies", return_value=dep_report(py, "missing")),
            patch("main.detect_hardware", return_value=hw_report()) as hw,
            redirect_stdout(out),
        ):
            result = main.main()
        self.assertEqual(result, 1)
        self.assertIn("pacchetto pip: missing", out.getvalue())
        self.assertIn("package_missing", out.getvalue())
        self.assertIn("sequenza controlli terminata con problemi", out.getvalue())
        hw.assert_called_once_with()

    def test_failed_metadata_lookup_explained(self):
        py = python_report("synthetic", True, ())
        out = StringIO()
        with (
            patch("main.inspect_environment", return_value=py),
            patch("main.plan_dependencies", return_value=dep_report(py, "inspection_failed")),
            patch("main.detect_hardware", return_value=hw_report()) as hw,
            redirect_stdout(out),
        ):
            result = main.main()
        self.assertEqual(result, 1)
        self.assertIn("diagnostica: inspection_incomplete", out.getvalue())
        hw.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
