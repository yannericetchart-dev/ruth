"""Official Ruth entrypoint: synthetic end-to-end orchestration tests."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import main
from environment.runtime.contracts import DependencyPlan, PackageState, PythonEnvironment


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


class EntrypointTests(unittest.TestCase):
    def check(self, good: bool, reasons: tuple[str, ...], expected_code: int):
        with TemporaryDirectory() as directory:
            python = python_report(str(Path(directory)), good, reasons)
            output = StringIO()
            with (
                patch("main.inspect_environment", return_value=python) as inspect,
                patch("main.plan_dependencies", return_value=dep_report(python)) as plan,
                redirect_stdout(output),
            ):
                status = main.main()
            self.assertEqual(status, expected_code)
            inspect.assert_called_once()
            self.assertEqual(inspect.call_args.args[0], Path(main.__file__).resolve().parent)
            printed = output.getvalue()
            self.assertIn("sequenza automatica", printed)
            self.assertIn("avvio controllo ambiente Python", printed)
            if good:
                plan.assert_called_once_with(main.DEFAULT_CHECK_PACKAGES, project_root=Path(main.__file__).resolve().parent)
                self.assertIn("ambiente Python conforme", printed)
                self.assertIn("pacchetto pip: present_unverified", printed)
                self.assertLess(printed.index("avvio controllo ambiente Python"), printed.index("avvio controllo dipendenze Python"))
            else:
                plan.assert_not_called()
                self.assertIn("controllo dipendenze saltato", printed)
                for reason in reasons:
                    self.assertIn(f"motivo: {reason}", printed)

    def test_ok_runs_python_then_dependencies(self):
        self.check(True, (), 0)

    def test_nonconforming_stops_before_package_metadata(self):
        self.check(False, ("configuration_missing", "prefix_mismatch"), 1)

    def test_missing_default_dependency_reports_failure(self):
        python = python_report("synthetic", True, ())
        buf = StringIO()
        with (
            patch("main.inspect_environment", return_value=python),
            patch("main.plan_dependencies", return_value=dep_report(python, "missing")),
            redirect_stdout(buf),
        ):
            code = main.main()
        self.assertEqual(code, 1)
        self.assertIn("pacchetto pip: missing", buf.getvalue())
        self.assertIn("sequenza controlli terminata con problemi", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
