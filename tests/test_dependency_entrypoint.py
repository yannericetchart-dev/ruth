"""Official Ruth CLI dependency mode, without inspecting any real packages."""
from contextlib import redirect_stdout
from io import StringIO
from unittest import TestCase
from unittest.mock import patch

import main
from environment.runtime.contracts import DependencyPlan, PackageState
from environment.dependencies.tests.test_plan import python_report


class DependencyEntrypointTests(TestCase):
    def run_mode(self, report):
        out = StringIO()
        with patch("main.plan_dependencies", return_value=report) as planned, redirect_stdout(out):
            code = main.main(["dependencies", "pip"])
        self.assertEqual(planned.call_count, 1)
        self.assertEqual(planned.call_args.args, (["pip"],))
        return code, out.getvalue()

    def test_present(self):
        report = DependencyPlan(python_report(), (
            PackageState("pip", "pip", "24.0", "present_unverified"),
        ), True)
        code, output = self.run_mode(report)
        self.assertEqual(code, 0)
        self.assertIn("avvio controllo dipendenze Python", output)
        self.assertIn("pacchetto pip: present_unverified", output)

    def test_missing_is_nonzero(self):
        report = DependencyPlan(python_report(), (
            PackageState("pip", "pip", None, "missing"),
        ), False)
        code, output = self.run_mode(report)
        self.assertEqual(code, 1)
        self.assertIn("pacchetto pip: missing", output)

    def test_bad_spec_is_usage_error(self):
        buf = StringIO()
        with patch("main.plan_dependencies", side_effect=ValueError("bad")), redirect_stdout(buf):
            code = main.main(["dependencies", "pkg>=1"])
        self.assertEqual(code, 2)
        self.assertIn("requisito non valido", buf.getvalue())

    def test_bad_mode_is_usage_error(self):
        with redirect_stdout(StringIO()):
            self.assertEqual(main.main(["unknown"]), 2)
            self.assertEqual(main.main(["dependencies"]), 2)

    def test_blocked_environment_explains_reason(self):
        report = DependencyPlan(python_report(False), (
            PackageState("pip", "pip", None, "blocked"),
        ), False)
        code, output = self.run_mode(report)
        self.assertEqual(code, 1)
        self.assertIn("controllo pacchetti bloccato", output)
        self.assertIn("prefix_mismatch", output)
