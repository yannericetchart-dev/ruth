"""End-to-end checks through the official Ruth entrypoint, with synthetic host observations."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import main
from environment.runtime.contracts import PythonEnvironment


class EntrypointTests(unittest.TestCase):
    def check(self, good: bool, reasons: tuple[str, ...], expected_code: int):
        with tempfile.TemporaryDirectory() as directory:
            report = PythonEnvironment(
                project_root=str(Path(directory)), interpreter="synthetic",
                python_version=(3, 12, 0), interpreter_in_project_venv=good,
                venv_status="ok" if good else "invalid_or_inactive", reasons=reasons,
            )
            output = StringIO()
            with patch("main.inspect_environment", return_value=report) as inspect, redirect_stdout(output):
                status = main.main()
            self.assertEqual(status, expected_code)
            self.assertEqual(inspect.call_count, 1)
            self.assertIn("avvio controllo ambiente Python", output.getvalue())
            if good:
                self.assertIn("ambiente Python conforme", output.getvalue())
            else:
                self.assertIn("ambiente Python non conforme", output.getvalue())
                for reason in reasons:
                    self.assertIn(f"motivo: {reason}", output.getvalue())

    def test_ok(self):
        self.check(True, (), 0)

    def test_nonconforming(self):
        self.check(False, ("configuration_missing", "prefix_mismatch"), 1)


if __name__ == "__main__":
    unittest.main()
