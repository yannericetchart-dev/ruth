"""Synthetic isolation scenarios; never examine the actual host environment."""
from dataclasses import FrozenInstanceError
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import environment
from environment.python import inspect
from environment.runtime.contracts import PythonEnvironment
import runtime


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.venv = self.root/'.venv'
        self.venv.mkdir()
        self.cfg = self.venv/'pyvenv.cfg'
        self.cfg.write_text('include-system-site-packages = false\n')
        self.observations = dict(prefix=self.venv, base_prefix=self.root/'base',
                                 interpreter='synthetic-python', python_version=(3, 12, 0))

    def report(self):
        return inspect(self.root, **self.observations)

    def test_isolated_project_interpreter(self):
        report = self.report()
        self.assertTrue(report.interpreter_in_project_venv)
        self.assertEqual(report.venv_status, 'ok')
        self.assertEqual(report.reasons, ())
        self.assertEqual(report.to_dict()['interpreter'], 'synthetic-python')
        with self.assertRaises(FrozenInstanceError):
            report.venv_status = 'changed'

    def test_global_interpreter_is_not_project_venv(self):
        self.observations['prefix'] = self.observations['base_prefix']
        report = self.report()
        self.assertFalse(report.interpreter_in_project_venv)
        self.assertEqual(set(report.reasons), {'prefix_mismatch', 'not_virtualenv'})

    def test_other_virtualenv_is_not_project_venv(self):
        self.observations['prefix'] = self.root/'other-venv'
        self.assertIn('prefix_mismatch', self.report().reasons)

    def test_shared_site_packages_are_rejected_case_insensitively(self):
        self.cfg.write_text('INCLUDE-SYSTEM-SITE-PACKAGES = TRUE\n')
        self.assertIn('shared_site_packages', self.report().reasons)

    def test_missing_configuration_is_rejected(self):
        self.cfg.unlink()
        self.assertIn('configuration_missing', self.report().reasons)

    def test_empty_ambiguous_or_invalid_configuration_is_rejected(self):
        for text in ('', 'home = synthetic\n', 'include-system-site-packages = maybe\n',
                     'include-system-site-packages = false\ninclude-system-site-packages = true\n'):
            with self.subTest(text=text):
                self.cfg.write_text(text)
                self.assertIn('configuration_invalid', self.report().reasons)

    def test_unreadable_configuration_fails_closed(self):
        with patch.object(Path, 'read_text', side_effect=PermissionError('synthetic')):
            self.assertEqual(self.report().reasons, ('inspection_failed',))

    def test_malformed_repeated_isolation_setting_is_rejected(self):
        for repeated in ('include-system-site-packages =\n',
                         'include-system-site-packages = false extra\n',
                         'include-system-site-packages\n'):
            with self.subTest(repeated=repeated):
                self.cfg.write_text('include-system-site-packages = false\n' + repeated)
                report = self.report()
                self.assertFalse(report.interpreter_in_project_venv)
                self.assertIn('configuration_invalid', report.reasons)

    def test_invalid_encoding_fails_closed(self):
        self.cfg.write_bytes(b'\xff')
        self.assertIn('inspection_failed', self.report().reasons)

    def test_unsupported_python_is_rejected(self):
        self.observations['python_version'] = (3, 9, 0)
        self.assertIn('python_too_old', self.report().reasons)

    def test_symlinked_venv_is_rejected(self):
        self.venv.rename(self.root/'actual-venv')
        self.venv.symlink_to(self.root/'actual-venv', target_is_directory=True)
        self.assertIn('venv_symlink', self.report().reasons)

    def test_parent_apis_preserve_common_contract(self):
        expected = self.report()
        with patch('environment.python.inspect', return_value=expected) as child:
            self.assertIs(environment.inspect(self.root), expected)
            self.assertIs(runtime.inspect_environment(self.root), expected)
            self.assertEqual(child.call_count, 2)
        self.assertIs(environment.PythonEnvironment, PythonEnvironment)
        self.assertIs(runtime.PythonEnvironment, PythonEnvironment)

    def test_public_entrypoints_execute_same_real_check(self):
        with patch('environment.python.runtime.inspection.sys.prefix', str(self.venv)), \
             patch('environment.python.runtime.inspection.sys.base_prefix', str(self.root/'base')), \
             patch('environment.python.runtime.inspection.sys.executable', 'synthetic-python'), \
             patch('environment.python.runtime.inspection.sys.version_info', (3, 12, 0)):
            for entrypoint in (inspect, environment.inspect, runtime.inspect_environment):
                with self.subTest(entrypoint=entrypoint.__module__):
                    self.assertEqual(entrypoint(self.root), self.report())
                    self.assertIsInstance(entrypoint(self.root), PythonEnvironment)
            self.cfg.write_text('include-system-site-packages = true\n')
            for entrypoint in (inspect, environment.inspect, runtime.inspect_environment):
                with self.subTest(entrypoint=entrypoint.__module__):
                    self.assertFalse(entrypoint(self.root).interpreter_in_project_venv)
                    self.assertIn('shared_site_packages', entrypoint(self.root).reasons)

    def test_inspection_does_not_write_configuration(self):
        before = self.cfg.read_bytes()
        self.report()
        self.assertEqual(self.cfg.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
