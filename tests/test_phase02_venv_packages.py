"""Offline tests for Ruth's project-only Python package inventory."""
from importlib import metadata
from pathlib import Path
import tempfile
import unittest

from ruth_runtime.venv_packages import (
    LOCK_PATH, REQUIREMENTS, inspect_python_environment,
)


class PythonEnvironmentTests(unittest.TestCase):
    def inspect(self, root, *, prefix_ok=True, providers=None, base=None):
        venv = root / ".venv"
        venv.mkdir(exist_ok=True)
        (venv / "pyvenv.cfg").write_text(
            "home = /usr/bin\ninclude-system-site-packages = false\n"
        )
        calls = []
        versions = providers or {}
        def provider(name):
            calls.append(name)
            if name not in versions:
                raise metadata.PackageNotFoundError(name)
            return versions[name]
        report = inspect_python_environment(
            project_root=root,
            prefix=venv if prefix_ok else root,
            base_prefix=base or Path("/usr"),
            interpreter=str(venv / "bin" / "python"),
            python_version=(3, 12, 3),
            package_version=provider,
        )
        return report, calls

    def test_external_python_must_not_inspect_host_packages(self):
        with tempfile.TemporaryDirectory() as td:
            report, calls = self.inspect(Path(td), prefix_ok=False)
            self.assertEqual(report.venv_status, "invalid_or_inactive")
            self.assertEqual(calls, [])
            self.assertFalse(report.ruth_runtime_ready)
            self.assertTrue(all(p.status == "blocked" for p in report.packages))

    def test_project_venv_reports_missing_dependencies_not_fake_ready(self):
        with tempfile.TemporaryDirectory() as td:
            report, calls = self.inspect(Path(td))
            self.assertTrue(report.interpreter_in_project_venv)
            self.assertEqual(len(report.packages), len(REQUIREMENTS))
            self.assertIn("llama-cpp-python", report.missing_runtime)
            self.assertIn("scikit-build-core", report.missing_build)
            self.assertFalse(report.runtime_dependencies_present)
            self.assertFalse(report.runtime_dependencies_verified)
            self.assertFalse(report.python_packages_installable)
            self.assertFalse(report.backend_gpu_verified)
            self.assertEqual(report.lock_status, "missing")
            self.assertEqual(len(calls), len(REQUIREMENTS))

    def test_installed_metadata_not_evidence_of_cuda_or_lock(self):
        with tempfile.TemporaryDirectory() as td:
            versions = {x[0]: ("0.3.36" if x[0] == "llama-cpp-python" else "99.0")
                        for x in REQUIREMENTS}
            report, _ = self.inspect(Path(td), providers=versions)
            self.assertTrue(report.runtime_dependencies_present)
            self.assertFalse(report.runtime_dependencies_verified)
            self.assertFalse(report.ruth_runtime_ready)
            self.assertTrue(all(p.status == "present_unverified" for p in report.packages))

    def test_incorrect_binding_version_is_incompatible(self):
        with tempfile.TemporaryDirectory() as td:
            versions = {x[0]: "0.0" for x in REQUIREMENTS}
            report, _ = self.inspect(Path(td), providers=versions)
            binding = next(p for p in report.packages if p.name == "llama-cpp-python")
            self.assertEqual(binding.status, "incompatible")
            self.assertIn("llama-cpp-python", report.missing_runtime)

    def test_global_site_packages_venv_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            report, _ = self.inspect(root)
            (root / ".venv" / "pyvenv.cfg").write_text(
                "home = /usr/bin\ninclude-system-site-packages = true\n"
            )
            def forbidden(_):
                self.fail("Must not consult exposed host site-packages")
            blocked = inspect_python_environment(
                project_root=root, prefix=root / ".venv",
                base_prefix=Path("/usr"), package_version=forbidden,
            )
            self.assertFalse(blocked.interpreter_in_project_venv)
            self.assertFalse(blocked.ruth_runtime_ready)

    def test_lock_filename_is_not_an_authorized_lock(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / LOCK_PATH).parent.mkdir()
            (root / LOCK_PATH).write_text('{"untrusted":true}')
            report, _ = self.inspect(root)
            self.assertEqual(report.lock_status, "unreviewed")
            self.assertFalse(report.python_packages_installable)

    def test_missing_venv_cfg_rejects_without_metadata_reads(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            def forbidden(_):
                self.fail("No installed metadata allowed from unverified environment")
            r = inspect_python_environment(
                project_root=root, prefix=root / ".venv",
                base_prefix=Path("/usr"), package_version=forbidden,
            )
            self.assertFalse(r.interpreter_in_project_venv)

    def test_environment_json_has_explicit_readiness_gates(self):
        with tempfile.TemporaryDirectory() as td:
            import json
            r, _ = self.inspect(Path(td))
            j = json.loads(json.dumps(r.to_dict()))
            self.assertEqual(j["runtime_dependencies_verified"], False)
            self.assertEqual(j["ruth_runtime_ready"], False)
            self.assertIn("missing_runtime", j)


if __name__ == "__main__":
    unittest.main()
