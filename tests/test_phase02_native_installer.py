"""No-network/no-native-GPU tests for the staged Ruth .venv activation."""
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from ruth_runtime.native_installer import (
    NATIVE_LOCK, PTH_FILE, RUNTIME_FOLDER,
    inspect_native_install_inputs, install_native_cuda,
)
from ruth_runtime.native_builder import BUILD_OUTPUT, BUILD_ROOT
from ruth_runtime.install_plan import SOURCE_COMMIT, LLAMA_CPP_SUBMODULE_COMMIT

HASH = "a" * 64


def fixture(root):
    base = root / ".venv"
    (base / "bin").mkdir(parents=True)
    (base / "lib/python3.12/site-packages").mkdir(parents=True)
    (root / "locks").mkdir()
    (root / NATIVE_LOCK).write_text(json.dumps({
        "schema": "ruth-native-wheel-lock-v1",
        "source_commit": SOURCE_COMMIT, "llama_cpp_commit": LLAMA_CPP_SUBMODULE_COMMIT,
        "python": "3.12", "cuda_architecture": "89",
        "wheel_filename": "llama_cpp_python-0.3.36-py3-none-linux_x86_64.whl",
        "wheel_sha256": HASH,
    }))
    dest = root / BUILD_ROOT / BUILD_OUTPUT
    wheels = dest / "wheels"
    wheels.mkdir(parents=True)
    (wheels / "llama_cpp_python-0.3.36-py3-none-linux_x86_64.whl").write_bytes(b"mock wheel")
    (dest / "build-result.json").write_text(json.dumps({
        "wheel_file": "llama_cpp_python-0.3.36-py3-none-linux_x86_64.whl",
        "wheel_sha256": HASH, "source_commit": SOURCE_COMMIT,
        "llama_cpp_commit": LLAMA_CPP_SUBMODULE_COMMIT,
        "cuda_architecture": "89", "ggml_cuda_library_packaged": True,
        "installed_in_ruth_venv": False,
    }))
    candidate = root / ".ruth-wheelhouse/python-dependencies-candidate"
    target = candidate / "wheels"
    target.mkdir(parents=True)
    rows = []
    names = ("diskcache", "jinja2", "markupsafe", "numpy",
             "packaging", "pathspec", "scikit-build-core", "typing-extensions")
    for name in names:
        filename = name + "-1.0-py3-none-any.whl"
        (target / filename).write_bytes(b"mock")
        rows.append({"filename": filename})
    (candidate / "candidate-lock.json").write_text(json.dumps({
        "approved": False, "files": rows,
    }))
    env = SimpleNamespace(
        interpreter_in_project_venv=True,
        python_version="3.12.3", project_root=str(root),
        interpreter=str(root / ".venv/bin/python"),
        packages=[SimpleNamespace(
            name=n, installed_version=None) for n in (
                "llama-cpp-python", "numpy", "typing-extensions",
                "diskcache", "jinja2", "MarkupSafe", "scikit-build-core",
            )],
    )
    return env


def good_pins(root):
    return True, (), None


def good_wheel(path):
    return True, HASH, None


class PipInstall:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        assert cmd[:4] == [
            str(Path(kwargs["cwd"]) / ".venv/bin/python"),
            "-m", "pip", "--isolated",
        ]
        assert "--target" in cmd
        assert "--no-index" in cmd and "--no-deps" in cmd
        assert "--no-cache-dir" in cmd and "--no-compile" in cmd
        assert "--user" not in cmd
        assert "wheel" not in cmd
        assert kwargs["env"]["PIP_NO_INDEX"] == "1"
        assert len(cmd[cmd.index("--target") + 2:]) == 9
        if not self.fail:
            target = Path(cmd[cmd.index("--target") + 1])
            (target / "llama_cpp").mkdir()
            (target / "llama_cpp/__init__.py").write_text("version='0.3.36'")
        return subprocess.CompletedProcess(cmd, int(self.fail), "", "mock")


class NativeInstallerTests(unittest.TestCase):
    def inspect(self, root, env, **kwargs):
        return inspect_native_install_inputs(
            project_root=root, environment=env, pins_check=good_pins,
            inspect_built=good_wheel, **kwargs,
        )

    def install(self, root, env, **kwargs):
        return install_native_cuda(
            project_root=root, environment=env,
            pins_check=good_pins, inspect_built=good_wheel, **kwargs,
        )

    def test_read_only_preflight_identifies_expected_wheels(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env = fixture(root)
            report, wheels = self.inspect(root, env)
            self.assertEqual(report.status, "ready_for_explicit_install_authorization")
            self.assertEqual(len(wheels), 9)
            self.assertFalse(report.installation_started)
            self.assertFalse(report.installed_in_ruth_venv)
            self.assertFalse(report.gpu_offload_executed)

    def test_no_consent_never_calls_pip(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env = fixture(root)
            report = self.install(
                root, env,
                run=lambda *a, **k: self.fail("pip should not run"),
            )
            self.assertEqual(report.status, "explicit_install_and_risk_consent_required")
            self.assertFalse(report.installation_started)
            self.assertFalse((root / ".venv" / RUNTIME_FOLDER).exists())

    def test_install_consent_alone_is_not_risk_acceptance(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env = fixture(root)
            report = self.install(
                root, env, authorize_install=True,
                run=lambda *a, **k: self.fail("pip should not run"),
            )
            self.assertEqual(report.status, "explicit_install_and_risk_consent_required")

    def test_diskcache_risk_acceptance_alone_is_not_install_consent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env = fixture(root)
            report = self.install(
                root, env, accept_diskcache_risk=True,
                run=lambda *a, **k: self.fail("pip should not run"),
            )
            self.assertEqual(report.status, "explicit_install_and_risk_consent_required")

    def test_success_promotes_checked_files_only_with_two_permissions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env = fixture(root)
            fake = PipInstall()
            checks = []
            def probe(python, location, *, staged, run, root):
                checks.append((staged, location))
                if not staged:
                    self.assertTrue((root / ".venv/lib/python3.12/site-packages" /
                                     PTH_FILE).exists())
                    self.assertTrue((root / ".venv" / RUNTIME_FOLDER).is_dir())
                return True
            report = self.install(
                root, env, authorize_install=True,
                accept_diskcache_risk=True, run=fake, probe=probe,
            )
            self.assertEqual(report.status, "native_cuda_binding_installed")
            self.assertTrue(report.installed_in_ruth_venv)
            self.assertTrue(report.gpu_offload_supported)
            self.assertFalse(report.gpu_offload_executed)
            self.assertFalse(report.inference_executed)
            self.assertFalse(report.fallback_automatically_selected)
            self.assertEqual(len(fake.calls), 1)
            self.assertEqual(len(checks), 2)
            self.assertEqual(checks[0][0], True)
            self.assertEqual(checks[1][0], False)
            self.assertTrue((root / ".venv" / RUNTIME_FOLDER).exists())
            self.assertEqual(
                (root / ".venv/lib/python3.12/site-packages" / PTH_FILE).read_text(),
                str((root / ".venv" / RUNTIME_FOLDER).resolve()) + "\n",
            )
            self.assertFalse((root / ".venv/.ruth-native-install.lock").exists())

    def test_pip_failure_leaves_venv_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env = fixture(root)
            report = self.install(
                root, env, authorize_install=True, accept_diskcache_risk=True,
                run=PipInstall(fail=True),
            )
            self.assertEqual(report.status, "staging_install_failed")
            self.assertFalse(report.installed_in_ruth_venv)
            self.assertFalse((root / ".venv" / RUNTIME_FOLDER).exists())
            self.assertFalse((root / ".venv/lib/python3.12/site-packages" /
                              PTH_FILE).exists())

    def test_failed_staged_probe_does_not_mutate_venv(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env = fixture(root)
            report = self.install(
                root, env, authorize_install=True, accept_diskcache_risk=True,
                run=PipInstall(), probe=lambda *a, **k: False,
            )
            self.assertEqual(report.status, "staged_cuda_probe_failed")
            self.assertFalse((root / ".venv" / RUNTIME_FOLDER).exists())
            self.assertFalse((root / ".venv/lib/python3.12/site-packages" /
                              PTH_FILE).exists())

    def test_failed_post_activation_probe_rolls_back(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env = fixture(root)
            report = self.install(
                root, env, authorize_install=True, accept_diskcache_risk=True,
                run=PipInstall(),
                probe=lambda python, loc, *, staged, run, root: staged,
            )
            self.assertEqual(report.status, "installation_rolled_back")
            self.assertTrue(report.rollback_performed)
            self.assertFalse((root / ".venv" / RUNTIME_FOLDER).exists())
            self.assertFalse((root / ".venv/lib/python3.12/site-packages" /
                              PTH_FILE).exists())

    def test_existing_venv_install_paths_not_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env = fixture(root)
            destination = root / ".venv" / RUNTIME_FOLDER
            destination.mkdir()
            (destination / "sentinel").write_text("preserve")
            report = self.install(
                root, env, authorize_install=True, accept_diskcache_risk=True,
                run=lambda *a, **k: self.fail("never run pip"),
            )
            self.assertEqual(report.status, "occupied_or_unsafe_venv_location")
            self.assertEqual((destination / "sentinel").read_text(), "preserve")

    def test_pinned_native_hash_mismatch_blocks_before_network(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env = fixture(root)
            report, wheels = inspect_native_install_inputs(
                project_root=root, environment=env, pins_check=good_pins,
                inspect_built=lambda x: (True, "b" * 64, None),
            )
            self.assertEqual(report.status, "native_wheel_unverified")
            self.assertEqual(wheels, ())

    def test_preexisting_runtime_blocks_shadowing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env = fixture(root)
            env.packages[0].installed_version = "0.3.36"
            report, wheels = self.inspect(root, env)
            self.assertEqual(report.status, "preexisting_runtime_packages")
            self.assertFalse(report.installation_started)

    def test_no_gpu_execution_or_inference_is_claimed_by_staged_import(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env = fixture(root)
            report = self.install(
                root, env, authorize_install=True, accept_diskcache_risk=True,
                run=PipInstall(), probe=lambda *a, **k: True,
            )
            self.assertTrue(report.gpu_offload_supported)
            self.assertFalse(report.gpu_offload_executed)
            self.assertFalse(report.inference_executed)


if __name__ == "__main__":
    unittest.main()
