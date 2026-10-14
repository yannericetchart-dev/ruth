"""Offline security gate tests for CVE-2025-69872 mitigation boundaries."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from zipfile import ZipFile

from ruth_runtime.dependency_resolution import inspect_wheel
from ruth_runtime.install_plan import SOURCE_COMMIT, LLAMA_CPP_SUBMODULE_COMMIT
from ruth_runtime.security_gate import inspect_upstream_cache_policy, review_diskcache_risk


LLAMA_MINIMAL = """
class Llama:
    def __init__(self):
        self.cache = None
    def set_cache(self, cache):
        self.cache = cache
"""
DISK_MINIMAL = """
class LlamaDiskCache:
    def __init__(self, cache_dir):
        self.cache = diskcache.Cache(cache_dir)
"""


def fake_run(cmd, **kwargs):
    assert cmd[1:4] == ["-m", "unittest", "discover"]
    assert cmd[-2:] == ["-p", "test_phase02_cuda_adapter.py"]
    return subprocess.CompletedProcess(
        cmd, 0, "", "............\nRan 12 tests in 0.04s\n\nOK\n")


def create_environment(tmp, *, source=True, cache_code=LLAMA_MINIMAL,
                       disk_code=DISK_MINIMAL, wheel=True):
    root = Path(tmp)
    origin = root / ".ruth-src/llama-cpp-python"
    package = origin / "llama_cpp"
    package.mkdir(parents=True)
    (package / "llama.py").write_text(cache_code)
    (package / "llama_cache.py").write_text(disk_code)
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_phase02_cuda_adapter.py").write_text("# test fixture")
    candidate = root / ".ruth-wheelhouse/python-dependencies-candidate"
    wheels = candidate / "wheels"
    wheels.mkdir(parents=True)
    file = wheels / "diskcache-5.6.3-py3-none-any.whl"
    with ZipFile(file, "w") as z:
        z.writestr("diskcache-5.6.3.dist-info/METADATA",
                   "Metadata-Version: 2.1\nName: diskcache\nVersion: 5.6.3\n")
    manifest = {
        "schema": "ruth-python-candidate-v1",
        "approved": False, "source_commit": SOURCE_COMMIT,
        "llama_cpp_commit": LLAMA_CPP_SUBMODULE_COMMIT,
        "files": [asdict(inspect_wheel(file))],
    }
    (candidate / "candidate-lock.json").write_text(json.dumps(manifest))
    env = SimpleNamespace(
        interpreter_in_project_venv=True, project_root=str(root),
        interpreter=str(root / ".venv/bin/python"),
    )
    plan = SimpleNamespace(
        source_verified=source, source_status="verified" if source else "missing",
        source_commit_actual=SOURCE_COMMIT,
        llama_cpp_commit_actual=LLAMA_CPP_SUBMODULE_COMMIT,
        source_directory=str(origin),
    )
    return env, plan, file


class SecurityGateTests(unittest.TestCase):
    def check(self, root, env, plan, run=fake_run):
        return review_diskcache_risk(
            project_root=root, environment=env, source_plan=plan, run=run
        )

    def test_known_upstream_cache_ast(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = root / "llama.py"
            b = root / "cache.py"
            a.write_text(LLAMA_MINIMAL)
            b.write_text(DISK_MINIMAL)
            self.assertEqual(inspect_upstream_cache_policy(a, b),
                             (True, True, None))

    def test_valid_controls_never_approve_vulnerable_package(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env, plan, _ = create_environment(td)
            r = self.check(root, env, plan)
            self.assertEqual(r.status, "controls_verified_residual_risk")
            self.assertTrue(r.mitigations_validated)
            self.assertEqual(r.adapter_tests_run, 12)
            self.assertTrue(r.adapter_contract_tests_passed)
            self.assertTrue(r.pinned_diskcache_wheel_verified)
            self.assertFalse(r.security_review_passed)
            self.assertFalse(r.lock_approved)
            self.assertFalse(r.installation_authorized)
            self.assertFalse(r.installation_started)
            self.assertFalse(r.native_build_started)
            self.assertFalse(r.fallback_automatically_selected)
            self.assertIn("malicious Python code", r.residual_risk)

    def test_upstream_constructor_enabling_cache_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            env, plan, _ = create_environment(
                td, cache_code=LLAMA_MINIMAL.replace(
                    "self.cache = None", "self.cache = object()"))
            r = self.check(Path(td), env, plan)
            self.assertEqual(r.status, "blocked")
            self.assertFalse(r.upstream_default_cache_disabled)

    def test_unknown_diskcache_interface_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            env, plan, _ = create_environment(
                td, disk_code=DISK_MINIMAL.replace("diskcache.Cache", "other.Cache"))
            r = self.check(Path(td), env, plan)
            self.assertFalse(r.disk_cache_attack_surface_identified)
            self.assertFalse(r.mitigations_validated)

    def test_modified_diskcache_wheel_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            env, plan, file = create_environment(td)
            with file.open("ab") as out:
                out.write(b"tampered")
            r = self.check(Path(td), env, plan)
            self.assertFalse(r.pinned_diskcache_wheel_verified)
            self.assertFalse(r.lock_approved)

    def test_wrong_checkout_commit_stops_prior_to_tests(self):
        with tempfile.TemporaryDirectory() as td:
            env, plan, _ = create_environment(td)
            plan.llama_cpp_commit_actual = "0" * 40
            r = self.check(Path(td), env, plan,
                           run=lambda *a, **kw: self.fail("no testing"))
            self.assertEqual(r.status, "blocked")
            self.assertFalse(r.verified_pinned_checkout)

    def test_wrong_interpreter_stops_prior_to_tests(self):
        with tempfile.TemporaryDirectory() as td:
            env, plan, _ = create_environment(td)
            env.interpreter = "/usr/bin/python3"
            r = self.check(Path(td), env, plan,
                           run=lambda *a, **kw: self.fail("no testing"))
            self.assertFalse(r.python_venv_verified)
            self.assertFalse(r.mitigations_validated)

    def test_partial_test_output_is_not_pass(self):
        with tempfile.TemporaryDirectory() as td:
            env, plan, _ = create_environment(td)
            def incomplete(cmd, **kwargs):
                return subprocess.CompletedProcess(
                    cmd, 0, "", "Ran 2 tests in 0.01s\n\nOK\n")
            r = self.check(Path(td), env, plan, incomplete)
            self.assertFalse(r.adapter_contract_tests_passed)
            self.assertEqual(r.adapter_tests_run, 2)

    def test_nonzero_test_exit_is_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            env, plan, _ = create_environment(td)
            def failing(cmd, **kw):
                return subprocess.CompletedProcess(
                    cmd, 1, "", "Ran 12 tests in 0.01s\n\nFAILED\n")
            r = self.check(Path(td), env, plan, failing)
            self.assertEqual(r.status, "blocked")

    def test_symlinked_candidate_manifest_is_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            env, plan, _ = create_environment(td)
            c = Path(td) / ".ruth-wheelhouse/python-dependencies-candidate"
            m = c / "candidate-lock.json"
            original = Path(td) / "outside.json"
            original.write_text(m.read_text())
            m.unlink()
            m.symlink_to(original)
            r = self.check(Path(td), env, plan)
            self.assertFalse(r.pinned_diskcache_wheel_verified)


if __name__ == "__main__":
    unittest.main()
