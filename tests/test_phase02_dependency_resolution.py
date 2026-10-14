"""Offline candidate-lock resolver tests. No network, pip install or CUDA."""
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from zipfile import ZipFile

from ruth_runtime.dependency_resolution import (
    ROOT_REQUIREMENTS, inspect_wheel, stage_dependency_candidate,
)


def env(root):
    return SimpleNamespace(
        interpreter_in_project_venv=True, project_root=str(root),
        python_version="3.12.3",
        interpreter=str(root / ".venv/bin/python"),
    )


def wheel(path, name, version="1.0"):
    label = name.replace("-", "_")
    path = path / (label + "-" + version + "-py3-none-any.whl")
    with ZipFile(path, "w") as z:
        z.writestr(
            label + "-" + version + ".dist-info/METADATA",
            "Metadata-Version: 2.1\nName: " + name + "\nVersion: " + version + "\n"
        )
    return path


class CandidateLockTests(unittest.TestCase):
    def test_without_explicit_approval_is_pure_readonly(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            r = stage_dependency_candidate(
                environment=env(root), source_verified=True,
                project_root=root, run=lambda *a, **k: self.fail("no pip"))
            self.assertEqual(r.status, "download_not_authorized")
            self.assertFalse((root / ".ruth-wheelhouse").exists())

    def test_source_unverified_blocks_even_with_approval(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            r = stage_dependency_candidate(
                authorize_network=True, environment=env(root),
                source_verified=False, project_root=root,
                run=lambda *a, **k: self.fail("no pip"))
            self.assertEqual(r.status, "blocked_preflight")
            self.assertFalse(r.downloaded)

    def test_resolves_transitives_as_wheels_and_writes_candidate_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            requested = []
            def downloader(command, **kwargs):
                requested.extend(command)
                self.assertEqual(command[:5], [
                    str(root / ".venv/bin/python"), "-m", "pip",
                    "--isolated", "download"
                ])
                self.assertIn("--only-binary=:all:", command)
                self.assertNotIn("install", command)
                self.assertEqual(set(ROOT_REQUIREMENTS).difference(command), set())
                output = Path(command[command.index("--dest") + 1])
                for name in ("numpy", "typing-extensions", "diskcache", "jinja2",
                             "MarkupSafe", "scikit-build-core", "packaging"):
                    wheel(output, name)
                return subprocess.CompletedProcess(command, 0, "", "")
            r = stage_dependency_candidate(
                authorize_network=True, environment=env(root),
                source_verified=True, project_root=root, run=downloader)
            self.assertEqual(r.status, "candidate_review_required", r.to_dict())
            self.assertTrue(r.downloaded)
            self.assertFalse(r.dependency_lock_ready)
            self.assertFalse(r.lock_approved)
            self.assertFalse(r.installation_started)
            self.assertFalse(r.compilation_started)
            self.assertFalse(r.fallback_automatically_selected)
            self.assertEqual(len(r.wheels), 7)
            self.assertEqual(len(set(x.sha256 for x in r.wheels)), 7)
            manifest = json.loads(Path(r.candidate_manifest).read_text())
            self.assertEqual(manifest["approved"], False)
            self.assertEqual(len(manifest["files"]), 7)
            self.assertFalse((root / ".ruth-wheelhouse" / ".candidate-download.lock").exists())

    def test_existing_candidate_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            existing = root / ".ruth-wheelhouse" / "python-dependencies-candidate"
            existing.mkdir(parents=True)
            (existing / "sentinel").write_text("keep")
            r = stage_dependency_candidate(
                authorize_network=True, environment=env(root),
                source_verified=True, project_root=root,
                run=lambda *a, **k: self.fail("no download"))
            self.assertEqual(r.status, "blocked_existing_candidate")
            self.assertEqual((existing / "sentinel").read_text(), "keep")

    def test_pip_failure_never_leaves_staging(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            r = stage_dependency_candidate(
                authorize_network=True, environment=env(root),
                source_verified=True, project_root=root,
                run=lambda command, **kwargs: subprocess.CompletedProcess(
                    command, 1, "", "failed"))
            self.assertEqual(r.status, "resolver_failed")
            self.assertFalse(r.lock_approved)
            self.assertEqual(list((root / ".ruth-wheelhouse").iterdir()), [])

    def test_prebuilt_llama_wheel_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            def fake_download(command, **kw):
                output = Path(command[command.index("--dest") + 1])
                for name in ("numpy", "typing-extensions", "diskcache", "jinja2",
                             "scikit-build-core", "llama-cpp-python"):
                    wheel(output, name)
                return subprocess.CompletedProcess(command, 0, "", "")
            r = stage_dependency_candidate(
                authorize_network=True, environment=env(root),
                source_verified=True, project_root=root, run=fake_download)
            self.assertEqual(r.status, "invalid_artifacts")
            self.assertFalse(r.fallback_automatically_selected)
            self.assertFalse((root / ".ruth-wheelhouse" / "python-dependencies-candidate").exists())

    def test_malformed_wheel_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            file = wheel(root, "numpy")
            with ZipFile(file, "w") as z:
                z.writestr("other-1.0.dist-info/METADATA", "Name: other\nVersion: 1.0\n")
            with self.assertRaises(ValueError):
                inspect_wheel(file)

    def test_wrong_venv_path_blocks_before_network(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wrong = env(root)
            wrong.interpreter = "/usr/bin/python3"
            r = stage_dependency_candidate(
                authorize_network=True, environment=wrong,
                source_verified=True, project_root=root,
                run=lambda *a, **k: self.fail("no pip"))
            self.assertEqual(r.status, "blocked_interpreter")


if __name__ == "__main__":
    unittest.main()
