"""Offline tests for automatic, pinned Git acquisition (no network or pip)."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from ruth_runtime.dependencies import DependencyCheck, DependencyReport
from ruth_runtime.install_plan import (
    build_installation_plan, SOURCE_COMMIT, LLAMA_CPP_SUBMODULE_COMMIT, SOURCE_REPOSITORY,
)
from ruth_runtime.source_build_plan import create_source_build_plan
from ruth_runtime.source_acquisition import acquire_pinned_sources
from ruth_runtime.toolkit import ToolkitCheck

NAMES = (
    "GNU/Linux x86_64", "Python >=3.10 (Ruth .venv)", "Ruth .venv",
    "nvidia-smi", "GPU/driver NVIDIA", "CUDA supportata dal driver",
)


def tool(name):
    return "/usr/bin/" + name


def installation():
    checks = DependencyReport(tuple(DependencyCheck(x, True, "ok", "fixture") for x in NAMES))
    cuda = ToolkitCheck(
        status="ok", expected_version="13.2", detected_version="13.2",
        nvcc_path="/usr/local/cuda-13.2/bin/nvcc",
        cmake_path="/usr/bin/cmake", cpp_compiler_path="/usr/bin/c++",
        cuda_architecture="89", native_build_prerequisites=True,
        detail="mock", remediation=None,
    )
    return build_installation_plan(
        dependencies=checks, toolkit=cuda,
        libc=("glibc", "2.39"), free_bytes=50_000_000_000,
        interpreter="/repo/.venv/bin/python",
    )


class GitRunner:
    """Simulate Git's side effects strictly in the injected temp folder."""
    def __init__(self, *, fail_stage=None, wrong_submodule=False):
        self.calls = []
        self.fail_stage = fail_stage
        self.wrong_submodule = wrong_submodule
        self.downloads = 0

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        cmd = argv[0]
        if cmd.startswith("/usr/bin/"):
            first = cmd.split("/")[-1]
            return subprocess.CompletedProcess(
                argv, 0, {
                    "git": "git version 2.43.0",
                    "cmake": "cmake version 3.28.3",
                    "c++": "c++ (Ubuntu 13.3.0) 13.3.0",
                    "make": "GNU Make 4.3",
                }[first] + "\n", "")
        if cmd != "git":
            raise AssertionError("unexpected non-git action")
        if "-c" in argv:
            self.assert_safe_write_command(argv, kwargs)
            if "clone" in argv:
                self.downloads += 1
                index = 1
                if self.fail_stage == index:
                    return subprocess.CompletedProcess(argv, 128, "", "mock failure")
                # git clone uses an existing empty temporary destination.
                target = Path(argv[-1])
                (target / ".git").mkdir()
            elif "checkout" in argv:
                index = 2
                if self.fail_stage == index:
                    return subprocess.CompletedProcess(argv, 128, "", "mock failure")
                directory = Path(argv[argv.index("-C") + 1])
                (directory / "pyproject.toml").write_text("[build-system]\n")
                (directory / "vendor" / "llama.cpp").mkdir(parents=True)
            elif "submodule" in argv:
                index = 3
                if self.fail_stage == index:
                    return subprocess.CompletedProcess(argv, 128, "", "mock failure")
                directory = Path(argv[argv.index("-C") + 1])
                (directory / "vendor" / "llama.cpp" / "CMakeLists.txt").touch()
            else:
                raise AssertionError(f"Unexpected writing command: {argv}")
            return subprocess.CompletedProcess(argv, 0, "", "")

        if argv[:2] != ["git", "-C"]:
            raise AssertionError("unexpected Git read format")
        directory = Path(argv[2])
        action = argv[3]
        if action == "rev-parse":
            reply = (
                ("0" * 40 if self.wrong_submodule else LLAMA_CPP_SUBMODULE_COMMIT)
                if directory.name == "llama.cpp" else SOURCE_COMMIT
            )
        elif action == "remote":
            reply = SOURCE_REPOSITORY + ".git"
        elif action == "status":
            reply = ""
        else:
            raise AssertionError(f"Unexpected read action {argv}")
        return subprocess.CompletedProcess(argv, 0, reply + ("\n" if reply else ""), "")

    def assert_safe_write_command(self, argv, kwargs):
        assert kwargs.get("timeout", 0) > 0
        assert kwargs.get("check") is False
        assert kwargs.get("env", {}).get("GIT_TERMINAL_PROMPT") == "0"
        assert "protocol.file.allow=never" in argv
        assert "protocol.ext.allow=never" in argv
        assert not any(word in argv for word in ("pip", "install", "apt", "curl"))


def plan(root, runner, *, toolkit_ok=True):
    p = installation()
    if not toolkit_ok:
        import dataclasses
        p = dataclasses.replace(p, primary_prerequisites_met=False)
    return create_source_build_plan(
        install_plan=p, project_root=root, which=tool, run=runner,
    )


class AutomaticAcquisitionTests(unittest.TestCase):
    def test_automatic_clone_checkout_and_verify_in_one_step(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runner = GitRunner()
            before = plan(root, runner)
            self.assertEqual(before.source_status, "missing")
            result = acquire_pinned_sources(before, project_root=root, run=runner)
            self.assertEqual(result.status, "acquired_verified", result.to_dict())
            self.assertEqual(result.source_status_after, "verified")
            self.assertEqual(result.source_commit, SOURCE_COMMIT)
            self.assertEqual(result.llama_cpp_commit, LLAMA_CPP_SUBMODULE_COMMIT)
            self.assertTrue(result.download_performed)
            self.assertFalse(result.build_started)
            self.assertFalse(result.python_install_started)
            self.assertFalse(result.fallback_automatically_selected)
            self.assertFalse(result.changed_existing_checkout)
            self.assertEqual(runner.downloads, 1)
            self.assertTrue(Path(result.source_directory).is_dir())
            self.assertFalse((root / ".ruth-src" / ".ruth-source-acquire.lock").exists())
            after = plan(root, runner)
            self.assertTrue(after.source_verified)
            self.assertFalse(after.ready_to_build)
            self.assertFalse(after.dependency_lock_ready)

    def test_idempotence_existing_verified_checkout_is_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runner = GitRunner()
            first = acquire_pinned_sources(plan(root, runner), project_root=root, run=runner)
            self.assertTrue(first.source_verified)
            count = runner.downloads
            second = acquire_pinned_sources(plan(root, runner), project_root=root, run=runner)
            self.assertEqual(second.status, "already_verified")
            self.assertEqual(runner.downloads, count)
            self.assertFalse(second.download_performed)

    def test_partial_clone_failure_never_promotes_or_leaves_staging(self):
        for stage in (1, 2, 3):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                runner = GitRunner(fail_stage=stage)
                out = acquire_pinned_sources(plan(root, runner), project_root=root,
                                             run=runner)
                self.assertEqual(out.status, "failed")
                self.assertFalse(out.source_verified)
                self.assertFalse((root / ".ruth-src" / "llama-cpp-python").exists())
                self.assertEqual(list((root / ".ruth-src").iterdir()), [])

    def test_wrong_submodule_commit_never_promoted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runner = GitRunner(wrong_submodule=True)
            out = acquire_pinned_sources(plan(root, runner), project_root=root, run=runner)
            self.assertEqual(out.status, "failed")
            self.assertEqual(out.source_status_after, "commit_mismatch")
            self.assertFalse((root / ".ruth-src" / "llama-cpp-python").exists())

    def test_existing_unverified_checkout_never_edited(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / ".ruth-src" / "llama-cpp-python"
            target.mkdir(parents=True)
            (target / "sentinel.txt").write_text("preserve")
            runner = GitRunner()
            out = acquire_pinned_sources(plan(root, runner), project_root=root, run=runner)
            self.assertEqual(out.status, "blocked")
            self.assertEqual((target / "sentinel.txt").read_text(), "preserve")
            self.assertEqual(runner.downloads, 0)

    def test_missing_build_prerequisites_prevent_download(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runner = GitRunner()
            before = plan(root, runner, toolkit_ok=False)
            out = acquire_pinned_sources(before, project_root=root, run=runner)
            self.assertEqual(out.status, "blocked")
            self.assertEqual(runner.downloads, 0)
            self.assertFalse((root / ".ruth-src").exists())

    def test_symlink_source_area_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            external = root / "unrelated"
            external.mkdir()
            (root / ".ruth-src").symlink_to(external, target_is_directory=True)
            runner = GitRunner()
            before = plan(root, runner)
            out = acquire_pinned_sources(before, project_root=root, run=runner)
            self.assertEqual(out.status, "blocked")
            self.assertEqual(runner.downloads, 0)
            self.assertEqual(list(external.iterdir()), [])

    def test_stale_lock_blocks_without_modifying_it(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            marker = root / ".ruth-src" / ".ruth-source-acquire.lock"
            marker.mkdir(parents=True)
            runner = GitRunner()
            out = acquire_pinned_sources(plan(root, runner), project_root=root, run=runner)
            self.assertEqual(out.status, "blocked")
            self.assertTrue(marker.exists())
            self.assertEqual(runner.downloads, 0)

    def test_path_spoof_prevents_download(self):
        with tempfile.TemporaryDirectory() as td:
            import dataclasses
            root = Path(td)
            runner = GitRunner()
            before = plan(root, runner)
            bad = dataclasses.replace(before, source_directory=str(root / "outside"))
            out = acquire_pinned_sources(bad, project_root=root, run=runner)
            self.assertEqual(out.status, "blocked")
            self.assertEqual(runner.downloads, 0)

    def test_explicit_cli_prepare_runs_automation_without_git_instructions(self):
        import dataclasses
        from io import StringIO
        from ruth_runtime.source_build_plan import main
        from ruth_runtime.source_acquisition import SourceAcquisitionResult
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            before = plan(root, GitRunner())
            after = dataclasses.replace(before, source_status="verified",
                                        source_verified=True)
            acquired = SourceAcquisitionResult(
                status="acquired_verified", source_directory=before.source_directory,
                source_status_before="missing", source_status_after="verified",
                source_verified=True, source_commit=SOURCE_COMMIT,
                llama_cpp_commit=LLAMA_CPP_SUBMODULE_COMMIT,
                download_performed=True, changed_existing_checkout=False,
                fallback_automatically_selected=False, build_started=False,
                python_install_started=False, error=None,
            )
            with patch("ruth_runtime.source_build_plan.create_source_build_plan",
                       side_effect=[before, after]), \
                 patch("ruth_runtime.source_acquisition.acquire_pinned_sources",
                       return_value=acquired) as acquire, \
                 patch("sys.argv", ["source_build_plan", "--prepare"]), \
                 patch("sys.stdout", new_callable=StringIO) as output:
                self.assertEqual(main(), 0)
                acquire.assert_called_once_with(before)
                self.assertIn('"operation": "prepare_sources"', output.getvalue())
                self.assertIn('"source_verified": true', output.getvalue())

    def test_no_network_in_default_cli_plan(self):
        from ruth_runtime.source_build_plan import main
        from io import StringIO
        from unittest.mock import patch
        runner = GitRunner()
        with tempfile.TemporaryDirectory() as td:
            ready = plan(Path(td), runner)
            with patch("ruth_runtime.source_build_plan.create_source_build_plan", return_value=ready), \
                 patch("sys.argv", ["source_build_plan"]), \
                 patch("sys.stdout", new_callable=StringIO) as captured:
                self.assertEqual(main(), 0)
                self.assertIn('"source_status": "missing"', captured.getvalue())
            self.assertEqual(runner.downloads, 0)


if __name__ == "__main__":
    unittest.main()
