"""Fully offline tests for the pinned CUDA native source build preflight."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from ruth_runtime.dependencies import DependencyCheck, DependencyReport
from ruth_runtime.install_plan import (
    SOURCE_COMMIT, LLAMA_CPP_SUBMODULE_COMMIT, SOURCE_REPOSITORY,
    build_installation_plan,
)
from ruth_runtime.source_build_plan import (
    MIN_CMAKE, SOURCE_RELATIVE_PATH, create_source_build_plan,
    inspect_checkout, parse_cmake_version,
)
from ruth_runtime.toolkit import ToolkitCheck


REQUIRED = (
    "GNU/Linux x86_64", "Python >=3.10 (Ruth .venv)",
    "Ruth .venv", "nvidia-smi", "GPU/driver NVIDIA",
    "CUDA supportata dal driver",
)


def pinned_install_plan(*, toolkit_ready=True):
    deps = DependencyReport(tuple(
        DependencyCheck(name, True, "ok", "mock") for name in REQUIRED
    ))
    t = ToolkitCheck(
        status="ok" if toolkit_ready else "missing",
        expected_version="13.2",
        detected_version="13.2" if toolkit_ready else None,
        nvcc_path="/usr/local/cuda-13.2/bin/nvcc" if toolkit_ready else None,
        cmake_path="/usr/bin/cmake", cpp_compiler_path="/usr/bin/c++",
        cuda_architecture="89",
        native_build_prerequisites=toolkit_ready,
        detail="mock", remediation=None,
    )
    return build_installation_plan(
        dependencies=deps, toolkit=t, libc=("glibc", "2.39"),
        free_bytes=60 * (1024**3), interpreter="/repo/.venv/bin/python",
    )


def available(name):
    return "/usr/bin/" + name


def fake_run(argv, **kwargs):
    if argv[0] == "/usr/bin/cmake":
        out = "cmake version 3.28.3"
    elif argv[0] == "/usr/bin/c++":
        out = "c++ (Ubuntu 13.3.0) 13.3.0"
    elif argv[0] == "/usr/bin/git":
        out = "git version 2.43.0"
    elif argv[0] == "/usr/bin/make":
        out = "GNU Make 4.3"
    elif argv[0] == "git":
        if argv[-2:] == ["rev-parse", "HEAD"]:
            out = (LLAMA_CPP_SUBMODULE_COMMIT
                   if argv[2].endswith("llama.cpp") else SOURCE_COMMIT)
        elif argv[-3:] == ["remote", "get-url", "origin"]:
            out = SOURCE_REPOSITORY + ".git"
        elif argv[-2:] == ["status", "--porcelain"]:
            raise AssertionError("Expected --untracked-files=all")
        elif argv[3] == "status":
            out = ""
        else:
            raise AssertionError(f"unexpected git command: {argv}")
    else:
        raise AssertionError(f"unexpected command: {argv}")
    return subprocess.CompletedProcess(argv, 0, out + ("\n" if out else ""), "")


class SourceBuildPlanTests(unittest.TestCase):
    def plan(self, *, which=available, run=fake_run, toolkit_ready=True,
             project_root=None):
        return create_source_build_plan(
            install_plan=pinned_install_plan(toolkit_ready=toolkit_ready),
            project_root=project_root or Path("/repo"),
            which=which, run=run,
        )

    def test_cmake_minimum_and_parser(self):
        self.assertEqual(MIN_CMAKE, (3, 21, 0))
        self.assertEqual(parse_cmake_version("cmake version 3.28.3\n"), (3, 28, 3))
        self.assertEqual(parse_cmake_version("cmake version 3.21"), (3, 21, 0))
        self.assertIsNone(parse_cmake_version("cmake: unknown version"))

    def test_toolchain_ok_checkout_missing_is_not_build_ready(self):
        with tempfile.TemporaryDirectory() as td:
            plan = self.plan(project_root=Path(td))
            self.assertTrue(plan.build_prerequisites_met)
            self.assertEqual(plan.source_status, "missing")
            self.assertFalse(plan.source_verified)
            self.assertFalse(plan.ready_to_build)
            self.assertFalse(plan.build_authorized)
            self.assertFalse(plan.download_authorized)
            self.assertEqual(plan.strategy, "cuda-toolkit-native-build")
            self.assertFalse(plan.fallback_automatically_selected)

    def test_pinned_checkout_instructions_and_cuda_arch(self):
        p = self.plan()
        self.assertEqual(p.source_commit_expected, SOURCE_COMMIT)
        self.assertEqual(p.llama_cpp_commit_expected, LLAMA_CPP_SUBMODULE_COMMIT)
        self.assertEqual(p.checkout_commands[0], "mkdir -p .ruth-src")
        self.assertIn("--no-checkout", p.checkout_commands[1])
        self.assertIn(SOURCE_COMMIT, p.checkout_commands[2])
        self.assertIn("submodule update --init --recursive", p.checkout_commands[3])
        self.assertIn("GGML_CUDA=ON", p.cuda_cmake_args)
        self.assertIn("CMAKE_CUDA_ARCHITECTURES=89", p.cuda_cmake_args)
        self.assertEqual(p.source_directory, str(Path("/repo") / SOURCE_RELATIVE_PATH))
        self.assertTrue(any("scikit-build-core" in r for r in p.python_build_requirements))

    def test_missing_git_does_not_attempt_git_operations(self):
        plan = self.plan(which=lambda name: None if name == "git" else available(name))
        self.assertEqual(plan.tools[0].status, "missing")
        self.assertFalse(plan.build_prerequisites_met)
        self.assertIn("install git", plan.tools[0].remediation)

    def test_old_cmake_blocks_native_build(self):
        def old_cmake(argv, **kwargs):
            if argv[0] == "/usr/bin/cmake":
                return subprocess.CompletedProcess(argv, 0, "cmake version 3.18.4", "")
            return fake_run(argv, **kwargs)
        plan = self.plan(run=old_cmake)
        self.assertEqual(plan.tools[1].status, "incompatible")
        self.assertFalse(plan.build_prerequisites_met)
        self.assertIn("3.21", plan.tools[1].remediation)

    def test_missing_cpp_compiler_blocks_native_build(self):
        plan = self.plan(
            which=lambda n: None if n == "c++" else available(n))
        self.assertEqual(plan.tools[2].status, "missing")
        self.assertFalse(plan.build_prerequisites_met)

    def test_missing_make_blocks_build_without_install(self):
        p = self.plan(
            which=lambda n: None if n == "make" else available(n))
        self.assertEqual(p.tools[3].name, "GNU Make")
        self.assertEqual(p.tools[3].status, "missing")
        self.assertIn("sudo apt install make", p.tools[3].remediation)
        self.assertFalse(p.build_prerequisites_met)
        self.assertFalse(p.build_authorized)

    def test_missing_toolkit_never_activates_fallback(self):
        plan = self.plan(toolkit_ready=False)
        self.assertFalse(plan.build_prerequisites_met)
        self.assertEqual(plan.toolkit_status, "missing")
        self.assertFalse(plan.fallback_automatically_selected)

    def test_pinned_nested_git_checkout(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / ".ruth-src" / "llama-cpp-python"
            child = source / "vendor" / "llama.cpp"
            child.mkdir(parents=True)
            (source / "pyproject.toml").touch()
            (child / "CMakeLists.txt").touch()
            report = self.plan(project_root=Path(td))
            self.assertEqual(report.source_status, "verified")
            self.assertEqual(report.source_commit_actual, SOURCE_COMMIT)
            self.assertEqual(report.llama_cpp_commit_actual,
                             LLAMA_CPP_SUBMODULE_COMMIT)
            self.assertTrue(report.source_verified)
            self.assertFalse(report.ready_to_build)  # wheel lock not ready

    def test_unexpected_submodule_commit_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source"
            child = source / "vendor" / "llama.cpp"
            child.mkdir(parents=True)
            def wrong(argv, **kwargs):
                if argv[0] == "git" and argv[-2:] == ["rev-parse", "HEAD"] \
                        and argv[2].endswith("llama.cpp"):
                    return subprocess.CompletedProcess(argv, 0, "0" * 40, "")
                return fake_run(argv, **kwargs)
            status, parent, nested = inspect_checkout(
                source, git_available=True, run=wrong)
            self.assertEqual(status, "commit_mismatch")
            self.assertEqual(parent, SOURCE_COMMIT)
            self.assertEqual(nested, "0" * 40)

    def test_dirty_tree_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source"
            child = source / "vendor" / "llama.cpp"
            child.mkdir(parents=True)
            def dirty(argv, **kwargs):
                if argv[0] == "git" and argv[3] == "status":
                    return subprocess.CompletedProcess(argv, 0, " M pyproject.toml", "")
                return fake_run(argv, **kwargs)
            status, _, _ = inspect_checkout(
                source, git_available=True, run=dirty)
            self.assertEqual(status, "dirty")

    def test_symlink_checkout_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            real = root / "actual"
            real.mkdir()
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            status, _, _ = inspect_checkout(alias, git_available=True,
                                            run=lambda *a, **k: self.fail("no git"))
            self.assertEqual(status, "unsafe_symlink")

    def test_git_probe_failure_nonfatal(self):
        def broken(argv, **kwargs):
            if argv[0] == "/usr/bin/git":
                raise OSError("broken")
            return fake_run(argv, **kwargs)
        p = self.plan(run=broken)
        self.assertEqual(p.tools[0].status, "probe_failed")
        self.assertFalse(p.build_prerequisites_met)

    def test_serializable_report_no_install_actions(self):
        import json
        p = self.plan()
        document = json.loads(json.dumps(p.to_dict()))
        self.assertFalse(document["ready_to_build"])
        self.assertFalse(document["build_authorized"])
        self.assertFalse(document["download_authorized"])
        self.assertFalse(document["fallback_automatically_selected"])
        self.assertEqual(document["toolkit_status"], "ok")


if __name__ == "__main__":
    unittest.main()
