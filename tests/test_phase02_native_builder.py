"""Offline tests for Ruth CUDA wheel staging, no compiler/GPU/network needed."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from ruth_runtime.dependency_resolution import inspect_wheel
from ruth_runtime.install_plan import SOURCE_COMMIT, LLAMA_CPP_SUBMODULE_COMMIT
from ruth_runtime.native_builder import (
    BUILD_OUTPUT, BUILD_ROOT, PIN_MANIFEST, build_native_cuda_wheel,
    inspect_cuda_wheel, verify_pinned_build_inputs,
)


def report(root, *, source_verified=True):
    source = root / ".ruth-src" / "llama-cpp-python"
    source.mkdir(parents=True, exist_ok=True)
    env = SimpleNamespace(
        interpreter_in_project_venv=True,
        project_root=str(root), python_version="3.12.3",
        interpreter=str(root / ".venv/bin/python"),
    )
    plan = SimpleNamespace(
        source_directory=str(source), source_status="verified",
        source_verified=source_verified, build_prerequisites_met=True,
        source_commit_actual=SOURCE_COMMIT,
        llama_cpp_commit_actual=LLAMA_CPP_SUBMODULE_COMMIT,
    )
    return env, plan


def wheel(path, *, cuda=True, llama=True, version="0.3.36"):
    file = path / f"llama_cpp_python-{version}-py3-none-linux_x86_64.whl"
    with ZipFile(file, "w") as z:
        z.writestr(f"llama_cpp_python-{version}.dist-info/METADATA",
                   f"Metadata-Version: 2.1\nName: llama-cpp-python\nVersion: {version}\n")
        if cuda:
            z.writestr("llama_cpp/lib/libggml-cuda.so", b"mock, not executable")
        if llama:
            z.writestr("llama_cpp/lib/libllama.so", b"mock, not executable")
    return file


def fake_copy(source, target):
    target.mkdir()
    (target / "pyproject.toml").write_text("[build-system]\n")
    return True


def pinned_tools(root):
    # Success fixture intentionally bypasses hash IO; verify_* tested separately.
    return True, (root / "a.whl", root / "b.whl", root / "c.whl"), None


class FakePip:
    def __init__(self, *, fail_stage=None, cuda=True):
        self.calls = []
        self.fail_stage = fail_stage
        self.cuda = cuda

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        assert cmd[:3] == [str(Path(kwargs["cwd"]) / ".venv/bin/python"),
                           "-m", "pip"]
        assert "--isolated" in cmd and "--no-index" in cmd
        assert "--no-deps" in cmd and "--no-cache-dir" in cmd
        assert "--allow-downloads" not in cmd
        assert "diskcache" not in " ".join(cmd)
        if "install" in cmd:
            assert "--target" in cmd
            assert "--no-compile" in cmd
            assert "--no-build-isolation" not in cmd
            if self.fail_stage == "tools":
                return subprocess.CompletedProcess(cmd, 1, "", "fail")
        elif "wheel" in cmd:
            assert "--no-build-isolation" in cmd
            assert kwargs["env"]["CMAKE_GENERATOR"] == "Unix Makefiles"
            assert kwargs["env"]["CMAKE_BUILD_PARALLEL_LEVEL"] == "2"
            assert "GGML_CUDA=ON" in kwargs["env"]["CMAKE_ARGS"]
            assert "CMAKE_CUDA_ARCHITECTURES=89" in kwargs["env"]["CMAKE_ARGS"]
            assert kwargs["env"]["PYTHONNOUSERSITE"] == "1"
            if self.fail_stage == "compile":
                return subprocess.CompletedProcess(cmd, 1, "", "CUDA error")
            wheel(Path(cmd[cmd.index("--wheel-dir") + 1]), cuda=self.cuda)
        else:
            raise AssertionError("Only isolated staged pip install and wheel allowed")
        return subprocess.CompletedProcess(cmd, 0, "", "")


class NativeBuilderTests(unittest.TestCase):
    def build(self, root, *, pip=None, source_ok=True, pins=pinned_tools,
              copier=fake_copy, environment=None):
        env, plan = report(root, source_verified=source_ok)
        cuda = root / "nvcc"
        cuda.touch()
        with patch("ruth_runtime.native_builder.NVCC", cuda):
            return build_native_cuda_wheel(
                project_root=root, environment=environment or env,
                source_plan=plan, run=pip or FakePip(),
                source_copy=copier, pins_check=pins,
                disk_usage=lambda _: SimpleNamespace(free=30 * 1024**3),
            )

    def test_builds_cuda_wheel_in_staging_not_project_venv(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fake = FakePip()
            result = self.build(root, pip=fake)
            self.assertEqual(result.status, "native_cuda_wheel_built", result.to_dict())
            self.assertTrue(result.pinned_wheels_verified)
            self.assertTrue(result.ggml_cuda_library_packaged)
            self.assertFalse(result.installed_in_ruth_venv)
            self.assertFalse(result.gpu_offload_executed)
            self.assertFalse(result.inference_executed)
            self.assertTrue(result.native_build_started)
            self.assertFalse(result.fallback_automatically_selected)
            self.assertEqual(len(fake.calls), 2)
            manifest = json.loads((root / BUILD_ROOT / BUILD_OUTPUT /
                                   "build-result.json").read_text())
            self.assertEqual(manifest["wheel_sha256"], result.wheel_sha256)
            self.assertTrue(Path(result.wheel_file).exists())
            self.assertFalse((root / BUILD_ROOT / ".native-cuda-build.lock").exists())

    def test_source_checkout_is_not_modified(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / ".ruth-src" / "llama-cpp-python"
            source.mkdir(parents=True)
            marker = source / "sentinel.txt"
            marker.write_text("unmodified")
            self.build(root)
            self.assertEqual(marker.read_text(), "unmodified")

    def test_tools_stage_failure_never_promotes_wheel(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = self.build(root, pip=FakePip(fail_stage="tools"))
            self.assertEqual(result.status, "build_tools_failed")
            self.assertFalse(result.native_build_started)
            self.assertFalse((root / BUILD_ROOT / BUILD_OUTPUT).exists())

    def test_compiler_failure_never_promotes_wheel_or_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = self.build(root, pip=FakePip(fail_stage="compile"))
            self.assertEqual(result.status, "cuda_compile_failed")
            self.assertTrue(result.native_build_started)
            self.assertFalse(result.fallback_automatically_selected)
            self.assertFalse((root / BUILD_ROOT / BUILD_OUTPUT).exists())

    def test_cpu_only_artifact_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = self.build(root, pip=FakePip(cuda=False))
            self.assertEqual(result.status, "cuda_backend_missing")
            self.assertFalse(result.ggml_cuda_library_packaged)
            self.assertFalse((root / BUILD_ROOT / BUILD_OUTPUT).exists())

    def test_invalid_source_never_calls_pip(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fake = FakePip()
            result = self.build(root, pip=fake, source_ok=False)
            self.assertEqual(result.status, "blocked_preflight")
            self.assertEqual(fake.calls, [])

    def test_wrong_project_interpreter_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env, _ = report(root)
            env.interpreter = "/usr/bin/python3"
            result = self.build(root, environment=env)
            self.assertEqual(result.status, "blocked_preflight")

    def test_missing_pinned_wheels_never_starts_build_tools(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fake = FakePip()
            result = self.build(
                root, pip=fake,
                pins=lambda r: (False, (), "hash mismatch"))
            self.assertEqual(result.status, "wheel_pins_mismatch")
            self.assertEqual(fake.calls, [])

    def test_previous_build_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            folder = root / BUILD_ROOT / BUILD_OUTPUT
            folder.mkdir(parents=True)
            marker = folder / "sentinel"
            marker.write_text("keep")
            result = self.build(root)
            self.assertEqual(result.status, "output_already_exists")
            self.assertEqual(marker.read_text(), "keep")

    def test_symlink_staging_is_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            outside = root / "outside"
            outside.mkdir()
            (root / BUILD_ROOT).symlink_to(outside, target_is_directory=True)
            result = self.build(root)
            self.assertEqual(result.status, "output_already_exists")
            self.assertEqual(list(outside.iterdir()), [])

    def test_cuda_wheel_rejects_wrong_binding_version(self):
        with tempfile.TemporaryDirectory() as td:
            result = inspect_cuda_wheel(wheel(Path(td), version="0.3.35"))
            self.assertFalse(result[0])

    def test_pinned_validator_rejects_missing_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            result = verify_pinned_build_inputs(Path(td))
            self.assertFalse(result[0])

    def test_pinned_validator_rejects_tampered_candidate_before_pip(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pin = root / PIN_MANIFEST
            pin.parent.mkdir(parents=True)
            pin.write_text('{"schema":"ruth-cuda-build-inputs-v1","files":[]}')
            result = verify_pinned_build_inputs(root)
            self.assertFalse(result[0])

    def test_pinned_wheel_file_compatible_metadata_inspection(self):
        with tempfile.TemporaryDirectory() as td:
            test_wheel = wheel(Path(td))
            self.assertEqual(inspect_wheel(test_wheel).version, "0.3.36")
            self.assertTrue(inspect_cuda_wheel(test_wheel)[0])


if __name__ == "__main__":
    unittest.main()
